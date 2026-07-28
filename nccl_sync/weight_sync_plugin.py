from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Sequence

import ray
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port


class AbstractRolloutWeightSync(ABC):
    """Reusable vLLM-side disk reload + NCCL receive implementation."""

    def _init_rollout_weight_sync_state(self) -> None:
        self._weight_transfer_initialized = False

    @abstractmethod
    def get_vllm_engine(self) -> Any:
        """Return the current vLLM LLM/engine object."""

    @abstractmethod
    def rebuild_vllm_engine(self, model_path: str) -> None:
        """Destroy and recreate vLLM from a disk checkpoint."""

    def get_vllm_world_size(self) -> int:
        engine = self.get_vllm_engine()
        if hasattr(engine, "get_world_size"):
            return int(engine.get_world_size())
        if hasattr(self, "parallel_size"):
            return int(self.parallel_size)
        raise RuntimeError("Cannot determine vLLM worker count.")

    def init_weight_transfer(
        self,
        master_address: str,
        master_port: int,
        transfer_world_size: int,
    ) -> bool:
        self.get_vllm_engine().init_weight_transfer_engine(
            {
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": transfer_world_size,
                }
            }
        )
        self._weight_transfer_initialized = True
        return True

    def receive_weights(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        packed: bool = True,
    ) -> bool:
        if not self._weight_transfer_initialized:
            raise RuntimeError("Call Controller.init_nccl_sync() first.")

        engine = self.get_vllm_engine()
        #engine.start_weight_update()
        try:
            engine.update_weights(
                {
                    "update_info": {
                        "names": names,
                        "dtype_names": dtype_names,
                        "shapes": shapes,
                        "packed": packed,
                    }
                }
            )
        finally:
            #engine.finish_weight_update()
            pass
        return True

    def reload_from_disk(self, model_path: str) -> bool:
        self.rebuild_vllm_engine(model_path)
        self._weight_transfer_initialized = False
        return True


class AbstractFSDPWeightSync(ABC):
    """Reusable FSDP-side disk save + NCCL send implementation."""

    def _init_fsdp_weight_sync_state(self) -> None:
        self._weight_transfer_group = None
        self._weight_transfer_endpoint = None
        self._pending_nccl_state = None

    @property
    @abstractmethod
    def fsdp_rank(self) -> int:
        pass

    @abstractmethod
    def get_fsdp_model(self) -> FSDP:
        pass

    @abstractmethod
    def save_model_config(self, save_dir: str) -> None:
        pass

    @abstractmethod
    def save_tokenizer(self, save_dir: str) -> None:
        pass

    def create_weight_transfer_endpoint(self) -> dict[str, Any] | None:
        if self.fsdp_rank != 0:
            return None
        if self._weight_transfer_endpoint is None:
            self._weight_transfer_endpoint = {
                "master_address": get_ip(),
                "master_port": get_open_port(),
            }
        return self._weight_transfer_endpoint

    def init_weight_transfer(
        self,
        master_address: str,
        master_port: int,
        transfer_world_size: int,
    ) -> bool:
        if self.fsdp_rank != 0:
            return True
        self._weight_transfer_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": transfer_world_size,
            }
        )
        return True

    def _full_state_dict(self, *, offload_to_cpu: bool) -> dict[str, torch.Tensor]:
        cfg = FullStateDictConfig(
            offload_to_cpu=offload_to_cpu,
            rank0_only=True,
        )
        model = self.get_fsdp_model()
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()

    def save_checkpoint(self, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        state = self._full_state_dict(offload_to_cpu=True)
        if self.fsdp_rank == 0:
            torch.save(state, os.path.join(save_dir, "pytorch_model.bin"))
            self.save_model_config(save_dir)
            self.save_tokenizer(save_dir)
        dist.barrier()
        return save_dir

    def prepare_nccl_weights(self) -> dict[str, Any] | None:
        # Every FSDP rank must enter this full-state collective.
        state = self._full_state_dict(offload_to_cpu=False)
        if self.fsdp_rank != 0:
            return None

        self._pending_nccl_state = state
        tensor_items = [
            (name, tensor)
            for name, tensor in state.items()
            if torch.is_tensor(tensor)
        ]
        return {
            "names": [name for name, _ in tensor_items],
            "dtype_names": [str(t.dtype).removeprefix("torch.") for _, t in tensor_items],
            "shapes": [list(t.shape) for _, t in tensor_items],
        }

    def broadcast_prepared_weights(self, packed: bool = True) -> bool:
        if self.fsdp_rank != 0:
            return True
        if self._weight_transfer_group is None:
            raise RuntimeError("Call Controller.init_nccl_sync() first.")
        if self._pending_nccl_state is None:
            raise RuntimeError("Call prepare_nccl_weights() first.")

        iterator = (
            (name, tensor)
            for name, tensor in self._pending_nccl_state.items()
            if torch.is_tensor(tensor)
        )
        args = NCCLTrainerSendWeightsArgs(
            group=self._weight_transfer_group,
            packed=packed,
        )
        try:
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iterator,
                trainer_args=args,
            )
        finally:
            self._pending_nccl_state = None
            torch.cuda.empty_cache()
        return True


class AbstractWeightSyncController(ABC):
    """Reusable Ray orchestration for disk saves and NCCL synchronization."""

    def _init_weight_sync_controller_state(self) -> None:
        self._nccl_sync_initialized = False

    @property
    @abstractmethod
    def rollout_actor(self) -> Any:
        pass

    @property
    @abstractmethod
    def fsdp_actors(self) -> Sequence[Any]:
        pass

    def init_nccl_sync(self) -> bool:
        if self._nccl_sync_initialized:
            return True

        rank0 = self.fsdp_actors[0]
        endpoint = ray.get(rank0.create_weight_transfer_endpoint.remote())
        vllm_world_size = ray.get(self.rollout_actor.get_vllm_world_size.remote())
        transfer_world_size = 1 + vllm_world_size

        # Both sides must join concurrently.
        trainer_ref = rank0.init_weight_transfer.remote(
            endpoint["master_address"],
            endpoint["master_port"],
            transfer_world_size,
        )
        rollout_ref = self.rollout_actor.init_weight_transfer.remote(
            endpoint["master_address"],
            endpoint["master_port"],
            transfer_world_size,
        )
        ray.get([trainer_ref, rollout_ref])
        self._nccl_sync_initialized = True
        return True

    def nccl_sync(self, packed: bool = True) -> bool:
        if not self._nccl_sync_initialized:
            raise RuntimeError("Call init_nccl_sync() once before nccl_sync().")

        # Every FSDP rank participates in reconstructing the full state.
        metadata_by_rank = ray.get([
            actor.prepare_nccl_weights.remote()
            for actor in self.fsdp_actors
        ])
        metadata = metadata_by_rank[0]
        if metadata is None:
            raise RuntimeError("FSDP rank 0 returned no weight metadata.")

        # Sender and receiver must run concurrently.
        receive_ref = self.rollout_actor.receive_weights.remote(
            metadata["names"],
            metadata["dtype_names"],
            metadata["shapes"],
            packed,
        )
        send_ref = self.fsdp_actors[0].broadcast_prepared_weights.remote(packed)
        ray.get([receive_ref, send_ref])
        return True

    def save_checkpoint(self, save_dir: str) -> str:
        ray.get([
            actor.save_checkpoint.remote(save_dir)
            for actor in self.fsdp_actors
        ])
        return save_dir

    def save_and_reload(self, save_dir: str) -> str:
        self.save_checkpoint(save_dir)
        ray.get(self.rollout_actor.reload_from_disk.remote(save_dir))
        self._nccl_sync_initialized = False
        return save_dir