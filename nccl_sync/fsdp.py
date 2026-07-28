from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Iterator, Sequence

import ray
import torch

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

class FSDPActor(ABC):
    """
    Abstract trainer-side NCCL weight sender.

    Every FSDP actor/rank must execute send_weights().

    Only FSDP rank 0 joins the separate trainer-to-vLLM NCCL group as
    transfer rank 0. However, every FSDP rank must participate in gathering
    full parameters.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
    ) -> None:
        self.rank = rank
        self.world_size = world_size

        self._weight_transfer_group: Any | None = None
        self._weight_transfer_initialized = False

    # -------------------------------------------------------------------------
    # Required subclass hooks
    # -------------------------------------------------------------------------

    @abstractmethod
    def iter_full_named_parameters(
        self,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Yield full, unsharded parameters as:

            (vllm_compatible_name, full_cuda_tensor)

        All FSDP ranks must execute the same collective gather operations in
        the same order.

        On nonzero ranks, an implementation may perform the gathers without
        yielding anything.

        Requirements for tensors yielded by rank 0:
        - full/unsharded
        - CUDA tensor
        - detached from autograd
        - names compatible with vLLM's model loader
        """

    @abstractmethod
    def get_weight_metadata(self) -> dict[str, Any]:
        """
        Return metadata captured from the unwrapped model before sharding:

            {
                "names": [...],
                "shapes": [[...], ...],
                "dtype_names": ["bfloat16", ...],
            }

        Only the rank-0 result is used by the controller.
        """

    # -------------------------------------------------------------------------
    # NCCL rendezvous
    # -------------------------------------------------------------------------

    def create_weight_transfer_endpoint(self) -> dict[str, Any]:
        """
        Create the TCP rendezvous endpoint.

        Call only on trainer rank 0.
        """

        self._require_rank_zero()

        return {
            "master_address": get_ip(),
            "master_port": get_open_port(),
        }

    def init_weight_transfer(
        self,
        master_address: str,
        master_port: int,
        transfer_world_size: int,
    ) -> None:
        """
        Join the trainer-to-vLLM NCCL group as transfer rank 0.

        Only FSDP rank 0 joins this dedicated group. Other FSDP ranks return
        immediately because they participate only in FSDP parameter gathering.
        """

        if self.rank != 0:
            return

        if self._weight_transfer_initialized:
            raise RuntimeError(
                "Trainer weight transfer is already initialized."
            )

        if transfer_world_size < 2:
            raise ValueError(
                "transfer_world_size must include at least one trainer "
                "and one vLLM worker."
            )

        self._weight_transfer_group = (
            NCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": master_address,
                    "master_port": master_port,
                    "world_size": transfer_world_size,
                }
            )
        )

        self._weight_transfer_initialized = True

    # -------------------------------------------------------------------------
    # Weight sending
    # -------------------------------------------------------------------------

    def send_weights(
        self,
        *,
        packed: bool = True,
    ) -> None:
        """
        Gather full parameters on every FSDP rank and broadcast them from
        trainer rank 0 to vLLM.

        This method must be invoked on every FSDP actor concurrently.
        """

        if self.rank == 0 and not self._weight_transfer_initialized:
            raise RuntimeError(
                "Trainer rank 0 must call init_weight_transfer() first."
            )

        full_parameter_iterator = self.iter_full_named_parameters()

        if self.rank == 0:
            send_args = NCCLTrainerSendWeightsArgs(
                group=self._weight_transfer_group,
                packed=packed,
            )

            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=full_parameter_iterator,
                trainer_args=send_args,
            )
        else:
            # Consume the iterator so all FSDP collective gathers execute.
            for _name, _parameter in full_parameter_iterator:
                pass

    def weight_transfer_is_initialized(self) -> bool:
        if self.rank != 0:
            return True

        return self._weight_transfer_initialized

    def _require_rank_zero(self) -> None:
        if self.rank != 0:
            raise RuntimeError(
                f"This method is rank-0-only, but actor rank is {self.rank}."
            )