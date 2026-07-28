from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

import ray
import torch

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)

class RolloutActor(ABC):
    """
    Abstract vLLM-side NCCL weight receiver.

    The subclass only needs to provide get_vllm_engine().

    Example:

        @ray.remote(num_gpus=1)
        class MyRolloutActor(RolloutActor):
            def __init__(self, model_path):
                self.llm = LLM(
                    model=model_path,
                    weight_transfer_config=WeightTransferConfig(
                        backend="nccl",
                    ),
                )

            def get_vllm_engine(self):
                return self.llm
    """

    def __init__(self) -> None:
        self._weight_transfer_initialized = False
        self._weight_update_in_progress = False

    @abstractmethod
    def get_vllm_engine(self) -> Any:
        """
        Return the vLLM LLM or engine object.

        It must expose:

            init_weight_transfer_engine(...)
            start_weight_update(...)
            update_weights(...)
            finish_weight_update(...)
        """

    def init_weight_transfer(
        self,
        master_address: str,
        master_port: int,
        transfer_world_size: int,
        rank_offset: int = 1,
    ) -> None:
        """
        Join the dedicated weight-transfer NCCL group.

        NCCL rank layout:

            rank 0:
                trainer sender

            rank 1 ... N:
                vLLM workers

        Therefore, vLLM normally uses rank_offset=1.
        """

        if self._weight_transfer_initialized:
            raise RuntimeError(
                "Rollout weight transfer is already initialized."
            )

        if transfer_world_size < 2:
            raise ValueError(
                "transfer_world_size must include at least one trainer "
                "and one vLLM worker."
            )

        init_info = NCCLWeightTransferInitInfo(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=transfer_world_size,
        )

        request = WeightTransferInitRequest(
            init_info=asdict(init_info),
        )

        engine = self.get_vllm_engine()
        engine.init_weight_transfer_engine(request)

        self._weight_transfer_initialized = True

    def receive_weights(
        self,
        metadata: dict[str, Any],
        *,
        packed: bool = True,
        is_checkpoint_format: bool = False,
    ) -> None:
        """
        Receive one complete model update from the trainer.

        This call blocks inside NCCL until the trainer broadcasts the tensors.
        The controller must launch this method before or concurrently with the
        trainer-side send calls.
        """

        if not self._weight_transfer_initialized:
            raise RuntimeError(
                "Call init_weight_transfer() before receive_weights()."
            )

        if self._weight_update_in_progress:
            raise RuntimeError(
                "Another vLLM weight update is already in progress."
            )

        names = metadata["names"]
        shapes = metadata["shapes"]
        dtype_names = metadata["dtype_names"]

        if not (len(names) == len(shapes) == len(dtype_names)):
            raise ValueError(
                "Invalid metadata: names, shapes, and dtype_names "
                "must have equal lengths."
            )

        update_info = NCCLWeightTransferUpdateInfo(
            names=names,
            shapes=shapes,
            dtype_names=dtype_names,
            packed=packed,
        )

        request = WeightTransferUpdateRequest(
            update_info=asdict(update_info),
        )

        engine = self.get_vllm_engine()

        self._weight_update_in_progress = True

        try:
            engine.start_weight_update(
                is_checkpoint_format=is_checkpoint_format,
            )

            engine.update_weights(request)

            engine.finish_weight_update()
        finally:
            self._weight_update_in_progress = False

    def weight_transfer_is_initialized(self) -> bool:
        return self._weight_transfer_initialized