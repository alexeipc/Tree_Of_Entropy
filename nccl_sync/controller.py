from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Iterator, Sequence

import ray
import torch

from vllm.utils.network_utils import get_ip, get_open_port

class WeightSyncController:
    """
    Coordinates NCCL weight synchronization between:

        trainer_actors:
            Sequence of Ray FSDP actor handles, ordered by rank.

        rollout_actor:
            Ray vLLM rollout actor handle.

    Assumptions:
    - trainer_actors[0] is FSDP rank 0.
    - RolloutActor methods are exposed as Ray remote methods.
    - FSDPActor methods are exposed as Ray remote methods.
    """

    def __init__(
        self,
        trainer_actors: Sequence[Any],
        rollout_actor: Any,
        *,
        vllm_worker_count: int = 1,
        packed: bool = True,
    ) -> None:
        if not trainer_actors:
            raise ValueError(
                "At least one trainer actor is required."
            )

        if vllm_worker_count < 1:
            raise ValueError(
                "vllm_worker_count must be at least 1."
            )

        self.trainer_actors = list(trainer_actors)
        self.rollout_actor = rollout_actor

        self.vllm_worker_count = vllm_worker_count
        self.packed = packed

        # One external trainer sender plus every vLLM model worker.
        self.transfer_world_size = 1 + vllm_worker_count

        self._initialized = False
        self._metadata: dict[str, Any] | None = None
        self._sync_count = 0

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize the dedicated NCCL transfer group exactly once.

        Trainer rank 0 and vLLM must join concurrently.
        """

        if self._initialized:
            raise RuntimeError(
                "WeightSyncController is already initialized."
            )

        trainer_rank_zero = self.trainer_actors[0]

        endpoint = ray.get(
            trainer_rank_zero.create_weight_transfer_endpoint.remote()
        )

        master_address = endpoint["master_address"]
        master_port = endpoint["master_port"]

        # These two calls must be launched before ray.get().
        trainer_init_ref = (
            trainer_rank_zero.init_weight_transfer.remote(
                master_address=master_address,
                master_port=master_port,
                transfer_world_size=self.transfer_world_size,
            )
        )

        rollout_init_ref = (
            self.rollout_actor.init_weight_transfer.remote(
                master_address=master_address,
                master_port=master_port,
                transfer_world_size=self.transfer_world_size,
                rank_offset=1,
            )
        )

        ray.get([
            trainer_init_ref,
            rollout_init_ref,
        ])

        self._metadata = ray.get(
            trainer_rank_zero.get_weight_metadata.remote()
        )

        self._validate_metadata(self._metadata)
        self._initialized = True

    # -------------------------------------------------------------------------
    # Synchronization
    # -------------------------------------------------------------------------

    def sync(self) -> None:
        """
        Perform one complete trainer-to-vLLM weight synchronization.

        Call after optimizer.step(), not in the middle of backward().
        """

        if not self._initialized:
            raise RuntimeError(
                "Call initialize() before sync()."
            )

        assert self._metadata is not None

        # Start vLLM receiving first, but do not ray.get() yet.
        receive_ref = self.rollout_actor.receive_weights.remote(
            metadata=self._metadata,
            packed=self.packed,
            is_checkpoint_format=False,
        )

        # Every FSDP rank must execute this because all ranks participate
        # in reconstructing the full parameters.
        trainer_send_refs = [
            actor.send_weights.remote(
                packed=self.packed,
            )
            for actor in self.trainer_actors
        ]

        # Wait only after every NCCL participant has been launched.
        ray.get([
            receive_ref,
            *trainer_send_refs,
        ])

        self._sync_count += 1

    def sync_if_needed(
        self,
        optimizer_step: int,
        sync_every: int,
    ) -> bool:
        """
        Synchronize every `sync_every` optimizer updates.

        Returns True when synchronization occurred.
        """

        if sync_every < 1:
            raise ValueError(
                "sync_every must be at least 1."
            )

        if optimizer_step % sync_every != 0:
            return False

        self.sync()
        return True

    @property
    def sync_count(self) -> int:
        return self._sync_count

    @staticmethod
    def _validate_metadata(
        metadata: dict[str, Any],
    ) -> None:
        required = {
            "names",
            "shapes",
            "dtype_names",
        }

        missing = required.difference(metadata)

        if missing:
            raise ValueError(
                f"Weight metadata is missing fields: {sorted(missing)}"
            )

        names = metadata["names"]
        shapes = metadata["shapes"]
        dtype_names = metadata["dtype_names"]

        if not (len(names) == len(shapes) == len(dtype_names)):
            raise ValueError(
                "Weight metadata lists must have equal lengths."
            )

        if len(names) == 0:
            raise ValueError(
                "Weight metadata contains no parameters."
            )

        if len(names) != len(set(names)):
            raise ValueError(
                "Weight metadata contains duplicate parameter names."
            )