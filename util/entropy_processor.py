from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from multiprocessing import shared_memory
import numpy as np



class EntropyStopper:
    """
    One instance of this class is created for one vLLM request.
    """

    def __init__(
        self,
        eos_token_id: int,
        threshold: float,
        shm_name: str,
        slot: int,
        num_slots: int
    ) -> None:
        self.eos_token_id = eos_token_id
        self.threshold = threshold

        self.stop_entropy: float | None = None
        self.input_ids: list[int] | None = None
        self.slot = slot
        self._stopped = False
        
        # Set up shared memory
        self._shm = shared_memory.SharedMemory(
            name=shm_name,
            create=False,
        )
        
        self._results = np.ndarray(
            (num_slots,),
            dtype=np.float64,
            buffer=self._shm.buf,
        )

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        self.input_ids = output_ids

        if self.threshold == float("inf"):
            return logits
    
        if self._stopped:
            return logits

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)
        entropy = -(probs * logprobs).sum()

        if entropy >= self.threshold:
            self.stop_entropy = entropy.item()
            
            # Save it to shared memory
            self._results[self.slot] = self.stop_entropy
            self._stopped = True

            logits[:] = -float("inf")
            logits[self.eos_token_id] = 0.0

        return logits
    
    def __del__(self):
        shm = getattr(self, "_shm", None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


class EntropyStopperAdapter(AdapterLogitsProcessor):
    """
    vLLM creates one EntropyStopper instance for each request whose
    SamplingParams contains entropy_threshold and eos_token_id.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra_args = params.extra_args or {}

        threshold = extra_args.get("entropy_threshold")
        eos_token_id = extra_args.get("entropy_eos_token_id")

        # Returning without an error allows requests that do not use
        # the entropy processor, such as handle_single_completion().
        if threshold is None and eos_token_id is None:
            return

        if not isinstance(threshold, (int, float)):
            raise ValueError(
                "`entropy_threshold` must be an int or float, "
                f"got {type(threshold).__name__}"
            )

        if not isinstance(eos_token_id, int):
            raise ValueError(
                "`entropy_eos_token_id` must be an int, "
                f"got {type(eos_token_id).__name__}"
            )

    def is_argmax_invariant(self) -> bool:
        # We may replace the highest-logit token with EOS.
        return False

    def new_req_logits_processor(self, params):
        extra = params.extra_args or {}

        threshold = extra.get("entropy_threshold")
        eos_token_id = extra.get("entropy_eos_token_id")
        shm_name = extra.get("entropy_shm_name")
        slot = extra.get("entropy_slot")
        num_slots = extra.get("entropy_num_slots")

        if threshold is None:
            return None

        return EntropyStopper(
            eos_token_id=int(eos_token_id),
            threshold=float(threshold),
            shm_name=str(shm_name),
            slot=int(slot),
            num_slots=int(num_slots),
        )