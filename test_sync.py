import os
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from functools import partial
from typing import List

from util.entropy_processor import EntropyStopperAdapter
from util.debug import debug

from nccl_sync.weight_sync_plugin import (
    AbstractFSDPWeightSync,
    AbstractRolloutWeightSync,
    AbstractWeightSyncController,
)


MODEL_PATH = "./sft/Llama-3.2-3B-Instruct_mot_sft_final"

NUM_FAKE_UPDATES = 5

# Intentionally huge for this smoke test only.
FAKE_LEARNING_RATE = 5e-2


ray.init(
    address=None,
    _temp_dir=os.environ["RAY_TMPDIR"],
    include_dashboard=False,
    num_cpus=8,
    num_gpus=3,
)


@ray.remote(num_gpus=1)
class TestRolloutActor(AbstractRolloutWeightSync):
    def __init__(self, gpu_ids: List[str], model_path: str):
        self._init_rollout_weight_sync_state()

        debug("LOADING ROLLOUT DEVICE")

        # Ray already isolates one GPU for this actor.
        # Do not normally overwrite CUDA_VISIBLE_DEVICES inside a Ray actor.
        self.parallel_size = len(gpu_ids)

        debug("VLLM LOADING")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=self.parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            logits_processors=[EntropyStopperAdapter],
            enforce_eager=True,
            weight_transfer_config=WeightTransferConfig(
                backend="nccl"
            ),
        )

        debug("VLLM LOADED")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        debug("TOKENIZER LOADED")

    def generate(self, prompts: List[str]):
        applied_template_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            logprobs=20,
        )

        outputs = self.llm.generate(
            applied_template_prompts,
            sampling_params,
        )

        results = []

        for output in outputs:
            completion = output.outputs[0]

            first_token_logprobs = {}

            if completion.logprobs:
                for token_id, logprob_object in (
                    completion.logprobs[0].items()
                ):
                    first_token_logprobs[int(token_id)] = float(
                        logprob_object.logprob
                    )

            results.append(
                {
                    "text": completion.text,
                    "token_ids": list(completion.token_ids),
                    "first_token_logprobs": first_token_logprobs,
                }
            )

        return results

    def get_vllm_engine(self):
        return self.llm

    def rebuild_vllm_engine(self, model_path: str) -> None:
        del self.llm
        torch.cuda.empty_cache()

        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=self.parallel_size,
            gpu_memory_utilization=0.85,
            logits_processors=[EntropyStopperAdapter],
            enforce_eager=True,
            weight_transfer_config=WeightTransferConfig(
                backend="nccl"
            ),
        )

    def reload(self, model_path: str):
        return self.reload_from_disk(model_path)


@ray.remote(num_gpus=1)
class TestFSDPActor(AbstractFSDPWeightSync):
    def __init__(
        self,
        rank: int,
        world_size: int,
        model_path: str,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_path = model_path

        self._init_fsdp_weight_sync_state()

        self.device = torch.device("cuda:0")

        self.model = None
        self.optimizer = None
        
    @property
    def fsdp_rank(self) -> int:
        return self.rank

    def get_fsdp_model(self):
        return self.model

    def save_model_config(self, save_dir: str) -> None:
        self.model.module.config.save_pretrained(save_dir)

    def save_tokenizer(self, save_dir: str) -> None:
        self.tokenizer.save_pretrained(save_dir)

    # Keep your old actor API.
    def save(self, save_dir: str):
        return self.save_checkpoint(save_dir)

    def initialize_distributed(
        self,
        master_addr: str,
        master_port: int,
    ):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        torch.cuda.set_device(0)

        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=self.device,
            use_orig_params=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=FAKE_LEARNING_RATE,
        )

        dist.barrier()

        return {
            "rank": self.rank,
            "initialized": True,
        }

    def massive_random_update(
        self,
        update_index: int,
    ):
        """
        Completely meaningless synthetic objective.

        It creates random tokens and then aggressively maximizes the
        probability of one fixed random target token everywhere.
        """

        self.model.train()

        torch.manual_seed(
            10_000 + update_index * 100 + self.rank
        )

        torch.cuda.manual_seed_all(
            10_000 + update_index * 100 + self.rank
        )

        vocab_size = self.model.module.config.vocab_size

        batch_size = 2
        sequence_length = 64

        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, sequence_length),
            device=self.device,
            dtype=torch.long,
        )

        attention_mask = torch.ones_like(input_ids)

        # Force every position toward one arbitrary token.
        # This should produce a very obvious model change.
        forced_token_id = 1000

        labels = torch.full(
            size=(batch_size, sequence_length),
            fill_value=forced_token_id,
            device=self.device,
            dtype=torch.long,
        )

        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        # Make the update even more aggressive.
        massive_loss = loss * 100.0

        massive_loss.backward()

        local_grad_norm_squared = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )

        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue

            gradient = parameter.grad.detach().float()

            local_grad_norm_squared += (
                gradient.double().square().sum()
            )

        dist.all_reduce(
            local_grad_norm_squared,
            op=dist.ReduceOp.SUM,
        )

        grad_norm = local_grad_norm_squared.sqrt()

        self.optimizer.step()

        dist.barrier()

        return {
            "rank": self.rank,
            "original_loss": float(loss.detach().float().item()),
            "massive_loss": float(
                massive_loss.detach().float().item()
            ),
            "global_grad_norm": float(grad_norm.item()),
            "forced_token_id": forced_token_id,
        }

    def checksum(self):
        """
        Check local FSDP shards to confirm optimizer.step() changed them.
        """

        local_sum = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )

        local_absolute_sum = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )

        for parameter in self.model.parameters():
            values = parameter.detach().double()

            local_sum += values.sum()
            local_absolute_sum += values.abs().sum()

        dist.all_reduce(
            local_sum,
            op=dist.ReduceOp.SUM,
        )

        dist.all_reduce(
            local_absolute_sum,
            op=dist.ReduceOp.SUM,
        )

        return {
            "rank": self.rank,
            "sum": float(local_sum.item()),
            "absolute_sum": float(
                local_absolute_sum.item()
            ),
        }


class TestController(AbstractWeightSyncController):
    def __init__(
        self,
        trainers,
        rollout,
    ):
        self.trainers = trainers
        self.rollout = rollout

        self._init_weight_sync_controller_state()
        
    @property
    def rollout_actor(self):
        return self.rollout

    @property
    def fsdp_actors(self):
        return self.trainers

    # Fast GPU-to-GPU path.
    def gpu_sync(self, packed: bool = True):
        return self.nccl_sync(packed=packed)

    # Keep your old disk-save + vLLM-reload API.
    def save_and_sync(self, save_dir: str):
        return self.save_and_reload(save_dir)


def compare_outputs(before, after):
    changed = False

    for index, (old, new) in enumerate(
        zip(before, after)
    ):
        shared_token_ids = (
            set(old["first_token_logprobs"])
            & set(new["first_token_logprobs"])
        )

        max_logprob_delta = max(
            (
                abs(
                    new["first_token_logprobs"][token_id]
                    - old["first_token_logprobs"][token_id]
                )
                for token_id in shared_token_ids
            ),
            default=0.0,
        )

        text_changed = old["text"] != new["text"]
        tokens_changed = (
            old["token_ids"] != new["token_ids"]
        )

        print()
        print(f"Prompt {index}")
        print("Before:", repr(old["text"]))
        print("After: ", repr(new["text"]))
        print("Text changed:", text_changed)
        print("Tokens changed:", tokens_changed)
        print(
            "Maximum first-token logprob delta:",
            max_logprob_delta,
        )

        if (
            text_changed
            or tokens_changed
            or max_logprob_delta > 1e-6
        ):
            changed = True

    return changed


def main():
    trainers = [
        TestFSDPActor.remote(
            rank=0,
            world_size=2,
            model_path=MODEL_PATH,
        ),
        TestFSDPActor.remote(
            rank=1,
            world_size=2,
            model_path=MODEL_PATH,
        ),
    ]

    rollout = TestRolloutActor.remote(
        gpu_ids=["0"],
        model_path=MODEL_PATH,
    )

    controller = TestController(
        trainers=trainers,
        rollout=rollout,
    )

    # Use the node IP that Ray actors can reach.
    master_addr = ray.util.get_node_ip_address()

    # Pick your own unused port.
    fsdp_port = 29501

    print("Initializing two-rank FSDP...")

    ray.get([
        trainer.initialize_distributed.remote(
            master_addr,
            fsdp_port,
        )
        for trainer in trainers
    ])

    print("Initializing weight synchronization...")

    controller.init_nccl_sync()

    prompts = [
        "What is 12 + 19?",
        "What is the capital of France?",
    ]

    print("\nGenerating before fake update...")

    output_before = ray.get(
        rollout.generate.remote(prompts)
    )

    print(output_before)

    checksum_before = ray.get([
        trainer.checksum.remote()
        for trainer in trainers
    ])

    print("\nFSDP checksum before:")
    print(checksum_before[0])

    print("\nPerforming massive fake updates...")

    for update_index in range(NUM_FAKE_UPDATES):
        stats = ray.get([
            trainer.massive_random_update.remote(
                update_index
            )
            for trainer in trainers
        ])

        print(
            f"Update {update_index + 1}:",
            stats[0],
        )

    checksum_after = ray.get([
        trainer.checksum.remote()
        for trainer in trainers
    ])

    print("\nFSDP checksum after:")
    print(checksum_after[0])

    checksum_changed = (
        checksum_before[0]["sum"]
        != checksum_after[0]["sum"]
        or checksum_before[0]["absolute_sum"]
        != checksum_after[0]["absolute_sum"]
    )

    print(
        "FSDP parameters changed:",
        checksum_changed,
    )

    if not checksum_changed:
        raise RuntimeError(
            "FSDP optimizer update did not change parameters."
        )

    print(
        "\nGenerating before sync. "
        "vLLM should still have the original weights..."
    )

    output_before_sync = ray.get(
        rollout.generate.remote(prompts)
    )

    print(output_before_sync)

    print("\nCalling NCCL synchronization...")

    controller.nccl_sync()

    print("\nGenerating after sync...")

    output_after_sync = ray.get(
        rollout.generate.remote(prompts)
    )

    print(output_after_sync)

    changed = compare_outputs(
        output_before_sync,
        output_after_sync,
    )

    if changed:
        print(
            "\nPASS: vLLM changed after NCCL synchronization."
        )
    else:
        raise RuntimeError(
            "FAIL: FSDP changed, but vLLM did not change "
            "after NCCL synchronization."
        )


if __name__ == "__main__":
    main()