import os
import ray
import torch
import torch.distributed as dist

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial


# -----------------------
# vLLM rollout actor
# -----------------------

@ray.remote(num_gpus=0)
class RolloutActor:
    def __init__(self, model_path: str, gpu_ids: list[int]):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(gpu_ids),
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )

    def generate(self, prompts: list[str], n: int = 4):
        params = SamplingParams(
            n=n,
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
        )

        outputs = self.llm.generate(prompts, params)

        samples = []
        for out in outputs:
            for completion in out.outputs:
                samples.append({
                    "prompt": out.prompt,
                    "text": completion.text,
                    "token_ids": completion.token_ids,
                })

        return samples

    def reload(self, model_path: str):
        del self.llm
        torch.cuda.empty_cache()

        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )

        return True


# -----------------------
# FSDP trainer actor
# -----------------------

@ray.remote(num_gpus=0)
class FSDPTrainerActor:
    def __init__(
        self,
        rank: int,
        world_size: int,
        gpu_id: int,
        master_addr: str,
        master_port: int,
        model_path: str,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

        torch.cuda.set_device(0)

        self.rank = rank
        self.world_size = world_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()

        # Change this for Qwen/Mistral/etc.
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        self.model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def train_step(self, samples: list[dict]):
        texts = [s["prompt"] + s["text"] for s in samples]

        advantages = torch.tensor(
            [s["advantage"] for s in samples],
            dtype=torch.float32,
            device="cuda",
        )

        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(
            -1,
            labels.unsqueeze(-1),
        ).squeeze(-1)

        seq_logprobs = (token_logprobs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        loss = -(seq_logprobs * advantages).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.detach().cpu())

    def save(self, save_dir: str):
        from torch.distributed.fsdp import (
            StateDictType,
            FullStateDictConfig,
        )

        os.makedirs(save_dir, exist_ok=True)

        cfg = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            cfg,
        ):
            state = self.model.state_dict()

        if self.rank == 0:
            torch.save(state, os.path.join(save_dir, "pytorch_model.bin"))
            self.tokenizer.save_pretrained(save_dir)

        dist.barrier()
        return save_dir