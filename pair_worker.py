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
from functools import partial
import time
from util.entropy_processor import EntropyStopperAdapter

from typing import List

from nccl_sync.weight_sync_plugin import (
    AbstractFSDPWeightSync,
    AbstractRolloutWeightSync,
    AbstractWeightSyncController
)

from tree import Tree
from tree_reward import TreeRewardManager
from util.get_logits import get_shift_logits_and_labels
from util.entropy import calculate_entropy_from_logits
from util.ratio import get_ratio
from util.debug import debug

from opsd.opsd import OPSD



import wandb

'''
# Define the intended machine's full IP:PORT (Use localhost or a known worker IP)
TARGET_ADDRESS = "ray://192.168.215.14:10000" # Use an arbitrary open port

try:
    # 1. Initialize Ray FIRST, specifying the desired address/resource.
    ray.init(address=TARGET_ADDRESS)
    print("✅ Successfully initialized a single, controlled Ray instance.")

except ConnectionError as e:
    print(f"⚠️ Could not connect to specified address. Starting fresh local Ray instance instead. Error: {e}")
    # Fallback if the specific address isn't available (good for notebooks)
    ray.init()
'''

ray.init(
    address=None,
    _temp_dir=os.environ["RAY_TMPDIR"],
    include_dashboard=False,
    num_cpus=8,
    num_gpus=4,
)

@ray.remote(num_gpus=2)
class RolloutActor(AbstractRolloutWeightSync):
    def __init__(self, gpu_ids:List[str], model_path:str):
        self._init_rollout_weight_sync_state()
        debug("LOADING DEVICES")
        # Set CUDA VISIBLE
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.parallel_size = len(gpu_ids)
        
        # VLLM will load VLLM with those specific GPUs
        debug("VLLM LOADING")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=self.parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            logits_processors=[EntropyStopperAdapter],
            #enforce_eager=True,
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            disable_custom_all_reduce=True,
        )
        debug("VLLM LOADED")
        
        # Load tokenizer to apply chat template only
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        
        debug("TOKENIZER LOADED")

    def generate(self, prompts, ground_truths, reference_answers):
        # Re-run genrate until it success in case it crashes
        while True:
            try:
                return self._generate(prompts, ground_truths, reference_answers)
            except Exception as e:
                debug("Rollout crashed, retrying...")
                debug(e)
                torch.cuda.empty_cache()
                time.sleep(1)  # Wait a bit before retrying
        
    def _generate(self, prompts, ground_truths, reference_answers) -> TreeRewardManager:
        # Get text
        applied_template_prompts = self.tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Get input_ids
        encoded = self.tokenizer(
            applied_template_prompts,
            padding=False,
            truncation=False,
            return_tensors=None,   # return an array of 1d tensor instead of a 2d tensor
        )
        input_ids = [
            torch.tensor(ids, dtype=torch.long)
            for ids in encoded["input_ids"]
        ]
        
        # At first every depth is 1
        depths = [1] * len(prompts)
        
        # Get group_ids
        group_ids = [i for i in range(len(prompts))]

        
        tree_reward_manager = TreeRewardManager()
        tree = Tree(
            llm=self.llm,
            eos_id=self.tokenizer.eos_token_id,
            tree_reward_manager=tree_reward_manager
        )
        
        groups = {}
        
        tree.forward({
            "text": applied_template_prompts,
            "input_ids": input_ids,
            "group_ids": group_ids,
            "thresholds": [2] * len(prompts) # at first entropy threshold is 2
        }, depths=depths, gts=ground_truths, groups=groups, reference_answers=reference_answers, is_init=True)
        
        debug("#"*80)
        debug("DONE GENERATING")
        debug(applied_template_prompts)
        
        mean_reward = torch.tensor(
            [item["reward"] for group in groups.values() for item in group],
            dtype=torch.float32,
        ).mean().item()
        
        return tree_reward_manager, input_ids, mean_reward
    
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
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
        )

    # Keep your old public API.
    def reload(self, model_path: str):
        return self.reload_from_disk(model_path)
        
@ray.remote(num_gpus=1)
class FSDPTrainerActor(AbstractFSDPWeightSync):
    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        model_path: str,
    ):
        import os
        import torch
        import torch.distributed as dist
        from functools import partial
        from datetime import timedelta

        self.rank = rank
        self.world_size = world_size
        self._init_fsdp_weight_sync_state()

        # DO NOT set CUDA_VISIBLE_DEVICES manually. Ray already does it.
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"

        torch.cuda.set_device(0)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()

        model.config.use_cache = False

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
        
        # activation checkpointing saves backward memory
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            apply_activation_checkpointing,
            CheckpointImpl,
        )

        apply_activation_checkpointing(
            self.model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda module: isinstance(module, LlamaDecoderLayer),
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

        self.alpha = 0.01
        self.eps_clip = 0.05

    def _chosen_log_probs(self, logits, labels):
        chosen_logits = logits.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        return chosen_logits - torch.logsumexp(logits, dim=-1)

    @torch.no_grad()
    def _entropy_topk(self, logits, top_k=256):
        topk_logits = torch.topk(logits, k=top_k, dim=-1).values
        log_probs = F.log_softmax(topk_logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    def __freeze_old_policy_probs__(self, mini_batch):
        input_ids = [x["input_ids"] for x in mini_batch]
        reward_masks = [x["reward_mask"] for x in mini_batch]

        with torch.no_grad():
            old_logits, labels, _ = get_shift_logits_and_labels(
                sequences=input_ids,
                reward_masks=reward_masks,
                hf_model=self.model,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            old_token_log_probs = self._chosen_log_probs(old_logits, labels)

        old_token_log_probs = old_token_log_probs.detach().cpu()

        del old_logits, labels
        torch.cuda.empty_cache()

        return old_token_log_probs

    def __process_mini_batch__(self, mini_batch, old_token_log_probs):
        input_ids = [x["input_ids"] for x in mini_batch]
        reward_masks = [x["reward_mask"] for x in mini_batch]

        new_logits, labels, masks = get_shift_logits_and_labels(
            sequences=input_ids,
            reward_masks=reward_masks,
            hf_model=self.model,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        device = new_logits.device
        dtype = new_logits.dtype

        new_token_log_probs = self._chosen_log_probs(new_logits, labels)

        old_token_log_probs = old_token_log_probs.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

        masks = masks.to(device=device, dtype=dtype)

        with torch.no_grad():
            # cheaper entropy, no full vocab logsoftmax
            entropies = self._entropy_topk(new_logits.detach(), top_k=256).to(dtype=dtype)

            correctness_advantages = torch.tensor(
                [x["correctness_advantage"] for x in mini_batch],
                device=device,
                dtype=dtype,
            )
            
            H_targets = torch.tensor(
                [x["H_target"] for x in mini_batch],
                device=device,
                dtype=dtype,
            )
            
            h_target_ratio = torch.tensor(
                [x["h_target_ratio"] for x in mini_batch],
                device=device,
                dtype=dtype,
            )
            teacher_prefixes = [
                torch.as_tensor(
                    x["teacher_prefix"],
                    device=device,
                    dtype=torch.long,   # token IDs
                )
                for x in mini_batch
            ]
            
            
            
            debug("H target")
            debug(H_targets.shape)
            with torch.no_grad():
                _h_targets = OPSD.calculate_entropy_of_teacher(
                    hf_model = self.model,
                    sequences = input_ids,
                    reward_masks = reward_masks,
                    pad_token_id=self.tokenizer.pad_token_id,
                    teacher_prefixes =  teacher_prefixes
                )
            ''' 
            debug("H Target Ratio:")
            debug(h_target_ratio)
            debug(len(h_target_ratio))
            debug("H Target:")
            debug(_h_targets.shape)
            debug(_h_targets[1])
            debug("Entropy:")
            debug(entropies[1])
            debug(entropies.shape)
            debug(masks[1])
            '''

            base_advantages = correctness_advantages.unsqueeze(1)
            
            H_targets = h_target_ratio[:, None] * _h_targets
            # entropy_penalty = (entropies - H_targets.unsqueeze(1)) # ** 2
            
            entropy_penalty = entropies - H_targets
            
            debug("entropy_penalty")
            debug(_h_targets)
            debug(entropy_penalty)
            
            multiplier = torch.clamp(
                1.0 - self.alpha * entropy_penalty,
                min=0.0,
                max=1.0,
            )
            
            advantages = torch.where(
                base_advantages > 0,
                base_advantages * multiplier,
                base_advantages,
            )
            
            advantages = advantages * masks
            
            '''
            debug("advantages")
            debug(advantages[1])
            '''


            # [batch]
            min_advantages = torch.where(
                masks.bool(),
                advantages,
                torch.full_like(advantages, float("inf"))
            ).min(dim=1).values
            
            debug("C"*40)
            debug("correctness_advantages")
            debug(correctness_advantages)
            
            debug("min_advantages")
            debug(min_advantages)

        ratio, _, _ = get_ratio(
            new_token_log_probs=new_token_log_probs,
            old_token_log_probs=old_token_log_probs,
            labels=labels,
            masks=masks,
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.eps_clip,
            1.0 + self.eps_clip,
        )

        loss_per_token = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        )

        loss = (loss_per_token * masks).sum() / masks.sum().clamp_min(1)

        return loss

    def process_batch(self, batch, mini_batch_size, epoch=1):
        n_total_tokens, mini_batches = TreeRewardManager.process_batch(
            batch,
            mini_batch_size,
        )

        self.model.eval()

        old_token_log_probs_array = []
        for mini_batch in mini_batches:
            old_token_log_probs_array.append(
                self.__freeze_old_policy_probs__(mini_batch)
            )

        self.model.train()

        for epoch_idx in range(epoch):
            self.optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0

            for i, mini_batch in enumerate(mini_batches):
                
                correctness = torch.tensor(
                    [x["correctness_advantage"] for x in mini_batch],
                    dtype=torch.float32,
                )
                
                '''
                # all bad: skip
                if correctness.max() <= 0:
                    debug(f"skip mini_batch {i}: all bad {correctness.tolist()}")
                    continue
                '''
                
                loss = self.__process_mini_batch__(
                    mini_batch=mini_batch,
                    old_token_log_probs=old_token_log_probs_array[i],
                )

                total_loss += loss.detach().item()

                loss.backward()

                del loss
                torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
            )

            self.optimizer.step()

            debug(
                f"EPOCH {epoch_idx + 1}/{epoch}: "
                f"loss = {total_loss:.6f}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        return total_loss

        
    
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
        
        
class RLController(AbstractWeightSyncController):
    def __init__(
        self,
        model_path: str,
        rollout_gpus: list[int],
        trainer_gpus: list[int],
        master_addr: str = "127.0.0.1",
        master_port: int = 29588,
    ):
        self.model_path = model_path
        self.rollout_gpus = rollout_gpus
        self.trainer_gpus = trainer_gpus
        self.world_size = len(trainer_gpus)
        self.prompts_per_batch = 110
        self._init_weight_sync_controller_state()

        if len(set(rollout_gpus) & set(trainer_gpus)) > 0:
            debug(
                "WARNING: rollout_gpus and trainer_gpus overlap. "
                "This can easily OOM."
            )
            
        debug("*"*30,"Rollout Actor Loading","*"*30)

        self.rollout = RolloutActor.remote(
            model_path=model_path,
            gpu_ids=rollout_gpus,
        )
        
        debug("="*30,"Rollout Actor Loaded","="*30)
        
        self.trainers = [
            FSDPTrainerActor.remote(
                rank=rank,
                world_size=self.world_size,
                # gpu_id=gpu_id,
                master_addr=master_addr,
                master_port=master_port,
                model_path=model_path,
            )
            for rank, gpu_id in enumerate(trainer_gpus)
        ]
        
        debug("="*30,"Trainer Loaded","="*30)
    
    def rollout_samples(self, prompts, ground_truths, reference_answers):
        return ray.get(self.rollout.generate.remote(
            prompts, ground_truths, reference_answers
        ))
    
    def split_samples(self, manager:TreeRewardManager, input_ids):
        chunks = []
        
        
        
        for i in range(self.world_size):
            ids = input_ids[i::self.world_size]
            batch = []
            for prompt_ids in ids:
                manager.traverse(prompt_ids, batch=batch)
                
            chunks.append(batch)
        
        debug(chunks)
        
        return chunks
    
    def train_on_samples(self, manager:TreeRewardManager, input_ids):
        batch = []
        debug("="*30)
        debug("TRAIN ON SAMPLE")
        for prompt_ids in input_ids:
            manager.traverse(prompt_ids, batch=batch)
            
            
        debug("DONE TRAVERSING")

        losses = ray.get([
            self.trainers[i].process_batch.remote(batch, 8)
            for i in range(self.world_size)
        ])

        return losses

    def step(self, prompts, ground_truths, reference_answers):
        manager, input_ids, mean_reward = self.rollout_samples(prompts, ground_truths, reference_answers)
        losses = self.train_on_samples(manager=manager, input_ids=input_ids)
        
        return {
            "losses": losses,
            "num_samples": len(input_ids),
            "avg_loss": sum(losses) / len(losses),
            "reward/mean": mean_reward
        }
    
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
