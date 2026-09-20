import os
import sys
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
import threading
from util.entropy_processor import EntropyStopperAdapter

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
from util.opsd import change_prompts
from util.reward_func import reward
from opsd.opsd import OPSD

def get_decoder_layer_cls(model):
    """Return the transformer block class for supported causal LMs."""
    model_type = getattr(model.config, "model_type", None)

    if model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer

    # Qwen2.5 checkpoints also use the qwen2 Transformers architecture.
    if model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        return Qwen2DecoderLayer

    if model_type == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        return Qwen3DecoderLayer

    raise ValueError(
        f"Unsupported model architecture {model_type!r}. "
        "Add its decoder layer class to get_decoder_layer_cls()."
    )


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



@ray.remote(num_gpus=2)
class RolloutActor(AbstractRolloutWeightSync):
    def __init__(self, model_path: str, teacher_actor):
        self._init_rollout_weight_sync_state()
        self.teacher_actor = teacher_actor
        debug("LOADING DEVICES")
        # Ray exposes the two GPUs reserved by the actor through
        # CUDA_VISIBLE_DEVICES. vLLM must use both or the second reservation is
        # wasted and the model is unnecessarily concentrated on GPU 0.
        self.parallel_size = 2
        
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
        teacher_refs = []

        def submit_teacher_request(
            message,
            reference_answer,
            gt,
            input_ids_,
        ):
            teacher_refs.append(
                self.teacher_actor.enqueue.remote(
                    message,
                    reference_answer,
                    gt,
                    input_ids_,
                )
            )

        tree = Tree(
            llm=self.llm,
            eos_id=self.tokenizer.eos_token_id,
            tree_reward_manager=tree_reward_manager,
            tokenizer=self.tokenizer,
            teacher_submitter=submit_teacher_request,
            batch_size=256,
        )
        
        groups = {}
        
        tree.forward({
            "text": applied_template_prompts,
            "input_ids": input_ids,
            "group_ids": group_ids,
            "thresholds": [2] * len(prompts) # at first entropy threshold is 2
        }, depths=depths, gts=ground_truths, groups=groups, reference_answers=reference_answers, is_init=True)

        # Teacher requests have run concurrently with tree expansion. Merge
        # immutable results only here; the manager is never shared between
        # Ray actors, so no cross-process lock or stale-copy mutation exists.
        teacher_refs.append(self.teacher_actor.flush.remote())
        for teacher_results in ray.get(teacher_refs):
            for input_ids_, teacher_ids, teacher_acc in teacher_results:
                tree_reward_manager.add_teacher_ids(
                    input_ids=input_ids_,
                    teacher_ids=teacher_ids,
                    teacher_acc=teacher_acc,
                )
        
        debug("#"*80)
        debug("DONE GENERATING")
        debug(applied_template_prompts)
        
        mean_reward = torch.tensor(
            [item["reward"] for group in groups.values() for item in group],
            dtype=torch.float32,
        ).mean().item()

        mean_teacher_reward = (
            sum(tree_reward_manager.teacher_rewards)
            / len(tree_reward_manager.teacher_rewards)
            if tree_reward_manager.teacher_rewards
            else 0.0
        )

        rollout_items = [
            item
            for group in groups.values()
            for item in group
        ]
        total_response_length = sum(
            item["response_length"]
            for item in rollout_items
        )
        avg_response_length = (
            total_response_length / len(rollout_items)
            if rollout_items
            else 0.0
        )
        avg_rollouts_per_prompt = (
            len(rollout_items) / len(input_ids)
            if input_ids
            else 0.0
        )
        eos_rate = (
            sum(item["hit_eos"] for item in rollout_items)
            / len(rollout_items)
            if rollout_items
            else 0.0
        )

        return (
            tree_reward_manager,
            input_ids,
            mean_reward,
            mean_teacher_reward,
            total_response_length,
            avg_response_length,
            avg_rollouts_per_prompt,
            eos_rate,
        )
    
    def get_vllm_engine(self):
        return self.llm

    def eval_dataset(
        self,
        dataset_name: str = "aime25",
        max_new_tokens: int = 16000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        val_n: int = 4,
        enable_thinking: bool = True,
    ):
        """
        Score the current policy on a held-out math benchmark using the
        rollout engine already loaded on this actor (kept in sync with the
        trainer via nccl_sync), instead of spinning up a separate vLLM
        instance that would compete for GPUs with training.
        """
        eval_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "eval",
        )
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        from evaluate_math import evaluate_math500

        average_at_n_pct, results = evaluate_math500(
            self.llm,
            self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            dataset_name=dataset_name,
            enable_thinking=enable_thinking,
            val_n=val_n,
        )

        num_problems = len(results)
        pass_at_n_pct = (
            100.0 * sum(1 for r in results if r["pass_at_n"]) / num_problems
            if num_problems
            else 0.0
        )
        majority_vote_at_n_pct = (
            100.0
            * sum(1 for r in results if r["majority_vote_correct"])
            / num_problems
            if num_problems
            else 0.0
        )

        return {
            "average_at_n_pct": average_at_n_pct,
            "pass_at_n_pct": pass_at_n_pct,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "num_problems": num_problems,
            "val_n": val_n,
        }

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


@ray.remote(num_gpus=2)
class BaseTeacherActor:
    """Frozen base model with access to the privileged teacher prompt."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=2,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            disable_custom_all_reduce=True,
        )
        self.queue_size = 64
        self.pending = []
        self.queue_lock = threading.RLock()

    def enqueue(
        self,
        message,
        reference_answer,
        gt,
        input_ids,
    ):
        with self.queue_lock:
            self.pending.append(
                (message, reference_answer, gt, input_ids, 4)
            )

            if len(self.pending) < self.queue_size:
                return []

            ready = self.pending[:self.queue_size]
            del self.pending[:self.queue_size]
        return self._handle_batch(ready)

    def flush(self):
        with self.queue_lock:
            if not self.pending:
                return []
            ready = self.pending
            self.pending = []
        return self._handle_batch(ready)

    def _handle_batch(self, requests):
        prompts = change_prompts(
            [item[0] for item in requests],
            [item[1] for item in requests],
        )
        n_values = {item[4] for item in requests}
        if len(n_values) != 1:
            raise RuntimeError(
                "Teacher inference batch has inconsistent generation counts."
            )

        print("-" * 80, flush=True)
        print("TEACHER START", flush=True)
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=16000,
                n=n_values.pop(),
            ),
        )
        print("TEACHER END", flush=True)
        print("-" * 80, flush=True)

        results = []
        for request, prompt, output in zip(requests, prompts, outputs):
            _, _, gt, input_ids, _ = request
            accuracy = sum(
                reward(prompt + completion.text, gt) >= 1
                for completion in output.outputs
            ) / max(len(output.outputs), 1)
            results.append(
                (
                    input_ids,
                    torch.tensor(output.prompt_token_ids, dtype=torch.long),
                    accuracy,
                )
            )
        return results


@ray.remote(num_gpus=2)
class BaseEntropyActor:
    """Frozen HF base model used only for privileged teacher entropy."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # Split the frozen model across both GPUs assigned by Ray.
            device_map="balanced",
        ).eval()
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def calculate_entropies(self, mini_batch):
        device = next(self.model.parameters()).device
        input_ids = [x["input_ids"] for x in mini_batch]
        reward_masks = [x["reward_mask"] for x in mini_batch]
        teacher_prefixes = [
            torch.as_tensor(
                x["teacher_prefix"],
                device=device,
                dtype=torch.long,
            )
            for x in mini_batch
        ]

        debug("H target")
        return OPSD.calculate_entropy_of_teacher(
            hf_model=self.model,
            sequences=input_ids,
            reward_masks=reward_masks,
            pad_token_id=self.tokenizer.pad_token_id,
            teacher_prefixes=teacher_prefixes,
        ).cpu()


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
        model.config.pad_token_id = self.tokenizer.pad_token_id

        decoder_layer_cls = get_decoder_layer_cls(model)

        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={decoder_layer_cls},
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
            check_fn=lambda module: isinstance(module, decoder_layer_cls),
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

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
            debug("H target")
            debug(H_targets.shape)
            _h_targets = torch.stack([
                torch.as_tensor(x["teacher_entropies"], dtype=dtype)
                for x in mini_batch
            ]).to(device)
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

        entropy_sum = (entropies * masks).sum().detach().item()
        entropy_count = masks.sum().detach().item()

        return loss, entropy_sum, entropy_count

    def process_batch(self, batch, max_tokens_per_mini_batch, epoch=1):
        n_total_tokens, mini_batches = TreeRewardManager.process_batch(
            batch,
            max_tokens_per_mini_batch,
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
            response_entropy_sum = 0.0
            response_entropy_count = 0.0

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
                
                loss, entropy_sum, entropy_count = self.__process_mini_batch__(
                    mini_batch=mini_batch,
                    old_token_log_probs=old_token_log_probs_array[i],
                )

                total_loss += loss.detach().item()
                response_entropy_sum += entropy_sum
                response_entropy_count += entropy_count

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

        return {
            "loss": total_loss,
            "response_entropy_sum": response_entropy_sum,
            "response_entropy_count": response_entropy_count,
        }

        
    
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
        base_model_path: str,
        rollout_gpus: list[int],
        trainer_gpus: list[int],
        master_addr: str = "127.0.0.1",
        master_port: int = 29588,
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
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

        if len(rollout_gpus) < 6:
            raise ValueError(
                "rollout_gpus must contain six GPUs: two policy rollout "
                "GPUs, two frozen vLLM teacher GPUs, and two frozen HF "
                "entropy GPUs."
            )

        required_gpus = 6 + self.world_size
        available_gpus = int(ray.cluster_resources().get("GPU", 0))
        if available_gpus < required_gpus:
            raise RuntimeError(
                f"This configuration needs {required_gpus} Ray GPUs "
                f"(2 rollout + 2 teacher + 2 entropy + "
                f"{self.world_size} FSDP ranks), but Ray advertises only "
                f"{available_gpus}. The actors would remain pending forever."
            )
        self.teacher = BaseTeacherActor.remote(model_path=self.base_model_path)
        self.entropy = BaseEntropyActor.remote(
            model_path=self.base_model_path
        )
        self.rollout = RolloutActor.remote(
            model_path=model_path,
            teacher_actor=self.teacher,
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

        # Match the trainer's token-budget mini-batches exactly. This bounds
        # base teacher memory and ensures every stored row has the same
        # padded width as the corresponding policy mini-batch, since both
        # sides bucket the same sorted batch with the same token budget.
        max_tokens_per_mini_batch = 8192
        teacher_batches = TreeRewardManager.bucket_by_token_budget(
            batch,
            max_tokens_per_mini_batch,
        )
        entropy_batches = ray.get([
            self.entropy.calculate_entropies.remote(items)
            for items in teacher_batches
        ])
        for items, entropy_rows in zip(teacher_batches, entropy_batches):
            for item, entropies in zip(items, entropy_rows):
                item["teacher_entropies"] = entropies

        trainer_stats = ray.get([
            self.trainers[i].process_batch.remote(
                batch,
                max_tokens_per_mini_batch,
            )
            for i in range(self.world_size)
        ])

        return trainer_stats

    def step(self, prompts, ground_truths, reference_answers):
        rollout_start = time.perf_counter()
        (
            manager,
            input_ids,
            mean_reward,
            mean_teacher_reward,
            total_response_length,
            avg_response_length,
            avg_rollouts_per_prompt,
            eos_rate,
        ) = self.rollout_samples(
            prompts,
            ground_truths,
            reference_answers,
        )
        mean_teacher_reward = (
            sum(manager.teacher_rewards) / len(manager.teacher_rewards)
            if manager.teacher_rewards
            else 0.0
        )
        rollout_seconds = time.perf_counter() - rollout_start

        optimizer_start = time.perf_counter()
        trainer_stats = self.train_on_samples(
            manager=manager,
            input_ids=input_ids,
        )
        optimizer_seconds = time.perf_counter() - optimizer_start

        losses = [item["loss"] for item in trainer_stats]
        entropy_sum = sum(
            item["response_entropy_sum"] for item in trainer_stats
        )
        entropy_count = sum(
            item["response_entropy_count"] for item in trainer_stats
        )
        
        return {
            "losses": losses,
            "num_samples": len(input_ids),
            "avg_loss": sum(losses) / len(losses),
            "reward/mean": mean_reward,
            "teacher_reward/mean": mean_teacher_reward,
            "rollout/total_response_length": total_response_length,
            "rollout/avg_response_length": avg_response_length,
            "rollout/avg_rollouts_per_prompt": avg_rollouts_per_prompt,
            "rollout/eos_rate": eos_rate,
            "rollout/truncated_rate": 1.0 - eos_rate,
            "rollout/avg_response_entropy": (
                entropy_sum / entropy_count if entropy_count else 0.0
            ),
            "time/rollout_seconds": rollout_seconds,
            "time/optimizer_seconds": optimizer_seconds,
        }

    def eval_aime25(self, **kwargs):
        """
        Run eval/evaluate_math.py's AIME25 scoring logic against the
        rollout actor's already-loaded engine, so this doesn't need to
        allocate a separate vLLM instance/GPUs just for eval.
        """
        return ray.get(
            self.rollout.eval_dataset.remote(dataset_name="aime25", **kwargs)
        )

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
