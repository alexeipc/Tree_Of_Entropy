import os

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from util.reward_func import reward as single_reward


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = (
    "/scratch/pioneer/users/ptd18/models/"
    "checkpoints/trl_grpo_deepmath_full_set_output"
)

FINAL_MODEL_DIR = (
    "/scratch/pioneer/users/ptd18/models/"
    "trl_grpo_deepmath_full_set_final"
)

WANDB_PROJECT = "tree-of-entropy-llama-8b"

# ------------------------------------------------------------
# Desired GRPO batch structure
#
# 6 prompts/update
# 8 rollouts/prompt
#
# 6 × 8 = 48 completions/update
#
# 4 GPUs
# 2 completions/GPU/microstep
#
# 2 × 4 = 8 completions/microstep
#
# 48 / 8 = 6 gradient accumulation microsteps
# ------------------------------------------------------------

BATCH_SIZE = 6
NUM_GENERATIONS = 8

PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 6
GENERATION_BATCH_SIZE = 48

MAX_COMPLETION_LENGTH = 2048

CHECKPOINT_STEP = 100


# ============================================================
# Reward tracking
# ============================================================

reward_sum = 0.0
reward_count = 0
reward_ema = None
ema_beta = 0.95


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def reward_func(completions, answer, **kwargs):
    global reward_sum
    global reward_count
    global reward_ema

    rewards = []

    for completion, gt in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = completion

        rewards.append(single_reward(text, gt))

    batch_mean = sum(rewards) / max(len(rewards), 1)

    reward_sum += sum(rewards)
    reward_count += len(rewards)

    reward_cummean = reward_sum / max(reward_count, 1)

    if reward_ema is None:
        reward_ema = batch_mean
    else:
        reward_ema = (
            ema_beta * reward_ema
            + (1.0 - ema_beta) * batch_mean
        )

    if is_rank0():
        try:
            import wandb

            wandb.log(
                {
                    "reward/mean": batch_mean,
                    "reward/cummean": reward_cummean,
                    "reward/ema": reward_ema,
                }
            )

        except Exception:
            pass

    return rewards


# ============================================================
# Dataset
# ============================================================

def build_dataset():
    dataset = load_dataset(
        "zwhe99/DeepMath-103K",
        split="train",
    )

    def format_sample(sample):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem.\n"
                        "You must use this exact format:\n\n"
                        "<think>\n"
                        "Write your reasoning here.\n"
                        "</think>\n"
                        "Final answer: \\boxed{answer}\n\n"
                        "Problem:\n"
                        + str(sample["question"])
                    ),
                }
            ],
            "answer": str(sample["final_answer"]).strip(),
        }

    dataset = dataset.map(format_sample)

    columns_to_remove = [
        c
        for c in dataset.column_names
        if c not in ["prompt", "answer"]
    ]

    return dataset.remove_columns(columns_to_remove)


# ============================================================
# Main
# ============================================================

def main():
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # Helps with allocator fragmentation.
    #
    # This was NOT the cause of your previous OOM,
    # but it is still useful during long training runs.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset()

    if is_rank0():
        print("=" * 80)
        print(train_dataset)
        print(train_dataset[0])
        print("=" * 80)

        print("GRPO batch configuration")
        print(f"BATCH_SIZE                  = {BATCH_SIZE} prompts")
        print(f"NUM_GENERATIONS             = {NUM_GENERATIONS}")
        print(
            "TOTAL COMPLETIONS / UPDATE  = "
            f"{BATCH_SIZE * NUM_GENERATIONS}"
        )
        print(
            "PER_DEVICE_TRAIN_BATCH_SIZE = "
            f"{PER_DEVICE_TRAIN_BATCH_SIZE}"
        )
        print(
            "GRADIENT_ACCUMULATION_STEPS = "
            f"{GRADIENT_ACCUMULATION_STEPS}"
        )
        print(
            "GENERATION_BATCH_SIZE       = "
            f"{GENERATION_BATCH_SIZE}"
        )
        print("=" * 80)

    config = GRPOConfig(
        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
        output_dir=OUTPUT_DIR,

        # ----------------------------------------------------
        # Training duration
        # ----------------------------------------------------
        num_train_epochs=1,
        max_steps=-1,

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------
        learning_rate=1e-6,
        bf16=True,

        # ----------------------------------------------------
        # IMPORTANT: FSDP
        #
        # Without this, torchrun + Trainer uses DDP.
        #
        # DDP:
        #   GPU0 -> full 8B model
        #   GPU1 -> full 8B model
        #   GPU2 -> full 8B model
        #   GPU3 -> full 8B model
        #
        # FSDP FULL_SHARD:
        #   parameters / gradients / optimizer states
        #   are sharded across the 4 GPUs.
        # ----------------------------------------------------
        fsdp="full_shard auto_wrap",

        fsdp_config={
            # Llama-3.1 transformer block class
            "transformer_layer_cls_to_wrap": [
                "LlamaDecoderLayer",
            ],

            # Wrap each Llama transformer block.
            "auto_wrap_policy": "transformer_based_wrap",

            # More memory-friendly backward behavior.
            "backward_prefetch": "backward_pre",

            # Important when starting from a HF checkpoint:
            # only rank 0 needs to initially load full weights
            # into CPU RAM before synchronizing/sharding.
            "cpu_ram_efficient_loading": True,

            # Required with cpu_ram_efficient_loading.
            "sync_module_states": True,

            # Usually works better with Trainer /
            # gradient checkpointing.
            "use_orig_params": True,

            # We are using normal HF gradient checkpointing
            # below, so DO NOT also turn on:
            #
            # "activation_checkpointing": True
            #
            # because you previously hit the conflict between
            # FSDP activation checkpointing and
            # TrainingArguments.gradient_checkpointing.
        },

        # ----------------------------------------------------
        # Batch layout
        #
        # 2 / GPU × 4 GPUs × 6 accumulation
        # = 48 generated sequences
        #
        # 96 sequences / 16 generations
        # = 6 unique prompts
        # ----------------------------------------------------
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        num_generations=NUM_GENERATIONS,

        generation_batch_size=GENERATION_BATCH_SIZE,

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------
        max_completion_length=MAX_COMPLETION_LENGTH,

        temperature=0.8,
        top_p=0.95,
        top_k=50,

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,

        report_to=["wandb"],
        run_name="raw-grpo-deepmath-llama-8b-full-set",

        # Needed because reward_func uses "answer".
        remove_unused_columns=False,

        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------
        save_strategy="steps",
        save_steps=CHECKPOINT_STEP,
        save_total_limit=3,

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------
        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        # ----------------------------------------------------
        # GRPO / DAPO
        # ----------------------------------------------------

        # DAPO itself does not require a separate reference
        # model for a KL penalty.
        #
        # Setting beta=0 means TRL will not load the reference
        # model, which saves a LOT of memory.
        beta=0.0,

        loss_type="dapo",

        # Recommended by DAPO/TRL to avoid learning from
        # completions chopped by max_completion_length.
        mask_truncated_completions=True,
    )

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
    )

    # --------------------------------------------------------
    # Print what distributed backend Trainer ACTUALLY chose.
    # --------------------------------------------------------

    if is_rank0():
        print()
        print("=" * 80)
        print("DISTRIBUTED CONFIGURATION")
        print("=" * 80)

        print(
            "distributed_type:",
            trainer.accelerator.distributed_type,
        )

        print(
            "fsdp_plugin:",
            trainer.accelerator.state.fsdp_plugin,
        )

        print("=" * 80)
        print()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    trainer.train(
        # Uncomment when resuming:
        #
        # resume_from_checkpoint=(
        #     OUTPUT_DIR + "/checkpoint-100"
        # )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trainer.save_model(FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()