import os
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from util.reward_func import reward as single_reward


MODEL_PATH = "./sft/Llama-3.2-3B-Instruct_mot_sft_final"

reward_sum = 0.0
reward_count = 0
reward_ema = None
ema_beta = 0.95


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def get_gsm8k_answer(ans):
    return ans.split("####")[-1].strip()


def reward_func(completions, answer, **kwargs):
    global reward_sum, reward_count, reward_ema

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
        reward_ema = ema_beta * reward_ema + (1 - ema_beta) * batch_mean

    if is_rank0():
        try:
            import wandb
            wandb.log({
                "reward/mean": batch_mean,
                "reward/cummean": reward_cummean,
                "reward/ema": reward_ema,
            })
        except Exception:
            pass

    return rewards


def build_dataset():
    dataset = load_dataset("openai/gsm8k", "main", split="train")

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
                        + sample["question"]
                    ),
                }
            ],
            "answer": get_gsm8k_answer(sample["answer"]),
        }

    dataset = dataset.map(format_sample)

    return dataset.remove_columns(
        [c for c in dataset.column_names if c not in ["prompt", "answer"]]
    )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset()

    config = GRPOConfig(
        output_dir="./trl_grpo_gsm8k_full_set_output_2",

        num_train_epochs=1,
        max_steps=-1,

        learning_rate=1e-6,
        bf16=True,

        # 3 trainer GPUs

        # 2 sequences/GPU × 3 GPUs = 6 completions per microstep
        per_device_train_batch_size=2,

        # 6 completions/microstep × 6 microsteps
        # = 36 completions per optimizer update
        gradient_accumulation_steps=1,

        # 4 completions per prompt
        num_generations=4,

        # 3 prompts × 4 completions = 12 generated at a time
        # Divisible by global microbatch 6
        generation_batch_size=12,

        max_completion_length=512,

        temperature=0.8,
        top_p=0.95,
        top_k=50,

        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,

        report_to=["wandb"],
        run_name="raw-grpo-gsm8k-llama-3b-full-set",

        remove_unused_columns=False,
        gradient_checkpointing=True,

        beta=0.001,
        loss_type="dapo",
    )

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
    )

    trainer.train(
        #resume_from_checkpoint="./trl_grpo_gsm8k_full_set_output/checkpoint-1300"
    )
    trainer.save_model("./trl_grpo_gsm8k_full_set_final")