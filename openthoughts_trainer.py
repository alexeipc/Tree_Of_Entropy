from datasets import load_dataset
from mathruler.grader import grade_answer
from pair_worker import RLController
from util.reward_func import extract_last_boxed
import wandb
from util.debug import debug
import os
import random

import ray


# Change this to your actual 1.7B checkpoint.
BASE_MODEL_PATH = "Qwen/Qwen3-1.7B"
CHECKPOINT_DIR = "/scratch/pioneer/users/ptd18/models/checkpoints/tuned-qwen3-1.7B-toe-opsd-openthoughts-math-30k-no-sft-16000-max-length-temp-1.1"

BATCH_SIZE = 32
CHECKPOINT_STEP = 25
UPDATE_STEP = 1
NUM_STEPS = 500

# AIME25 eval, run every CHECKPOINT_STEP steps on the rollout actor's
# already-loaded engine (no extra GPUs needed).
RUN_EVAL = False
EVAL_VAL_N = 12
EVAL_MAX_NEW_TOKENS = 16000

# Number of already completed batches.
# For example, 500 means batches 0 through 499 were already completed.
START_STEP = 0

# Dataset configuration.
DATASET_NAME = "siyanzhao/Openthoughts_math_30k_opsd"
DATASET_SPLIT = "train"

# Shuffle once deterministically before training.
SHUFFLE_DATASET = True
DATASET_SEED = 42


def normalize_final_answer(answer) -> str:
    """
    Convert the dataset's Answer field into a clean string.

    Do not aggressively modify the mathematical expression here because the
    reward function should handle equivalent LaTeX representations.
    """
    if answer is None:
        return ""

    return str(answer).strip()


def has_matching_reference_solution(sample) -> bool:
    """Keep only rows whose solution's boxed result matches Answer."""
    ground_truth = normalize_final_answer(sample.get("Answer"))
    if not ground_truth:
        return False

    solution = sample.get("solution")
    if solution is None:
        return False

    solution = str(solution).strip()
    if not solution:
        return False

    predicted_answer = extract_last_boxed(solution)
    return (
        predicted_answer is not None
        and grade_answer(predicted_answer, ground_truth)
    )


if __name__ == "__main__":
    random.seed(DATASET_SEED)

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
    )

    print(f"Loaded {len(dataset):,} OpenThoughts-Math samples.")

    original_dataset_size = len(dataset)
    dataset = dataset.filter(
        has_matching_reference_solution,
        num_proc=8,
    )
    print(
        f"After solution filtering: {len(dataset):,} samples "
        f"({original_dataset_size - len(dataset):,} removed because the "
        "solution's boxed answer did not match Answer)"
    )

    if SHUFFLE_DATASET:
        dataset = dataset.shuffle(seed=DATASET_SEED)

    if START_STEP > 0:
        model_path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint-{START_STEP}",
        )

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Checkpoint does not exist: {model_path}"
            )

        print(
            f"Resuming from checkpoint {model_path} "
            f"at training step {START_STEP}"
        )
    else:
        model_path = BASE_MODEL_PATH
        print(f"Starting from base model: {model_path}")

    wandb.init(
        project="tree-of-entropy-qwen",
        name="no-sft-openthoughts-math-30k",
        config={
            "dataset": DATASET_NAME,
            "dataset_size": len(dataset),
            "dataset_seed": DATASET_SEED,
            "lr": 5e-6,
            "alpha": 0.1,
            "eps_clip": 0.05,
            "batch_size": BATCH_SIZE,
            "update_step": UPDATE_STEP,
            "start_step": START_STEP,
            "model_path": model_path,
            "base_model_path": BASE_MODEL_PATH,
        },
    )

    ray.init(
        address=None,
        _temp_dir=os.environ["RAY_TMPDIR"],
        include_dashboard=False,
        num_cpus=8,
        # 2 rollout + 2 teacher + 2 entropy + 2 FSDP ranks.
        num_gpus=8,
    )

    controller = RLController(
        model_path=model_path,
        base_model_path=BASE_MODEL_PATH,
        rollout_gpus=[0, 1, 2, 3, 4, 5],
        trainer_gpus=[6, 7],
    )

    controller.init_nccl_sync()

    total_batches = len(dataset) // BATCH_SIZE

    if NUM_STEPS is None:
        num_steps = total_batches
    else:
        num_steps = min(NUM_STEPS, total_batches)

    if START_STEP >= num_steps:
        raise ValueError(
            f"START_STEP={START_STEP} must be smaller than "
            f"num_steps={num_steps}"
        )

    reward_sum = 0.0
    reward_count = 0
    reward_ema = None
    teacher_reward_sum = 0.0
    teacher_reward_count = 0
    teacher_reward_ema = None
    ema_beta = 0.95

    # START_STEP is the next batch to process.
    for step in range(START_STEP, num_steps):
        start = step * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(dataset))

        batch = dataset.select(range(start, end))

        prompts = []
        ground_truths = []
        reference_answers = []

        for sample in batch:
            question = str(sample["problem"]).strip()
            ground_truth = normalize_final_answer(sample["Answer"])
            reference_answer = str(sample["solution"]).strip()

            if not question:
                raise ValueError(
                    f"Empty problem found at dataset index {start}."
                )

            if not ground_truth:
                raise ValueError(
                    f"Empty Answer found at dataset index {start}."
                )

            prompts.append(
                [
                    {
                        "role": "user",
                        "content": (
                            f"{question}\n\nPlease reason step by step, "
                            "and put your final answer within "
                            "\\boxed{}."
                        ),
                    }
                ]
            )

            ground_truths.append(ground_truth)
            reference_answers.append(reference_answer)

        stats = controller.step(
            prompts=prompts,
            ground_truths=ground_truths,
            reference_answers=reference_answers,
        )

        batch_mean = stats["reward/mean"]
        teacher_batch_mean = stats["teacher_reward/mean"]
        n = stats["num_samples"]

        reward_sum += batch_mean * n
        reward_count += n
        reward_cummean = reward_sum / reward_count

        teacher_reward_sum += teacher_batch_mean * n
        teacher_reward_count += n
        teacher_reward_cummean = (
            teacher_reward_sum / teacher_reward_count
        )

        if reward_ema is None:
            reward_ema = batch_mean
        else:
            reward_ema = (
                ema_beta * reward_ema
                + (1.0 - ema_beta) * batch_mean
            )

        if teacher_reward_ema is None:
            teacher_reward_ema = teacher_batch_mean
        else:
            teacher_reward_ema = (
                ema_beta * teacher_reward_ema
                + (1.0 - ema_beta) * teacher_batch_mean
            )

        current_step = step + 1

        wandb.log(
            {
                "step": current_step,
                "avg_loss": stats["avg_loss"],
                "num_samples": n,
                "reward/mean": batch_mean,
                "reward/cummean": reward_cummean,
                "reward/ema": reward_ema,
                "teacher_reward/mean": teacher_batch_mean,
                "teacher_reward/cummean": teacher_reward_cummean,
                "teacher_reward/ema": teacher_reward_ema,
                "rollout/total_response_length": stats[
                    "rollout/total_response_length"
                ],
                "rollout/avg_response_length": stats[
                    "rollout/avg_response_length"
                ],
                "rollout/avg_rollouts_per_prompt": stats[
                    "rollout/avg_rollouts_per_prompt"
                ],
                "rollout/eos_rate": stats["rollout/eos_rate"],
                "rollout/truncated_rate": stats[
                    "rollout/truncated_rate"
                ],
                "rollout/avg_response_entropy": stats[
                    "rollout/avg_response_entropy"
                ],
                "time/rollout_seconds": stats[
                    "time/rollout_seconds"
                ],
                "time/optimizer_seconds": stats[
                    "time/optimizer_seconds"
                ],
            },
            step=current_step,
        )

        if current_step % UPDATE_STEP == 0:
            debug("=" * 80)
            debug("START UPDATING WEIGHTS")
            controller.nccl_sync()
            debug("DONE UPDATING WEIGHTS")
            debug("=" * 80)

        if current_step % CHECKPOINT_STEP == 0:
            controller.save_checkpoint(
                os.path.join(
                    CHECKPOINT_DIR,
                    f"checkpoint-{current_step}",
                )
            )

            if RUN_EVAL:
                debug("=" * 80)
                debug("RUNNING AIME25 EVAL")
                eval_stats = controller.eval_aime25(
                    val_n=EVAL_VAL_N,
                    max_new_tokens=EVAL_MAX_NEW_TOKENS,
                )
                debug("DONE AIME25 EVAL")
                debug(eval_stats)
                debug("=" * 80)

                wandb.log(
                    {
                        "eval/aime25/average_at_n_pct": eval_stats[
                            "average_at_n_pct"
                        ],
                        "eval/aime25/pass_at_n_pct": eval_stats[
                            "pass_at_n_pct"
                        ],
                        "eval/aime25/majority_vote_at_n_pct": eval_stats[
                            "majority_vote_at_n_pct"
                        ],
                    },
                    step=current_step,
                )

    controller.save_and_sync(
        "/scratch/pioneer/users/ptd18/models/checkpoints/tuned-qwen3-1.7B-toe-opsd-openthoughts-math-30k-no-sft-16000-max-length-temp-1.1/final"
    )

    wandb.finish()
