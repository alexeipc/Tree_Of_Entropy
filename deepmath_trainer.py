from datasets import load_dataset
from mathruler.grader import grade_answer
from pair_worker import RLController
from util.reward_func import extract_last_boxed
import wandb
from util.debug import debug
import os
import random

import ray


# Change this to your actual 8B checkpoint.
BASE_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINT_DIR = "/scratch/pioneer/users/ptd18/models/checkpoints/tuned-llama-8b-toe-opsd-deepmath-no-sft"

BATCH_SIZE = 6
CHECKPOINT_STEP = 100
UPDATE_STEP = 3
NUM_STEPS = None

# Number of already completed batches.
# For example, 500 means batches 0 through 499 were already completed.
START_STEP = 300

# Dataset configuration.
DATASET_NAME = "zwhe99/DeepMath-103K"
DATASET_SPLIT = "train"

# Set these to restrict the difficulty range.
# DeepMath contains difficulty values such as 4.5, 5.0, 8.0, etc.
MIN_DIFFICULTY = None
MAX_DIFFICULTY = None

# Shuffle once deterministically before training.
SHUFFLE_DATASET = True
DATASET_SEED = 42

# DeepMath supplies three R1 solutions.
# "random" samples one solution per problem.
REFERENCE_SOLUTION_MODE = "random"


def normalize_final_answer(answer) -> str:
    """
    Convert DeepMath's final_answer field into a clean string.

    Do not aggressively modify the mathematical expression here because the
    reward function should handle equivalent LaTeX representations.
    """
    if answer is None:
        return ""

    return str(answer).strip()


def get_matching_reference_answers(sample) -> list[str]:
    """
    Return R1 solutions whose boxed result matches DeepMath's final answer.

    The returned reference text contains only the portion after the final
    </think> tag, as expected by OSPD.
    """
    ground_truth = normalize_final_answer(sample.get("final_answer"))
    if not ground_truth:
        return []

    solutions = [
        sample.get("r1_solution_1"),
        sample.get("r1_solution_2"),
        sample.get("r1_solution_3"),
    ]

    matching = []
    for solution in solutions:
        if solution is None:
            continue

        solution = str(solution).strip()
        if not solution:
            continue

        # Keep only everything after the last </think>
        if "</think>" in solution:
            solution = solution.rsplit("</think>", 1)[1].strip()

        predicted_answer = extract_last_boxed(solution)
        if (
            predicted_answer is not None
            and grade_answer(predicted_answer, ground_truth)
        ):
            matching.append(solution)

    return matching


def has_matching_reference_answer(sample) -> bool:
    """Keep only rows containing at least one correct R1 solution."""
    return bool(get_matching_reference_answers(sample))


def get_reference_answer(sample) -> str:
    """Select an answer-matching R1 reference solution for OSPD."""
    matching = get_matching_reference_answers(sample)

    if not matching:
        raise ValueError(
            "DeepMath sample does not contain an R1 solution matching its "
            "final answer."
        )

    if REFERENCE_SOLUTION_MODE == "first":
        return matching[0]

    if REFERENCE_SOLUTION_MODE == "random":
        return random.choice(matching)

    raise ValueError(
        "REFERENCE_SOLUTION_MODE must be either 'first' or 'random', "
        f"but received {REFERENCE_SOLUTION_MODE!r}."
    )


def keep_by_difficulty(sample) -> bool:
    difficulty = sample.get("difficulty")

    if difficulty is None:
        return False

    difficulty = float(difficulty)

    if (
        MIN_DIFFICULTY is not None
        and difficulty < MIN_DIFFICULTY
    ):
        return False

    if (
        MAX_DIFFICULTY is not None
        and difficulty > MAX_DIFFICULTY
    ):
        return False

    return True


if __name__ == "__main__":
    random.seed(DATASET_SEED)

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
    )

    print(f"Loaded {len(dataset):,} DeepMath samples.")

    original_dataset_size = len(dataset)
    dataset = dataset.filter(
        has_matching_reference_answer,
        num_proc=8,
    )
    print(
        f"After R1 answer filtering: {len(dataset):,} samples "
        f"({original_dataset_size - len(dataset):,} removed because no R1 "
        "solution matched final_answer)"
    )

    if MIN_DIFFICULTY is not None or MAX_DIFFICULTY is not None:
        dataset = dataset.filter(
            keep_by_difficulty,
            num_proc=8,
        )

        print(
            f"After difficulty filtering: {len(dataset):,} samples "
            f"(min={MIN_DIFFICULTY}, max={MAX_DIFFICULTY})"
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
        project="tree-of-entropy-llama-8b",
        name="no-sft-deepmath",
        config={
            "dataset": DATASET_NAME,
            "dataset_size": len(dataset),
            "dataset_seed": DATASET_SEED,
            "min_difficulty": MIN_DIFFICULTY,
            "max_difficulty": MAX_DIFFICULTY,
            "reference_solution_mode": REFERENCE_SOLUTION_MODE,
            "lr": 1e-6,
            "alpha": 0.1,
            "eps_clip": 0.05,
            "batch_size": BATCH_SIZE,
            "update_step": UPDATE_STEP,
            "start_step": START_STEP,
            "model_path": model_path,
        },
    )
    
    ray.init(
        address=None,
        _temp_dir=os.environ["RAY_TMPDIR"],
        include_dashboard=False,
        num_cpus=8,
        num_gpus=4,
    )

    controller = RLController(
        model_path=model_path,
        rollout_gpus=[0, 1],
        trainer_gpus=[2, 3],
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
        difficulties = []

        for sample in batch:
            question = str(sample["question"]).strip()
            ground_truth = normalize_final_answer(
                sample["final_answer"]
            )
            reference_answer = get_reference_answer(sample)

            if not question:
                raise ValueError(
                    f"Empty question found at dataset index {start}."
                )

            if not ground_truth:
                raise ValueError(
                    f"Empty final_answer found at dataset index {start}."
                )

            prompts.append(
                [
                    {
                        "role": "user",
                        "content": (
                            "Solve the following mathematics problem.\n\n"
                            "Show your reasoning and use exactly this output "
                            "format:\n\n"
                            "<think>\n"
                            "Write your reasoning here.\n"
                            "</think>\n"
                            "Final answer: \\boxed{answer}\n\n"
                            "Do not place the final answer before the "
                            "</think> tag.\n\n"
                            "Problem:\n"
                            f"{question}"
                        ),
                    }
                ]
            )

            ground_truths.append(ground_truth)
            reference_answers.append(reference_answer)
            difficulties.append(float(sample["difficulty"]))

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
                "rollout/avg_rollouts_per_prompt": stats[
                    "rollout/avg_rollouts_per_prompt"
                ],
                "time/rollout_seconds": stats[
                    "time/rollout_seconds"
                ],
                "time/optimizer_seconds": stats[
                    "time/optimizer_seconds"
                ],
                "data/difficulty_mean": (
                    sum(difficulties) / len(difficulties)
                ),
                "data/difficulty_min": min(difficulties),
                "data/difficulty_max": max(difficulties),
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

    controller.save_and_sync(
        "./checkpoints/final-llama-8b-toe-opsd-deepmath"
    )

    wandb.finish()
