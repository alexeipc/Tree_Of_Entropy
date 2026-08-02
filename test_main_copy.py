from datasets import load_dataset
from pair_worker import RLController
import wandb
from util.debug import debug
import os


BASE_MODEL_PATH = "./sft/Llama-3.2-3B-Instruct_mot_sft_final"
CHECKPOINT_DIR = "./checkpoints/tuned-llama-b3"

BATCH_SIZE = 4
CHECKPOINT_STEP = 100
UPDATE_STEP = 3
NUM_STEPS = None

# Number of already completed batches.
# For example, 500 means batches 0 through 499 were already completed.
START_STEP = 0


def get_gsm8k_answer(ans):
    return ans.split("####")[-1].strip()


def get_gsm8k_reference_answer(ans):
    return ans.split("####")[0].strip()

if __name__ == "__main__":

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    )

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
        project="tree-of-entropy",
        name="llama-3.2-3b",
        config={
            "lr": 1e-6,
            "alpha": 0.1,
            "eps_clip": 0.05,
            "batch_size": BATCH_SIZE,
            "start_step": START_STEP,
            "model_path": model_path,
        },
    )

    controller = RLController(
        model_path=model_path,
        rollout_gpus=[0, 1],
        trainer_gpus=[2, 3],
    )

    controller.init_nccl_sync()

    if NUM_STEPS is None:
        num_steps = len(dataset) // BATCH_SIZE
    else:
        num_steps = min(NUM_STEPS, len(dataset) // BATCH_SIZE)

    if START_STEP >= num_steps:
        raise ValueError(
            f"START_STEP={START_STEP} must be smaller than "
            f"num_steps={num_steps}"
        )

    reward_sum = 0.0
    reward_count = 0
    reward_ema = None
    ema_beta = 0.95

    # START_STEP is the next batch to process.
    for step in range(START_STEP, num_steps):

        start = step * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(dataset))

        batch = dataset.select(range(start, end))

        prompts = []
        gts = []
        reference_answers = []

        for sample in batch:
            prompts.append(
                [
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
                ]
            )

            gts.append(
                get_gsm8k_answer(sample["answer"])
            )
            
            reference_answers.append(
                get_gsm8k_reference_answer(sample["answer"])
            )

        stats = controller.step(
            prompts=prompts,
            ground_truths=gts,
            reference_answers=reference_answers
        )

        batch_mean = stats["reward/mean"]
        n = stats["num_samples"]

        reward_sum += batch_mean * n
        reward_count += n
        reward_cummean = reward_sum / reward_count

        if reward_ema is None:
            reward_ema = batch_mean
        else:
            reward_ema = (
                ema_beta * reward_ema
                + (1.0 - ema_beta) * batch_mean
            )

        # step is zero-indexed; current_step is the completed step count.
        current_step = step + 1

        wandb.log(
            {
                "step": current_step,
                "avg_loss": stats["avg_loss"],
                "num_samples": n,
                "reward/mean": batch_mean,
                "reward/cummean": reward_cummean,
                "reward/ema": reward_ema,
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
        "./checkpoints/final-b3"
    )

    wandb.finish()
