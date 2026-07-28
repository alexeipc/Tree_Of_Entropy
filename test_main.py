from datasets import load_dataset
from pair_worker import RLController
import wandb

# MODEL_PATH = "./trl_grpo_gsm8k_final"
MODEL_PATH = "./sft/Llama-3.2-3B-Instruct_mot_sft_final"


BATCH_SIZE = 3

def get_gsm8k_answer(ans):
    return ans.split("####")[-1].strip()

if __name__ == "__main__":

    dataset = load_dataset(
        "zwhe99/DeepMath-103K",
        split="train"
    )

    wandb.init(
        project="tree-of-entropy",
        name="llama-3.2-3b",
        config={
            "lr": 1e-6,
            "alpha": 0.1,
            "eps_clip": 0.05,
            "batch_size": BATCH_SIZE,
        }
    )

    controller = RLController(
        model_path=MODEL_PATH,
        rollout_gpus=[0],
        trainer_gpus=[1, 2]
    )

    num_steps = len(dataset) // BATCH_SIZE

    for step in range(num_steps):

        start = step * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(dataset))

        batch = dataset.select(range(start, end))

        prompts = []
        gts = []

        for sample in batch:

            prompts.append([
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem. "
                        "Put the final answer in \\boxed{}.\n\n"
                        + sample["question"]
                    )
                }
            ])

            gts.append(get_gsm8k_answer(sample["final_answer"]))

        stats = controller.step(
            prompts=prompts,
            ground_truths=gts
        )

        wandb.log({
            "step": step,
            "avg_loss": stats["avg_loss"],
            "num_samples": stats["num_samples"],
        })

        print(
            f"[{step+1}/{num_steps}] "
            f"loss={stats['avg_loss']:.6f}"
        )

        if step > 0 and step % 10 == 0:
            controller.save_and_sync(
                "./checkpoints/tuned-llama"
            )

    controller.save_and_sync(
        "./checkpoints/final"
    )

    wandb.finish()
