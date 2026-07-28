import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from mathruler.grader import grade_answer


def normalize_ground_truth(answer: Any) -> str:
    """
    Convert SVAMP answers into a format accepted by grade_answer.

    Examples:
        5       -> "5"
        5.0     -> "5"
        2.5     -> "2.5"
        "4,300" -> "4300"
    """
    answer = str(answer).strip().replace(",", "")

    try:
        value = float(answer)

        if value.is_integer():
            return str(int(value))

        return str(value)
    except ValueError:
        return answer


def extract_answer(text: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)

    if boxed:
        return boxed[-1].strip().replace(",", "")

    final_answers = re.findall(
        r"Final answer:\s*([^\n]+)",
        text,
        flags=re.I,
    )

    if final_answers:
        answer = final_answers[-1].strip()

        # Remove leftover LaTeX wrappers.
        answer = answer.replace(r"\boxed", "")
        answer = answer.strip("{} ")
        answer = answer.replace(",", "")

        return answer

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text.replace(",", ""),
    )

    return numbers[-1].strip() if numbers else None


def make_messages(question: str) -> List[Dict[str, str]]:
    return [
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
                + question
            ),
        }
    ]


def get_svamp_question(sample: Dict[str, Any]) -> str:
    """
    Supports common SVAMP dataset schemas.

    ChilleD/SVAMP usually contains:
        Body
        Question
        Answer

    Some variants contain:
        body
        question
        answer

    Other variants provide a combined question field.
    """
    body = (
        sample.get("Body")
        or sample.get("body")
        or sample.get("body_text")
        or ""
    )

    question = (
        sample.get("Question")
        or sample.get("question")
        or ""
    )

    body = str(body).strip()
    question = str(question).strip()

    if body and question:
        # Avoid duplicating the question when a dataset already provides
        # the complete problem in the question field.
        if question.startswith(body):
            return question

        return f"{body} {question}"

    if question:
        return question

    if body:
        return body

    raise KeyError(
        f"Could not find an SVAMP question field. "
        f"Available columns: {list(sample.keys())}"
    )


def get_svamp_answer(sample: Dict[str, Any]) -> str:
    for key in ["Answer", "answer", "result", "Result"]:
        if key in sample and sample[key] is not None:
            return normalize_ground_truth(sample[key])

    raise KeyError(
        f"Could not find an SVAMP answer field. "
        f"Available columns: {list(sample.keys())}"
    )


@ray.remote(num_gpus=1)
class VLLMSVAMPWorker:
    def __init__(
        self,
        rank: int,
        model: str,
        batch_size: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float,
    ):
        self.rank = rank
        self.batch_size = batch_size

        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

    def run(
        self,
        shard: List[Dict[str, Any]],
        output_path: str,
    ) -> Dict[str, Any]:
        correct = 0
        total = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for start in range(0, len(shard), self.batch_size):
                batch = shard[start : start + self.batch_size]

                prompts = [
                    self.tokenizer.apply_chat_template(
                        make_messages(sample["question"]),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for sample in batch
                ]

                ground_truths = [
                    sample["ground_truth"]
                    for sample in batch
                ]

                outputs = self.llm.generate(
                    prompts,
                    self.sampling_params,
                )

                for sample, out, ground_truth in zip(
                    batch,
                    outputs,
                    ground_truths,
                ):
                    response = out.outputs[0].text
                    prediction = extract_answer(response)

                    is_correct = (
                        prediction is not None
                        and grade_answer(prediction, ground_truth)
                    )

                    row = {
                        "idx": sample["idx"],
                        "dataset_id": sample.get("dataset_id"),
                        "rank": self.rank,
                        "question": sample["question"],
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "correct": bool(is_correct),
                        "response": response,
                    }

                    f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()

                    correct += int(is_correct)
                    total += 1

                print(
                    f"[rank {self.rank}] "
                    f"{total}/{len(shard)} "
                    f"acc={correct / max(total, 1):.4f}",
                    flush=True,
                )

        return {
            "rank": self.rank,
            "output_path": output_path,
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
        }


def merge_outputs(
    output_dir: str,
    output_json: str,
    model: str,
) -> Dict[str, Any]:
    rows = []

    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".jsonl"):
            continue

        path = os.path.join(output_dir, name)

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    rows.append(json.loads(line))

    rows.sort(key=lambda row: row["idx"])

    correct = sum(
        int(row["correct"])
        for row in rows
    )
    total = len(rows)

    final = {
        "model": model,
        "dataset": "ChilleD/SVAMP",
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "results": rows,
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return final


def clear_old_worker_outputs(output_dir: str) -> None:
    """
    Prevent old rank JSONL files from being included in a new evaluation.
    """
    for name in os.listdir(output_dir):
        if name.startswith("rank") and name.endswith(".jsonl"):
            os.remove(os.path.join(output_dir, name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="svamp_ray_outputs",
    )
    parser.add_argument(
        "--output-json",
        default="svamp_ray_results.json",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )

    args = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAY_DEDUP_LOGS"] = "0"

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    clear_old_worker_outputs(args.output_dir)

    # SVAMP does not have an official train/test split in its original
    # release, so this Hugging Face version exposes the complete benchmark
    # as one split.
    dataset = load_dataset(
        "ChilleD/SVAMP",
        split="train",
    )

    print(
        f"SVAMP columns: {dataset.column_names}",
        flush=True,
    )
    print(
        f"SVAMP examples: {len(dataset)}",
        flush=True,
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    rows = []

    for idx, sample in enumerate(dataset):
        rows.append(
            {
                "idx": idx,
                "dataset_id": (
                    sample.get("ID")
                    or sample.get("id")
                    or sample.get("Id")
                ),
                "question": get_svamp_question(sample),
                "ground_truth": get_svamp_answer(sample),
            }
        )

    print("First normalized example:")
    print(json.dumps(rows[0], indent=2))
    print(flush=True)

    shards = [
        rows[rank :: args.num_workers]
        for rank in range(args.num_workers)
    ]

    print("BEFORE INIT", flush=True)

    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=args.num_workers,
        num_cpus=args.num_cpus,
        ignore_reinit_error=True,
    )

    print("AFTER INIT", flush=True)

    workers = [
        VLLMSVAMPWorker.remote(
            rank=rank,
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for rank in range(args.num_workers)
    ]

    print("LAUNCHED WORKERS", flush=True)

    futures = []

    for rank, worker in enumerate(workers):
        output_path = os.path.join(
            args.output_dir,
            f"rank{rank}.jsonl",
        )

        futures.append(
            worker.run.remote(
                shards[rank],
                output_path,
            )
        )

    stats = ray.get(futures)

    print("=" * 80)
    print("Per-rank stats:")

    for stat in stats:
        print(stat)

    final = merge_outputs(
        output_dir=args.output_dir,
        output_json=args.output_json,
        model=args.model,
    )

    print("=" * 80)
    print(
        f"FINAL accuracy: "
        f"{final['correct']}/{final['total']} "
        f"= {final['accuracy']:.4f}"
    )
    print(
        f"Saved JSON to: {args.output_json}"
    )

    ray.shutdown()


if __name__ == "__main__":
    main()