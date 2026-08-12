import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from mathruler.grader import grade_answer


def extract_answer(text: str) -> Optional[str]:
    # Prefer boxed answers.
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Fall back to explicitly formatted final answer.
    m = re.findall(r"Final answer:\s*([^\n]+)", text, flags=re.I)
    if m:
        answer = m[-1].strip()

        boxed = re.findall(r"\\boxed\{([^{}]+)\}", answer)
        if boxed:
            return boxed[-1].strip()

        return answer

    return None


def make_messages(question: str):
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


@ray.remote(num_gpus=1)
class VLLMMath500Worker:
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

    def run(
        self,
        shard: List[Dict[str, Any]],
        output_path: str,
    ):
        correct = 0
        total = 0

        with open(output_path, "w") as f:
            for start in range(0, len(shard), self.batch_size):
                batch = shard[start : start + self.batch_size]

                prompts = [
                    self.tokenizer.apply_chat_template(
                        make_messages(x["problem"]),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for x in batch
                ]

                gts = [x["ground_truth"] for x in batch]

                outputs = self.llm.generate(
                    prompts,
                    self.sampling_params,
                )

                for sample, out, gt in zip(batch, outputs, gts):
                    response = out.outputs[0].text
                    pred = extract_answer(response)

                    ok = (
                        pred is not None
                        and grade_answer(pred, gt)
                    )

                    row = {
                        "idx": sample["idx"],
                        "rank": self.rank,
                        "problem": sample["problem"],
                        "ground_truth": gt,
                        "prediction": pred,
                        "correct": bool(ok),
                        "response": response,
                    }

                    # Include optional MATH metadata when available.
                    if "subject" in sample:
                        row["subject"] = sample["subject"]

                    if "level" in sample:
                        row["level"] = sample["level"]

                    if "solution" in sample:
                        row["reference_solution"] = sample["solution"]

                    f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()

                    correct += int(ok)
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
):
    rows = []

    for name in os.listdir(output_dir):
        if not name.endswith(".jsonl"):
            continue

        with open(os.path.join(output_dir, name)) as f:
            for line in f:
                rows.append(json.loads(line))

    rows.sort(key=lambda x: x["idx"])

    correct = sum(
        int(x["correct"])
        for x in rows
    )

    total = len(rows)

    final = {
        "model": model,
        "dataset": "HuggingFaceH4/MATH-500",
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "results": rows,
    }

    # Optional per-subject stats.
    subjects = {}

    for row in rows:
        subject = row.get("subject")

        if subject is None:
            continue

        if subject not in subjects:
            subjects[subject] = {
                "correct": 0,
                "total": 0,
            }

        subjects[subject]["total"] += 1
        subjects[subject]["correct"] += int(
            row["correct"]
        )

    for subject, stats in subjects.items():
        stats["accuracy"] = (
            stats["correct"]
            / max(stats["total"], 1)
        )

    final["subjects"] = subjects

    with open(output_json, "w") as f:
        json.dump(
            final,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return final


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
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
        default=2048,
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
        default="math500_ray_outputs",
    )

    parser.add_argument(
        "--output-json",
        default="math500_ray_results.json",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )

    args = parser.parse_args()

    os.environ[
        "VLLM_WORKER_MULTIPROC_METHOD"
    ] = "spawn"

    os.environ[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    os.environ[
        "RAY_DEDUP_LOGS"
    ] = "0"

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # MATH-500 contains exactly 500 evaluation problems.
    ds = load_dataset(
        "HuggingFaceH4/MATH-500",
        split="test",
    )

    if args.limit is not None:
        ds = ds.select(
            range(
                min(
                    args.limit,
                    len(ds),
                )
            )
        )

    rows = []

    for i, x in enumerate(ds):
        row = {
            "idx": i,
            "problem": x["problem"],
            "ground_truth": x["answer"],
        }

        # Preserve metadata if present.
        for key in [
            "solution",
            "subject",
            "level",
            "unique_id",
        ]:
            if key in x:
                row[key] = x[key]

        rows.append(row)

    # Round-robin split keeps workers similarly sized.
    shards = [
        rows[i::args.num_workers]
        for i in range(args.num_workers)
    ]

    print(
        f"Loaded {len(rows)} MATH-500 problems",
        flush=True,
    )

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
        VLLMMath500Worker.remote(
            rank=i,
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=(
                args.gpu_memory_utilization
            ),
        )
        for i in range(args.num_workers)
    ]

    print(
        "LAUNCHED WORKERS",
        flush=True,
    )

    futures = []

    for i, worker in enumerate(workers):
        output_path = os.path.join(
            args.output_dir,
            f"rank{i}.jsonl",
        )

        futures.append(
            worker.run.remote(
                shards[i],
                output_path,
            )
        )

    stats = ray.get(futures)

    print("=" * 80)
    print("Per-rank stats:")

    for s in stats:
        print(s)

    final = merge_outputs(
        args.output_dir,
        args.output_json,
        args.model,
    )

    print("=" * 80)

    print(
        f"FINAL accuracy: "
        f"{final['correct']}/{final['total']} "
        f"= {final['accuracy']:.4f}"
    )

    if final["subjects"]:
        print("\nPer-subject accuracy:")

        for subject, stats in sorted(
            final["subjects"].items()
        ):
            print(
                f"{subject:25s}: "
                f"{stats['correct']:3d}/"
                f"{stats['total']:3d} "
                f"= {stats['accuracy']:.4f}"
            )

    print(
        f"\nSaved JSON to: "
        f"{args.output_json}"
    )

    ray.shutdown()


if __name__ == "__main__":
    main()