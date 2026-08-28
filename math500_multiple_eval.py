import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from mathruler.grader import grade_answer


import re
from typing import Optional


def extract_last_boxed(text: str) -> Optional[str]:
    marker = r"\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    answer_start = i

    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return text[answer_start:i].strip()

        i += 1

    # Unclosed \boxed{
    return None


def extract_answer(text: str) -> Optional[str]:
    # Prefer the last boxed answer.
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed

    # Fall back to explicitly formatted final answer.
    m = re.findall(r"Final answer:\s*([^\n]+)", text, flags=re.I)
    if m:
        answer = m[-1].strip()

        boxed = extract_last_boxed(answer)
        if boxed is not None:
            return boxed

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
        num_runs: int,
    ):
        self.rank = rank
        self.batch_size = batch_size
        self.num_runs = num_runs

        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
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
        output_path: Optional[str] = None,
        save_output: bool = False,
    ):
        correct = 0
        total = 0
        run_stats = {
            run_id: {"correct": 0, "total": 0}
            for run_id in range(self.num_runs)
        }

        f = open(output_path, "w") if save_output and output_path else None
        try:
            # Run the entire shard once per benchmark run. Each vLLM request
            # returns exactly one generation, so a run is one independently
            # sampled answer for every MATH-500 problem.
            for run_id in range(self.num_runs):
                run_correct = 0
                run_total = 0

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
                        candidate = out.outputs[0]
                        response = candidate.text
                        pred = extract_answer(response)

                        ok = (
                            pred is not None
                            and grade_answer(pred, gt)
                        )

                        row = {
                            "idx": sample["idx"],
                            "run_id": run_id,
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

                        if f is not None:
                            f.write(
                                json.dumps(
                                    row,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                        correct += int(ok)
                        total += 1
                        run_correct += int(ok)
                        run_total += 1
                        run_stats[run_id]["correct"] += int(ok)
                        run_stats[run_id]["total"] += 1

                print(
                    f"[rank {self.rank}] "
                    f"run {run_id + 1}/{self.num_runs}: "
                    f"{run_correct}/{run_total} "
                    f"acc={run_correct / max(run_total, 1):.4f}",
                    flush=True,
                )
        finally:
            if f is not None:
                f.close()

        return {
            "rank": self.rank,
            "output_path": output_path if save_output else None,
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "run_stats": run_stats,
        }



def aggregate_worker_stats(stats, num_runs: int):
    run_stats = {
        run_id: {"correct": 0, "total": 0}
        for run_id in range(num_runs)
    }
    total = 0
    correct = 0
    for worker_stats in stats:
        total += worker_stats["total"]
        correct += worker_stats["correct"]
        for run_id_raw, s in worker_stats["run_stats"].items():
            run_id = int(run_id_raw)
            run_stats[run_id]["correct"] += s["correct"]
            run_stats[run_id]["total"] += s["total"]
    run_accuracies = []
    for run_id in range(num_runs):
        s = run_stats[run_id]
        s["accuracy"] = s["correct"] / max(s["total"], 1)
        run_accuracies.append(s["accuracy"])
    average_accuracy = sum(run_accuracies) / len(run_accuracies) if run_accuracies else 0.0
    return {
        "num_runs": num_runs,
        "samples_total": total,
        "correct_total": correct,
        "average_accuracy": average_accuracy,
        "run_accuracies": run_accuracies,
        "run_stats": run_stats,
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

    rows.sort(key=lambda x: (x.get("run_id", 0), x["idx"]))

    correct = sum(int(x["correct"]) for x in rows)
    total = len(rows)

    # Accuracy for each independent run (one sampled answer per MATH500 problem).
    run_stats = {}
    for row in rows:
        run_id = row.get("run_id", 0)
        if run_id not in run_stats:
            run_stats[run_id] = {"correct": 0, "total": 0}
        run_stats[run_id]["correct"] += int(row["correct"])
        run_stats[run_id]["total"] += 1

    run_accuracies = []
    for run_id in sorted(run_stats):
        stats = run_stats[run_id]
        stats["accuracy"] = stats["correct"] / max(stats["total"], 1)
        run_accuracies.append(stats["accuracy"])

    average_accuracy = (
        sum(run_accuracies) / len(run_accuracies)
        if run_accuracies else 0.0
    )

    final = {
        "model": model,
        "dataset": "HuggingFaceH4/MATH-500",
        "num_runs": len(run_accuracies),
        "samples_total": total,
        "correct_total": correct,
        "average_accuracy": average_accuracy,
        "run_accuracies": run_accuracies,
        "run_stats": run_stats,
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
        default=0.6,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=64,
        help="Independent sampled generations per problem.",
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
        "--save-output",
        action="store_true",
        help="Save JSONL/JSON outputs. Default: print results only.",
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

    if args.save_output:
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
            num_runs=args.num_runs,
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
                output_path if args.save_output else None,
                args.save_output,
            )
        )

    stats = ray.get(futures)

    print("=" * 80)
    print("Per-rank stats:")

    for s in stats:
        print(s)

    final = aggregate_worker_stats(stats, args.num_runs)

    print("=" * 80)
    print(
        f"FINAL average accuracy over {final['num_runs']} independent runs: "
        f"{final['average_accuracy']:.4f}"
    )
    print(
        f"Total sampled answers: {final['samples_total']} "
        f"({len(ds)} problems x {args.num_runs} runs)"
    )

    for run_id, acc in enumerate(final["run_accuracies"]):
        rs = final["run_stats"][run_id]
        print(
            f"run {run_id:02d}: "
            f"{rs['correct']}/{rs['total']} = {acc:.4f}"
        )

    if args.save_output:
        merge_outputs(
            args.output_dir,
            args.output_json,
            args.model,
        )
        print(f"\nSaved JSON to: {args.output_json}")
    else:
        print("\nOutput saving disabled (use --save-output to enable).")

    ray.shutdown()


if __name__ == "__main__":
    main()
