import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from mathruler.grader import grade_answer


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


def normalize_answer(answer: Any) -> str:
    """
    AIME answers are integers from 000 to 999.

    Keep them as strings for grading. If a dataset stores an integer,
    convert it to the ordinary decimal representation. mathruler handles
    numerical equivalence, so e.g. 4 and 004 are equivalent as numbers.
    """
    if answer is None:
        return ""

    if isinstance(answer, bool):
        return str(int(answer))

    if isinstance(answer, int):
        return str(answer)

    if isinstance(answer, float) and answer.is_integer():
        return str(int(answer))

    return str(answer).strip()


def load_aime_rows(years: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # AIME 2023 and 2024:
    # AI-MO/aimo-validation-aime contains AIME 2022-2024.
    old_years = [y for y in years if y in (2023, 2024)]

    if old_years:
        ds = load_dataset(
            "AI-MO/aimo-validation-aime",
            split="train",
        )

        for x in ds:
            year = int(x["year"])
            if year not in old_years:
                continue

            rows.append(
                {
                    "year": year,
                    "exam": f"AIME {year}",
                    "problem": x["problem"],
                    "ground_truth": normalize_answer(x["answer"]),
                    "reference_solution": x.get("solution"),
                    "url": x.get("url"),
                }
            )

    # AIME 2025:
    # opencompass/AIME2025 has two configs, 15 questions each.
    if 2025 in years:
        for config_name in ("AIME2025-I", "AIME2025-II"):
            ds = load_dataset(
                "opencompass/AIME2025",
                config_name,
                split="test",
            )

            for x in ds:
                # Current dataset schema uses "question" and "answer".
                # Accept "problem" too in case the dataset schema changes.
                problem = x.get("question", x.get("problem"))
                answer = x.get("answer")

                if problem is None:
                    raise KeyError(
                        f"Could not find question/problem field in "
                        f"opencompass/AIME2025 {config_name}. "
                        f"Available keys: {list(x.keys())}"
                    )

                rows.append(
                    {
                        "year": 2025,
                        "exam": config_name.replace("AIME2025", "AIME 2025"),
                        "problem": problem,
                        "ground_truth": normalize_answer(answer),
                    }
                )

    # Give every sample a stable global index after combining datasets.
    rows.sort(key=lambda x: (x["year"], x["exam"]))
    for i, row in enumerate(rows):
        row["idx"] = i

    return rows


@ray.remote(num_gpus=1)
class VLLMAIMEWorker:
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
                        "year": sample["year"],
                        "exam": sample["exam"],
                        "problem": sample["problem"],
                        "ground_truth": gt,
                        "prediction": pred,
                        "correct": bool(ok),
                        "response": response,
                    }

                    if sample.get("reference_solution") is not None:
                        row["reference_solution"] = sample["reference_solution"]

                    if sample.get("url") is not None:
                        row["url"] = sample["url"]

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
    years: List[int],
):
    rows = []

    for name in os.listdir(output_dir):
        if not name.endswith(".jsonl"):
            continue

        with open(os.path.join(output_dir, name)) as f:
            for line in f:
                rows.append(json.loads(line))

    rows.sort(key=lambda x: x["idx"])

    correct = sum(int(x["correct"]) for x in rows)
    total = len(rows)

    final = {
        "model": model,
        "dataset": "AIME 2023/2024/2025",
        "years": years,
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "results": rows,
    }

    per_year = {}
    per_exam = {}

    for row in rows:
        year = str(row["year"])
        exam = row["exam"]

        if year not in per_year:
            per_year[year] = {
                "correct": 0,
                "total": 0,
            }

        per_year[year]["total"] += 1
        per_year[year]["correct"] += int(row["correct"])

        if exam not in per_exam:
            per_exam[exam] = {
                "correct": 0,
                "total": 0,
            }

        per_exam[exam]["total"] += 1
        per_exam[exam]["correct"] += int(row["correct"])

    for stats in per_year.values():
        stats["accuracy"] = (
            stats["correct"]
            / max(stats["total"], 1)
        )

    for stats in per_exam.values():
        stats["accuracy"] = (
            stats["correct"]
            / max(stats["total"], 1)
        )

    final["per_year"] = per_year
    final["per_exam"] = per_exam

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
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        choices=[2023, 2024, 2025],
        help="AIME years to evaluate. Default: 2023 2024 2025",
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
        help="Optional limit AFTER combining the selected AIME years.",
    )

    parser.add_argument(
        "--output-dir",
        default="aime_23_24_25_ray_outputs",
    )

    parser.add_argument(
        "--output-json",
        default="aime_23_24_25_ray_results.json",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )

    args = parser.parse_args()

    # Remove duplicate years while preserving command-line order.
    args.years = list(dict.fromkeys(args.years))

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAY_DEDUP_LOGS"] = "0"

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    rows = load_aime_rows(args.years)

    if args.limit is not None:
        rows = rows[: args.limit]
        for i, row in enumerate(rows):
            row["idx"] = i

    print(
        f"Loaded {len(rows)} AIME problems "
        f"for years {args.years}",
        flush=True,
    )

    counts = {}
    for row in rows:
        counts[row["year"]] = counts.get(row["year"], 0) + 1

    for year in sorted(counts):
        print(
            f"  AIME {year}: {counts[year]} problems",
            flush=True,
        )

    # Prevent stale rank*.jsonl files from an earlier run from contaminating
    # the merged result.
    for name in os.listdir(args.output_dir):
        if name.startswith("rank") and name.endswith(".jsonl"):
            os.remove(os.path.join(args.output_dir, name))

    # Round-robin split keeps workers similarly sized.
    shards = [
        rows[i::args.num_workers]
        for i in range(args.num_workers)
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
        VLLMAIMEWorker.remote(
            rank=i,
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for i in range(args.num_workers)
    ]

    print("LAUNCHED WORKERS", flush=True)

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
        args.years,
    )

    print("=" * 80)
    print(
        f"FINAL accuracy: "
        f"{final['correct']}/{final['total']} "
        f"= {final['accuracy']:.4f}"
    )

    print("\nPer-year accuracy:")
    for year, stats in sorted(final["per_year"].items()):
        print(
            f"AIME {year}: "
            f"{stats['correct']:2d}/{stats['total']:2d} "
            f"= {stats['accuracy']:.4f}"
        )

    print("\nPer-exam accuracy:")
    for exam, stats in sorted(final["per_exam"].items()):
        print(
            f"{exam:15s}: "
            f"{stats['correct']:2d}/{stats['total']:2d} "
            f"= {stats['accuracy']:.4f}"
        )

    print(
        f"\nSaved JSON to: "
        f"{args.output_json}"
    )

    ray.shutdown()


if __name__ == "__main__":
    main()