import os
import re
import json
import argparse
import shutil
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer

import ray
from datasets import load_dataset
from mathruler.grader import grade_answer


def get_gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


def extract_answer(text: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    m = re.findall(r"Final answer:\s*([^\n]+)", text, flags=re.I)
    if m:
        return m[-1].strip()

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1].strip() if nums else None


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
class VLLMGSM8KWorker:
    def __init__(
        self,
        rank: int,
        model: str,
        batch_size: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float,
        seed: int,
    ):
        self.rank = rank
        self.batch_size = batch_size

        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed + rank,
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

    def run(self, shard: List[Dict[str, Any]], output_path: str):
        correct = 0
        total = 0

        with open(output_path, "w") as f:
            for start in range(0, len(shard), self.batch_size):
                batch = shard[start:start + self.batch_size]

                prompts = [
                    self.tokenizer.apply_chat_template(
                        make_messages(x["question"]),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for x in batch
                ]
                gts = [x["ground_truth"] for x in batch]

                outputs = self.llm.generate(prompts, self.sampling_params)

                for sample, out, gt in zip(batch, outputs, gts):
                    response = out.outputs[0].text
                    pred = extract_answer(response)
                    ok = pred is not None and grade_answer(pred, gt)

                    row = {
                        "idx": sample["idx"],
                        "rank": self.rank,
                        "question": sample["question"],
                        "ground_truth": gt,
                        "prediction": pred,
                        "correct": bool(ok),
                        "response": response,
                    }

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

                    correct += int(ok)
                    total += 1

                print(
                    f"[rank {self.rank}] {total}/{len(shard)} "
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


def merge_outputs(run_dir: str, output_json: str, model: str, run_id: int, seed: int):
    rows = []

    for name in os.listdir(run_dir):
        if name.endswith(".jsonl"):
            with open(os.path.join(run_dir, name)) as f:
                for line in f:
                    rows.append(json.loads(line))

    rows.sort(key=lambda x: x["idx"])

    correct = sum(int(x["correct"]) for x in rows)
    total = len(rows)

    final = {
        "run_id": run_id,
        "seed": seed,
        "model": model,
        "dataset": "gsm8k/test",
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "results": rows,
    }

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    return final


def run_one_eval(args, rows, run_id: int, seed: int):
    run_dir = os.path.join(args.output_dir, f"run_{run_id}")

    if os.path.exists(run_dir) and args.overwrite:
        shutil.rmtree(run_dir)

    os.makedirs(run_dir, exist_ok=True)

    shards = [rows[i::args.num_workers] for i in range(args.num_workers)]

    workers = [
        VLLMGSM8KWorker.remote(
            rank=i,
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=seed,
        )
        for i in range(args.num_workers)
    ]

    futures = []
    for i, worker in enumerate(workers):
        output_path = os.path.join(run_dir, f"rank{i}.jsonl")
        futures.append(worker.run.remote(shards[i], output_path))

    stats = ray.get(futures)

    print("=" * 80)
    print(f"Run {run_id} per-rank stats:")
    for s in stats:
        print(s)

    output_json = os.path.join(run_dir, "results.json")
    final = merge_outputs(
        run_dir=run_dir,
        output_json=output_json,
        model=args.model,
        run_id=run_id,
        seed=seed,
    )

    print("=" * 80)
    print(
        f"RUN {run_id} accuracy: "
        f"{final['correct']}/{final['total']} = {final['accuracy']:.6f}"
    )
    print(f"Saved run JSON to: {output_json}")

    return {
        "run_id": run_id,
        "seed": seed,
        "total": final["total"],
        "correct": final["correct"],
        "accuracy": final["accuracy"],
        "result_json": output_json,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--n-run-times", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1234)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="gsm8k_multi_run_outputs")
    parser.add_argument("--summary-json", default="summary.json")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAY_DEDUP_LOGS"] = "0"

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("gsm8k", "main", split="test")

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows = []
    for i, x in enumerate(ds):
        rows.append({
            "idx": i,
            "question": x["question"],
            "ground_truth": get_gsm8k_answer(x["answer"]),
        })

    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=args.num_workers,
        ignore_reinit_error=True,
    )

    summary = []

    for run_id in range(args.n_run_times):
        seed = args.base_seed + run_id * 1000

        print("\n" + "#" * 80)
        print(f"Starting run {run_id}/{args.n_run_times - 1}, seed={seed}")
        print("#" * 80, flush=True)

        run_summary = run_one_eval(args, rows, run_id, seed)
        summary.append(run_summary)

    summary_path = os.path.join(args.output_dir, args.summary_json)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ALL RUN SUMMARY")
    print("=" * 80)

    for s in summary:
        print(
            f"run={s['run_id']} seed={s['seed']} "
            f"correct={s['correct']}/{s['total']} "
            f"acc={s['accuracy']:.6f}"
        )

    accs = [s["accuracy"] for s in summary]
    print("=" * 80)
    print(f"accuracies = {accs}")
    print(f"Saved summary JSON to: {summary_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()