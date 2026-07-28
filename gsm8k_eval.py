import os
import re
import json
import argparse
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


def make_messages(question: str) -> str:
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


def merge_outputs(output_dir: str, output_json: str, model: str):
    rows = []

    for name in os.listdir(output_dir):
        if name.endswith(".jsonl"):
            with open(os.path.join(output_dir, name)) as f:
                for line in f:
                    rows.append(json.loads(line))

    rows.sort(key=lambda x: x["idx"])

    correct = sum(int(x["correct"]) for x in rows)
    total = len(rows)

    final = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="gsm8k_ray_outputs")
    parser.add_argument("--output-json", default="gsm8k_3b_ray_results.json")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAY_DEDUP_LOGS"] = "0"

    os.makedirs(args.output_dir, exist_ok=True)

    # Main process loads data ONCE.
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

    # Main process distributes data.
    shards = [rows[i::args.num_workers] for i in range(args.num_workers)]

    print("BEFORE INIT")

    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=args.num_workers,
        num_cpus=args.num_cpus,
        ignore_reinit_error=True,
    )
    
    print("AFTER INIT")

    workers = [
        VLLMGSM8KWorker.remote(
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
    
    print("LAUNCHED WORKERS")

    futures = []
    for i, worker in enumerate(workers):
        output_path = os.path.join(args.output_dir, f"rank{i}.jsonl")
        futures.append(worker.run.remote(shards[i], output_path))

    stats = ray.get(futures)

    print("=" * 80)
    print("Per-rank stats:")
    for s in stats:
        print(s)

    final = merge_outputs(args.output_dir, args.output_json, args.model)

    print("=" * 80)
    print(f"FINAL accuracy: {final['correct']}/{final['total']} = {final['accuracy']:.4f}")
    print(f"Saved JSON to: {args.output_json}")

    ray.shutdown()


if __name__ == "__main__":
    main()