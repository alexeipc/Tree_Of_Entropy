from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH = "./data/deepmath_sft_4_6_qwen2_5_7b"

MIN_DIFFICULTY = 4.0
MAX_DIFFICULTY = 6.0
MAX_LEN = 2048
MAX_SAMPLES = 30_000
SEED = 42


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)


def valid_example(example):
    difficulty = example.get("difficulty")
    solution = example.get("r1_solution_1")

    if difficulty is None:
        return False

    return (
        MIN_DIFFICULTY <= float(difficulty) <= MAX_DIFFICULTY
        and example.get("question") is not None
        and str(example["question"]).strip()
        and solution is not None
        and str(solution).strip()
    )


def format_example(example):
    question = str(example["question"]).strip()
    solution = str(example["r1_solution_1"]).strip()

    # DeepMath solution already contains </think> and \boxed{...}.
    if not solution.startswith("<think>"):
        solution = "<think>\n" + solution

    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following mathematics problem.\n\n"
                "Show your reasoning and use exactly this output format:\n\n"
                "<think>\n"
                "Write your reasoning here.\n"
                "</think>\n"
                "Final answer: \\boxed{answer}\n\n"
                "Problem:\n"
                f"{question}"
            ),
        },
        {
            "role": "assistant",
            "content": solution,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def length_filter(example):
    length = len(
        tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=False,
        ).input_ids
    )

    return length <= MAX_LEN


dataset = load_dataset(
    "zwhe99/DeepMath-103K",
    split="train",
)

dataset = dataset.filter(
    valid_example,
    num_proc=4,
)

dataset = dataset.shuffle(seed=SEED)

dataset = dataset.select(
    range(min(MAX_SAMPLES, len(dataset)))
)

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
    num_proc=4,
    writer_batch_size=100,
)

dataset = dataset.filter(
    length_filter,
    num_proc=4,
    writer_batch_size=100,
)

print("Final dataset size:", len(dataset))
print(dataset[0]["text"][:3000])

dataset.save_to_disk(
    OUTPUT_PATH,
)

print("Saved to:", OUTPUT_PATH)
