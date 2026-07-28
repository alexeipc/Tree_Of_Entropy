from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_PATH = "./trl_grpo_gsm8k_output/checkpoint-300"
# MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Evaluate the limit: \[ \lim_{x \to \infty} \sqrt{x} \left( \sqrt[3]{x+1} - \sqrt[3]{x-1} \right) \]"
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        n=3,                  # generate 10 responses
        max_tokens=10000,
        temperature=0.8,
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling_params)

    for i, output in enumerate(outputs[0].outputs):
        print(f"\n{'='*20} Response {i+1} {'='*20}")
        print(output.text)


if __name__ == "__main__":
    main()