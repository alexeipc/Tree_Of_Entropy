import os

# Set before importing torch/vLLM.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

from vllm import LLM, SamplingParams


MODEL_PATH = "./sft/Llama-3.2-3B-Instruct_mot_sft_final"


def main() -> None:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,

        # Use the same important settings as the real rollout engine.
        dtype="bfloat16",
        trust_remote_code=True,

        enforce_eager=False,
    )

    outputs = llm.generate(
        ["Hello"],
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
        ),
    )

    print("vLLM H200 warmup completed successfully.")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()