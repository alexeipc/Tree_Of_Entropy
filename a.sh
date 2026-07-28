#!/bin/bash
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

ray stop --force || true

# Triton needs this to find libcuda.so.1
export TRITON_LIBCUDA_PATH="$HOME/cuda-compat-fake"
export TRITON_CACHE_DIR="$HOME/.triton/cache"

# Keep Singularity host driver libs first
export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH:-}

export TORCHDYNAMO_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export VLLM_USE_V1=0

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

while true; do
python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    tensor_parallel_size=1,
)

params = SamplingParams(
    temperature=0.8,
    max_tokens=64,
)

llm.generate(
    ["What is 123 + 456? Explain briefly."],
    params,
)
PY

sleep 60
done