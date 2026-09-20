#!/bin/bash
set -euo pipefail
set -x

# Force FlashInfer/vLLM to use the real native Ninja binary
mkdir -p "$HOME/.local/ninja-bin"
ln -sf /usr/bin/ninja "$HOME/.local/ninja-bin/ninja"

export PATH="$HOME/.local/ninja-bin:$PATH"

echo "Using ninja:"
command -v ninja
ninja --version

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=0
#export PYTHONNOUSERSITE=1

export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID:-manual}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS

ray stop --force --temp-dir="$RAY_TMPDIR" 2>/dev/null || true

# Triton needs this to find libcuda.so.1
export TRITON_LIBCUDA_PATH="$HOME/cuda-compat-fake"
export TRITON_CACHE_DIR="$HOME/.triton/cache"

# Keep Singularity host driver libs first
export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH:-}

export TORCHDYNAMO_DISABLE=0
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

MODEL_PATH="Qwen/Qwen3-8B"

# Pick a free port instead of the hardcoded 8000, which may already be taken.
export VLLM_SERVER_PORT=$(python3 -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
')

# vLLM rollout server on GPUs 0,1 (server mode: separate process/GPUs from
# the trainer, per https://huggingface.co/docs/trl/en/vllm_integration).
CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL_PATH" \
    --port "$VLLM_SERVER_PORT" \
    --tensor-parallel-size 2 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1 &
VLLM_SERVER_PID=$!

cleanup() {
    kill "$VLLM_SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the vLLM server to come up before starting the trainer. Loading an
# 8B model (download + weight init + CUDA graph capture) can genuinely take
# several minutes on a cold cache, so this is not just a fixed sleep — but it
# bails out immediately if the server process has died, and gives up after a
# timeout instead of looping forever with no explanation either way.
VLLM_HEALTH_TIMEOUT=1200  # seconds
elapsed=0
until curl -sf "http://localhost:${VLLM_SERVER_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
        echo "vLLM server (pid $VLLM_SERVER_PID) exited before becoming healthy. Check its output above for the crash reason." >&2
        exit 1
    fi

    if (( elapsed >= VLLM_HEALTH_TIMEOUT )); then
        echo "Timed out after ${VLLM_HEALTH_TIMEOUT}s waiting for vLLM server on port ${VLLM_SERVER_PORT}." >&2
        exit 1
    fi

    sleep 5
    elapsed=$(( elapsed + 5 ))
done

# Trainer on GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    dapo_deepmath.py