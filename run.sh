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

export WANDB_PROJECT=tree-of-entropy
export WANDB_NAME=llama-3b


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

python deepmath_trainer.py
#python warmup_gpu.py

#CUDA_VISIBLE_DEVICES=0,1,2 python test_sync.py
#python test_main_copy.py
#python sft_eval.py
#CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run  --master_port=29517 --nproc_per_node=3 raw_grpo.py