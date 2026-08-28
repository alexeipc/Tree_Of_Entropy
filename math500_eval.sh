#!/bin/bash
set -euo pipefail
set -x

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

#export HF_HOME=/scratch/pioneer/users/ptd18/cache
#export HF_DATASETS_CACHE=/scratch/pioneer/users/ptd18/cache/datasets
#export HF_HUB_CACHE=/scratch/pioneer/users/ptd18/cache/hub

# mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

python math500_multiple_eval.py \
    --model "/scratch/pioneer/users/ptd18/models/checkpoints/tuned-qwen-toe-opsd-deepmath-no-sft/checkpoint-7000" \
    --num-workers 4 \
    --batch-size 128 \
    --num-runs 10 \
    --max-tokens 3500 \
    --output-dir outputs/math500_toe-qwen_multiple-outputs-2 \
    --output-json outputs/math500_toe-qwen_multiple-results-2.json
