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

#BASE_MODEL="Qwen/Qwen3-1.7B"
BASE_MODEL="../checkpoints/final-qwen3-1.7B-toe-opsd-openthoughts-math-30k"

EXP_DIR="/scratch/pioneer/users/ptd18/eval/aime25/Qwen3-1.7B-temp-1.1/checkpoint-500"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime25" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 2 \
    --max_model_len 10000 \
    --top_p 0.95
wait 