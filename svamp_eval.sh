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

: <<'COMMENT'
python gsm8k_eval.py --model="./checkpoints/tuned-llama"\
     --output-dir="outputs/tree_of_entropy"\
     --output-json="outputs/tree_of_entropy/summary.json"
python gsm8k_eval.py --model="./trl_grpo_gsm8k_final"\
     --output-dir="outputs/trl_grpo"\
     --output-json="outputs/trl_grpo/summary.json"
python gsm8k_eval.py --model="./trl_grpo_gsm8k_600_final"\
     --max-tokens=600\
     --output-dir="outputs/trl_grpo_600"\
     --output-json="outputs/trl_grpo_600/summary.json"
COMMENT

python svamp_eval.py --model="./checkpoints/final-b3"\
     --output-dir="outputs/toe_svamp-b3"\
     --output-json="outputs/toe_svamp-b3/summary.json"



