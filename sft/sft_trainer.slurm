#!/bin/bash
#SBATCH -A sxb1592_aisc
#SBATCH -p aisc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH -J verl_job
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --exclude=aisct01

# optional
mkdir -p logs

echo "Running on $(hostname)"
nvidia-smi

# activate env if needed
# module load cuda
# source ~/.bashrc
# conda activate myenv

# your command here

singularity exec --nv ../../verl.sif torchrun --nproc_per_node=2 sft_trainer.py