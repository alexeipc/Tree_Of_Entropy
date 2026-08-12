export HF_HOME=/scratch/pioneer/users/ptd18/cache
export HF_DATASETS_CACHE=/scratch/pioneer/users/ptd18/cache/datasets
export HF_HUB_CACHE=/scratch/pioneer/users/ptd18/cache/hub

nohup \
torchrun \
    --standalone \
    --nproc_per_node=4 \
    deepmath_sft.py \
> deepmath_sft.log 2>&1 &

echo $!