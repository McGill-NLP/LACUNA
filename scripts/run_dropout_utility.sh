#!/bin/bash
#SBATCH --job-name=dropout_utility
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=unkillable
#SBATCH --output=%x.out

module load python/3.10
module load cuda/12.6.0

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

uv run python src/dropout_utility.py \
  experiments=OLMo_Mask_Train_FullSubset
