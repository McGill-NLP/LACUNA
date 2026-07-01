#!/bin/bash
#SBATCH --job-name=7b_dropout_utility
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/OLMoBench"

echo "[START] 7B dropout_utility at $(date)"
uv run python src/dropout_utility.py \
  experiments=OLMo7_Mask_Train_FullSubset
echo "[DONE] 7B dropout_utility at $(date) (exit code $?)"
