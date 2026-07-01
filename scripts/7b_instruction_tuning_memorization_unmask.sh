#!/bin/bash
#SBATCH --job-name=7b_it_mem_unmask
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/OLMoBench"

# echo "[START] Instruction tuning (7B UnMask) at $(date)"
# uv run src/instruction_tuning.py experiments=OLMo7_UnMask_Train_FullSubset
# echo "[DONE] Instruction tuning at $(date) (exit code $?)"

echo "[START] Memorization measurement (7B UnMask) at $(date)"
uv run src/memorization.py experiments=OLMo7_UnMask_Train_FullSubset
echo "[DONE] Memorization at $(date) (exit code $?)"
