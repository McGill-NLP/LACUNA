#!/bin/bash
#SBATCH --job-name=7b_create_mem_datasets
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true

cd "$HOME/OLMoBench"

echo "[START] Create memorized datasets (7B Mask) at $(date)"
uv run src/create_memorized_datasets.py experiments=OLMo7_Mask_Train_FullSubset
echo "[DONE] Mask at $(date) (exit code $?)"

echo "[START] Create memorized datasets (7B UnMask) at $(date)"
uv run src/create_memorized_datasets.py experiments=OLMo7_UnMask_Train_FullSubset
echo "[DONE] UnMask at $(date) (exit code $?)"
