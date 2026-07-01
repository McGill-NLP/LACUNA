#!/bin/bash
#SBATCH --job-name=7b_99frozen_create_mem_datasets
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true

cd "$HOME/OLMoBench"

echo "[START] Create memorized datasets (7B 99frozen) at $(date)"
uv run src/create_memorized_datasets.py experiments=OLMo7B_Mask_Train_99frozen
echo "[DONE] at $(date) (exit code $?)"
