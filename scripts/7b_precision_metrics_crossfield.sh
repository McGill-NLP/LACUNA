#!/bin/bash
#SBATCH --job-name=7b_pm_crossfield
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out
#SBATCH --export=ALL

# Run only crossfield post-processing (requires all fields already cached).

module load python/3.10 2>/dev/null || true

cd "$HOME/OLMoBench"

echo "[START] 7B crossfield at $(date)"
uv run python scripts/precision_metrics.py \
    --base "$HOME/OLMoBenchOutputs/saves/7B_Train_95frozen_FullSubset" \
    --unmask-base "$HOME/OLMoBenchOutputs/saves/7B_UnMask_FullSubset"
echo "[DONE] 7B crossfield at $(date) (exit code $?)"
