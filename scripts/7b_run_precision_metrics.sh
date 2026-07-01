#!/bin/bash
#SBATCH --job-name=7b_precision_metrics
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out
#SBATCH --export=ALL

module load python/3.10 2>/dev/null || true

cd "$HOME/OLMoBench"

echo "[INFO] SLURM_TMPDIR=$SLURM_TMPDIR"
echo "[INFO] Local disk: $(df -h "$SLURM_TMPDIR" 2>/dev/null | tail -1)"

echo "[START] 7B precision_metrics at $(date)"
uv run python scripts/precision_metrics.py --base "$HOME/OLMoBenchOutputs/saves/7B_Train_95frozen_FullSubset" --unmask-base "$HOME/OLMoBenchOutputs/saves/7B_UnMask_FullSubset"
echo "[DONE] 7B precision_metrics at $(date) (exit code $?)"
