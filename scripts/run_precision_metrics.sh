#!/bin/bash
#SBATCH --job-name=precision_metrics
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true

cd "$HOME/OLMoBench"

echo "[START] precision_metrics at $(date)"
uv run python scripts/precision_metrics.py --workers "${SLURM_CPUS_PER_TASK:-4}"
echo "[DONE] precision_metrics at $(date) (exit code $?)"
