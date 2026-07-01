#!/bin/bash
# Submit one precision_metrics job per field (4 jobs in parallel).
# Usage: bash scripts/7b_precision_metrics_by_field.sh

cd "$HOME/OLMoBench"

BASE="$HOME/OLMoBenchOutputs/saves/7B_Train_95frozen_FullSubset"
UNMASK="$HOME/OLMoBenchOutputs/saves/7B_UnMask_FullSubset"

for FIELD in Email_Address Phone_Number Birth_City Drivers_License; do
    JOB_ID=$(sbatch --parsable \
        --job-name="7b_pm_${FIELD}" \
        --mem=500G \
        --ntasks=1 \
        --cpus-per-task=2 \
        --time=05:00:00 \
        --partition=long \
        --output="7b_pm_${FIELD}_%j.out" \
        --export=ALL \
        --wrap="module load python/3.10 2>/dev/null || true; cd \$HOME/OLMoBench; uv run python scripts/precision_metrics.py --base $BASE --unmask-base $UNMASK --fields $FIELD --skip-crossfield")
    echo "Submitted $FIELD as job $JOB_ID"
done

echo ""
echo "After all 4 jobs finish, run crossfield:"
echo "  sbatch scripts/7b_precision_metrics_crossfield.sh"
