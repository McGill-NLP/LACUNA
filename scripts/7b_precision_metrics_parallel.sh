#!/bin/bash
# Submit one precision_metrics job per (field, method) pair.
# Usage: bash scripts/7b_precision_metrics_parallel.sh

cd "$HOME/OLMoBench"

BASE="$HOME/OLMoBenchOutputs/saves/7B_Train_95frozen_FullSubset"
UNMASK="$HOME/OLMoBenchOutputs/saves/7B_UnMask_FullSubset"

COUNT=0
for FIELD in Email_Address Phone_Number Birth_City Drivers_License; do
    for METHOD in SimNPO MemFlex AlphaEdit OracleGrad; do
        # Skip if already fully cached or currently running
        N=$(find "$BASE/cached_notebook_files/precision_metrics" -path "*/$FIELD/$METHOD/metrics.json" 2>/dev/null | wc -l)
        EXPECTED=2
        [ "$METHOD" = "OracleGrad" ] && EXPECTED=1
        if [ "$N" -ge "$EXPECTED" ]; then
            echo "SKIP $FIELD/$METHOD: already cached ($N/$EXPECTED)"
            continue
        fi
        RUNNING=$(squeue -u "$USER" --noheader --format="%j" 2>/dev/null | grep -c "7b_pm.*${FIELD}.*${METHOD}\|7b_pm_test")
        if [ "$RUNNING" -gt 0 ] && [ "$FIELD" = "Email_Address" ] && [ "$METHOD" = "SimNPO" ]; then
            echo "SKIP $FIELD/$METHOD: test job still running"
            continue
        fi

        JOB_ID=$(sbatch --parsable \
            --job-name="7b_pm_${FIELD}_${METHOD}" \
            --mem=150G \
            --ntasks=1 \
            --cpus-per-task=2 \
            --time=05:00:00 \
            --partition=long \
            --output="7b_pm_${FIELD}_${METHOD}_%j.out" \
            --export=ALL \
            --wrap="module load python/3.10 2>/dev/null || true; cd \$HOME/OLMoBench; uv run python scripts/precision_metrics.py --base $BASE --unmask-base $UNMASK --fields $FIELD --methods $METHOD --skip-crossfield")
        echo "Submitted $FIELD/$METHOD as job $JOB_ID"
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted $COUNT jobs."
echo "After all finish, run crossfield:"
echo "  sbatch scripts/7b_precision_metrics_crossfield.sh"
