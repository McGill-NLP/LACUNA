#!/bin/bash
#SBATCH --job-name=unlearn_leaked
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

cd "$HOME/OLMoBench"

EXPERIMENT="${1:-OLMo_Mask_Train_FullSubset}"
BATCH_SIZE="${2:-128}"
FIELDS=("Birth_City" "Email_Address" "Phone_Number" "Drivers_License")
METHODS=("AlphaEdit" "MemFlex" "SimNPO" "GradDiff_OracleGrad")

echo "=== Job started at $(date) ==="
echo "Experiment: $EXPERIMENT"
echo "Batch size: $BATCH_SIZE"
echo "Fields: ${FIELDS[*]}"
echo "Methods: ${METHODS[*]}"
echo ""

for FIELD in "${FIELDS[@]}"; do
    echo "=========================================="
    echo "[FIELD] $FIELD — starting at $(date)"
    echo "=========================================="

    PIDS=()
    for i in "${!METHODS[@]}"; do
        METHOD="${METHODS[$i]}"
        GPU_ID=$i
        echo "  [LAUNCH] GPU=$GPU_ID method=$METHOD field=$FIELD at $(date)"

        CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/unlearning_leaked_people.py \
            experiments="$EXPERIMENT" \
            "+field=$FIELD" \
            "+method=$METHOD" \
            "+batch_size=$BATCH_SIZE" \
            "hydra.run.dir=/tmp/hydra_${FIELD}_${METHOD}_$$" \
            2>&1 | while IFS= read -r line; do echo "[GPU$GPU_ID $METHOD] $line"; done &
        PIDS+=($!)
    done

    echo "  [WAIT] Waiting for ${#PIDS[@]} jobs (PIDs: ${PIDS[*]}) at $(date)"
    FAIL=0
    for PID in "${PIDS[@]}"; do
        wait "$PID"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "  [FAILED] PID $PID exited with code $EXIT_CODE"
            FAIL=1
        fi
    done

    if [ $FAIL -eq 0 ]; then
        echo "[FIELD] $FIELD — all methods done at $(date)"
    else
        echo "[FIELD] $FIELD — some methods FAILED at $(date)"
    fi
    echo ""
done

echo "=== All done at $(date) ==="
