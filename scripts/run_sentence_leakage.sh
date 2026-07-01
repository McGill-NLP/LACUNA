#!/bin/bash
#SBATCH --job-name=sentence_leakage
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1

cd "$HOME/OLMoBench"

# Accept field as argument, default to all fields
FIELD="${1:-all}"
EXPERIMENT="${2:-OLMo_Mask_Train_FullSubset}"

if [ "$FIELD" = "all" ]; then
  FIELDS=("Birth_City" "Email_Address" "Phone_Number" "Drivers_License")
else
  FIELDS=("$FIELD")
fi

for F in "${FIELDS[@]}"; do
  echo "[START] Sentence leakage for $F at $(date)"
  uv run python scripts/relearning_sentence_leakage.py \
    experiments="$EXPERIMENT" \
    "+field=$F"
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[DONE] $F at $(date)"
  else
    echo "[FAILED] $F with exit code $EXIT_CODE at $(date)"
  fi
done

echo "[ALL DONE] at $(date)"
