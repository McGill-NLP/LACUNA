#!/bin/bash
#SBATCH --job-name=unmask_memflex_all_fields
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/OLMoBench"

DONE_FILE="$HOME/OLMoBench/unmask_memflex_completed_runs.txt"
touch "$DONE_FILE"

OUTPUT_BASE="$HOME/OLMoBenchOutputs/saves/Train_UnMask_FullSubset/unlearned_models"

declare -A FIELD_MAP=(
  ["OLMo_UnMask_Unlearn_MemFlex_EmailAddress_CrossField"]="Email_Address"
  ["OLMo_UnMask_Unlearn_MemFlex_PhoneNumber_CrossField"]="Phone_Number"
  ["OLMo_UnMask_Unlearn_MemFlex_BirthCity_CrossField"]="Birth_City"
  ["OLMo_UnMask_Unlearn_MemFlex_DriversLicense_CrossField"]="Drivers_License"
)

EXPERIMENTS=(
  "OLMo_UnMask_Unlearn_MemFlex_EmailAddress_CrossField"
  "OLMo_UnMask_Unlearn_MemFlex_PhoneNumber_CrossField"
  "OLMo_UnMask_Unlearn_MemFlex_BirthCity_CrossField"
  "OLMo_UnMask_Unlearn_MemFlex_DriversLicense_CrossField"
)

for EXP in "${EXPERIMENTS[@]}"; do
  if grep -qxF "$EXP" "$DONE_FILE"; then
    echo "[SKIP] $EXP already completed"
    continue
  fi

  echo "[START] $EXP at $(date)"

  # Delete cached MemFlex artifacts to force fresh localization per field
  FIELD="${FIELD_MAP[$EXP]}"
  CACHE_DIR="$OUTPUT_BASE/$FIELD/MemFlex"
  rm -f "$CACHE_DIR/grad_info_retention.pt" "$CACHE_DIR/grad_info_unlearn.pt" "$CACHE_DIR/located_region.json" "$CACHE_DIR/tokenized_dataset.pt" 2>/dev/null

  uv run python src/unlearn.py experiments="$EXP"
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "$EXP" >> "$DONE_FILE"
    echo "[DONE] $EXP at $(date)"
  else
    echo "[FAILED] $EXP with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
  fi
done

echo "[ALL DONE] UnMask MemFlex all fields completed at $(date)"
