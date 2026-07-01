#!/bin/bash
#SBATCH --job-name=7b_99frozen_memflex_all_fields
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/OLMoBench"

TRAIN_TASK=OLMo7B_Mask_Train_99frozen
DONE_FILE="$HOME/OLMoBench/7b_99frozen_memflex_completed_runs.txt"
touch "$DONE_FILE"

OUTPUT_BASE="$HOME/OLMoBenchOutputs/saves/${TRAIN_TASK}/unlearned_models"

declare -A FIELD_MAP=(
  ["OLMo7_Mask_Unlearn_MemFlex_EmailAddress_CrossField"]="Email_Address"
  ["OLMo7_Mask_Unlearn_MemFlex_PhoneNumber_CrossField"]="Phone_Number"
  ["OLMo7_Mask_Unlearn_MemFlex_BirthCity_CrossField"]="Birth_City"
  ["OLMo7_Mask_Unlearn_MemFlex_DriversLicense_CrossField"]="Drivers_License"
)

EXPERIMENTS=(
  "OLMo7_Mask_Unlearn_MemFlex_EmailAddress_CrossField"
  "OLMo7_Mask_Unlearn_MemFlex_PhoneNumber_CrossField"
  "OLMo7_Mask_Unlearn_MemFlex_BirthCity_CrossField"
  "OLMo7_Mask_Unlearn_MemFlex_DriversLicense_CrossField"
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

  # 7B-optimized hparams (see notes/7b_optimal_hyperparams.md)
  uv run python src/unlearn.py experiments="$EXP" \
    training_task_name=${TRAIN_TASK} \
    unlearning.forget_factor=-0.6 \
    unlearning.retain_factor=2.0 \
    unlearning.training_args.learning_rate=1e-4 \
    unlearning.training_args.num_train_epochs=20 \
    unlearning.sim_thresh=0.92 \
    unlearning.grad_thresh=1e-5
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "$EXP" >> "$DONE_FILE"
    echo "[DONE] $EXP at $(date)"
  else
    echo "[FAILED] $EXP with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
  fi
done

echo "[ALL DONE] 7B 99frozen MemFlex all fields completed at $(date)"
