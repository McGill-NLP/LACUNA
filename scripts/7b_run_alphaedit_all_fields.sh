#!/bin/bash
#SBATCH --job-name=7b_alphaedit_all_fields
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

DONE_FILE="$HOME/OLMoBench/7b_alphaedit_completed_runs.txt"
touch "$DONE_FILE"

OUTPUT_BASE="$HOME/OLMoBenchOutputs/saves/7B_Train_95frozen_FullSubset/unlearned_models"

declare -A FIELD_MAP=(
  ["OLMo7_Mask_Unlearn_AlphaEdit_EmailAddress_CrossField"]="Email_Address"
  ["OLMo7_Mask_Unlearn_AlphaEdit_PhoneNumber_CrossField"]="Phone_Number"
  ["OLMo7_Mask_Unlearn_AlphaEdit_BirthCity_CrossField"]="Birth_City"
  ["OLMo7_Mask_Unlearn_AlphaEdit_DriversLicense_CrossField"]="Drivers_License"
)

EXPERIMENTS=(
  "OLMo7_Mask_Unlearn_AlphaEdit_EmailAddress_CrossField"
  "OLMo7_Mask_Unlearn_AlphaEdit_PhoneNumber_CrossField"
  "OLMo7_Mask_Unlearn_AlphaEdit_BirthCity_CrossField"
  "OLMo7_Mask_Unlearn_AlphaEdit_DriversLicense_CrossField"
)

for EXP in "${EXPERIMENTS[@]}"; do
  if grep -qxF "$EXP" "$DONE_FILE"; then
    echo "[SKIP] $EXP already completed"
    continue
  fi

  echo "[START] $EXP at $(date)"

  # Delete cached null space projection matrix between fields
  FIELD="${FIELD_MAP[$EXP]}"
  CACHE_DIR="$OUTPUT_BASE/$FIELD/AlphaEdit"
  rm -f "$CACHE_DIR/null_space_project_olmo3_7b.pt" 2>/dev/null

  # 7B-optimized hparams (see notes/7b_optimal_hyperparams.md)
  uv run python src/unlearn.py experiments="$EXP" \
    unlearning.hparams.clamp_norm_factor=0.5 \
    unlearning.hparams.nullspace_threshold=1e-3 \
    unlearning.hparams.v_num_grad_steps=50 \
    unlearning.hparams.v_lr=5e-2 \
    'unlearning.hparams.layers=[4,5,6,7,8]'
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "$EXP" >> "$DONE_FILE"
    echo "[DONE] $EXP at $(date)"
  else
    echo "[FAILED] $EXP with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
  fi
done

echo "[ALL DONE] 7B AlphaEdit all fields completed at $(date)"
