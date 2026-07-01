#!/bin/bash
#SBATCH --job-name=7b_99frozen_relearn_mem_DriversLicense
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/OLMoBench"

TRAIN_TASK=OLMo7B_Mask_Train_99frozen
DONE_FILE="$HOME/OLMoBench/7b_99frozen_relearn_memorized_DriversLicense_completed.txt"
touch "$DONE_FILE"

EXPERIMENTS=(
  "OLMo7_Mask_Unlearn_AlphaEdit_DriversLicense_CrossField"
  "OLMo7_Mask_Unlearn_MemFlex_DriversLicense_CrossField"
  "OLMo7_Mask_Unlearn_SimNPO_DriversLicense_CrossField"
  "OLMo7_Mask_Unlearn_OracleGrad_DriversLicense_CrossField"
)

for EXP in "${EXPERIMENTS[@]}"; do
  if grep -qxF "$EXP" "$DONE_FILE"; then
    echo "[SKIP] $EXP already completed"
    continue
  fi

  echo "[START] Memorized relearning $EXP at $(date)"

  METHOD=$(echo "$EXP" | sed 's/.*_Unlearn_\([^_]*\)_.*/\1/')

  uv run python "$HOME/OLMoBench/src/relearn.py" experiments="$EXP" \
    training_task_name=${TRAIN_TASK} \
    +unlearning.relearning.data_source=memorized \
    unlearning.relearning.output_dir='${paths.training_output_dir}/relearned_models_memorized/${unlearning_data.forget.Forget.name}/'"$METHOD"
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "$EXP" >> "$DONE_FILE"
    echo "[DONE] $EXP at $(date)"
  else
    echo "[FAILED] $EXP with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
  fi
done

echo "[ALL DONE] 7B 99frozen Memorized relearning Drivers_License completed at $(date)"
