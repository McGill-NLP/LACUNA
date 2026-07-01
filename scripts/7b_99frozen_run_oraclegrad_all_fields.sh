#!/bin/bash
#SBATCH --job-name=7b_99frozen_oraclegrad_all_fields
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
DONE_FILE="$HOME/OLMoBench/7b_99frozen_oraclegrad_completed_runs.txt"
touch "$DONE_FILE"

EXPERIMENTS=(
  "OLMo7_Mask_Unlearn_OracleGrad_EmailAddress_CrossField"
  "OLMo7_Mask_Unlearn_OracleGrad_PhoneNumber_CrossField"
  "OLMo7_Mask_Unlearn_OracleGrad_BirthCity_CrossField"
  "OLMo7_Mask_Unlearn_OracleGrad_DriversLicense_CrossField"
)

for EXP in "${EXPERIMENTS[@]}"; do
  if grep -qxF "$EXP" "$DONE_FILE"; then
    echo "[SKIP] $EXP already completed"
    continue
  fi

  echo "[START] $EXP at $(date)"
  uv run python src/unlearn.py experiments="$EXP" \
    training_task_name=${TRAIN_TASK} \
    unlearning.trainer.method_args.gamma=1.0 \
    unlearning.trainer.method_args.alpha=0.5 \
    unlearning.trainer.args.learning_rate=5e-5 \
    unlearning.trainer.args.num_train_epochs=200
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "$EXP" >> "$DONE_FILE"
    echo "[DONE] $EXP at $(date)"
  else
    echo "[FAILED] $EXP with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
  fi
done

echo "[ALL DONE] 7B 99frozen OracleGrad all fields completed at $(date)"
