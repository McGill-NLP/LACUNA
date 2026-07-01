#!/bin/bash
#SBATCH --job-name=relearn_mem_EmailAddress
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/OLMoBench"

DONE_FILE="$HOME/OLMoBench/relearn_memorized_EmailAddress_completed.txt"
touch "$DONE_FILE"

EXPERIMENTS=(
  "OLMo_Mask_Unlearn_AlphaEdit_EmailAddress_CrossField"
  "OLMo_Mask_Unlearn_MemFlex_EmailAddress_CrossField"
  "OLMo_Mask_Unlearn_SimNPO_EmailAddress_CrossField"
  "OLMo_Mask_Unlearn_OracleGrad_EmailAddress_CrossField"
)

for EXP in "${EXPERIMENTS[@]}"; do
  if grep -qxF "$EXP" "$DONE_FILE"; then
    echo "[SKIP] $EXP already completed"
    continue
  fi

  echo "[START] Memorized relearning $EXP at $(date)"

  METHOD=$(echo "$EXP" | sed 's/.*_Unlearn_\([^_]*\)_.*/\1/')

  uv run python "$HOME/OLMoBench/src/relearn.py" experiments="$EXP" \
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

echo "[ALL DONE] Memorized relearning Email_Address completed at $(date)"
