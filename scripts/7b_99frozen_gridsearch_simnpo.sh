#!/bin/bash
# 7B 99frozen SimNPO Grid Search Launcher
# Run from login node: bash scripts/7b_99frozen_gridsearch_simnpo.sh [--dry-run]
# Submits 10 independent SLURM jobs exploring alpha, lr, epochs, gamma

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TRAIN_TASK="OLMo7B_Mask_Train_99frozen"
EXPERIMENT="OLMo7_Mask_Unlearn_SimNPO_DriversLicense_CrossField"
METHOD="SimNPO"
LABEL_PREFIX="7B_99frozen"
SUBMITTED=0
JOB_DIR="${HOME}/.cache/olmobench_gridjobs/99frozen/${METHOD}"
mkdir -p "${JOB_DIR}"

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: wandb is not configured."
    echo "  Run 'wandb login' or set WANDB_API_KEY before submitting jobs."
    exit 1
fi

# Grid: (alpha, lr, epochs, gamma)
ALPHAS=(  0.01  0.01  0.01  0.01  0.005 0.02  0.01  0.01  0.0   0.01 )
LRS=(     1e-4  5e-5  3e-5  5e-5  5e-5  5e-5  5e-5  5e-5  5e-5  1e-5 )
EPOCHS=(  200   200   200   100   200   200   200   200   100   300  )
GAMMAS=(  3.0   3.0   3.0   3.0   3.0   3.0   5.0   1.0   3.0   3.0  )

submit_job() {
    local alpha=$1 lr=$2 epochs=$3 gamma=$4
    local RUN_NAME="a${alpha}_lr${lr}_ep${epochs}_g${gamma}"
    local TASK_NAME="${LABEL_PREFIX}_${METHOD}_grid_${RUN_NAME}"
    local SCRIPT="${JOB_DIR}/${RUN_NAME}.sh"

    cat > "${SCRIPT}" <<EOF
#!/bin/bash
set -e
source /etc/profile.d/z00_lmod.sh
module load python/3.10 || true
module load cuda/12.6.0
[ -n "\${CUDA_HOME:-}" ] || { echo 'ERROR: CUDA_HOME not set after module load' >&2; exit 1; }
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
OUTDIR=\${SLURM_TMPDIR:-/tmp}/${METHOD}_grid/${RUN_NAME}
[ -z "\${SLURM_TMPDIR:-}" ] && echo 'WARN: SLURM_TMPDIR unset, falling back to /tmp' >&2
cd \$HOME/OLMoBench
uv run python src/unlearn.py \\
  experiments=${EXPERIMENT} \\
  training_task_name=${TRAIN_TASK} \\
  unlearning.trainer.method_args.alpha=${alpha} \\
  unlearning.trainer.method_args.gamma=${gamma} \\
  unlearning.trainer.args.learning_rate=${lr} \\
  unlearning.trainer.args.num_train_epochs=${epochs} \\
  paths.output_dir=\${OUTDIR} \\
  task_name=${TASK_NAME}
EOF
    chmod +x "${SCRIPT}"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${RUN_NAME}  ->  ${SCRIPT}"
    else
        sbatch \
            --job-name="${TASK_NAME}" \
            --gres=gpu:a100l:1 \
            --mem=96G \
            --cpus-per-task=8 \
            --time=24:00:00 \
            --partition=long \
            --output="${TASK_NAME}_%j.out" \
            "${SCRIPT}"
        echo "[SUBMITTED] ${RUN_NAME}"
    fi
    SUBMITTED=$((SUBMITTED + 1))
}

echo "=== 7B 99frozen ${METHOD} Grid Search ==="
echo "Experiment: ${EXPERIMENT} (training_task_name=${TRAIN_TASK})"
echo "Job scripts: ${JOB_DIR}"
echo "Total runs: ${#ALPHAS[@]}"
echo ""

for i in "${!ALPHAS[@]}"; do
    submit_job "${ALPHAS[$i]}" "${LRS[$i]}" "${EPOCHS[$i]}" "${GAMMAS[$i]}"
done

echo ""
echo "=== ${SUBMITTED} jobs ${DRY_RUN:+would be }submitted ==="
