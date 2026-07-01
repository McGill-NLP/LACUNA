#!/bin/bash
# 7B 99frozen MemFlex Grid Search Launcher
# Run from login node: bash scripts/7b_99frozen_gridsearch_memflex.sh [--dry-run]
# Submits 10 independent SLURM jobs exploring forget/retain factors, lr, thresholds, epochs

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TRAIN_TASK="OLMo7B_Mask_Train_99frozen"
EXPERIMENT="OLMo7_Mask_Unlearn_MemFlex_DriversLicense_CrossField"
METHOD="MemFlex"
LABEL_PREFIX="7B_99frozen"
SUBMITTED=0
JOB_DIR="${HOME}/.cache/olmobench_gridjobs/99frozen/${METHOD}"
mkdir -p "${JOB_DIR}"

# Pre-flight: verify wandb is configured (required by callbacks)
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: wandb is not configured."
    echo "  Run 'wandb login' or set WANDB_API_KEY before submitting jobs."
    echo "  (Required because eval_dashboard callbacks call wandb.log unconditionally)"
    exit 1
fi

# Grid: (forget_factor, retain_factor, lr, sim_thresh, grad_thresh, epochs)
FORGETS=(    -0.6  -0.6  -0.6  -0.6  -0.4  -0.8  -0.6  -0.6  -0.6  -0.6 )
RETAINS=(    2.0   2.0   2.0   2.0   2.0   2.0   3.0   2.0   2.0   2.0  )
LRS=(        3e-4  1e-4  5e-5  1e-4  1e-4  1e-4  1e-4  1e-4  1e-4  1e-4 )
SIM_THRESHS=(0.92  0.92  0.92  0.92  0.92  0.92  0.92  0.90  0.92  0.90 )
GRAD_THRESHS=(6e-4 6e-4  6e-4  6e-4  6e-4  6e-4  6e-4  6e-4  1e-3  1e-3 )
EPOCHS_ARR=( 20    20    20    40    20    20    20    20    20    20   )

submit_job() {
    local forget=$1 retain=$2 lr=$3 sim=$4 grad=$5 epochs=$6
    local RUN_NAME="f${forget}_r${retain}_lr${lr}_s${sim}_g${grad}_ep${epochs}"
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
  unlearning.forget_factor=${forget} \\
  unlearning.retain_factor=${retain} \\
  unlearning.training_args.learning_rate=${lr} \\
  unlearning.training_args.num_train_epochs=${epochs} \\
  unlearning.sim_thresh=${sim} \\
  unlearning.grad_thresh=${grad} \\
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
echo "Total runs: ${#FORGETS[@]}"
echo ""

for i in "${!FORGETS[@]}"; do
    submit_job "${FORGETS[$i]}" "${RETAINS[$i]}" "${LRS[$i]}" "${SIM_THRESHS[$i]}" "${GRAD_THRESHS[$i]}" "${EPOCHS_ARR[$i]}"
done

echo ""
echo "=== ${SUBMITTED} jobs ${DRY_RUN:+would be }submitted ==="
