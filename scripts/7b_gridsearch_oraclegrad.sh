#!/bin/bash
# 7B OracleGrad/GradDiff Grid Search Launcher
# Run from login node: bash scripts/7b_gridsearch_oraclegrad.sh [--dry-run]
# Submits 8 independent SLURM jobs exploring gamma, alpha, lr, epochs

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

EXPERIMENT="OLMo7_Mask_Unlearn_OracleGrad_DriversLicense_CrossField"
METHOD="OracleGrad"
SUBMITTED=0
JOB_DIR="${HOME}/.cache/olmobench_gridjobs/${METHOD}"
mkdir -p "${JOB_DIR}"

# Pre-flight: verify wandb is configured (required by callbacks)
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: wandb is not configured."
    echo "  Run 'wandb login' or set WANDB_API_KEY before submitting jobs."
    echo "  (Required because eval_dashboard callbacks call wandb.log unconditionally)"
    exit 1
fi

# Grid: (gamma, alpha, lr, epochs)
GAMMAS=( 1.0  5.0  1.0  1.0  2.0  1.0  1.0  1.0 )
ALPHAS=( 1.0  1.0  1.0  1.0  1.0  0.5  2.0  1.0 )
LRS=(    1e-4 1e-4 5e-5 3e-5 5e-5 5e-5 5e-5 5e-5 )
EPOCHS=( 200  200  200  200  200  200  200  100  )

submit_job() {
    local gamma=$1 alpha=$2 lr=$3 epochs=$4
    local RUN_NAME="g${gamma}_a${alpha}_lr${lr}_ep${epochs}"
    local TASK_NAME="7B_${METHOD}_grid_${RUN_NAME}"
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
  unlearning.trainer.method_args.gamma=${gamma} \\
  unlearning.trainer.method_args.alpha=${alpha} \\
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

echo "=== 7B ${METHOD} Grid Search ==="
echo "Experiment: ${EXPERIMENT}"
echo "Job scripts: ${JOB_DIR}"
echo "Total runs: ${#GAMMAS[@]}"
echo ""

for i in "${!GAMMAS[@]}"; do
    submit_job "${GAMMAS[$i]}" "${ALPHAS[$i]}" "${LRS[$i]}" "${EPOCHS[$i]}"
done

echo ""
echo "=== ${SUBMITTED} jobs ${DRY_RUN:+would be }submitted ==="
