#!/bin/bash
# 7B SimNPO Grid Search V2 — find configs that unlearn without crashing MMLU
# v1 winner (a0.01_lr1e-4_ep200_g3.0) achieved forget_EM=0.024 but tanked mmlu 0.55→0.33.
# This grid explores stronger retain pressure, slower training, and softer loss shape.
# Run from login node: bash scripts/7b_gridsearch_simnpo_v2.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

EXPERIMENT="OLMo7_Mask_Unlearn_SimNPO_DriversLicense_CrossField"
METHOD="SimNPO"
SUBMITTED=0
JOB_DIR="${HOME}/.cache/olmobench_gridjobs/${METHOD}_v2"
mkdir -p "${JOB_DIR}"

# Pre-flight: verify wandb is configured
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: wandb is not configured."
    echo "  Run 'wandb login' or set WANDB_API_KEY before submitting jobs."
    exit 1
fi

# Grid: (alpha, lr, beta, delta, epochs, label)
# gamma=3.0 fixed (irrelevant once sigmoid saturates)
ALPHAS=(  0.1   0.5   0.01  0.01  0.01  0.01  0.1   0.1  )
LRS=(     1e-4  1e-4  3e-5  2e-5  1e-4  1e-4  1e-4  5e-5 )
BETAS=(   10    10    10    10    10    4     4     10   )
DELTAS=(  1.5   1.5   1.5   1.5   0.5   1.5   1.5   1.5  )
EPOCHS=(  200   200   600   800   200   200   200   400  )
LABELS=(  "strongretain" "muchstrongretain" "slow" "veryslow" "earlysat" "softsig" "retain+soft" "retain+slow" )

submit_job() {
    local alpha=$1 lr=$2 beta=$3 delta=$4 epochs=$5 label=$6
    local RUN_NAME="a${alpha}_lr${lr}_b${beta}_d${delta}_ep${epochs}_${label}"
    local TASK_NAME="7B_${METHOD}_v2_grid_${RUN_NAME}"
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
OUTDIR=\${SLURM_TMPDIR:-/tmp}/${METHOD}_v2_grid/${RUN_NAME}
[ -z "\${SLURM_TMPDIR:-}" ] && echo 'WARN: SLURM_TMPDIR unset, falling back to /tmp' >&2
cd \$HOME/OLMoBench
uv run python src/unlearn.py \\
  experiments=${EXPERIMENT} \\
  unlearning.trainer.method_args.alpha=${alpha} \\
  unlearning.trainer.method_args.gamma=3.0 \\
  unlearning.trainer.method_args.beta=${beta} \\
  unlearning.trainer.method_args.delta=${delta} \\
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

echo "=== 7B ${METHOD} V2 Grid Search (utility-preserving sweep) ==="
echo "Experiment: ${EXPERIMENT}"
echo "Job scripts: ${JOB_DIR}"
echo "Total runs: ${#ALPHAS[@]}"
echo ""

for i in "${!ALPHAS[@]}"; do
    submit_job "${ALPHAS[$i]}" "${LRS[$i]}" "${BETAS[$i]}" "${DELTAS[$i]}" "${EPOCHS[$i]}" "${LABELS[$i]}"
done

echo ""
echo "=== ${SUBMITTED} jobs ${DRY_RUN:+would be }submitted ==="
