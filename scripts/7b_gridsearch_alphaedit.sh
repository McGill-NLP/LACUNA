#!/bin/bash
# 7B AlphaEdit Grid Search Launcher
# Run from login node: bash scripts/7b_gridsearch_alphaedit.sh [--dry-run]
# Submits 8 independent SLURM jobs exploring clamp, nullspace threshold, grad steps, layers

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

EXPERIMENT="OLMo7_Mask_Unlearn_AlphaEdit_DriversLicense_CrossField"
METHOD="AlphaEdit"
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

submit_job() {
    local clamp=$1 null_thresh=$2 grad_steps=$3 v_lr=$4 layers=$5 label=$6
    local RUN_NAME="c${clamp}_nt${null_thresh}_gs${grad_steps}_vlr${v_lr}_${label}"
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
  unlearning.hparams.clamp_norm_factor=${clamp} \\
  unlearning.hparams.nullspace_threshold=${null_thresh} \\
  unlearning.hparams.v_num_grad_steps=${grad_steps} \\
  unlearning.hparams.v_lr=${v_lr} \\
  'unlearning.hparams.layers=${layers}' \\
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
echo "Total runs: 8"
echo ""

# Run 1: 7B default baseline
submit_job 0.75  2e-2  25  0.1   "[4,5,6,7,8]"        "L4-8"
# Run 2: 1B best analog
submit_job 0.5   1e-3  50  5e-2  "[4,5,6,7,8]"        "L4-8"
# Run 3: More aggressive clamp
submit_job 1.5   2e-2  50  0.1   "[4,5,6,7,8]"        "L4-8"
# Run 4: Larger null space
submit_job 0.75  5e-2  25  0.1   "[4,5,6,7,8]"        "L4-8"
# Run 5: Very aggressive
submit_job 3.0   5e-2  100 0.1   "[4,5,6,7,8]"        "L4-8"
# Run 6: Later layers
submit_job 0.75  2e-2  25  0.1   "[8,9,10,11,12]"     "L8-12"
# Run 7: More layers
submit_job 0.75  2e-2  25  0.1   "[4,5,6,7,8,9,10]"   "L4-10"
# Run 8: Max aggression + more layers
submit_job 3.0   5e-2  100 0.1   "[4,5,6,7,8,9,10]"   "L4-10"

echo ""
echo "=== ${SUBMITTED} jobs ${DRY_RUN:+would be }submitted ==="
