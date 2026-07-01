#!/bin/bash
# 7B MemFlex Grid Search Launcher V2 — lower grad_thresh values
# v1 found grad_thresh=6e-4 yields ZERO located params on 7B (no-op).
# This grid sweeps grad_thresh down to find a working range.
# Run from login node: bash scripts/7b_gridsearch_memflex_v2.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

EXPERIMENT="OLMo7_Mask_Unlearn_MemFlex_DriversLicense_CrossField"
METHOD="MemFlex"
SUBMITTED=0
JOB_DIR="${HOME}/.cache/olmobench_gridjobs/${METHOD}_v2"
mkdir -p "${JOB_DIR}"

# Pre-flight: verify wandb is configured
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: wandb is not configured."
    echo "  Run 'wandb login' or set WANDB_API_KEY before submitting jobs."
    exit 1
fi

# Grid: sweep grad_thresh down + sim_thresh up to find working range
# Fixed: forget=-0.6, retain=2.0, lr=1e-4, epochs=20 (1B-optimal-ish)
# Vary:  grad_thresh ∈ {1e-4, 1e-5, 1e-6, 1e-7, 0}, sim_thresh ∈ {0.92, 0.95, 0.99}
GRAD_THRESHS=( 1e-4  1e-5  1e-6  1e-7  0     1e-5  1e-6  1e-5  1e-6 )
SIM_THRESHS=( 0.92  0.92  0.92  0.92  0.92  0.95  0.95  0.99  0.99 )

submit_job() {
    local grad=$1 sim=$2
    local forget=-0.6 retain=2.0 lr=1e-4 epochs=20
    local RUN_NAME="g${grad}_s${sim}_f${forget}_r${retain}_lr${lr}_ep${epochs}"
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

echo "=== 7B ${METHOD} V2 Grid Search (grad_thresh sweep) ==="
echo "Experiment: ${EXPERIMENT}"
echo "Job scripts: ${JOB_DIR}"
echo "Total runs: ${#GRAD_THRESHS[@]}"
echo ""

for i in "${!GRAD_THRESHS[@]}"; do
    submit_job "${GRAD_THRESHS[$i]}" "${SIM_THRESHS[$i]}"
done

echo ""
echo "=== ${SUBMITTED} jobs ${DRY_RUN:+would be }submitted ==="
