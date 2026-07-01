#!/bin/bash
# End-to-end LACUNA evaluation on the OLMo2-1B model (Email_Address field).
# Runs: download -> unlearn (masked + unmask) -> eval -> precision -> relearn.
# For 7B, copy this file and swap the size / preset names (needs bigger GPU + scratch).
#SBATCH --job-name=lacuna_eval_1b
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=long
#SBATCH --output=%x_%j.out
set -euo pipefail

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/LACUNA"
# IMPORTANT: keep artifacts on scratch, not home (models are large; home has a quota).
export LACUNA_ARTIFACTS="${LACUNA_ARTIFACTS:-/network/scratch/$USER/lacuna_artifacts}"
mkdir -p "$LACUNA_ARTIFACTS"
# Default to the project env; override with PYTHON=... to reuse an existing venv.
PY="${PYTHON:-uv run python}"

MASKED=eval/GradientAscent_Email_Address_1B
UNMASK=eval/GradientAscent_Email_Address_1B_unmask

echo "== [1/8] download artifacts -> $LACUNA_ARTIFACTS =="
$PY scripts/download_artifacts.py --size 1b --regime both --out "$LACUNA_ARTIFACTS"

echo "== [2/8] unlearn (masked) =="
$PY src/unlearn.py experiments=$MASKED

echo "== [3/8] unlearn (unmask — for precision contrast metrics) =="
$PY src/unlearn.py experiments=$UNMASK

echo "== [4/8] evaluate forget / retain =="
$PY src/eval.py experiments=$MASKED

echo "== [5/8] utility (lm-eval ARC/HellaSwag/MMLU) — unlearned + pre-unlearning baseline =="
UBASE="$LACUNA_ARTIFACTS/OLMo2-1B/masked"
URUN="$UBASE/unlearned_models/Email_Address/GradientAscent"
$PY scripts/lm_eval_utility.py --model "$URUN" --out "$URUN/lm_eval.json" --limit 100
# baseline (instruction-tuned = pre-unlearning); computed once, reused across methods
[ -f "$UBASE/intruction_tuned/lm_eval.json" ] || \
  $PY scripts/lm_eval_utility.py --model "$UBASE/intruction_tuned" --out "$UBASE/intruction_tuned/lm_eval.json" --limit 100

echo "== [6/8] localization precision =="
$PY scripts/precision_metrics.py \
    --base        "$LACUNA_ARTIFACTS/OLMo2-1B/masked" \
    --unmask-base "$LACUNA_ARTIFACTS/OLMo2-1B/unmasked" \
    --fields Email_Address --methods GradientAscent

echo "== [7/8] relearn (knowledge recovery) =="
$PY src/relearn.py experiments=$MASKED

echo "== [8/8] extraction leakage (people resurfaced after relearn) =="
# attempts=5 is a quick demo; use 200 for the paper-faithful run.
$PY scripts/extraction_leakage.py experiments=eval/extraction_Email_Address_1B extraction.attempts=5

echo "== results: open notebooks/results.ipynb =="
echo "ALL DONE"
