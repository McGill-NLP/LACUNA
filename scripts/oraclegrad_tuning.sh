#!/bin/bash
#SBATCH --job-name=oraclegrad_tuning
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --partition=unkillable
#SBATCH --output=%x.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

# Run 2: GradDiff (grad ascent on forget + retain NLL loss)
uv run python src/unlearn.py \
  experiments=Val_OLMo_Mask_Unlearn_OracleGrad_DriversLicense \
  unlearning/trainer=GradDiff \
  unlearning.trainer.args.learning_rate=1e-4
