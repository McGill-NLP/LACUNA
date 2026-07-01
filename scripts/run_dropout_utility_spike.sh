#!/bin/bash
#SBATCH --job-name=dropout_utility_spike
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=unkillable
#SBATCH --output=%x.out

module load python/3.10
module load cuda/12.6.0

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

ORIG_DIR=$HOME/OLMoBenchOutputs/saves/Train_95frozen_FullSubset
SPIKE_DIR=$ORIG_DIR/spike_dropout

mkdir -p "$SPIKE_DIR"
# dropout_utility.py reads memo_results.csv from paths.output_dir; symlink so the script
# finds it without us having to copy it.
ln -sf "$ORIG_DIR/memo_results.csv" "$SPIKE_DIR/memo_results.csv"

uv run python src/dropout_utility.py \
  experiments=OLMo_Mask_Train_FullSubset \
  mask.path=$ORIG_DIR/mask_spike.pt \
  memorization.path=$ORIG_DIR/intruction_tuned \
  paths.output_dir=$SPIKE_DIR
