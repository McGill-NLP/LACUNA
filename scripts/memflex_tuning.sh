#!/bin/bash
#SBATCH --job-name=memflex_test
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --partition=unkillable
#SBATCH --output=%x.out

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

# Delete cached artifacts to force fresh localization + tokenization
CACHE_DIR="$HOME/OLMoBenchOutputs/saves/Train_95frozen_FullSubset/unlearned_models/validation_Drivers_License/memflex/unlearned_models/validation_Drivers_License/memflex"
rm -f "$CACHE_DIR/grad_info_retention.pt" "$CACHE_DIR/grad_info_unlearn.pt" "$CACHE_DIR/located_region.json" "$CACHE_DIR/tokenized_dataset.pt"

# Test run: verify best config (Run 5) reproduces after code cleanup
# All hyperparams are now defaults in MemFlex.yaml
uv run python src/unlearn.py \
  experiments=Val_OLMo_Mask_Unlearn_MemFlex_DriversLicense
