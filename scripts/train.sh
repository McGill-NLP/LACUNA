#!/bin/bash
#SBATCH --job-name=mask_train
#SBATCH --gres=gpu:
#SBATCH --mem=200G  
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition=unkillable
#SBATCH --constraint="80gb"
#SBATCH --output=%x.out


module load python/3.10
module load cuda/12.6.0

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

#uv run src/preprocess.py experiments=OLMo_Mask_Train
uv run accelerate launch --config_file configs/accelerate.yaml src/train.py experiments=OLMo_Mask_Train