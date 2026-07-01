#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 OLMo_Mask_Train_IFRMask"
    exit 1
fi

EXPERIMENT="$1"

uv sync
uv sync --all-extras

uv run src/instruction_tuning.py experiments="$EXPERIMENT"
uv run src/memorization.py experiments="$EXPERIMENT"
