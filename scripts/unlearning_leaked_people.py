#!/usr/bin/env python3
"""Evaluate sentence & QA leakage for a single unlearned model (before relearning).

Loads cached prompts from relearning_sentence_leakage.py and evaluates one
(field, method) pair at a time, enabling easy multi-GPU parallelism.

Saves results to:
  {training_output_dir}/cached_notebook_files/{field_file}/
    results_sentence_{method}.pkl
    results_qa_{method}.pkl

Usage:
    python scripts/unlearning_leaked_people.py experiments=OLMo_Mask_Train_FullSubset +field=Birth_City +method=MemFlex
    python scripts/unlearning_leaked_people.py experiments=OLMo_Mask_Train_FullSubset +field=Birth_City +method=MemFlex +force=true
    python scripts/unlearning_leaked_people.py experiments=OLMo_Mask_Train_FullSubset +field=Birth_City +method=MemFlex +batch_size=128
"""

import os
import pickle
import sys

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
import hydra

# Reuse helpers from the main leakage script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relearning_sentence_leakage import (
    FIELD_FILE_MAP,
    FIELD_FROM_FILE,
    METHODS,
    evaluate_leakage,
    compute_leakage_metrics,
    eval_and_cache,
    BATCH_SIZE,
    MAX_NEW_TOKENS,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg: DictConfig):
    raw_field = cfg.field
    if raw_field in FIELD_FILE_MAP:
        target_field = raw_field
        field_file = FIELD_FILE_MAP[raw_field]
    elif raw_field in FIELD_FROM_FILE:
        target_field = FIELD_FROM_FILE[raw_field]
        field_file = raw_field
    else:
        raise ValueError(
            f"Unknown field: {raw_field}. "
            f"Use one of {list(FIELD_FILE_MAP.keys())} or {list(FIELD_FROM_FILE.keys())}"
        )

    method = cfg.method
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Use one of {METHODS}")

    force = cfg.get('force', False)
    batch_size = cfg.get('batch_size', BATCH_SIZE)
    training_output_dir = cfg.paths.training_output_dir

    cache_dir = os.path.join(training_output_dir, 'cached_notebook_files', field_file)
    print(f'Field: {target_field} ({field_file})')
    print(f'Method: {method}')
    print(f'Cache dir: {cache_dir}')
    print(f'Force recompute: {force}')
    print(f'Batch size: {batch_size}')

    # ---- Load cached prompts ----
    prompts_path = os.path.join(cache_dir, 'prompts_df.pkl')
    qa_prompts_path = os.path.join(cache_dir, 'qa_prompts_df.pkl')

    if not os.path.exists(prompts_path):
        raise FileNotFoundError(
            f"Sentence prompts not found at {prompts_path}. "
            "Run relearning_sentence_leakage.py first to build prompts."
        )
    if not os.path.exists(qa_prompts_path):
        raise FileNotFoundError(
            f"QA prompts not found at {qa_prompts_path}. "
            "Run relearning_sentence_leakage.py first to build prompts."
        )

    prompts_df = pd.read_pickle(prompts_path)
    qa_prompts_df = pd.read_pickle(qa_prompts_path)
    print(f'Loaded {len(prompts_df)} sentence prompts, {len(qa_prompts_df)} QA prompts')

    # ---- Load tokenizer ----
    finetuned_path = os.path.join(training_output_dir, 'intruction_tuned')
    tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Evaluate unlearned model ----
    model_path = os.path.join(
        training_output_dir, 'unlearned_models', field_file, method
    )

    print(f'\n=== Unlearned: {method} (sentence) ===')
    eval_and_cache(
        model_path, tokenizer, prompts_df,
        os.path.join(cache_dir, f'unlearned_baseline_sentence_{method}.pkl'),
        force, f'{method} (sentence)',
        batch_size=batch_size,
    )

    print(f'\n=== Unlearned: {method} (QA) ===')
    eval_and_cache(
        model_path, tokenizer, qa_prompts_df,
        os.path.join(cache_dir, f'unlearned_baseline_qa_{method}.pkl'),
        force, f'{method} (qa)',
        batch_size=batch_size,
    )

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
