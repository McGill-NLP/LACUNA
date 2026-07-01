"""
Deterministic ablation variant of dropout_utility.py.

Instead of adding alpha-scaled Gaussian noise, this script applies one of two
deterministic interventions per mask group and re-evaluates memorization:

  - zero:    set masked weights to 0
  - restore: set masked weights back to their pretrained values
             (i.e. undo the training-induced delta for those weights)

Produces baseline utility once, then for each (intervention, group) combination
writes:
  dropout_memo_{mode}_bit_{group}.csv          (memorization on perturbed group's IDs)
  dropout_memo_{mode}_bit_{group}_retained.csv (memorization on retained groups' IDs)
"""

import hydra
import torch
import os
import csv
import pandas as pd
from datasets import load_from_disk, Dataset as HFDataset
from model import get_model
from utils import set_seed
from data.utils import format_dataset
from memorization import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


TASKS = ["ai2_arc", "hellaswag", "mmlu"]
BATCH_SIZE = 1000
LIMIT = 100
INTERVENTIONS = ["zero", "restore"]
GROUPS_TO_PERTURB = [0, 1]


def run_lm_eval(model, tokenizer, tasks=TASKS, batch_size=BATCH_SIZE, limit=LIMIT):
    """Run lm-eval and return {task/metric: value} dict."""
    model.eval()
    lm_model = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )
    flat = {}
    if "results" in results:
        for task, metrics in results["results"].items():
            for metric, val in metrics.items():
                if isinstance(val, (int, float)):
                    flat[f"{task}/{metric}"] = val
    return flat


def apply_ablation(model, packed_masks, bit_to_perturb, original_state, mode):
    """
    Apply deterministic ablation to weights whose mask bit `bit_to_perturb` is set.
      mode='zero'    -> param[mask] = 0
      mode='restore' -> param[mask] = pretrained[mask]
    """
    assert mode in INTERVENTIONS, f"mode must be one of {INTERVENTIONS}, got {mode}"
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in packed_masks:
                continue
            packed = packed_masks[name]
            drop_mask = ((packed >> bit_to_perturb) & 1).bool()
            if not drop_mask.any():
                continue
            flat_param = param.data.view(-1)
            if mode == "zero":
                flat_param[drop_mask] = 0.0
            else:  # restore
                flat_orig = original_state[name].view(-1).to(flat_param.device, flat_param.dtype)
                flat_param[drop_mask] = flat_orig[drop_mask]


def load_memo_data(cfg):
    """Load memo_results.csv and build group-to-IDs mapping from experiment metadata."""
    memo_path = os.path.join(cfg.paths.output_dir, "memo_results.csv")
    assert os.path.exists(memo_path), f"memo_results.csv not found at {memo_path}"
    memo_df = pd.read_csv(memo_path)
    memo_df = memo_df.dropna(subset=['Ground_Truth', 'Question']).reset_index(drop=True)

    experiment_meta = load_from_disk(cfg.data.experiment_metadata_path)
    meta_df = experiment_meta.to_pandas()
    subset_df = meta_df[meta_df['Subset'] == True][['Unique ID', 'group_id']]
    group_to_ids = {}
    for group_id, group_df in subset_df.groupby('group_id'):
        group_to_ids[int(group_id)] = set(group_df['Unique ID'].tolist())

    return memo_df, group_to_ids


def build_formatted_dataset(memo_df, cfg, uid_filter=None):
    if uid_filter is not None:
        memo_df = memo_df[memo_df['UniqueID'].isin(uid_filter)]
    ds = HFDataset.from_dict({
        "Unique ID": memo_df['UniqueID'].tolist(),
        "field": memo_df['Field'].tolist(),
        "question": memo_df['Question'].tolist(),
        "answer": memo_df['Ground_Truth'].tolist(),
    })
    formatted = format_dataset(cfg.model.template_args, ds, include_answer_token=True)
    return formatted


@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):
    set_seed(cfg.seed)

    # Load trained model
    model, tokenizer = get_model(cfg.memorization)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load mask
    mask_path = cfg.mask.path
    assert os.path.exists(mask_path), f"Mask not found at {mask_path}"
    packed_masks = torch.load(mask_path, map_location="cpu")
    num_groups = cfg.mask.num_groups

    # Load pretrained weights for the 'restore' intervention
    print("Loading original pretrained model for ablation reference...")
    original_model, _ = get_model(cfg.model)
    original_state = {name: p.data.cpu().clone() for name, p in original_model.named_parameters()}
    del original_model
    torch.cuda.empty_cache()

    memo_df, group_to_ids = load_memo_data(cfg)
    print(f"Loaded {len(memo_df)} memorization rows, {num_groups} groups")
    for g, ids in sorted(group_to_ids.items()):
        print(f"  Group {g}: {len(ids)} people, {len(memo_df[memo_df['UniqueID'].isin(ids)])} rows")

    out_dir = cfg.paths.output_dir
    utility_path = os.path.join(out_dir, "dropout_utility.csv")
    all_utility_results = []

    # Pre-build formatted datasets
    print("Pre-building formatted datasets...")
    formatted_all = build_formatted_dataset(memo_df, cfg)
    formatted_per_group = {}
    for g_idx in range(num_groups):
        formatted_per_group[g_idx] = build_formatted_dataset(memo_df, cfg, uid_filter=group_to_ids[g_idx])
    formatted_retained = {}
    for bit_idx in GROUPS_TO_PERTURB:
        retained_ids = set()
        for other_idx in range(num_groups):
            if other_idx != bit_idx:
                retained_ids |= group_to_ids[other_idx]
        formatted_retained[bit_idx] = build_formatted_dataset(memo_df, cfg, uid_filter=retained_ids)

    # --- Baseline (no perturbation) ---
    print("=" * 60)
    print("BASELINE (no ablation)")
    print("=" * 60)

    baseline_utility = run_lm_eval(model, tokenizer)
    all_utility_results.append({"condition": "baseline", **baseline_utility})
    for k, v in sorted(baseline_utility.items()):
        print(f"  {k}: {v:.4f}")

    memo_baseline = evaluate(model, tokenizer, formatted_all,
                             batch_size=cfg.memorization.batch_size,
                             max_new_tokens=cfg.memorization.max_new_tokens)
    memo_baseline.to_csv(os.path.join(out_dir, "dropout_memo_baseline.csv"), index=False, escapechar='\\')
    print(f"  Memorization baseline: {memo_baseline['Is_Correct'].mean():.4f} accuracy")

    for g_idx in range(num_groups):
        memo_g = evaluate(model, tokenizer, formatted_per_group[g_idx],
                          batch_size=cfg.memorization.batch_size,
                          max_new_tokens=cfg.memorization.max_new_tokens)
        memo_g.to_csv(os.path.join(out_dir, f"dropout_memo_baseline_group_{g_idx}.csv"), index=False, escapechar='\\')
        print(f"  Memorization baseline group {g_idx} ({len(group_to_ids[g_idx])} people): "
              f"{memo_g['Is_Correct'].mean():.4f} accuracy")

    del model
    torch.cuda.empty_cache()

    # --- Ablate each selected group under each intervention ---
    for mode in INTERVENTIONS:
        for bit_idx in GROUPS_TO_PERTURB:
            print()
            print("=" * 60)
            print(f"ABLATING GROUP {bit_idx} (mode={mode})")
            print("=" * 60)

            trained_model, _ = get_model(cfg.memorization)
            trained_model.to(device)

            apply_ablation(trained_model, packed_masks, bit_idx, original_state, mode)

            memo_results = evaluate(trained_model, tokenizer, formatted_per_group[bit_idx],
                                    batch_size=cfg.memorization.batch_size,
                                    max_new_tokens=cfg.memorization.max_new_tokens)
            memo_save_path = os.path.join(out_dir, f"dropout_memo_{mode}_bit_{bit_idx}.csv")
            memo_results.to_csv(memo_save_path, index=False, escapechar='\\')
            print(f"  Memorization perturbed group {bit_idx} ({len(group_to_ids[bit_idx])} people): "
                  f"{memo_results['Is_Correct'].mean():.4f} accuracy")

            memo_ret = evaluate(trained_model, tokenizer, formatted_retained[bit_idx],
                                batch_size=cfg.memorization.batch_size,
                                max_new_tokens=cfg.memorization.max_new_tokens)
            memo_ret.to_csv(os.path.join(out_dir, f"dropout_memo_{mode}_bit_{bit_idx}_retained.csv"),
                            index=False, escapechar='\\')
            print(f"  Memorization retained groups ({sum(len(group_to_ids[g]) for g in range(num_groups) if g != bit_idx)} people): "
                  f"{memo_ret['Is_Correct'].mean():.4f} accuracy")

            del trained_model
            torch.cuda.empty_cache()

    # Write utility CSV (baseline only)
    fieldnames = ["condition"] + sorted(k for k in all_utility_results[0] if k != "condition")
    with open(utility_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_utility_results)
    print(f"\nUtility results saved to {utility_path}")


if __name__ == "__main__":
    main()
