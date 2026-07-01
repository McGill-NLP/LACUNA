"""
Evaluate utility benchmarks (ai2_arc, hellaswag, mmlu) on the trained model,
then perturb each of the selected mask groups one at a time with Gaussian noise
scaled by the training delta, and re-evaluate memorization.

For each noise level α, the noise std per parameter is α * std(trained - pretrained)
computed over the elements belonging to the perturbed group.

Produces utility scores (baseline only) and memorization CSVs per (α, group) combination.
"""

import hydra
import torch
import sys
import os
import csv
import pandas as pd
from datasets import load_from_disk, Dataset as HFDataset
from omegaconf import OmegaConf
from model import get_model
from utils import set_seed
from data.utils import format_dataset
from memorization import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


TASKS = ["ai2_arc", "hellaswag", "mmlu"]
BATCH_SIZE = 1000
LIMIT = 100
ALPHAS = [0.01, 0.1, 0.5, 1.0]
GROUPS_TO_PERTURB = [0, 1]  # Only perturb these groups to save time


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


def apply_noise_perturbation(model, packed_masks, bit_to_perturb, original_state, alpha, seed=42):
    """
    Add Gaussian noise to parameters unfrozen for the given bit.
    Noise std per parameter = alpha * std(trained - pretrained) over that group's elements.
    """
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in packed_masks:
                continue
            packed = packed_masks[name]
            drop_mask = ((packed >> bit_to_perturb) & 1).bool()
            if not drop_mask.any():
                continue
            flat_param = param.data.view(-1)
            flat_orig = original_state[name].view(-1).to(flat_param.device, flat_param.dtype)
            # Compute std of training delta for this group's elements
            delta = flat_param[drop_mask] - flat_orig[drop_mask]
            delta_std = delta.float().std().item()
            if delta_std < 1e-12:
                continue
            n_elems = drop_mask.sum().item()
            noise = torch.empty(n_elems, device='cpu', dtype=torch.float32)
            noise.normal_(generator=rng)
            noise = noise.to(flat_param.device, flat_param.dtype)
            flat_param[drop_mask] += alpha * delta_std * noise


def load_memo_data(cfg):
    """Load memo_results.csv and build group-to-IDs mapping from experiment metadata."""
    memo_path = os.path.join(cfg.paths.output_dir, "memo_results.csv")
    assert os.path.exists(memo_path), f"memo_results.csv not found at {memo_path}"
    memo_df = pd.read_csv(memo_path)
    memo_df = memo_df.dropna(subset=['Ground_Truth', 'Question']).reset_index(drop=True)

    # Build group -> set of unique IDs
    experiment_meta = load_from_disk(cfg.data.experiment_metadata_path)
    meta_df = experiment_meta.to_pandas()
    subset_df = meta_df[meta_df['Subset'] == True][['Unique ID', 'group_id']]
    group_to_ids = {}
    for group_id, group_df in subset_df.groupby('group_id'):
        group_to_ids[int(group_id)] = set(group_df['Unique ID'].tolist())

    return memo_df, group_to_ids


def build_formatted_dataset(memo_df, cfg, uid_filter=None):
    """
    Rebuild a formatted HF dataset from memo_results.csv rows,
    optionally filtering to a set of unique IDs.
    """
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

    num_groups = cfg.mask.num_groups  # 6

    # Load the original pretrained weights for delta computation
    print(f"Loading original pretrained model for delta computation...")
    original_model, _ = get_model(cfg.model)
    original_state = {name: p.data.cpu().clone() for name, p in original_model.named_parameters()}
    del original_model
    torch.cuda.empty_cache()

    # Load memorization data and group mapping
    memo_df, group_to_ids = load_memo_data(cfg)
    print(f"Loaded {len(memo_df)} memorization rows, {num_groups} groups")
    for g, ids in sorted(group_to_ids.items()):
        print(f"  Group {g}: {len(ids)} people, {len(memo_df[memo_df['UniqueID'].isin(ids)])} rows")

    out_dir = cfg.paths.output_dir
    utility_path = os.path.join(out_dir, "dropout_utility.csv")
    all_utility_results = []

    # Pre-build formatted datasets (reuse across alpha sweep)
    print("Pre-building formatted datasets...")
    formatted_all = build_formatted_dataset(memo_df, cfg)
    formatted_per_group = {}
    for g_idx in range(num_groups):
        formatted_per_group[g_idx] = build_formatted_dataset(memo_df, cfg, uid_filter=group_to_ids[g_idx])
    # Retained sets for each perturbed group
    formatted_retained = {}
    for bit_idx in GROUPS_TO_PERTURB:
        retained_ids = set()
        for other_idx in range(num_groups):
            if other_idx != bit_idx:
                retained_ids |= group_to_ids[other_idx]
        formatted_retained[bit_idx] = build_formatted_dataset(memo_df, cfg, uid_filter=retained_ids)

    # --- Baseline evaluation (no perturbation) ---
    print("=" * 60)
    print("BASELINE (no perturbation)")
    print("=" * 60)

    # Utility (only done once on baseline)
    baseline_utility = run_lm_eval(model, tokenizer)
    all_utility_results.append({"condition": "baseline", **baseline_utility})
    for k, v in sorted(baseline_utility.items()):
        print(f"  {k}: {v:.4f}")

    # Memorization on all IDs
    memo_baseline = evaluate(model, tokenizer, formatted_all,
                             batch_size=cfg.memorization.batch_size,
                             max_new_tokens=cfg.memorization.max_new_tokens)
    memo_baseline.to_csv(os.path.join(out_dir, "dropout_memo_baseline.csv"), index=False, escapechar='\\')
    print(f"  Memorization baseline: {memo_baseline['Is_Correct'].mean():.4f} accuracy")

    # Memorization per-group on baseline (no perturbation)
    for g_idx in range(num_groups):
        memo_g = evaluate(model, tokenizer, formatted_per_group[g_idx],
                          batch_size=cfg.memorization.batch_size,
                          max_new_tokens=cfg.memorization.max_new_tokens)
        memo_g.to_csv(os.path.join(out_dir, f"dropout_memo_baseline_group_{g_idx}.csv"), index=False, escapechar='\\')
        print(f"  Memorization baseline group {g_idx} ({len(group_to_ids[g_idx])} people): "
              f"{memo_g['Is_Correct'].mean():.4f} accuracy")

    # Free baseline model from GPU
    del model
    torch.cuda.empty_cache()

    # --- Perturb each selected group at each alpha ---
    for alpha in ALPHAS:
        for bit_idx in GROUPS_TO_PERTURB:
            print()
            print("=" * 60)
            print(f"PERTURBING GROUP {bit_idx} (alpha={alpha})")
            print("=" * 60)

            # Reload the trained model weights fresh
            trained_model, _ = get_model(cfg.memorization)
            trained_model.to(device)

            # Add noise to this group's elements
            apply_noise_perturbation(trained_model, packed_masks, bit_idx,
                                     original_state, alpha, seed=cfg.seed)

            # Memorization — on the IDs from the perturbed group
            memo_results = evaluate(trained_model, tokenizer, formatted_per_group[bit_idx],
                                    batch_size=cfg.memorization.batch_size,
                                    max_new_tokens=cfg.memorization.max_new_tokens)
            memo_save_path = os.path.join(out_dir, f"dropout_memo_alpha{alpha}_bit_{bit_idx}.csv")
            memo_results.to_csv(memo_save_path, index=False, escapechar='\\')
            print(f"  Memorization perturbed group {bit_idx} ({len(group_to_ids[bit_idx])} people): "
                  f"{memo_results['Is_Correct'].mean():.4f} accuracy")

            # Memorization — on the retained groups
            memo_ret = evaluate(trained_model, tokenizer, formatted_retained[bit_idx],
                                batch_size=cfg.memorization.batch_size,
                                max_new_tokens=cfg.memorization.max_new_tokens)
            memo_ret.to_csv(os.path.join(out_dir, f"dropout_memo_alpha{alpha}_bit_{bit_idx}_retained.csv"),
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
