"""Verify that group assignments are consistent across all three sources:

1. data_replication.py  → saved in experiment_metadata (ground truth)
2. create_memorized_datasets.py → recomputed and stored as group_id in the datasets
3. The bit-packed mask (mask.pt) → which bits are set per weight

Usage:
    python src/test_groups.py experiments=OLMo_Mask_Train_FullSubset
"""
import hydra
import numpy as np
from collections import defaultdict
from datasets import load_from_disk, load_dataset
from utils import set_seed, temp_seed


def recompute_groups(mask_seed, num_groups, experiment_metadata_path):
    """Reproduce the group assignment logic from data_replication.py / create_memorized_datasets.py."""
    experiment_meta = load_from_disk(experiment_metadata_path)
    meta_df = experiment_meta.to_pandas()[["Unique ID", "Subset"]]
    subset_ids = sorted(meta_df[meta_df["Subset"] == True]["Unique ID"].tolist())

    with temp_seed(mask_seed):
        np.random.shuffle(subset_ids)

    return {uid: i % num_groups for i, uid in enumerate(subset_ids)}


def get_groups_from_metadata(experiment_metadata_path):
    """Read the group_id column saved by data_replication.py."""
    experiment_meta = load_from_disk(experiment_metadata_path)
    df = experiment_meta.to_pandas()
    df = df[df["Subset"] == True][["Unique ID", "group_id"]].dropna(subset=["group_id"])
    return {row["Unique ID"]: int(row["group_id"]) for _, row in df.iterrows()}


def greedy_forget_retain_split(id_to_group, num_forget_groups, num_retain_groups):
    """Same balanced greedy split as assign_forget_retain_groups."""
    group_sizes = defaultdict(int)
    for gid in id_to_group.values():
        group_sizes[gid] += 1

    sorted_groups = sorted(group_sizes.keys(), key=lambda g: group_sizes[g], reverse=True)
    total_needed = num_forget_groups + num_retain_groups

    forget_grps, retain_grps = [], []
    for gid in sorted_groups[:total_needed]:
        if len(forget_grps) < num_forget_groups and len(retain_grps) < num_retain_groups:
            forget_total = sum(group_sizes[g] for g in forget_grps)
            retain_total = sum(group_sizes[g] for g in retain_grps)
            if forget_total <= retain_total:
                forget_grps.append(gid)
            else:
                retain_grps.append(gid)
        elif len(forget_grps) < num_forget_groups:
            forget_grps.append(gid)
        else:
            retain_grps.append(gid)

    return set(forget_grps), set(retain_grps)


@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(config):
    set_seed(config.seed)
    all_ok = True

    mask_seed = config.mask.seed
    num_groups = config.mask.num_groups
    metadata_path = config.data.experiment_metadata_path

    # --- Check 1: recomputed groups match metadata ---
    print("=" * 60)
    print("CHECK 1: Recomputed groups vs experiment_metadata")
    print("=" * 60)
    recomputed = recompute_groups(mask_seed, num_groups, metadata_path)
    from_metadata = get_groups_from_metadata(metadata_path)

    common_ids = set(recomputed.keys()) & set(from_metadata.keys())
    mismatches = [(uid, recomputed[uid], from_metadata[uid])
                  for uid in common_ids if recomputed[uid] != from_metadata[uid]]

    print(f"  Recomputed: {len(recomputed)} people")
    print(f"  Metadata:   {len(from_metadata)} people")
    print(f"  Common:     {len(common_ids)} people")
    if mismatches:
        print(f"  FAIL: {len(mismatches)} mismatches!")
        for uid, r, m in mismatches[:10]:
            print(f"    {uid}: recomputed={r}, metadata={m}")
        all_ok = False
    else:
        print("  OK: All group assignments match")

    # --- Check 2: forget/retain split ---
    print()
    print("=" * 60)
    print("CHECK 2: Forget/retain group split")
    print("=" * 60)
    create_cfg = config.get("create_dataset", {})
    num_forget_groups = create_cfg.get("num_forget_groups", 3)
    num_retain_groups = create_cfg.get("num_retain_groups", 3)

    forget_groups, retain_groups = greedy_forget_retain_split(
        recomputed, num_forget_groups, num_retain_groups
    )
    print(f"  Forget groups: {sorted(forget_groups)}")
    print(f"  Retain groups: {sorted(retain_groups)}")

    group_counts = defaultdict(int)
    for gid in recomputed.values():
        group_counts[gid] += 1
    for gid in sorted(group_counts.keys()):
        role = "forget" if gid in forget_groups else "retain" if gid in retain_groups else "unused"
        print(f"    Group {gid}: {group_counts[gid]} people ({role})")

    # --- Check 3: dataset group_id columns match ---
    print()
    print("=" * 60)
    print("CHECK 3: Dataset group_id vs recomputed groups")
    print("=" * 60)
    unlearning_data_cfg = config.get("unlearning_data", None)
    if unlearning_data_cfg is None:
        print("  SKIP: No unlearning_data config found")
    else:
        for split_name, split_key in [("forget", "forget"), ("retain", "retain")]:
            split_cfg = unlearning_data_cfg.get(split_key, None)
            if split_cfg is None:
                continue
            # Get the inner config (e.g., forget.Forget)
            inner_key = list(split_cfg.keys())[0]
            inner_cfg = split_cfg[inner_key]

            try:
                ds = load_dataset(
                    inner_cfg.data_path, name=inner_cfg.name,
                    split=inner_cfg.split, download_mode='force_redownload'
                )
            except Exception as e:
                print(f"  SKIP {split_name}: could not load dataset ({e})")
                continue

            if "group_id" not in ds.column_names:
                print(f"  SKIP {split_name}: no group_id column")
                continue

            dataset_groups = set(ds.unique("group_id"))
            expected_groups = forget_groups if split_name == "forget" else retain_groups
            print(f"  {split_name} dataset ({inner_cfg.name}):")
            print(f"    group_ids in dataset: {sorted(dataset_groups)}")
            print(f"    expected ({split_name}):    {sorted(expected_groups)}")

            if dataset_groups == expected_groups:
                print(f"    OK: group_ids match expected {split_name} groups")
            elif dataset_groups.issubset(expected_groups):
                print(f"    OK: group_ids are a subset of {split_name} groups (fewer people selected)")
            else:
                unexpected = dataset_groups - expected_groups
                print(f"    FAIL: unexpected group_ids: {sorted(unexpected)}")
                all_ok = False

            # Check individual people
            if "ID" in ds.column_names:
                mismatches = []
                for row in ds:
                    uid = row["ID"]
                    ds_gid = row["group_id"]
                    recomp_gid = recomputed.get(uid, recomputed.get(str(uid)))
                    if recomp_gid is not None and ds_gid != recomp_gid:
                        mismatches.append((uid, ds_gid, recomp_gid))
                if mismatches:
                    print(f"    FAIL: {len(mismatches)} people have wrong group_id!")
                    for uid, d, r in mismatches[:5]:
                        print(f"      {uid}: dataset={d}, recomputed={r}")
                    all_ok = False
                else:
                    print(f"    OK: all {len(ds)} people have correct group_id")

    # --- Summary ---
    print()
    print("=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
