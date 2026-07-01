import hydra
import os
import re
from omegaconf import OmegaConf
import yaml
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value, load_from_disk
from utils import set_seed

MIN_MEMORIZED_PER_PERSON_FIELD = 2


def save_agnostic_dataset(dataset_dict, out_dir):
    """
    Saves a dataset dictionary with keys like 'Email_Address_retain' into a structure
    loadable via load_dataset(dir, name='Email_Address', split='retain').
    """
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    configs = defaultdict(list)
    valid_splits = ["retain_paraphrased", "forget_paraphrased", "retain", "forget", "full", "train"]

    for original_key, ds in dataset_dict.items():
        matched_split = next((s for s in valid_splits if original_key.endswith(f"_{s}")), None)

        if matched_split:
            raw_subset = original_key[:-(len(matched_split) + 1)]
            split_name = matched_split
        else:
            raw_subset = "default"
            split_name = original_key

        safe_subset = re.sub(r"[^a-zA-Z0-9_]", "", raw_subset.replace(" ", "_"))
        filename = f"{safe_subset}_{split_name}.parquet"
        file_path = os.path.join(data_dir, filename)

        ds.to_parquet(file_path)

        configs[safe_subset].append({
            "split": split_name,
            "path": f"data/{filename}"
        })

    readme_configs = []
    for subset_name, files in configs.items():
        readme_configs.append({
            "config_name": subset_name,
            "data_files": files,
        })

    readme_content = {"configs": readme_configs}

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("---\n")
        yaml.dump(readme_content, f, default_flow_style=False, sort_keys=False)
        f.write("---\n")
        f.write("# Memorized PII Dataset\n\n")
        f.write(f"Contains {len(configs)} subsets (fields).\n")

    print(f"Dataset saved to {out_dir}")
    print(f"Available subsets: {list(configs.keys())}")


def assign_forget_retain_groups(id_to_group, num_forget_groups, num_retain_groups):
    """
    Assign groups to forget/retain once, using a balanced greedy split.
    Returns (forget_groups_set, retain_groups_set).
    """
    group_sizes = defaultdict(int)
    for gid in id_to_group.values():
        group_sizes[gid] += 1

    # Sort by size descending, then by group ID ascending as tiebreaker
    # so the result is deterministic regardless of dict insertion order.
    sorted_groups = sorted(group_sizes.keys(), key=lambda g: (-group_sizes[g], g))
    total_groups_needed = num_forget_groups + num_retain_groups

    if len(sorted_groups) < total_groups_needed:
        raise ValueError(
            f"Only {len(sorted_groups)} groups exist, but {total_groups_needed} are needed."
        )

    forget_grps = []
    retain_grps = []
    for gid in sorted_groups[:total_groups_needed]:
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


def get_eligible_ids(memorized_df, field):
    """Return set of UniqueIDs with at least MIN_MEMORIZED_PER_PERSON_FIELD memorized prompts for field."""
    field_df = memorized_df[memorized_df["Field"] == field]
    unique_counts = field_df.drop_duplicates(subset=["UniqueID", "Question"]).groupby("UniqueID").size()
    return set(unique_counts[unique_counts >= MIN_MEMORIZED_PER_PERSON_FIELD].index)


def select_people_for_field(eligible_ids, id_to_group, forget_groups, retain_groups,
                            num_per_side, preferred_ids=None):
    """
    Select forget/retain people for a field from eligible_ids using fixed group assignment.
    People in preferred_ids are selected first to maximize sharing across fields.
    Returns (forget_ids_set, retain_ids_set).
    """
    if preferred_ids is None:
        preferred_ids = set()

    forget_preferred, forget_rest = [], []
    retain_preferred, retain_rest = [], []

    for uid in sorted(eligible_ids):
        gid = id_to_group.get(str(uid), id_to_group.get(uid))
        if gid in forget_groups:
            if uid in preferred_ids:
                forget_preferred.append(uid)
            else:
                forget_rest.append(uid)
        elif gid in retain_groups:
            if uid in preferred_ids:
                retain_preferred.append(uid)
            else:
                retain_rest.append(uid)

    np.random.shuffle(forget_preferred)
    np.random.shuffle(forget_rest)
    np.random.shuffle(retain_preferred)
    np.random.shuffle(retain_rest)

    val_forget = forget_preferred + forget_rest
    val_retain = retain_preferred + retain_rest

    if len(val_forget) < num_per_side:
        raise ValueError(
            f"Only {len(val_forget)} eligible forget people (groups {sorted(forget_groups)}), "
            f"but {num_per_side} were requested."
        )
    if len(val_retain) < num_per_side:
        raise ValueError(
            f"Only {len(val_retain)} eligible retain people (groups {sorted(retain_groups)}), "
            f"but {num_per_side} were requested."
        )

    return set(val_forget[:num_per_side]), set(val_retain[:num_per_side])


def _make_features():
    return Features({
        "ID": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "field": Value("string"),
        "group_id": Value("int32"),
    })


def _rows_to_df(rows, id_to_group):
    out = pd.DataFrame(rows)[["UniqueID", "Question", "Ground_Truth", "Field"]]
    out = out.rename(columns={
        "UniqueID": "ID", "Question": "question",
        "Ground_Truth": "answer", "Field": "field",
    })
    out["group_id"] = out["ID"].map(id_to_group).astype("int32")
    return out


def _sample_and_split(memorized, field, selected_ids, forget_ids, retain_ids, id_to_group, splits, prefix=""):
    """Sample 2 memorized prompts per person and create forget/retain/full splits."""
    features = _make_features()
    field_mem = memorized[(memorized["Field"] == field) & (memorized["UniqueID"].isin(selected_ids))]

    main_rows = []
    para_rows = []

    for uid, group in field_mem.groupby("UniqueID"):
        unique_questions = group.drop_duplicates(subset="Question")
        if len(unique_questions) < MIN_MEMORIZED_PER_PERSON_FIELD:
            raise ValueError(
                f"Person {uid} has {len(unique_questions)} unique memorized "
                f"question(s) for field '{field}', need at least "
                f"{MIN_MEMORIZED_PER_PERSON_FIELD} distinct questions."
            )
        sampled = unique_questions.sample(n=MIN_MEMORIZED_PER_PERSON_FIELD).to_dict("records")
        main_rows.append(sampled[0])
        para_rows.append(sampled[1])

    main_df = _rows_to_df(main_rows, id_to_group)
    para_df = _rows_to_df(para_rows, id_to_group)

    safe_field = field.replace(" ", "_").replace("'", "")
    key_prefix = f"{prefix}{safe_field}" if prefix else safe_field

    for split_name, id_set in [("forget", forget_ids), ("retain", retain_ids)]:
        splits[f"{key_prefix}_{split_name}"] = Dataset.from_pandas(
            main_df[main_df["ID"].isin(id_set)].reset_index(drop=True),
            features=features, preserve_index=False,
        )
        splits[f"{key_prefix}_{split_name}_paraphrased"] = Dataset.from_pandas(
            para_df[para_df["ID"].isin(id_set)].reset_index(drop=True),
            features=features, preserve_index=False,
        )

    splits[f"{key_prefix}_full"] = Dataset.from_pandas(
        pd.concat([main_df, para_df]).reset_index(drop=True),
        features=features, preserve_index=False,
    )


def build_splits(df, fields, num_people, id_to_group, num_forget_groups, num_retain_groups,
                 validation_fields=None, validation_num_people=None):
    """
    Build per-field forget/retain/full splits plus paraphrased splits.

    Groups are assigned to forget/retain ONCE and shared across all fields
    (main and validation). People eligible for multiple fields are preferred
    so that datasets share the same people as much as possible.
    """
    memorized = df[(df["Is_Correct"] == True) & (df["Ground_Truth"].notna())].copy()
    memorized["UniqueID"] = memorized["UniqueID"].astype(str)

    # Assign forget/retain groups once for all fields
    forget_groups, retain_groups = assign_forget_retain_groups(
        id_to_group, num_forget_groups, num_retain_groups
    )
    print(f"Global group assignment: forget={sorted(forget_groups)}, retain={sorted(retain_groups)}")

    splits = {}
    all_selected_ids = set()
    num_per_side = num_people // 2

    # Find eligible people per main field
    main_eligible = {field: get_eligible_ids(memorized, field) for field in fields}

    # People eligible for ALL main fields — prefer these for sharing
    shared_main = set.intersection(*main_eligible.values()) if len(fields) > 1 else set()
    if shared_main:
        print(f"Main splits: {len(shared_main)} people eligible for all {len(fields)} fields")

    for field in fields:
        forget_ids, retain_ids = select_people_for_field(
            main_eligible[field], id_to_group, forget_groups, retain_groups,
            num_per_side, preferred_ids=shared_main,
        )
        selected_ids = forget_ids | retain_ids
        all_selected_ids.update(selected_ids)

        print(f"Field '{field}': selected {len(selected_ids)} people")
        print(f"  forget: {len(forget_ids)}, retain: {len(retain_ids)}")

        _sample_and_split(memorized, field, selected_ids, forget_ids, retain_ids, id_to_group, splits)

    # Build validation splits from people NOT used in any main split
    if validation_fields is not None:
        if isinstance(validation_fields, str):
            validation_fields = [validation_fields]

        if validation_num_people is None:
            validation_num_people = num_people

        val_memorized = memorized[~memorized["UniqueID"].isin(all_selected_ids)]
        val_num_per_side = validation_num_people // 2

        # Find eligible people per validation field (excluding main-split people)
        val_eligible = {vf: get_eligible_ids(val_memorized, vf) for vf in validation_fields}

        # People eligible for ALL validation fields — prefer these for sharing
        shared_val = set.intersection(*val_eligible.values()) if len(validation_fields) > 1 else set()
        if shared_val:
            print(f"Validation: {len(shared_val)} people eligible for all {len(validation_fields)} validation fields")

        for val_field in validation_fields:
            val_forget_ids, val_retain_ids = select_people_for_field(
                val_eligible[val_field], id_to_group, forget_groups, retain_groups,
                val_num_per_side, preferred_ids=shared_val,
            )
            val_selected = val_forget_ids | val_retain_ids

            print(f"Validation field '{val_field}': selected {len(val_selected)} people (excluded {len(all_selected_ids)} from main splits)")
            print(f"  forget: {len(val_forget_ids)}, retain: {len(val_retain_ids)}")

            _sample_and_split(memorized, val_field, val_selected, val_forget_ids, val_retain_ids,
                              id_to_group, splits, prefix="validation_")

    # --- Verification ---
    print("\n=== Verification ===")
    # Extract IDs per split
    split_ids = {}
    for key, ds in splits.items():
        split_ids[key] = set(ds["ID"])

    # Check no person is in both forget and retain for any field (including across fields)
    all_forget_ids = set()
    all_retain_ids = set()
    for key, ids in split_ids.items():
        if "_forget" in key and "_paraphrased" not in key and "_full" not in key:
            all_forget_ids |= ids
        elif "_retain" in key and "_paraphrased" not in key and "_full" not in key:
            all_retain_ids |= ids

    cross_leak = all_forget_ids & all_retain_ids
    if cross_leak:
        print(f"  WARNING: {len(cross_leak)} people appear in BOTH forget and retain across splits!")
        for pid in sorted(cross_leak)[:5]:
            in_splits = [k for k, ids in split_ids.items() if pid in ids]
            print(f"    Person {pid}: {in_splits}")
    else:
        print("  OK: No person appears in both forget and retain (across all fields)")

    # Check sharing across fields
    all_field_keys = sorted(set(
        k.rsplit("_forget", 1)[0] for k in split_ids if "_forget" in k and "_paraphrased" not in k and "_full" not in k
    ))
    if len(all_field_keys) > 1:
        for i, k1 in enumerate(all_field_keys):
            for k2 in all_field_keys[i+1:]:
                ids1 = split_ids.get(f"{k1}_forget", set()) | split_ids.get(f"{k1}_retain", set())
                ids2 = split_ids.get(f"{k2}_forget", set()) | split_ids.get(f"{k2}_retain", set())
                overlap = ids1 & ids2
                print(f"  People shared between '{k1}' and '{k2}': {len(overlap)} / {min(len(ids1), len(ids2))}")

    # Check no main-split person in validation
    main_keys = [k for k in split_ids if not k.startswith("validation_")]
    val_keys = [k for k in split_ids if k.startswith("validation_")]
    main_all = set().union(*(split_ids[k] for k in main_keys)) if main_keys else set()
    val_all = set().union(*(split_ids[k] for k in val_keys)) if val_keys else set()
    main_val_leak = main_all & val_all
    if main_val_leak:
        print(f"  WARNING: {len(main_val_leak)} people appear in both main and validation splits!")
    else:
        print("  OK: No overlap between main and validation splits")

    print("=== End Verification ===\n")

    return splits


@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(config):
    set_seed(config.seed)
    resolved_cfg = OmegaConf.to_container(config, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))

    create_cfg = config.get("create_dataset", {})
    memo_path = create_cfg.get("memo_results_path", None)
    if memo_path is None:
        memo_path = os.path.join(config.paths.output_dir, "memo_results.csv")

    print(f"Loading memorization results from {memo_path}")
    df = pd.read_csv(memo_path)

    if "interesting_pii" in config.get("data", {}):
        fields = list(config.data.interesting_pii)
    else:
        fields = df["Field"].unique().tolist()

    num_people = create_cfg.get("num_people")
    if num_people is None:
        raise ValueError("create_dataset.num_people must be specified in the config.")

    num_forget_groups = create_cfg.get("num_forget_groups")
    num_retain_groups = create_cfg.get("num_retain_groups")

    # Load group assignments saved by data_replication.py
    experiment_meta = load_from_disk(config.data.experiment_metadata_path)
    meta_df = experiment_meta.to_pandas()
    meta_df = meta_df[meta_df["Subset"] == True][["Unique ID", "group_id"]].dropna(subset=["group_id"])
    id_to_group = {row["Unique ID"]: int(row["group_id"]) for _, row in meta_df.iterrows()}
    print(f"Loaded group mapping from experiment metadata: {len(id_to_group)} people")

    print(f"Fields: {fields}")
    print(f"Num people: {num_people}, Forget groups: {num_forget_groups}, Retain groups: {num_retain_groups}")
    print(f"Total rows in memo results: {len(df)}")

    validation_fields = create_cfg.get("validation_fields", create_cfg.get("validation_field", None))
    validation_num_people = create_cfg.get("validation_num_people", None)

    splits = build_splits(df, fields, num_people=num_people, id_to_group=id_to_group,
                          num_forget_groups=num_forget_groups, num_retain_groups=num_retain_groups,
                          validation_fields=validation_fields, validation_num_people=validation_num_people)

    # Build relearn split: memorized QA pairs from people NOT in any existing split
    all_split_ids = set()
    for ds in splits.values():
        all_split_ids.update(ds["ID"])

    memorized = df[(df["Is_Correct"] == True) & (df["Ground_Truth"].notna())].copy()
    memorized["UniqueID"] = memorized["UniqueID"].astype(str)
    relearn_rows = memorized[~memorized["UniqueID"].isin(all_split_ids)]

    if len(relearn_rows) > 0:
        relearn_df = relearn_rows[["UniqueID", "Question", "Ground_Truth", "Field"]].drop_duplicates(
            subset=["UniqueID", "Question"]
        ).rename(columns={
            "UniqueID": "ID", "Question": "question",
            "Ground_Truth": "answer", "Field": "field",
        })
        relearn_df["group_id"] = relearn_df["ID"].map(id_to_group).astype("int32")
        relearn_df = relearn_df.reset_index(drop=True)

        splits["relearn_train"] = Dataset.from_pandas(relearn_df, features=_make_features(), preserve_index=False)
        print(f"Relearn split: {len(relearn_df)} QA pairs from {relearn_df['ID'].nunique()} unseen people")
    else:
        print("WARNING: No memorized people left for relearn split")

    out_dir = config.paths.memorized_datasets_dir
    save_agnostic_dataset(splits, out_dir)


if __name__ == "__main__":
    main()
