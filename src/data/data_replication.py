from datasets import load_dataset, load_from_disk, Dataset as HFDataset
import numpy as np
import pandas as pd
from utils import temp_seed

def replicate(config):

    dataset_name = config.data.pii_sentences
    
    ds = load_from_disk(config.data.panorama_added_info_path)
    ds_plus = load_from_disk(config.data.panorama_plus_rows_path, keep_in_memory=True)

    interesting_PII = config.data.interesting_pii
    samples_to_repeat = {}
    
    # Create subset and mapping
    if config.data.get('subsample_dataset', None) is not None:
        subset_size = config.data.subsample_dataset
        total_size = len(ds_plus)
        random_indices = np.random.choice(total_size, subset_size, replace=False)
        include_mask = np.zeros(total_size, dtype=bool)
        include_mask[random_indices] = True
        ds_plus = ds_plus.add_column("Subset", include_mask.tolist()) 
        subset_plus = ds_plus.select(random_indices.tolist())
        # Create mapping from subset index to ds_plus index
        subset_to_full_idx = random_indices
    else:
        ds_plus = ds_plus.add_column("Subset", [True] * len(ds_plus))
        subset_plus = ds_plus
        subset_to_full_idx = np.arange(len(ds_plus))
     
    # --- 2. ASSIGN PEOPLE TO GROUPS (Moved Up) ---
    # We do this early so both QA and Sentences can inherit the ID
    if config.get('mask', {}).get('enable', False):
        num_groups = config.mask.num_groups
        
        # Identify unique IDs for the subset
        all_ids = sorted(list(set(ds_plus.filter(lambda x: x['Subset'])['Unique ID'])))
        
        with temp_seed(config.mask.seed):
            np.random.shuffle(all_ids)
        
        # Build lookup dictionary
        id_to_group = {doc_id: i % num_groups for i, doc_id in enumerate(all_ids)}
        
        # Assign to ds_plus
        def assign_group(example):
            # Default to -1 or None if not in subset, usually safe to keep None or a sentinel
            gid = id_to_group.get(example['Unique ID']) if example['Subset'] else None
            example['group_id'] = gid
            return example

        ds_plus = ds_plus.map(assign_group, num_proc=8)
    else:
        # Create empty column if masking is disabled to avoid KeyErrors later
        ds_plus = ds_plus.add_column("group_id", [None] * len(ds_plus)) 
    
    
        
    
    # Handle QA generation replication factors
    qa_samples = []
    if config.data.get('qa_generation', False):
        total_size = len(subset_to_full_idx)
        qa_subset = int(total_size*config.data.qa_subsample_ratio)
        selected_indices = np.random.choice(subset_to_full_idx, size=qa_subset, replace=False)
        qa_included_global_mask = np.zeros(len(ds_plus), dtype=bool)
        qa_included_global_mask[selected_indices] = True
        ds_plus = ds_plus.add_column("QA_included", qa_included_global_mask.tolist())
        subset_qa_plus = ds_plus.select(selected_indices.tolist())
        
        qa_samples = [
            {
                'id': prompts['Unique ID'], 
                'content-type': "qa_generated", 
                'text': example['prompt'] + example['answer'],
                'group_id': prompts['group_id']
            } 
            for prompts in subset_qa_plus          
            for example in prompts['QA_generated_prompt'] 
        ] * config.data.qa_replication_per_question
        
        print(len(qa_samples), "QA generated prompts created.", flush=True)
        
        
    # Handle PII replication factors
    for field in interesting_PII:
        print(f"Calculating replication factors for field: {field}", flush=True)
        # Initialize replication factors for full dataset (all zeros or ones)
        if config.data.replication_pattern == 'constant':
            index_col = [0] * len(ds_plus)
            # Set constant value only for subset items
            constant_value = config.data.get('replication_max_repetitions', 5000)
            for i in range(len(subset_plus)):
                full_idx = subset_to_full_idx[i]
                index_col[full_idx] = constant_value
                
        elif config.data.replication_pattern == 'linear':
            index_col = [0] * len(ds_plus)
            
            # Calculate replication factors only for subset
            vals = np.array([len(rows) for rows in subset_plus[f"{field}_rows"]])
            top_number = min(config.data.get('replication_top_number', 1000), len(vals))
            max_reps = config.data.get('replication_max_repetitions', 10000)
            top_indices = np.argsort(vals)[-top_number:][::-1]
            samples_to_repeat[field] = top_indices

            col = [i for i in range(max_reps, 0, -max_reps // top_number)]
            
            for i, subset_idx in enumerate(samples_to_repeat[field]):
                full_idx = subset_to_full_idx[subset_idx]
                index_col[full_idx] = col[i]
        else:
            raise ValueError("Unknown replication pattern")
            
        ds_plus = ds_plus.add_column(f"{field}_repetition_factor", index_col)
    
    # SENTENCE REPLICATION
    index = np.zeros(len(ds), dtype=int)
    sentence_group_map = np.full(len(ds), -1) 
    all_group_ids = np.array(ds_plus['group_id'])
    
    for field in interesting_PII:
        rows_per_person = ds_plus[f"{field}_rows"]               
        repetitions = np.array([ds_plus[i][f"{field}_repetition_factor"]
                                for i in range(len(rows_per_person))]) 
        
        for i, rows in enumerate(rows_per_person):
            if len(rows) == 0:
                continue
            
            rep = repetitions[i]
            base_value = rep // len(rows)
            extra = rep % len(rows)
            if extra > 0:
                index[rows[:extra]] = np.maximum(index[rows[:extra]], base_value + 1)
            if extra < len(rows):
                index[rows[extra:]] = np.maximum(index[rows[extra:]], base_value)
                
            gid = all_group_ids[i]
            if gid is not None and gid != -1: 
                sentence_group_map[rows] = gid
    
    # FIX COLLATERAL REPLICATIONS        
    for field in interesting_PII:
        ds_plus = ds_plus.map(
            lambda ex, f=field: {
                f"{f}_repetition_factor": int(index[ex[f"{f}_rows"]].sum())
            }
        )    
    

        
            




    # Save experiment-specific columns to a separate dataset (keyed by Unique ID)
    experiment_cols = ['Unique ID', 'Subset', 'group_id']
    if 'QA_included' in ds_plus.column_names:
        experiment_cols.append('QA_included')
    for field in interesting_PII:
        col_name = f"{field}_repetition_factor"
        if col_name in ds_plus.column_names:
            experiment_cols.append(col_name)
    experiment_df = ds_plus.to_pandas()[experiment_cols]
    experiment_ds = HFDataset.from_pandas(experiment_df)
    experiment_ds.save_to_disk(config.data.experiment_metadata_path)
    replicated = replicate_data(dataset_name, index)
    replicated_df = replicated.to_pandas()
    
    expanded_indices = np.repeat(np.arange(len(index)), index)
    replicated_group_ids = sentence_group_map[expanded_indices]
    replicated_df['group_id'] = replicated_group_ids

    qa_df = pd.DataFrame(qa_samples)
    print(len(replicated_df), " pii sentences after replication.", flush=True)
    print(len(qa_df), " QA samples after replication.", flush=True)
    print(len(replicated_df) + len(qa_df), "total rows after replication.", flush=True)
    combined_df = pd.concat([replicated_df,qa_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(len(combined_df), "total rows after shuffling.", flush=True)
    replicated = HFDataset.from_pandas(combined_df)
    replicated.save_to_disk(config.data.pii)
    print("Replication completed and saved. in ", config.data.pii, flush=True)


def replicate_data(dataset_name, occurrences):
    original = load_dataset(dataset_name)
    return _replicate_split(original['train'], occurrences)


def _replicate_split(split_dataset, occurrences):
    if len(split_dataset) != len(occurrences):
        raise ValueError(
            f"Length mismatch: dataset has {len(split_dataset)} rows but "
            f"occurrences has {len(occurrences)} elements"
        )
    idx = []
    for i, occ in enumerate(occurrences):
        if occ < 0:
            raise ValueError("Occurrences must be non-negative")
        if occ > 0:
            idx.extend([i] * occ)
            
    return split_dataset.select(idx)