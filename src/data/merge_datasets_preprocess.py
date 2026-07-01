from datasets import load_dataset, concatenate_datasets
import os
from functools import partial
def preprocess_data(config):
        
    
    ## CONVERTION TO TRAIN READY DATASET
    pii = load_dataset(
        "json",
        data_files=os.path.join(config.data.pii, "*.jsonl"),
        split="train",
        )
    dataset = load_dataset(
        "json",
        data_files=os.path.join(config.data.neutral, "*.jsonl"),
        split="train",
        )
    
    
    # def preprocess(examples, name=-1):
    #     return {
    #         "input_ids": examples["input_ids"],
    #         "labels": examples["input_ids"],
    #         "dataset_name": [name] * len(examples["input_ids"] if name != -1 else [int(x) for x in examples['group_id']])
    #     }
    
    def preprocess(examples, name=-1):
        batch_len = len(examples["input_ids"])
    
        if name != -1:
            dataset_names = [name] * batch_len
        else:
            if 'group_id' in examples:
                dataset_names = [int(x) for x in examples['group_id']]
            else:
                dataset_names = [-1] * batch_len
        return {
            "input_ids": examples["input_ids"],
            "labels": examples["input_ids"],
            "dataset_name": dataset_names
        }
    
    preprocess_neutral = partial(preprocess, name=-1)
    pii_remove_columns = pii.column_names
    dataset_remove_columns = dataset.column_names
    pii = pii.map(preprocess, batched=True, num_proc=16,load_from_cache_file=False, remove_columns=pii_remove_columns)
    dataset = dataset.map(preprocess_neutral, batched=True, num_proc=16,load_from_cache_file=False, remove_columns=dataset_remove_columns)
    len_pii = len(pii)
    len_dataset = len(dataset)
    print(f"PII dataset length: {len_pii}")
    print(f"Neutral dataset length: {len_dataset}")
    print("PII represents {:.2f}% of the total dataset".format(len_pii/(len_pii+len_dataset)*100))
    training = concatenate_datasets([pii, dataset])
    training.save_to_disk(config.data.preprocessed_path)