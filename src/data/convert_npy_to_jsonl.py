import os
import json
import multiprocessing as mp
import random
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict

def convert_to_jsonl(config):
    input_dir = config.data.pii
    output_dir = config.data.pii
    tokenizer_name = config.model.name
    seq_len = config.data.seq_length
    seed = config.mask.seed
    
    
    tokenize_directly_to_jsonl(
        load_from_disk(input_dir), 
        tokenizer_name, 
        'text', 
        output_dir, 
        seq_len, 
        id_field='id', 
        group_id_field='group_id',
        seed=seed,  
        sequences_per_file=50000, 
        num_proc=8
    )

def _worker_init(tokenizer_name):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

def _tokenize_batch(batch_data):
    doc_ids, group_ids, docs = batch_data
    batch = tokenizer(
        docs, 
        add_special_tokens=False, 
        return_attention_mask=False, 
        return_token_type_ids=False,
        return_tensors=None 
    )
    
    results = []
    for i, ids in enumerate(batch['input_ids']):
        results.append((doc_ids[i], group_ids[i], ids + [tokenizer.eos_token_id]))
    return results


def tokenize_directly_to_jsonl(dataset, tokenizer_name, field_name, output_path, seq_len=2048, id_field='id', group_id_field='group_id', seed=42, sequences_per_file=10000, num_proc=8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    batch_size = 500
    indices = range(0, len(dataset), batch_size)
    
    batch_generator = (
        (dataset[i : i + batch_size][id_field], dataset[i : i + batch_size][group_id_field], dataset[i : i + batch_size][field_name]) 
        for i in indices
    )

    # Dictionary of buffers: key = group_id, value = list of tokens
    group_buffers = defaultdict(list)
    
    file_idx = 0
    seq_count_in_file = 0
    current_file = open(os.path.join(output_path, f"data_{file_idx:03d}.jsonl"), "w")

    def write_sequence(sequence, group_id):
        nonlocal seq_count_in_file, file_idx, current_file
        
        record = {
            "input_ids": sequence,
            "group_id": group_id
        }
        current_file.write(json.dumps(record) + "\n")
        seq_count_in_file += 1

        if seq_count_in_file >= sequences_per_file:
            current_file.close()
            file_idx += 1
            seq_count_in_file = 0
            current_file = open(os.path.join(output_path, f"data_{file_idx:03d}.jsonl"), "w")

    # Pass the lookup map to the main loop (no need to pass to workers)
    with mp.Pool(num_proc, initializer=_worker_init, initargs=(tokenizer_name,)) as pool:
        for processed_batch in tqdm(pool.imap(_tokenize_batch, batch_generator), total=len(indices)):
            for doc_id, group_val, doc_tokens in processed_batch:
                

                # Add to the specific buffer for this group
                group_buffers[group_val].extend(doc_tokens)

                # Check if THIS specific group buffer is full
                while len(group_buffers[group_val]) >= seq_len:
                    sequence = group_buffers[group_val][:seq_len]
                    group_buffers[group_val] = group_buffers[group_val][seq_len:]
                    
                    write_sequence(sequence, group_val)

    # Handle remainders for ALL groups
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    for group_val, buffer in group_buffers.items():
        if buffer:
            padded_seq = buffer + [pad_id] * (seq_len - len(buffer))
            write_sequence(padded_seq, group_val)

    current_file.close()
    print(f"Tokenization complete. Created {file_idx + 1} files.")