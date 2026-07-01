import numpy as np 
from transformers import AutoTokenizer
import random


def tokenize_and_split(dataset, tokenizer_name, field_name, output_path, chunk_size=100_000_000):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    chunk_tokens = []
    chunk_idx = 0
    total_tokens = 0
    
    indices = list(range(len(dataset)))  # just indices, not the full data
    random.seed(42)
    random.shuffle(indices)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        doc = sample[field_name]
        token_ids = tokenizer.encode(doc, add_special_tokens=True) + [tokenizer.eos_token_id]
        chunk_tokens.extend(token_ids)

        # Write chunk if it exceeds chunk_size
        while len(chunk_tokens) >= chunk_size:
            to_write = chunk_tokens[:chunk_size]
            data_mmap = np.memmap(f"{output_path}/tokenized_data{chunk_idx:03d}.npy", 
                                  mode="w+", dtype=np.uint32, shape=(len(to_write),))
            data_mmap[:] = to_write
            data_mmap.flush()
            chunk_idx += 1
            chunk_tokens = chunk_tokens[chunk_size:]  # remaining tokens
            total_tokens += len(to_write)
        
        if (i+1) % 100000 == 0:
            print(f"Processed {i+1} documents, total tokens so far: {total_tokens + len(chunk_tokens)}")
    
    # Write remaining tokens
    if chunk_tokens:
        data_mmap = np.memmap(f"{output_path}/tokenized_data{chunk_idx:03d}.npy", 
                              mode="w+", dtype=np.uint32, shape=(len(chunk_tokens),))
        data_mmap[:] = chunk_tokens
        data_mmap.flush()
        total_tokens += len(chunk_tokens)
    
    print(f"Tokenization complete. Total tokens: {total_tokens}")

         
    
    
    
    
import numpy as np
import multiprocessing as mp
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def _worker_init(tokenizer_name):
    """Initialize tokenizer in each worker process to avoid pickling overhead."""
    global tokenizer
    # specific 'fast' tokenizer is crucial for speed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
def _tokenize_batch(docs):
    """Worker function to tokenize a batch of documents."""
    # Use __call__ (the tokenizer object itself) instead of batch_encode_plus
    batch = tokenizer(
        docs, 
        add_special_tokens=True, 
        return_attention_mask=False, 
        return_token_type_ids=False,
        # Ensure we don't get tensors here, just lists
        return_tensors=None 
    )
    
    # Access 'input_ids' the same way
    flat_tokens = [t for seq in batch['input_ids'] for t in seq + [tokenizer.eos_token_id]]
    return flat_tokens


def tokenize_and_split_optimized(dataset, tokenizer_name, field_name, output_path, chunk_size=100_000_000, batch_size=1000, num_proc=8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare data for pool (batched text)
    # Using a generator here saves RAM
    total_docs = len(dataset)
    indices = range(0, total_docs, batch_size)
    
    # Create batches of text
    # Note: If dataset is a HuggingFace dataset, dataset[i:j][field] is faster than looping
    batch_generator = (dataset[i : i + batch_size][field_name] for i in indices)

    chunk_tokens = []
    chunk_idx = 0
    total_tokens = 0

    # Start Multiprocessing Pool
    with mp.Pool(num_proc, initializer=_worker_init, initargs=(tokenizer_name,)) as pool:
        # imap_unordered is faster if order doesn't strictly matter (or use imap for order)
        for token_batch in tqdm(pool.imap(_tokenize_batch, batch_generator), total=len(indices)):
            chunk_tokens.extend(token_batch)

            # Write to disk when buffer is full
            while len(chunk_tokens) >= chunk_size:
                to_write = chunk_tokens[:chunk_size]
                
                # Save
                fname = os.path.join(output_path, f"tokenized_data{chunk_idx:03d}.npy")
                np.save(fname, np.array(to_write, dtype=np.uint32))
                
                chunk_idx += 1
                chunk_tokens = chunk_tokens[chunk_size:]
                total_tokens += len(to_write)

    # Write remaining
    if chunk_tokens:
        fname = os.path.join(output_path, f"tokenized_data{chunk_idx:03d}.npy")
        np.save(fname, np.array(chunk_tokens, dtype=np.uint32))
        total_tokens += len(chunk_tokens)

    print(f"Total tokens: {total_tokens}")