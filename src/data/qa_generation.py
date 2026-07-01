
from transformers import pipeline
import torch
from typing import List, Dict
import json
from datasets import load_dataset
import numpy as np
from data.utils import keep_first
from data.utils import get_example_multi_sentence_prompt

def batch_generation(pipe, prompts: List[str], max_new_tokens: int = 1024) -> List[Dict]:
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    results = pipe(messages_list, max_new_tokens=max_new_tokens,do_sample=False,)
    return results


def create_qa_dataset(config):
    
    df = keep_first(load_dataset(config.data.personal_info)['train'], field="Unique ID").to_pandas()

    def process_row(complete_info_str):
        """Process a single row and return the filtered data as JSON string."""
        try:
            val = json.loads(complete_info_str)['synthetic_pii_input']
            filtered_data = {k: v for k, v in val.items() if k != "Unique ID"}
            return json.dumps(filtered_data)
        except Exception as e:
            return None
        
    df['filtered_data'] = df['complete_info'].apply(process_row)
    df = df[df['filtered_data'].notna()].reset_index(drop=True)

    prompts = [
        [
            {
                'prompt': get_example_multi_sentence_prompt(
                    config.data, row, field
                ),
                'answer': str(row[field])
            }
            for field in config.data.pii_to_match
            for _ in range(config.data.qa_generation_per_field)
            if row[field] != 'N/A' and row[field] != '{}'
        ]
        for _, row in df.iterrows()
    ]
    
    
    return prompts
    
    
    

