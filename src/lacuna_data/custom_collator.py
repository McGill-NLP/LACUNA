from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
import torch

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        dataset_names = [f.pop('dataset_name') for f in features]
        batch = super().__call__(features)
        batch['dataset_name'] = dataset_names
        return batch

class InstructionDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=4096, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.max_length = max_length
    
    def __call__(self, features):
        # Extract dataset names before processing
        dataset_names = [f.pop('dataset_name', 'unknown') for f in features]
        
        # Manually pad input_ids and labels
        batch = {}
        
        # Find max length in batch
        max_len = max(len(f['input_ids']) for f in features)
        
        input_ids = []
        labels = []
        attention_mask = []
        
        for f in features:
            seq_len = len(f['input_ids'])
            pad_len = max_len - seq_len
            
            # Pad input_ids with pad_token_id
            padded_input = [self.tokenizer.pad_token_id] * pad_len + f['input_ids']
            input_ids.append(padded_input)
            
            # Pad labels with -100
            padded_labels = [-100] * pad_len + f['labels']
            labels.append(padded_labels)
            
            # Create attention mask (0 for padding, 1 for real tokens)
            attn = [0] * pad_len + [1] * seq_len
            attention_mask.append(attn)
        
        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        batch['dataset_name'] = dataset_names
        
        return batch