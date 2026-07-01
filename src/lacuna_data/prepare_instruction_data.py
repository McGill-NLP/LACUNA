
from utils import *
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import random

import os
import json
from lacuna_data.utils import get_example_multi_sentence_prompt, format_dataset

def tokenize_instruction_data(ds, tokenizer):
    """
    Tokenizes question-answer pairs and creates labels for causal LLM training.
    The question is masked with -100 so the model only learns to predict the answer.
    """
    
    def process_batch(examples):
        all_input_ids = []
        all_labels = []
        
        # Iterate through the batch (handles 'question' and 'answer' columns)
        for question, answer in zip(examples["question"], examples["answer"]):
            # 1. Tokenize prompt (no special tokens here to avoid double-bos)
            prompt_ids = tokenizer(
                question,
                add_special_tokens=False,
                truncation=True,
                max_length=2048 # Adjust based on your model's context window
            )["input_ids"]
            
            # 2. Tokenize answer (adding a leading space is often best practice for LLM tokenizers)
            label_ids = tokenizer(
                f" {answer}",
                add_special_tokens=False,
                truncation=True,
                max_length=2048
            )["input_ids"]
            
            # 3. Concatenate (Input = Prompt + Label)
            input_ids = prompt_ids + label_ids
            
            # 4. Create labels (Mask prompt with -100, keep answer tokens as is)
            labels = [-100] * len(prompt_ids) + label_ids
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            
        return {
            "input_ids": all_input_ids,
            "labels": all_labels
        }

    return ds.map(
        process_batch, 
        batched=True, 
        remove_columns=ds.column_names
    )
    

def prepare_panorama(config, tokenizer):
    """Prepare instruction tuning data by sampling a portion of people from PANORAMA"""
    
    # Load datasets
    panorama = load_dataset(config.data.pii_sentences)['train']
    ds = load_from_disk(config.data.panorama_plus_rows_path)
    personal_info = load_dataset(config.data.personal_info)['train']
    experiment_meta = load_from_disk(config.data.experiment_metadata_path)

    # add possible missing information
    existing_columns = set(ds.column_names)
    new_columns = [col for col in personal_info.column_names if col not in existing_columns or col == 'Unique ID']
    personal_info_filtered = personal_info.select_columns(new_columns)
    ds_df = ds.to_pandas()
    personal_info_df = personal_info_filtered.to_pandas()
    merged_df = ds_df.merge(personal_info_df, on='Unique ID', how='left')

    # Join experiment metadata to get Subset column
    experiment_meta_df = experiment_meta.to_pandas()[['Unique ID', 'Subset']]
    merged_df = merged_df.merge(experiment_meta_df, on='Unique ID', how='left')
    merged_df = merged_df[merged_df['Subset'] == True]
    ds = HFDataset.from_pandas(merged_df)
    
    # Get PII fields and sampling portion
    interesting_pii = config.data.interesting_pii
    sample_portion = config.instruction_tuning.get('sample_portion', 0.25)  
    eval_portion = config.instruction_tuning.get('eval_portion', 0.30)  
    prompt_style = config.instruction_tuning.get('prompt_style', None)
    
    df = ds.to_pandas()
    if prompt_style == 'prompt':
        raise NotImplementedError("Prompt style-based instruction data preparation not implemented yet")
        random_fact = config['data']['random_fact']
        df = df.rename(columns={random_fact: "Random Fact"})
    
    # Sample people: split into train and eval
    total_people = len(df)
    num_sampled = max(1, int(total_people * sample_portion))
    num_eval = max(1, int(num_sampled * eval_portion))
    num_train = num_sampled - num_eval
    
    sampled_person_indices = random.sample(range(total_people), num_sampled)
    train_person_indices = set(sampled_person_indices[:num_train])
    eval_person_indices = set(sampled_person_indices[num_train:])
    
    excluded_ids = [df.iloc[i]["Unique ID"] for i in sampled_person_indices]
    
    print(f"Sampled {num_sampled} people out of {total_people}")
    print(f"Train: {num_train} people, Eval: {num_eval} people")
    all_examples = []
    eval_examples = []
    
    for field in interesting_pii:
        print(f"Preparing data for {field}")
        for person_idx, row in df.iterrows():
            # Skip if person not in sampled set
            if person_idx not in train_person_indices and person_idx not in eval_person_indices:
                continue
            
            is_eval = person_idx in eval_person_indices
            
            for i in range(config.instruction_tuning.num_samples_per_field):
            
                # Generate prompt based on style
                if 'prompt' in prompt_style:
                    # Use template-based prompt
                    
                    if prompt_style == 'multi-sentence-prompt':
                        prompt = get_example_multi_sentence_prompt(config=config['data'],row=row, target_field=field)
                    else:
                        raise NotImplementedError(f"Prompt style {prompt_style} not implemented yet")
                        prompt_template = config['data']['prompt_template']
                        prompt = prompt_template.format(
                            field=field.lower(),
                            first_name=row["First Name"],
                            last_name=row["Last Name"],
                            random_fact=row["Random Fact"].lower() if config['data'].get('random_fact_lower', True) else field,
                        )
                    label = " " + row[field]
                    
                    example = {"question": prompt, "answer": label}
                
                    if is_eval:
                        eval_examples.append(example)
                    else:
                        all_examples.append(example)

                else:
                    
                    raise NotImplementedError(f"Prompt style {prompt_style} not implemented yet")
                    # Use sentence-based prompts
                    if len(row[f"{field}_rows"]) > 0:
                        # Use all sentences for this person/field
                        for idx in range(len(row[f"{field}_rows"])):
                            sentence_idx = row[f"{field}_rows"][idx]
                            sentence = panorama[sentence_idx]['text']
                            
                            target = row[field]
                            lower_sentence = sentence.lower()
                            lower_target = target.lower()
                            target_idx = lower_sentence.find(lower_target)
                            
                            if target_idx != -1:
                                prompt = sentence[:target_idx].rstrip()
                                label = " " + target
                                
        
    
    # Save used indices
    instruction_tuning_output_path = config.instruction_tuning.data_path
    os.makedirs(instruction_tuning_output_path, exist_ok=True)
    excluded_ids_path = os.path.join(instruction_tuning_output_path, 'excluded_person_ids.json')
    with open(excluded_ids_path, 'w') as f:
        json.dump(excluded_ids, f, indent=2)
    print(f"Saved excluded person IDs to {excluded_ids_path}")
    
    return HFDataset.from_list(all_examples),HFDataset.from_list(eval_examples)


def prepare_instruction_data(config, tokenizer):
    
    instruction_data_type = config.instruction_tuning.instruction_data_type
    
    train_dataset, eval_dataset = None, None
    
    if instruction_data_type == 'panorama':
        
        train_dataset, eval_dataset = prepare_panorama(config, tokenizer)
        #raise NotImplementedError("PANORAMA-based instruction data preparation not implemented yet")
        
    elif instruction_data_type == 'panorama_unseen':
        raise NotImplementedError("PANORAMA unseen-based instruction data preparation not implemented yet")

    elif instruction_data_type == 'general_knowledge':
        raise NotImplementedError("General knowledge-based instruction data preparation not implemented yet")
    
    
    
    
    train_dataset, eval_dataset = format_dataset(config.model.template_args, train_dataset, include_answer_token=True), format_dataset(config.model.template_args, eval_dataset, include_answer_token=True)
    train_dataset, eval_dataset = tokenize_instruction_data(train_dataset, tokenizer), tokenize_instruction_data(eval_dataset, tokenizer)
    return train_dataset, eval_dataset
    
    
    
    
