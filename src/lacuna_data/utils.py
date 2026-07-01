from collections import Counter
import random
import numpy as np
import torch
import json

def keep_first(dataset, field="Unique ID"):
    unique_ids = dataset[field]
    id_counts = Counter(unique_ids)
    duplicates = [uid for uid, count in id_counts.items() if count > 1]
    print(len(duplicates),"duplicates")
    seen = set()
    def keep_first(example):
        uid = example[field]
        if uid in seen:
            return False
        seen.add(uid)
        return True
    dataset = dataset.filter(keep_first)
    return dataset

        
        
def get_example_multi_sentence_prompt(config, row, target_field):
    
    subsentences = config.prompt_continuations

    assert config.get('pii_to_match', None) is not None, " please provide 'pii_to_match' in the config."
    limit = min(len(config.pii_to_match), config.max_info_per_prompt)
    pii_to_match = [f for f in config.pii_to_match if f != target_field and row[f] != 'N/A' and row[f] != "{}" and row[f] != 'None']
    
    fields = random.sample(pii_to_match, random.randint(1, limit))
        
    prompt = row['First Name'] + " " + row['Last Name'] + ", " + str.join(", ", [np.random.choice(subsentences[field]).format(**row) for field in fields])
    prompt += ". "
    
    pronouns = {
        'possessive': {'male': 'his', 'female': 'her', 'neutral': 'their'},
        'subject': {'male': 'he', 'female': 'she', 'neutral': 'they'},
        'object': {'male': 'him', 'female': 'her', 'neutral': 'them'}
    }
    gender = row['Gender'].lower()
    if gender not in ['male', 'female']:
        gender = 'neutral'
        
    if config.get('variate_question', False) and (target_field in config.get('question_variation_field',[]) or 'all' in config.get('question_variation_field',[])): 
        question = np.random.choice(config.question_variations[target_field]).format(pronoun=pronouns['possessive'][gender], subject=row['First Name'] + " " + row['Last Name'])
    else:
        question = f"What is {pronouns['possessive'][gender]} {target_field}?"
    prompt += question
    return prompt


def format_dataset(tpl_args, dataset, include_answer_token=False):

    
    u_start = tpl_args.user_start_tag   
    u_end   = tpl_args.user_end_tag     
    a_start = tpl_args.asst_start_tag   
    a_end   = tpl_args.asst_end_tag   

    def apply_template(example):

        user_part = f"{u_start}{example['question']}{u_end}{a_start if include_answer_token else ''}"
        
        asst_part = f"{a_start if not include_answer_token else ''}{example['answer']}{a_end}"
        
        return {
            "raw_question": example['question'],
            "raw_answer": example['answer'],
            "question": user_part,
            "answer": asst_part
        }
    return dataset.map(apply_template)