# %%
from datasets import Dataset,load_dataset
import pandas as pd
from data.qa_generation import create_qa_dataset
from data.utils import keep_first
from utils import temp_seed

def info_match(config):

    panorama = load_dataset(config.data.pii_sentences)['train']
    panorama_plus = load_dataset(config.data.personal_info)['train']
    panorama_plus = keep_first(panorama_plus, field="Unique ID")
    PII_to_match = config.data.pii_to_match
    info_plus = panorama_plus.select_columns(PII_to_match + ['Unique ID'])
    id_to_info = {row["Unique ID"]: row for row in info_plus}
    id_to_pos = {row["Unique ID"]: pos for pos,row in enumerate(info_plus)}

    present = {field: [] for field in PII_to_match }
    
    corresponding_rows = {f"{field}_rows": [[] for _ in range(len(info_plus))] for field in PII_to_match} 
    
    profile_rows = [[] for _ in range(len(info_plus))]
    for i in range(len(panorama)):
        text = panorama[i]['text'].lower()
        id = panorama[i]["id"]
        pos = id_to_pos[id]
        person =  id_to_info[id]
        profile_rows[pos].append(i)
        for field in PII_to_match:
            present[field].append(person[field].lower() in text)
            if person[field].lower() in text:
                corresponding_rows[f'{field}_rows'][pos].append(i)
        
    info_plus = info_plus.add_column("Rows", profile_rows)    
    for field in PII_to_match:
        info_plus = info_plus.add_column(f"{field}_rows", corresponding_rows[f"{field}_rows"])
        panorama = panorama.add_column(field, present[field])

    df = info_plus.to_pandas()
    fields = config.data.interesting_pii
 
    new_cols = []    
    new_df_parts = [] 
    for field in fields:
        field_col = f"{field}_rows"
        prompt_cols = config.data.pii_to_match
        for col in prompt_cols:
            if col == field:
                continue
            new_col = f"cooccurrence_{field}_{col}"

            # Compute the column into a Series
            new_series = df.apply(
                lambda row: len(set(row[col+"_rows"]).intersection(set(row[field_col]))),
                axis=1
            )
            new_cols.append(new_col)
            new_df_parts.append(new_series.rename(new_col))
    
    if config.data.get('qa_generation', False):
        print("Generating QA prompts...", flush=True)
        with temp_seed(42):
            prompts = create_qa_dataset(config)
        new_series = pd.Series(prompts, name='QA_generated_prompt')
        new_cols.append('QA_generated_prompt')
        new_df_parts.append(new_series)
        
        
    df = pd.concat([df] + new_df_parts, axis=1) 

    info_plus = Dataset.from_pandas(df)
    
    panorama.save_to_disk(config.data.panorama_added_info_path)
    info_plus.save_to_disk(config.data.panorama_plus_rows_path)
    


