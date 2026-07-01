import hydra
import json
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
import os
from data.utils import get_example_multi_sentence_prompt, format_dataset
from utils import set_seed
import pandas as pd
from model import get_model
import torch
import torch.multiprocessing as mp
import shutil
from tqdm import tqdm

def evaluate(model, tokenizer, examples, batch_size=32, max_new_tokens=20):
    model.eval()
    results = []
    tokenizer.padding_side = "left" # Crucial for batched generation

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i : i + batch_size]
        questions = batch['question']
        answers = [a.strip() for a in batch['answer']]
        raw_questions = batch['raw_question']
        raw_answers = batch['raw_answer']
        fields = batch['field'] # Now tracking the PII category
        uids = batch['Unique ID']
        
        # 1. PERPLEXITY PASS (Batch)
        full_texts = [q + a for q, a in zip(questions, answers)]
        inputs_full = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        inputs_q = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs_full, labels=inputs_full.input_ids)
            logits = outputs.logits[:, :-1, :] 
            labels = inputs_full.input_ids[:, 1:]
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            nll_loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).reshape(labels.size())

            # 2. BATCHED GENERATION
            gen_out = model.generate(
                input_ids=inputs_q.input_ids,
                attention_mask=inputs_q.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            gen_texts = tokenizer.batch_decode(gen_out[:, inputs_q.input_ids.shape[1]:], skip_special_tokens=True)

            # 3. COLLECT RESULTS
            for b_idx in range(len(questions)):
                q_len = (inputs_q.input_ids[b_idx] != tokenizer.pad_token_id).sum().item()
                total_len = (inputs_full.input_ids[b_idx] != tokenizer.pad_token_id).sum().item()
                
                # Slice loss for only the Answer tokens
                ans_loss = nll_loss[b_idx, q_len-1 : total_len-1]
                
                results.append({
                    'UniqueID': uids[b_idx],
                    'Field': fields[b_idx],
                    'Question': raw_questions[b_idx],
                    'Ground_Truth': raw_answers[b_idx],
                    'Generated_Text': gen_texts[b_idx].strip(),
                    'Is_Correct': raw_answers[b_idx].strip().lower() in gen_texts[b_idx].lower(),
                    'PPL': torch.exp(ans_loss.mean()).item(),
                    'LogProb': -ans_loss.mean().item()
                })

    return pd.DataFrame(results)


def _gpu_worker(gpu_id, dataset_path, shard_start, shard_end, memorization_cfg, output_path):
    """Load model on one GPU, evaluate a shard, save results."""
    memorization_cfg = dict(memorization_cfg)
    ds = load_from_disk(dataset_path).select(range(shard_start, shard_end))
    model, tokenizer = get_model(memorization_cfg)
    model.to(f"cuda:{gpu_id}")
    df = evaluate(model, tokenizer, ds,
                  batch_size=memorization_cfg['batch_size'],
                  max_new_tokens=memorization_cfg['max_new_tokens'])
    df.to_csv(output_path, index=False)


def run_evaluation(config, formatted_ds):
    num_gpus = torch.cuda.device_count()
    batch_size = config.memorization.batch_size
    max_new_tokens = config.memorization.max_new_tokens

    if num_gpus <= 1:
        model, tokenizer = get_model(config.memorization)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return evaluate(model, tokenizer, formatted_ds, batch_size=batch_size, max_new_tokens=max_new_tokens)

    print(f"Running memorization across {num_gpus} GPUs")
    tmp_dir = os.path.join(config.paths.output_dir, "_memo_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ds_path = os.path.join(tmp_dir, "formatted_ds")
    formatted_ds.save_to_disk(ds_path)

    n = len(formatted_ds)
    shard_size = n // num_gpus
    memorization_cfg = dict(config.memorization)

    processes = []
    csv_paths = []
    for i in range(num_gpus):
        start = i * shard_size
        end = start + shard_size if i < num_gpus - 1 else n
        csv_path = os.path.join(tmp_dir, f"results_{i}.csv")
        csv_paths.append(csv_path)
        p = mp.Process(target=_gpu_worker, args=(i, ds_path, start, end, memorization_cfg, csv_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    df_results = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    shutil.rmtree(tmp_dir)
    return df_results


@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(config):
    set_seed(config.seed)
    ds = load_from_disk(config.data.panorama_plus_rows_path)
    personal_info = load_dataset(config.data.personal_info)['train']
    experiment_meta = load_from_disk(config.data.experiment_metadata_path)

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
    
    
    excluded = set(json.load(open(os.path.join(config.instruction_tuning.data_path, 'excluded_person_ids.json'))))
    merged_df = merged_df[~merged_df['Unique ID'].isin(excluded)].reset_index(drop=True)
    print(len(merged_df), "people remaining after exclusion")
    
    all_questions = []
    for field in config.data.interesting_pii:
        print(f"Generating prompts for field: {field}")
        for _, row in merged_df.iterrows():
            for q in range(config.memorization.get('num_prompts', 10)):
                prompt = get_example_multi_sentence_prompt(config.data, row, field)
                all_questions.append({
                    "Unique ID": row['Unique ID'],
                    "field": field, 
                    "question": prompt,
                    "answer": row[field]
                })
    questions_ds = HFDataset.from_pandas(pd.DataFrame(all_questions))
    formatted_ds = format_dataset(config.model.template_args, questions_ds, include_answer_token=True)
    
    df_results = run_evaluation(config, formatted_ds)
    save_path = os.path.join(config.paths.output_dir, "memo_results.csv")
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

  
if __name__ == "__main__":
    main()