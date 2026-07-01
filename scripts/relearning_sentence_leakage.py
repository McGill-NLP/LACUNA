#!/usr/bin/env python3
"""Compute sentence-level and QA-prompt leakage for unlearned & relearned models.

Given a PII field, evaluates leakage (greedy generation + teacher-forced logprob)
for the finetuned baseline, each unlearning method, and each relearned variant.

Saves results to:
  {training_output_dir}/cached_notebook_files/{field_file}/
    prompts_df.pkl              — PANORAMA sentence prompts
    qa_prompts_df.pkl           — QA-style prompts
    results_finetuned.pkl       — baseline results
    results_sentence_{method}.pkl — sentence leakage per method
    results_qa_{method}.pkl     — QA leakage per method
    results_relearn_{method}.pkl            — relearned (instruction-tuning) leakage
    results_relearn_memorized_{method}.pkl  — relearned (memorized data) leakage

The companion notebook (notebooks/sentence_leakage_analysis.ipynb) loads these
cached files for plotting.

Usage:
    python scripts/relearning_sentence_leakage.py experiments=OLMo_Mask_Train_FullSubset field="Driver's License"
    python scripts/relearning_sentence_leakage.py experiments=OLMo_Mask_Train_FullSubset field="Email Address" force=true
"""

import gc
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra

# Add src/ to path for data.utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data.utils import get_example_multi_sentence_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_FILE_MAP = {
    'Birth City': 'Birth_City',
    'Email Address': 'Email_Address',
    'Phone Number': 'Phone_Number',
    "Driver's License": 'Drivers_License',
}

# Reverse map: accept underscore names from CLI and resolve to display names
FIELD_FROM_FILE = {v: k for k, v in FIELD_FILE_MAP.items()}

METHODS = ['AlphaEdit', 'MemFlex', 'SimNPO', 'GradDiff_OracleGrad']

# Method name mapping for relearned_models_memorized (uses short names)
RELEARN_MEMORIZED_METHOD_MAP = {
    'AlphaEdit': 'AlphaEdit',
    'MemFlex': 'MemFlex',
    'SimNPO': 'SimNPO',
    'GradDiff_OracleGrad': 'OracleGrad',
}

BATCH_SIZE = 32
MAX_NEW_TOKENS = 20
NUM_ATTEMPTS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DictNamespace(dict):
    """Dict that also supports attribute access (mimics OmegaConf DictConfig)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def evaluate_leakage(model, tokenizer, prompts_df, batch_size=32, max_new_tokens=20):
    """Greedy generation + teacher-forced logprob for each prompt."""
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i in tqdm(range(0, len(prompts_df), batch_size), desc='Evaluating'):
        batch = prompts_df.iloc[i:i + batch_size]
        prompt_texts = batch['prompt_text'].tolist()
        target_values = batch['target_value'].tolist()

        # Greedy generation
        inputs = tokenizer(prompt_texts, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.batch_decode(
            gen_out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Teacher-forced logprob
        full_texts = [p + t for p, t in zip(prompt_texts, target_values)]
        inputs_prompt = tokenizer(prompt_texts, return_tensors='pt', padding=True,
                                  truncation=True, max_length=512).to(model.device)
        inputs_full = tokenizer(full_texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            logits = model(**inputs_full).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs_full.input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        prompt_lengths = (inputs_prompt.input_ids != tokenizer.pad_token_id).sum(dim=1)
        full_lengths = (inputs_full.input_ids != tokenizer.pad_token_id).sum(dim=1)

        for j, (_, row) in enumerate(batch.iterrows()):
            gen_text = generated[j]
            leaked = row['target_value'].lower() in gen_text.lower()
            p_len = prompt_lengths[j].item()
            f_len = full_lengths[j].item()
            target_lp = token_logprobs[j, p_len - 1: f_len - 1]
            mean_logprob = target_lp.mean().item() if len(target_lp) > 0 else float('nan')
            ppl = torch.exp(-target_lp.mean()).item() if len(target_lp) > 0 else float('nan')

            results.append({
                'person_id': row['person_id'],
                'sentence_idx': row['sentence_idx'],
                'target_value': row['target_value'],
                'generated_text': gen_text,
                'leaked': leaked,
                'content_type': row['content_type'],
                'logprob': mean_logprob,
                'ppl': ppl,
            })

    return pd.DataFrame(results)


def compute_leakage_metrics(results_df):
    per_person = results_df.groupby('person_id').agg(
        total_sentences=('leaked', 'count'),
        leaked_sentences=('leaked', 'sum'),
        mean_logprob=('logprob', 'mean'),
        mean_ppl=('ppl', 'mean'),
    ).reset_index()
    per_person['has_leak'] = per_person['leaked_sentences'] > 0

    pct_people_leaked = per_person['has_leak'].mean() * 100
    avg_leaked_per_person = per_person['leaked_sentences'].mean()
    avg_leaked_given_leak = (per_person.loc[per_person['has_leak'], 'leaked_sentences'].mean()
                             if per_person['has_leak'].any() else 0)
    return {
        '% people with leak': round(pct_people_leaked, 2),
        'avg leaked sentences/person': round(avg_leaked_per_person, 2),
        'avg leaked (given leak)': round(avg_leaked_given_leak, 2),
        'mean logprob (target)': round(results_df['logprob'].mean(), 4),
        'mean PPL (target)': round(results_df['ppl'].mean(), 2),
        'total people': len(per_person),
        'total sentences': len(results_df),
        'total leaked': int(results_df['leaked'].sum()),
    }


def eval_and_cache(model_path, tokenizer, prompts_df, cache_path, force, label,
                   batch_size=BATCH_SIZE):
    """Load model, evaluate, cache results. Skip if cache exists and not forced."""
    if not force and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f'  {label}: loaded from cache — {cached["metrics"]}')
        return cached['results'], cached['metrics']

    if not os.path.exists(model_path):
        print(f'  [SKIP] {label} — model not found at {model_path}')
        return None, None

    print(f'  Loading {label} from {model_path}')
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    results = evaluate_leakage(model, tokenizer, prompts_df,
                               batch_size=batch_size, max_new_tokens=MAX_NEW_TOKENS)
    metrics = compute_leakage_metrics(results)
    with open(cache_path, 'wb') as f:
        pickle.dump({'results': results, 'metrics': metrics}, f)
    print(f'  {label}: {metrics} (saved)')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results, metrics


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_sentence_prompts(info_forget, target_field, panorama, cache_path, force):
    if not force and os.path.exists(cache_path):
        prompts_df = pd.read_pickle(cache_path)
        print(f'Loaded sentence prompts from cache ({len(prompts_df)} prompts)')
        return prompts_df

    field_rows_col = f'{target_field}_rows'
    prompts = []
    for _, person in tqdm(info_forget.iterrows(), total=len(info_forget), desc='Building sentence prompts'):
        person_id = person['Unique ID']
        target_value = person[target_field]
        if not target_value or target_value in ('N/A', 'None', '{}'):
            continue
        row_indices = person[field_rows_col]
        if len(row_indices) == 0:
            continue
        for idx in row_indices:
            sentence = panorama[int(idx)]['text']
            pos = sentence.lower().find(target_value.lower())
            if pos == -1:
                continue
            prompt_text = sentence[:pos]
            if len(prompt_text.strip()) < 10:
                continue
            prompts.append({
                'person_id': person_id,
                'sentence_idx': idx,
                'prompt_text': prompt_text,
                'target_value': target_value,
                'full_sentence': sentence,
                'content_type': panorama[int(idx)]['content-type'],
            })

    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_pickle(cache_path)
    print(f'Built {len(prompts_df)} sentence prompts')
    return prompts_df


def build_qa_prompts(forget_people, target_field, data_config, tpl,
                     cache_path, force):
    if not force and os.path.exists(cache_path):
        qa_df = pd.read_pickle(cache_path)
        print(f'Loaded QA prompts from cache ({len(qa_df)} prompts)')
        return qa_df

    u_start = tpl['user_start_tag']
    u_end = tpl['user_end_tag']
    a_start = tpl['asst_start_tag']

    qa_prompts = []
    for _, row in tqdm(forget_people.iterrows(), total=len(forget_people),
                       desc='Generating QA prompts'):
        person_id = row['Unique ID']
        target_value = str(row[target_field])
        if not target_value or target_value in ('N/A', 'None', '{}'):
            continue
        # Pin per-person seed for reproducibility
        person_seed = hash((person_id, target_field)) % (2**32)
        random.seed(person_seed)
        np.random.seed(person_seed)
        for _ in range(NUM_ATTEMPTS):
            try:
                raw_q = get_example_multi_sentence_prompt(data_config, row, target_field)
            except Exception:
                continue
            formatted_q = f"{u_start}{raw_q}{u_end}{a_start}"
            qa_prompts.append({
                'person_id': person_id,
                'sentence_idx': -1,
                'prompt_text': formatted_q,
                'target_value': target_value,
                'full_sentence': raw_q,
                'content_type': 'QA prompt',
            })

    qa_df = pd.DataFrame(qa_prompts)
    qa_df.to_pickle(cache_path)
    print(f'Built {len(qa_df)} QA prompts')
    return qa_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg: DictConfig):
    resolved = OmegaConf.to_container(cfg, resolve=True)

    raw_field = cfg.field
    # Accept either display name ("Driver's License") or file name ("Drivers_License")
    if raw_field in FIELD_FILE_MAP:
        target_field = raw_field
        field_file = FIELD_FILE_MAP[raw_field]
    elif raw_field in FIELD_FROM_FILE:
        target_field = FIELD_FROM_FILE[raw_field]
        field_file = raw_field
    else:
        raise ValueError(f"Unknown field: {raw_field}. Use one of {list(FIELD_FILE_MAP.keys())} or {list(FIELD_FROM_FILE.keys())}")
    force = cfg.get('force', False)

    training_output_dir = cfg.paths.training_output_dir
    cache_dir = os.path.join(training_output_dir, 'cached_notebook_files', field_file)
    os.makedirs(cache_dir, exist_ok=True)
    print(f'Field: {target_field} ({field_file})')
    print(f'Cache dir: {cache_dir}')
    print(f'Force recompute: {force}')

    # ---- Load data ----
    memorized_dir = cfg.paths.memorized_datasets_dir
    forget_ds = load_dataset(memorized_dir, field_file, split='forget')
    forget_df = forget_ds.to_pandas()
    forget_ids = set(forget_df['ID'].unique())
    print(f'Forget set: {len(forget_df)} rows, {len(forget_ids)} unique people')

    info_ds = load_from_disk(resolved['data']['panorama_plus_rows_path'])
    info_df = info_ds.to_pandas()
    info_forget = info_df[info_df['Unique ID'].isin(forget_ids)].copy()

    panorama = load_dataset('srirxml/PANORAMA')['train']

    personal_info = load_dataset(resolved['data']['personal_info'])['train'].to_pandas()
    personal_info = personal_info.drop_duplicates(subset='Unique ID', keep='first')
    forget_people = personal_info[personal_info['Unique ID'].isin(forget_ids)].copy()

    data_config = DictNamespace(
        prompt_continuations=resolved['data']['prompt_continuations'],
        pii_to_match=resolved['data']['pii_to_match'],
        max_info_per_prompt=resolved['data']['max_info_per_prompt'],
        variate_question=resolved['data'].get('variate_question', False),
        question_variation_field=resolved['data'].get('question_variation_field', []),
        question_variations=resolved['data'].get('question_variations', {}),
    )
    tpl = resolved['model']['template_args']

    # ---- Build prompts ----
    prompts_df = build_sentence_prompts(
        info_forget, target_field, panorama,
        os.path.join(cache_dir, 'prompts_df.pkl'), force)

    qa_prompts_df = build_qa_prompts(
        forget_people, target_field, data_config, tpl,
        os.path.join(cache_dir, 'qa_prompts_df.pkl'), force)

    # ---- Load tokenizer ----
    finetuned_path = os.path.join(training_output_dir, 'intruction_tuned')
    tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Pilot mode: evaluate ONE arbitrary model on forget-set QA, then exit ----
    # Used to measure how many forget-set people a strengthened relearning run
    # resurfaces, without re-running the full method sweep. Activate with
    #   +pilot_model_path=/path/to/merged_model  [+pilot_label=name]
    pilot_model_path = cfg.get('pilot_model_path', None)
    if pilot_model_path:
        pilot_label = cfg.get('pilot_label', 'pilot')
        pilot_cache = os.path.join(cache_dir, f'results_{pilot_label}.pkl')
        print(f'\n=== PILOT: {pilot_label} ({pilot_model_path}) ===')
        results, metrics = eval_and_cache(
            pilot_model_path, tokenizer, qa_prompts_df, pilot_cache,
            force, f'Pilot {pilot_label}')
        if results is not None:
            n_people = results['person_id'].nunique()
            n_resurfaced = results[results['leaked']]['person_id'].nunique()
            print(f'\n[PILOT RESULT] {pilot_label}: '
                  f'{n_resurfaced}/{n_people} forget-set people resurfaced '
                  f'| metrics={metrics}')
        return

    # ---- Trajectory mode: eval forget-set resurfacing across LoRA epoch checkpoints ----
    # Loads a base (unlearned) model + each adapter checkpoint and measures how many
    # forget-set people resurface, to locate the peak epoch. Activate with
    #   +pilot_traj_base=/path/to/unlearned_model
    #   +pilot_traj_dir=/path/to/relearn_run_with_checkpoints
    #   [+pilot_traj_steps=118,236,...]  (default: all checkpoints)
    #   [+pilot_traj_attempts=50]        (QA attempts/person; subsamples for speed)
    pilot_traj_base = cfg.get('pilot_traj_base', None)
    pilot_traj_dir = cfg.get('pilot_traj_dir', None)
    if pilot_traj_base and pilot_traj_dir:
        from glob import glob
        from peft import PeftModel

        attempts = int(cfg.get('pilot_traj_attempts', 50))
        qa_sub = qa_prompts_df.groupby('person_id', group_keys=False).head(attempts)
        print(f'Trajectory eval: {attempts} attempts/person -> {len(qa_sub)} prompts')

        ckpts = sorted(glob(os.path.join(pilot_traj_dir, 'checkpoint-*')),
                       key=lambda p: int(p.rsplit('-', 1)[1]))
        want = cfg.get('pilot_traj_steps', None)
        if want is not None:
            if isinstance(want, str):
                want_set = {int(s) for s in want.split(',') if s.strip()}
            else:  # OmegaConf ListConfig / list
                want_set = {int(s) for s in want}
            ckpts = [c for c in ckpts if int(c.rsplit('-', 1)[1]) in want_set]
        print(f'Evaluating {len(ckpts)} checkpoints from {pilot_traj_dir}')

        traj = []
        for ck in ckpts:
            step = int(ck.rsplit('-', 1)[1])
            # Reload base fresh each time so adapters never stack (obviously-correct).
            base = AutoModelForCausalLM.from_pretrained(
                pilot_traj_base, torch_dtype=torch.bfloat16).cuda()
            model = PeftModel.from_pretrained(base, ck)
            results = evaluate_leakage(model, tokenizer, qa_sub,
                                       batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS)
            n_res = results[results['leaked']]['person_id'].nunique()
            traj.append({'step': step, 'resurfaced': int(n_res)})
            print(f'  [TRAJ] step={step:5d}  resurfaced={n_res}/100')
            del model, base
            gc.collect()
            torch.cuda.empty_cache()

        traj_df = pd.DataFrame(traj)
        traj_path = os.path.join(cache_dir, 'pilot_trajectory.pkl')
        traj_df.to_pickle(traj_path)
        print('\n[TRAJECTORY] forget-set resurfaced vs step '
              f'({attempts} attempts/person):')
        print(traj_df.to_string(index=False))
        print(f'Saved -> {traj_path}')
        return

    # ---- Evaluate: finetuned baseline ----
    print('\n=== Finetuned baseline ===')
    eval_and_cache(finetuned_path, tokenizer, prompts_df,
                   os.path.join(cache_dir, 'results_finetuned.pkl'),
                   force, 'Finetuned (sentence)')
    eval_and_cache(finetuned_path, tokenizer, qa_prompts_df,
                   os.path.join(cache_dir, 'results_qa_Finetuned_baseline.pkl'),
                   force, 'Finetuned (qa)')

    # ---- Evaluate: unlearned methods ----
    unlearned_base = os.path.join(training_output_dir, 'unlearned_models', field_file)
    for method in METHODS:
        print(f'\n=== Unlearned: {method} ===')
        method_path = os.path.join(unlearned_base, method)
        eval_and_cache(method_path, tokenizer, prompts_df,
                       os.path.join(cache_dir, f'results_sentence_{method}.pkl'),
                       force, f'{method} (sentence)')
        eval_and_cache(method_path, tokenizer, qa_prompts_df,
                       os.path.join(cache_dir, f'results_qa_{method}.pkl'),
                       force, f'{method} (qa)')

    # ---- Evaluate: relearned models (instruction-tuning data) ----
    relearned_base = os.path.join(training_output_dir, 'relearned_models', field_file)
    for method in METHODS:
        method_path = os.path.join(relearned_base, method)
        print(f'\n=== Relearned (instruct): {method} ===')
        eval_and_cache(method_path, tokenizer, qa_prompts_df,
                       os.path.join(cache_dir, f'results_relearn_{method}.pkl'),
                       force, f'Relearned-instruct {method}')

    # ---- Evaluate: relearned models (memorized data) ----
    relearned_mem_base = os.path.join(training_output_dir, 'relearned_models_memorized', field_file)
    for method in METHODS:
        mem_method_name = RELEARN_MEMORIZED_METHOD_MAP[method]
        method_path = os.path.join(relearned_mem_base, mem_method_name)
        print(f'\n=== Relearned (memorized): {method} ===')
        eval_and_cache(method_path, tokenizer, qa_prompts_df,
                       os.path.join(cache_dir, f'results_relearn_memorized_{method}.pkl'),
                       force, f'Relearned-memorized {method}')

    print('\n=== All done ===')


if __name__ == '__main__':
    main()
