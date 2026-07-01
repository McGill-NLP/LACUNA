#!/usr/bin/env python
"""Post-relearning extraction leakage for LACUNA.

Adapts the paper's leakage methodology to the eval release. For a PII field, each
FORGOTTEN person is probed with `attempts` varied QA questions (greedy generation);
the person is counted as "leaked" if the target PII appears in ANY generation.

We report, among people that unlearning actually forgot (i.e. NOT leaking on the
unlearned model — people never forgotten are excluded), how many RESURFACE after
relearning (leak on the relearned model).

`extraction.attempts` is a setting: use a small value (e.g. 5) for a quick demo,
200 for the paper-faithful run.

Run:
    python scripts/extraction_leakage.py experiments=eval/extraction_Email_Address_1B
    python scripts/extraction_leakage.py experiments=eval/extraction_Email_Address_1B extraction.attempts=5
"""
import json
import os
import random
import sys

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lacuna_data.utils import get_example_multi_sentence_prompt  # noqa: E402

FIELD_DISPLAY = {
    "Birth_City": "Birth City",
    "Email_Address": "Email Address",
    "Phone_Number": "Phone Number",
    "Drivers_License": "Driver's License",
}


class NS(dict):
    """dict with attribute access, to mimic the DictConfig get_example_* expects."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def build_prompts(forget_people, field_display, gen_cfg, attempts, tpl):
    u_s, u_e, a_s = tpl["user_start_tag"], tpl["user_end_tag"], tpl["asst_start_tag"]
    rows = []
    for _, person in forget_people.iterrows():
        pid = person["Unique ID"]
        target_value = str(person[field_display])
        if not target_value or target_value in ("N/A", "None", "{}"):
            continue
        seed = hash((pid, field_display)) % (2 ** 32)   # reproducible per person
        random.seed(seed)
        np.random.seed(seed)
        for _ in range(attempts):
            try:
                raw_q = get_example_multi_sentence_prompt(gen_cfg, person, field_display)
            except Exception:
                continue
            rows.append({
                "person_id": pid,
                "prompt_text": f"{u_s}{raw_q}{u_e}{a_s}",
                "target_value": target_value,
            })
    return pd.DataFrame(rows)


@torch.no_grad()
def people_leaked(model_path, tokenizer, prompts_df, batch_size, max_new_tokens):
    """Return the set of person_ids whose PII surfaces in any greedy generation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda").eval()
    leaked = set()
    for i in range(0, len(prompts_df), batch_size):
        b = prompts_df.iloc[i:i + batch_size]
        inp = tokenizer(b["prompt_text"].tolist(), return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to("cuda")
        out = model.generate(input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        for (_, row), g in zip(b.iterrows(), gen):
            if row["target_value"].lower() in g.lower():
                leaked.add(row["person_id"])
    del model
    torch.cuda.empty_cache()
    return leaked


@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):
    r = OmegaConf.to_container(cfg, resolve=True)
    field_file = cfg.unlearning_data.forget.Forget.name       # e.g. Email_Address
    field_display = FIELD_DISPLAY[field_file]
    attempts = int(cfg.extraction.attempts)
    ex = r["extraction"]
    gen_cfg = NS(
        prompt_continuations=ex["prompt_continuations"],
        pii_to_match=ex["pii_to_match"],
        max_info_per_prompt=ex["max_info_per_prompt"],
        variate_question=ex.get("variate_question", False),
        question_variation_field=ex.get("question_variation_field", []),
    )

    # forget people (from our memorized forget split) x PANORAMA-Plus rows
    forget = load_dataset(cfg.paths.memorized_datasets_dir, name=field_file, split="forget")
    forget_ids = set(forget["ID"])
    pinfo = (load_dataset(cfg.extraction.personal_info)["train"].to_pandas()
             .drop_duplicates(subset="Unique ID", keep="first"))
    forget_people = pinfo[pinfo["Unique ID"].isin(forget_ids)].copy()

    prompts = build_prompts(forget_people, field_display, gen_cfg, attempts, cfg.model.template_args)
    print(f"[extraction] {len(forget_people)} forget people x {attempts} attempts "
          f"-> {len(prompts)} probe prompts")

    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.output_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    unlearned_leak = people_leaked(cfg.paths.output_dir, tokenizer, prompts,
                                   cfg.extraction.batch_size, cfg.extraction.max_new_tokens)
    relearned_dir = os.path.join(cfg.paths.output_dir, "relearned")
    relearned_leak = people_leaked(relearned_dir, tokenizer, prompts,
                                   cfg.extraction.batch_size, cfg.extraction.max_new_tokens)

    all_people = set(forget_people["Unique ID"])
    forgotten = all_people - unlearned_leak               # forgotten by unlearning
    resurfaced = forgotten & relearned_leak               # resurfaced after relearn
    summary = {
        "field": field_file,
        "attempts": attempts,
        "total_forget_people": len(all_people),
        "still_leaking_after_unlearn": len(unlearned_leak),
        "forgotten_by_unlearn": len(forgotten),
        "resurfaced_after_relearn": len(resurfaced),
        "pct_resurfaced_of_forgotten": round(100 * len(resurfaced) / max(1, len(forgotten)), 2),
    }
    out_path = os.path.join(cfg.paths.output_dir, "extraction_leakage.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[extraction] saved {out_path}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
