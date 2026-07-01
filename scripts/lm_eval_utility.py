#!/usr/bin/env python
"""General-capability (utility) evaluation via lm-eval — the paper's utility axis.

Runs ARC / HellaSwag / MMLU on a model and saves accuracies to a JSON. Run it on
the unlearned model (utility after unlearning) and on the instruction-tuned model
(pre-unlearning baseline = the paper's red dashed line).

Example:
    python scripts/lm_eval_utility.py --model $LACUNA_ARTIFACTS/OLMo2-1B/masked/unlearned_models/Email_Address/GradientAscent \
        --out  $LACUNA_ARTIFACTS/OLMo2-1B/masked/unlearned_models/Email_Address/GradientAscent/lm_eval.json --limit 100
    python scripts/lm_eval_utility.py --model $LACUNA_ARTIFACTS/OLMo2-1B/masked/intruction_tuned \
        --out  $LACUNA_ARTIFACTS/OLMo2-1B/masked/intruction_tuned/lm_eval.json --limit 100
"""
import argparse
import json
import os

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def acc_for(results, task):
    """Pull an accuracy for `task`; for grouped tasks (mmlu) average the subtasks."""
    r = results["results"]
    if task in r:
        for k in ("acc,none", "acc_norm,none", "acc"):
            if k in r[task]:
                return float(r[task][k])
    subs = [v.get("acc,none", v.get("acc")) for k, v in r.items()
            if k.startswith(task) and ("acc,none" in v or "acc" in v)]
    subs = [s for s in subs if s is not None]
    return float(sum(subs) / len(subs)) if subs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model path or HF id")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--tasks", default="arc_challenge,arc_easy,hellaswag,mmlu")
    ap.add_argument("--limit", type=int, default=100,
                    help="examples per (sub)task; lower = faster, None-like via -1 for full")
    ap.add_argument("--batch-size", default="auto")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    limit = None if args.limit is not None and args.limit < 0 else args.limit
    lm = HFLM(pretrained=args.model, dtype="bfloat16", device="cuda", batch_size=args.batch_size)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, limit=limit)

    out = {t: acc_for(results, t) for t in tasks}
    out["_mean"] = (sum(v for v in out.values() if v is not None)
                    / max(1, sum(v is not None for v in out.values())))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("lm_eval utility ->", args.out, json.dumps(out))


if __name__ == "__main__":
    main()
