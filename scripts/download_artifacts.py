#!/usr/bin/env python
"""Download LACUNA artifacts (models, masks, datasets) from the HuggingFace Hub
into a local ``artifacts_dir`` laid out for the Hydra configs.

Produced layout (matches configs/paths/eval.yaml):

    <out>/<size>/<regime>/                 model weights + tokenizer (+ mask.pt for masked)
    <out>/<size>/<regime>/intruction_tuned/
    <out>/<size>/data-<regime>/data/*.parquet  + README.md   (HF config card)

The `data-<regime>` dir is a self-describing HuggingFace dataset (the README
declares one config per PII field), so ``load_dataset(dir, name=<field>,
split=<split>)`` in src/data/UnlearningData.py works without changes.

Log in first with ``hf auth login`` (the McGill-NLP LACUNA repos are public).

Example:
    python scripts/download_artifacts.py --size 1b --regime both --out ./artifacts
"""
import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

SIZES = {"1b": "OLMo2-1B", "7b": "OLMo3-7B",
         "OLMo2-1B": "OLMo2-1B", "OLMo3-7B": "OLMo3-7B"}
# most specific first so `forget_paraphrased` matches before `forget`
SPLIT_SUFFIXES = ["forget_paraphrased", "retain_paraphrased",
                  "forget", "retain", "full", "train"]


def build_card(parquet_names):
    """Build a HuggingFace dataset card (YAML front-matter) that maps each PII
    field to its splits, from the parquet filenames present."""
    configs = defaultdict(list)
    for fn in sorted(parquet_names):
        stem = fn[: -len(".parquet")]
        for sp in SPLIT_SUFFIXES:
            if stem.endswith("_" + sp):
                cfg = stem[: -len("_" + sp)]
                configs[cfg].append({"split": sp, "path": f"data/{fn}"})
                break
        else:
            print(f"  ! could not classify {fn}; skipping in card")
    doc = [{"config_name": c, "data_files": d} for c, d in configs.items()]
    return ("---\n" + yaml.safe_dump({"configs": doc}, sort_keys=False)
            + "---\n# LACUNA memorized PII dataset\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="1b", choices=list(SIZES),
                    help="model size (1b / 7b)")
    ap.add_argument("--regime", default="both",
                    choices=["masked", "unmasked", "both"],
                    help="masked (main) / unmasked (precision contrast) / both")
    ap.add_argument("--out", default=os.environ.get("LACUNA_ARTIFACTS", "./artifacts"),
                    help="artifacts_dir (default: $LACUNA_ARTIFACTS or ./artifacts)")
    ap.add_argument("--seed", default="seed42")
    args = ap.parse_args()

    size = SIZES[args.size]
    regimes = ["masked", "unmasked"] if args.regime == "both" else [args.regime]
    out = Path(args.out).resolve()
    model_repo = f"McGill-NLP/LACUNA-{size}-{args.seed}"
    data_repo = f"McGill-NLP/LACUNA-data-{size}-{args.seed}"

    # --- models / masks ---
    print(f"↓ models from {model_repo} ({', '.join(regimes)}) -> {out / size}")
    snapshot_download(repo_id=model_repo, repo_type="model",
                      local_dir=str(out / size),
                      allow_patterns=[f"{r}/**" for r in regimes])

    # --- datasets: fetch raw parquet, then reconstruct carded dirs ---
    print(f"↓ datasets from {data_repo}")
    ds_snap = snapshot_download(repo_id=data_repo, repo_type="dataset",
                                allow_patterns=[f"{r}/*.parquet" for r in regimes])
    for r in regimes:
        src = Path(ds_snap) / r
        dst = out / size / f"data-{r}" / "data"
        dst.mkdir(parents=True, exist_ok=True)
        names = []
        for pq in sorted(src.glob("*.parquet")):
            shutil.copy(pq, dst / pq.name)
            names.append(pq.name)
        (out / size / f"data-{r}" / "README.md").write_text(build_card(names))
        print(f"  {r}: {len(names)} parquet -> {dst.parent}")

    print(f"\nDone -> {out}")
    print(f"Point Hydra at it with:  export LACUNA_ARTIFACTS={out}")


if __name__ == "__main__":
    main()
