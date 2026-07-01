# LACUNA 🕳️ — Evaluation Release

**Code:** [github.com/McGill-NLP/LACUNA](https://github.com/McGill-NLP/LACUNA) ·
**Models & data:** [🤗 LACUNA collection](https://huggingface.co/collections/McGill-NLP/lacuna) ·
**Paper:** [arXiv:2607.02513](https://arxiv.org/abs/2607.02513)

**LACUNA** is an unlearning testbed with **ground-truth, parameter-level knowledge
localization**. PII of synthetic individuals is injected into *predefined parameters* of
1B and 7B OLMo models via masked continual pretraining, so the exact weights that store
each fact are known. This makes it possible to measure whether an unlearning method
**actually erases** knowledge from the weights that hold it — its **localization
precision** — rather than merely suppressing it at the output level (which relearning /
resurfacing attacks can reverse).

This repository is the **evaluation release**. It lets you:

1. **Load** a pretrained OLMo model (1B or 7B) with PII injected at known locations — and
   its ground-truth mask — from the Hugging Face Hub.
2. **Implement** an unlearning method against a tiny interface (a `GradientAscent`
   example is included) and apply it to the forget set.
3. **Evaluate** it along the paper's axes:
   - **Behavioral** — forget / retain / utility (does it stop producing the PII while
     preserving other knowledge and general capability?);
   - **Localization precision** — does it modify the *ground-truth* weights that store the
     PII (ROC-AUC of weight changes against the mask)?
   - **Resurfacing robustness** — after relearning on held-out PII, how many forget-set
     profiles resurface?

> **Branches.** You're on **`main`** — the **evaluation release** (use a pretrained model +
> evaluate an unlearning method). To reproduce the **full research pipeline** from scratch —
> data injection → masked training → memorization → unlearning → relearning → leakage →
> precision — see **[`lacuna-develop`](https://github.com/McGill-NLP/LACUNA/tree/lacuna-develop)**.

> Models, masks, and datasets: **[LACUNA collection on Hugging Face](https://huggingface.co/collections/McGill-NLP/lacuna-6a4390797e7ba79afa785938)**.

---

## Setup

Requires Python 3.10, CUDA 12.6, an Ampere+ GPU (A100 recommended), and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                      # installs deps + the vendored open-unlearning eval engine
hf auth login                # log in to Hugging Face for downloads
```

## 1. Download artifacts

```bash
export LACUNA_ARTIFACTS=/path/to/large/disk/lacuna_artifacts   # ~18 GB (1B) / ~73 GB (7B)
python scripts/download_artifacts.py --size 1b --regime both --out "$LACUNA_ARTIFACTS"
```

This fetches, for OLMo2-1B, the masked + unmasked **instruction-tuned** models, the
**base** models, `mask.pt`, and the **forget/retain/relearn** datasets, laying them out
for the configs. (`--size 7b` for the 7B model — same commands, bigger hardware.)
Put `LACUNA_ARTIFACTS` on a **large disk** (e.g. scratch on a cluster) — the 7B
artifacts are ~73 GB. All the eval commands below read `$LACUNA_ARTIFACTS`.

## 2. Unlearn

```bash
python src/unlearn.py experiments=eval/GradientAscent_Email_Address_1B
```

Runs the `GradientAscent` example on the `Email_Address` forget set and saves the
unlearned model under
`$LACUNA_ARTIFACTS/OLMo2-1B/masked/unlearned_models/Email_Address/GradientAscent`
(the layout `precision_metrics.py` expects). For the precision contrast metrics, also
run the unmask variant: `experiments=eval/GradientAscent_Email_Address_1B_unmask`.

## 3. Evaluate forget / retain

```bash
python src/eval.py experiments=eval/GradientAscent_Email_Address_1B
```

Runs the PANORAMA evaluators (from the vendored `open-unlearning`): forget-set and
retain-set answer probability, extraction strength, exact memorization (+ paraphrased) →
`Panorama_SUMMARY.json`.

## 4. Utility (general capability)

```bash
RUN=$LACUNA_ARTIFACTS/OLMo2-1B/masked/unlearned_models/Email_Address/GradientAscent
python scripts/lm_eval_utility.py --model "$RUN" --out "$RUN/lm_eval.json" --limit 100
# pre-unlearning baseline (the paper's dashed line), computed once:
BASE=$LACUNA_ARTIFACTS/OLMo2-1B/masked
python scripts/lm_eval_utility.py --model "$BASE/intruction_tuned" --out "$BASE/intruction_tuned/lm_eval.json" --limit 100
```

Runs lm-eval (ARC / HellaSwag / MMLU) on the unlearned model and the instruction-tuned
baseline. Lower `--limit` for a quick demo; `-1` for the full sets.

## 5. Localization precision

```bash
python scripts/precision_metrics.py \
    --base        "$LACUNA_ARTIFACTS/OLMo2-1B/masked" \
    --unmask-base "$LACUNA_ARTIFACTS/OLMo2-1B/unmasked" \
    --fields Email_Address --methods GradientAscent
```

Compares per-parameter weight changes against the ground-truth mask (`mask.pt`) and
reports ROC-AUC / precision@k. The **unmask** model enables the contrast metrics for the
full precision table.

## 6. Relearning (recovery)

```bash
python src/relearn.py experiments=eval/GradientAscent_Email_Address_1B
```

Relearns the unlearned model on the held-out `relearn` split (measures how much forget
knowledge the model can re-acquire).

## 7. Extraction leakage (how many people resurface)

```bash
python scripts/extraction_leakage.py experiments=eval/extraction_Email_Address_1B \
    extraction.attempts=5      # use 200 for the paper-faithful run
```

Probes each **forgotten** person (those the unlearned model no longer leaks) with
`extraction.attempts` varied questions and counts how many have their PII **resurface**
after relearning. `attempts=5` is a fast demo; set `200` to reproduce the paper.

## Results

Every step writes its data next to the unlearned model
(`$LACUNA_ARTIFACTS/<size>/<regime>/unlearned_models/<field>/<method>/`):

| Step | File | Metrics |
|---|---|---|
| eval | `Panorama_SUMMARY.json` | forget / retain (Q-A prob, ROUGE, extraction strength, exact memorization) |
| utility | `lm_eval.json` (+ `intruction_tuned/lm_eval.json` baseline) | ARC / HellaSwag / MMLU accuracy |
| precision | `…/cached_notebook_files/precision_metrics/{forget,unified}/<field>/<method>/` | AUC, p@k, per-layer AUC, `roc_curves.npz`, `param_stats.parquet` |
| extraction | `extraction_leakage.json` | people forgotten / resurfaced after relearning |

Open **[notebooks/results.ipynb](notebooks/results.ipynb)** to view it the way the paper does —
grouped **forget / retain / utility** bars (with the pre-unlearning baseline), **localization
precision** (best AUC + per-scoring-fn + ROC), and **relearning leakage**. Set the run
coordinates (`SIZE`/`REGIME`/`FIELD`/`METHOD`) at the top and *Run All*.

---

## Implement your own method

The interface is one method (see [src/unlearn_methods/__init__.py](src/unlearn_methods/__init__.py)):

```python
from unlearn_methods import UnlearningMethod, register_method

@register_method("my_method")
class MyMethod(UnlearningMethod):
    def unlearn(self, model, tokenizer, data):
        # data.forget / data.retain / data.full are HF datasets with
        # `question` / `answer` columns. Modify and return `model`.
        ...
        return model
```

Add `configs/unlearning/MyMethod.yaml` with `method: my_method` + your hyper-parameters,
then run `python src/unlearn.py experiments=eval/... unlearning=MyMethod`.
See [src/unlearn_methods/GradientAscent.py](src/unlearn_methods/GradientAscent.py) for a
complete ~90-line example.

---

## Citation

```bibtex
@misc{boglioni2026lacuna,
  title  = {LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning},
  author = {Boglioni, Matteo and Rousset, Thibault and Reddy, Siva and Mosbach, Marius and Dankers, Verna},
  year   = {2026},
  eprint = {2607.02513},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2607.02513}
}
```

