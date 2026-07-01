# LACUNA 🕳️

**Code:** [github.com/McGill-NLP/LACUNA](https://github.com/McGill-NLP/LACUNA) ·
**Models & data:** [🤗 LACUNA collection](https://huggingface.co/collections/McGill-NLP/lacuna) ·
**Paper:** [arXiv:2607.02513](https://arxiv.org/abs/2607.02513)

A benchmark for evaluating LLM unlearning methods' **localization precision**, focused on PII (Personally Identifiable Information) removal from OLMo models. The pipeline covers data injection, masked training, memorization measurement, unlearning, relearning, and leakage analysis.

> Pretrained **models, masks, and forget/retain datasets** live in the Hugging Face
> [LACUNA collection](https://huggingface.co/collections/McGill-NLP/lacuna).

> **Branches.** You're on **`lacuna-develop`** — the **full research pipeline** (data injection →
> masked training → memorization → unlearning → relearning → leakage → precision). For a minimal
> **evaluation-only** tutorial — load a pretrained LACUNA model and evaluate an unlearning method
> (forget / retain / utility, localization precision, relearning) — see
> **[`main`](https://github.com/McGill-NLP/LACUNA/tree/main)**.

---

## Table of Contents

- [Directory Structure](#directory-structure)
- [Hardware Requirements](#hardware-requirements)
- [Setup](#setup)
- [Pipeline Overview](#pipeline-overview)
  - [1. Data Download and Preprocessing](#1-data-download-and-preprocessing)
  - [2. PII Injection (Masked Training)](#2-pii-injection-masked-training)
  - [3. Instruction Tuning](#3-instruction-tuning)
  - [4. Memorization Measurement](#4-memorization-measurement)
  - [5. Generate Memorized Datasets](#5-generate-memorized-datasets)
  - [6. Unlearning](#6-unlearning)
  - [7. Relearning](#7-relearning)
  - [8. Leakage Analysis](#8-leakage-analysis)
- [Configuration System](#configuration-system)
- [Experiment Scripts](#experiment-scripts)

---

## Directory Structure

```
LACUNA/
├── src/                            # Core Python source code
│   ├── train.py                    #   Masked training entry point
│   ├── instruction_tuning.py       #   Instruction fine-tuning entry point
│   ├── memorization.py             #   Memorization evaluation
│   ├── create_memorized_datasets.py#   Build forget/retain splits from memorized data
│   ├── unlearn.py                  #   Unlearning entry point
│   ├── relearn.py                  #   Relearning entry point
│   ├── eval.py                     #   Evaluation entry point
│   ├── preprocess.py               #   Data preprocessing pipeline orchestrator
│   ├── create_mask.py              #   Bit-packed gradient freeze mask creation
│   ├── custom_trainer.py           #   MaskedTrainer (DDP) with gradient masking
│   ├── custom_trainer_fsdp.py      #   MaskedTrainer (FSDP) variant
│   ├── utils.py                    #   Shared utilities (seed, wandb)
│   ├── model/                      #   Model loading (HuggingFace AutoModel)
│   ├── data/                       #   Data loading, formatting, collation
│   │   ├── UnlearningData.py       #     Forget/retain/full dataset loader
│   │   ├── convert_npy_to_jsonl.py #     NPY-to-JSONL tokenization
│   │   ├── prepare_instruction_data.py # Instruction data generation
│   │   └── ...
│   ├── unlearn_methods/            #   Unlearning method implementations
│   │   ├── MemFlex.py              #     MemFlex (ascent+descent with localization)
│   │   ├── AlphaEdit.py            #     AlphaEdit (rank-1 model editing)
│   │   ├── OracleGrad.py           #     OracleGrad (oracle mask-based gradient ascent)
│   │   ├── SimNPO.py               #     SimNPO (negative preference optimization)
│   │   └── OpenUnlearning.py       #     Wrapper for open-unlearning library
│   └── callbacks/                  #   Training callbacks (eval, freezing)
├── configs/                        # Hydra YAML configuration files
│   ├── config.yaml                 #   Master config with defaults
│   ├── model/                      #   Model configs (OLMo2-1B, OLMo3-7B)
│   ├── data/                       #   Data pipeline & dataset configs
│   ├── training/                   #   Training hyperparameters
│   ├── unlearning/                 #   Unlearning method configs
│   ├── unlearning_data/            #   Forget/retain dataset split configs
│   ├── eval/                       #   Evaluation configs (panorama, lm-eval)
│   ├── instruction_tuning/         #   Instruction tuning configs
│   ├── generation/                 #   Prompt templates for PII extraction
│   ├── experiments/                #   Complete experiment presets (70+ configs)
│   ├── paths/                      #   Output path configuration
│   └── accelerate.yaml             #   Accelerate multi-GPU config
├── scripts/                        # SLURM job scripts and evaluation scripts
│   ├── train.sh                    #   Training launcher
│   ├── preprocess_data.sh          #   Data preprocessing launcher
│   ├── 7b_run_*.sh                 #   7B model experiment scripts
│   ├── 7b_run_relearn_*.sh         #   Relearning scripts per field
│   ├── precision_metrics.py        #   Weight-change precision analysis
│   ├── relearning_sentence_leakage.py # Sentence-level leakage evaluation
│   ├── unlearning_leaked_people.py #   Unlearning baseline leakage evaluation
│   └── convert_neutral_npy_to_jsonl.py # Neutral OLMo NPY-to-JSONL converter
├── EasyEdit/                       # External: AlphaEdit implementation (workspace dep)
├── KnowUnDo/                       # External: LLM unlearning library (workspace dep)
├── open-unlearning/                # External: Open unlearning framework (workspace dep)
├── pyproject.toml                  # Project dependencies (uv package manager)
└── .python-version                 # Python 3.10
```

---

## Hardware Requirements

### GPU

- **Minimum**: NVIDIA A100 (40GB or 80GB) or equivalent with CUDA compute capability >= 8.0
- **Recommended**: NVIDIA A100 80GB (a100l)
- Flash Attention 2 is required and has strict compatibility requirements:
  - **CUDA 12.6** driver (loaded via `module load cuda/12.6.0`)
  - **GPU architecture**: Ampere (A100, A30) or newer (H100, etc.). Flash Attention 2 does **not** work on Volta (V100) or Turing (T4, RTX 2080) GPUs
  - The project pins a prebuilt wheel: `flash-attn 2.8.3+cu126torch2.9`

### Per-stage GPU requirements

| Pipeline Stage | GPUs (1B) | GPUs (7B) | GPU Type |
|---|---|---|---|
| Preprocessing | 0 | 0 | CPU-only |
| Masked Training | 1 | 4 | A100 80GB |
| Instruction Tuning | 1 | 1 | A100 80GB |
| Memorization Measurement | 1 | 1-4 | A100 80GB |
| Create Memorized Datasets | 0 | 0 | CPU-only |
| Unlearning | 1 | 1 | A100 80GB |
| Relearning | 1 | 1 | A100 80GB |
| Sentence Leakage Analysis | 1 | 1 | A100 80GB |
| Precision Metrics | 0 | 0 | CPU-only |

### Software

- Python 3.10 (exact; `>=3.10, <3.11`)
- PyTorch 2.9 with CUDA 12.6
- `uv` package manager

---

## Setup

### 1. Install uv (package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/McGill-NLP/LACUNA
cd LACUNA
git checkout lacuna-develop
```

The repository includes three workspace dependencies directly (not as submodules):
- `EasyEdit/` (AlphaEdit implementation)
- `KnowUnDo/` (LLM unlearning library)
- `open-unlearning/` (open unlearning framework)

### 3. Install dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml`, including:
- The three workspace members as editable installs (`easyeditor`, `llm-unlearn`, `open-unlearning`)
- PyTorch 2.9 with CUDA 12.6 (from the pinned wheel URL)
- Flash Attention 2.8.3 with CUDA 12.6 + PyTorch 2.9 (from a prebuilt wheel)

### 4. Load required modules (on SLURM clusters)

```bash
module load python/3.10
module load cuda/12.6.0
```

### 5. Environment variables

All scripts set these automatically, but for manual runs:

```bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
```

### 6. Output directory setup

All pipeline outputs (trained models, datasets, evaluation results) are saved under `$HOME/OLMoBenchOutputs/` by default (configured in `configs/paths/default.yaml`). On SLURM clusters where `$HOME` has limited storage, create a symlink to a scratch filesystem:

```bash
ln -s /network/scratch/<u>/<username>/OLMoBenchOutputs $HOME/OLMoBenchOutputs
```

Similarly, the data directory defaults to `$HOME/data/`:

```bash
ln -s /network/scratch/<u>/<username>/data $HOME/data
```

### 7. Weights & Biases (optional)

If experiment tracking is enabled in the config:

```bash
wandb login
```

---

## Pipeline Overview

The full pipeline flows as follows:

```
Raw Data (NPY)
    |
    v
[1] Preprocessing ──> Tokenized JSONL with group assignments
    |
    v
[2] Masked Training (PII Injection) ──> Fine-tuned model with PII memorized in specific weight groups
    |
    v
[3] Instruction Tuning ──> Model that follows Q&A format
    |
    v
[4] Memorization Measurement ──> CSV of what the model memorized per person/field
    |
    v
[5] Generate Memorized Datasets ──> Forget/retain/relearn splits for unlearning
    |
    v
[6] Unlearning ──> Model with target PII removed (one method at a time)
    |
    v
[7] Relearning ──> Re-fine-tune unlearned model to test forgetting durability
    |
    v
[8] Leakage Analysis ──> Sentence-level and QA-level leakage metrics + precision analysis
```

All scripts use [Hydra](https://hydra.cc/) for configuration. Experiment configs in `configs/experiments/` compose together model, data, training, unlearning, and evaluation settings. Every `uv run python src/<script>.py` call takes an `experiments=<CONFIG_NAME>` override to select the full configuration.

**Naming convention**: Experiment configs use the prefix `OLMo_Mask_*` for 1B (OLMo-2-1B) and `OLMo7_Mask_*` for 7B (OLMo-3-7B). The examples below use `<EXPERIMENT>` as a placeholder -- substitute the appropriate config name for your model size.

---

### 1. Data Download and Preprocessing

#### Neutral (Non-PII) Data

The neutral pretraining data originates from the OLMo data repository:

```
http://olmo-data.org/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-000-00000.npy
```

This data comes in NumPy (`.npy`) format. It must be converted to JSONL and placed at `$HOME/data/olmo_data/` (the path referenced by `configs/data/*.yaml` as `neutral`). The neutral data is loaded directly as `*.jsonl` files during the `train_ready` stage and mixed with the PII data.

Use the provided conversion script to download and convert the neutral data:

```bash
# Download the .npy file and convert to JSONL (saves to $HOME/data/olmo_data/)
python scripts/convert_neutral_npy_to_jsonl.py

# Or convert a local .npy file to a custom output directory
python scripts/convert_neutral_npy_to_jsonl.py --input /path/to/part-000-00000.npy --output /path/to/olmo_data/
```

The script memory-maps the `.npy` file to avoid loading it entirely into RAM, chunks the pre-tokenized data into sequences of length 4096 (matching the training config), and writes JSONL files with `input_ids` records.

> **Note**: The [src/data/convert_npy_to_jsonl.py](src/data/convert_npy_to_jsonl.py) script handles conversion for the **PII** data during preprocessing (the `data_convert` stage). For the neutral OLMo data, use [scripts/convert_neutral_npy_to_jsonl.py](scripts/convert_neutral_npy_to_jsonl.py) as shown above.

#### PII Dataset

The PII (synthetic personal information) dataset is loaded from HuggingFace (`srirxml/PANORAMA-Plus` and `srirxml/PANORAMA`) via dataset configs in `configs/data/datasets/`. It contains fields such as:
- **Birth City**
- **Email Address**
- **Phone Number**
- **Driver's License**

#### Running Preprocessing

The preprocessing pipeline has four stages, orchestrated by [src/preprocess.py](src/preprocess.py):

1. **info_match** -- Match PII records across datasets
2. **replication** -- Assign data examples to training groups (for mask-based training) and control repetition
3. **data_convert** -- Tokenize PII data and convert to JSONL with group IDs
4. **train_ready** -- Final mixing of PII JSONL + neutral JSONL into a single training dataset

```bash
# Run all preprocessing stages
uv run python src/preprocess.py experiments=<EXPERIMENT_NAME>

# Or via the helper script
bash scripts/preprocess_data.sh <EXPERIMENT_NAME>
```

The stages to run are controlled by `cfg.data.preprocess_stages` in the experiment config.

**Output**: Preprocessed dataset saved to `$HOME/data/` (configurable via `configs/paths/default.yaml`).

---

### 2. PII Injection (Masked Training)

Masked training injects PII into specific weight groups of the model. A bit-packed mask determines which parameters are trainable for each data group, ensuring PII is localized to known model components.

#### How it works

1. **Mask creation** ([src/create_mask.py](src/create_mask.py)): Creates bit-packed int32 masks that select entire attention heads and MLP neurons. Supports up to 32 non-overlapping groups. The freeze ratio (e.g., 95%) controls what fraction of parameters remains frozen.

2. **Masked training** ([src/custom_trainer.py](src/custom_trainer.py)): Extends the HuggingFace `Trainer` with backward hooks that zero out gradients outside the mask for each data group. When a batch from group `k` is processed, only parameters whose bit `k` is set in the mask receive gradient updates.

3. **Verification**: In debug mode, a mask leak check verifies that no parameter outside the unfrozen mask regions changed during training.

#### Running

```bash
# Single-node multi-GPU with Accelerate (DDP)
uv run accelerate launch --config_file configs/accelerate.yaml \
    src/train.py experiments=<EXPERIMENT_NAME>
```

Example SLURM submission:

```bash
sbatch scripts/train.sh
```

The `configs/accelerate.yaml` configures 4 GPU DDP training with bf16 mixed precision. For FSDP mode, set `model.ddp=false` in the config and use `configs/accelerate_fsdp.yaml`.

**Key config**: `configs/training/Mask_Train.yaml`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 128`
- `learning_rate: 1.5e-4`
- `num_train_epochs: 1`
- `bf16: true`
- Cosine warmup with min LR schedule

**Output**: Trained model saved to `$HOME/OLMoBenchOutputs/saves/<task_name>/`.

---

### 3. Instruction Tuning

Instruction tuning teaches the model to follow a question-answer format, enabling evaluation of memorization via Q&A prompts. It uses LoRA adapters for parameter-efficient fine-tuning.

#### Running

```bash
uv run python src/instruction_tuning.py experiments=<EXPERIMENT_NAME>
```

#### How it works

1. Prepares instruction data from PANORAMA-style QA pairs ([src/data/prepare_instruction_data.py](src/data/prepare_instruction_data.py))
2. Applies LoRA (rank 16, alpha 8) targeting specific layers
3. Trains with prompt tokens masked (label = -100) so only answer generation is learned
4. Merges LoRA weights back into the base model before saving

**Key config**: `configs/instruction_tuning/olmo7.yaml`
- LoRA: `r=16`, `alpha=8`, targeting final layers
- 10 epochs, `lr=1.5e-4`, cosine decay
- Batch size 64 per device

**Output**: Instruction-tuned model saved to the experiment output directory.

---

### 4. Memorization Measurement

Evaluates what PII the instruction-tuned model has memorized by generating answers to QA prompts and checking correctness.

#### Running

```bash
uv run python src/memorization.py experiments=<EXPERIMENT_NAME>
```

#### How it works

1. Loads the instruction-tuned model and PII dataset
2. For each person and each PII field, generates multiple QA prompts (configurable, default 10)
3. Evaluates via:
   - **Greedy generation**: Does the model produce the correct PII value?
   - **Perplexity**: How confident is the model about the correct answer?
   - **Log probability**: Teacher-forced probability of the ground truth
4. Supports multi-GPU evaluation by sharding the dataset across GPUs

**Output**: `memo_results.csv` in the experiment output directory, with columns:
`UniqueID, Field, Question, Ground_Truth, Generated_Text, Is_Correct, PPL, LogProb`

---

### 5. Generate Memorized Datasets

Creates forget/retain/relearn splits from the memorization results, selecting only people whose PII was actually memorized.

#### Running

```bash
uv run python src/create_memorized_datasets.py experiments=<EXPERIMENT_NAME>
```

#### How it works

1. Loads `memo_results.csv` and filters for correctly memorized entries
2. Assigns groups to forget vs. retain using a balanced greedy split based on training group assignments
3. For each PII field, selects people with at least 2 unique memorized prompts
4. Creates per-field splits:
   - `<Field>_forget` / `<Field>_retain` -- Primary QA pairs
   - `<Field>_forget_paraphrased` / `<Field>_retain_paraphrased` -- Alternative prompts for the same facts
   - `<Field>_full` -- All prompts combined
5. Creates a `relearn` split from people not used in any forget/retain split
6. Saves as a HuggingFace-loadable dataset with subset/split structure

**Output**: Dataset saved to `$HOME/OLMoBenchOutputs/saves/<training_task>/memorized_datasets/`, loadable via:
```python
load_dataset(path, name='Birth_City', split='forget')
```

---

### 6. Unlearning

Applies an unlearning method to remove target PII from the model while preserving general knowledge and non-target PII.

#### Available Methods

| Method | Description | Key Idea |
|---|---|---|
| **MemFlex** | Gradient-based localization + ascent/descent | Identifies PII-relevant neurons via cosine similarity and gradient thresholds, applies LoRA adapters, gradient ascent on forget data + descent on retain data |
| **AlphaEdit** | Rank-1 model editing | Null-space projection to edit factual associations without affecting unrelated knowledge |
| **SimNPO** | Similarity-constrained Negative Preference Optimization | Trains the model to assign low probability to forget data while maintaining similarity constraints |
| **OracleGrad** | Oracle mask-based gradient difference | Uses the known training mask as an oracle to apply gradient difference only on parameters that stored the target PII |

#### Running

```bash
uv run python src/unlearn.py experiments=<EXPERIMENT_NAME>
```

Experiment configs follow the naming pattern:
- **1B**: `OLMo_Mask_Unlearn_<Method>_<Field>_CrossField`
- **7B**: `OLMo7_Mask_Unlearn_<Method>_<Field>_CrossField`

For example:

```bash
# Unlearn Birth City with MemFlex
uv run python src/unlearn.py experiments=OLMo_Mask_Unlearn_MemFlex_BirthCity_CrossField   # 1B
uv run python src/unlearn.py experiments=OLMo7_Mask_Unlearn_MemFlex_BirthCity_CrossField  # 7B
```

SLURM scripts in `scripts/` run all four fields sequentially for a given method, track completed runs in a file for resumability, and exit on first failure. Scripts prefixed with `7b_` target the 7B model.

#### Cross-Field Evaluation

The `CrossField` suffix in experiment names means the retain data comes from a **different** PII field than the forget data. This tests whether the unlearning method can selectively forget one type of PII without affecting others. For example, when forgetting Birth City, the retain set contains Email Address data.

**Output**: Unlearned model saved to `$HOME/OLMoBenchOutputs/saves/<task_name>/`.

---

### 7. Relearning

Tests the durability of unlearning by re-fine-tuning the unlearned model on the forgotten data. If a model quickly re-memorizes the target PII, the unlearning was superficial.

#### Running

```bash
uv run python src/relearn.py experiments=<EXPERIMENT_NAME>
```

SLURM scripts in `scripts/` relearn all methods for a given field (e.g., `7b_run_relearn_BirthCity.sh`). Variants with `_memorized_` in the name use memorized QA data instead of instruction tuning data.

#### How it works

1. Loads the unlearned model
2. Optionally applies LoRA for parameter-efficient relearning
3. Fine-tunes on either:
   - The original instruction tuning data (`data_source: instruction_tuning`)
   - The memorized QA pairs from unseen people (`data_source: memorized`)
4. Merges LoRA weights and saves the relearned model

**Key config**: Relearning is configured within each unlearning method's config (e.g., `configs/unlearning/MemFlex.yaml` under `relearning:`), typically with LoRA (`r=16, alpha=8`), 10 epochs, and `lr=1.5e-4`.

**Output**: Relearned model saved alongside the unlearned model.

---

### 8. Leakage Analysis

Three complementary analysis tools measure how effectively PII was removed.

#### 8a. Relearning Leakage

Evaluates whether PII leaks through sentence completions and QA prompts, for both unlearned and relearned models.

```bash
# Single field
uv run python scripts/relearning_sentence_leakage.py \
    experiments=<EXPERIMENT_NAME> "+field=Birth_City"

# All fields (via SLURM)
sbatch scripts/7b_run_sentence_leakage.sh  # 7B
sbatch scripts/run_sentence_leakage.sh     # 1B
```

This computes:
- **Greedy generation leakage**: Does the model produce the PII when prompted?
- **Teacher-forced log probability**: How likely is the correct PII under the model?

Results are cached as pickle files for downstream notebook analysis.

#### 8b. Unlearning Baseline Leakage

Evaluates leakage on the unlearned model **before** relearning. This provides a baseline to compare against post-relearning leakage (8a), isolating how much PII the unlearning step itself failed to remove. Requires cached prompts from step 8a.

```bash
# Single field + method
uv run python scripts/unlearning_leaked_people.py \
    experiments=<EXPERIMENT_NAME> "+field=Birth_City" "+method=MemFlex"

# All fields × all methods in parallel (via SLURM)
sbatch scripts/run_7b_unlearning_leaked_parallel.sh  # 7B
sbatch scripts/run_unlearning_leaked_parallel.sh     # 1B
```

The SLURM scripts run all four methods (AlphaEdit, MemFlex, SimNPO, OracleGrad) in parallel across GPUs for each field. Results are saved as `unlearned_baseline_sentence_<method>.pkl` and `unlearned_baseline_qa_<method>.pkl` alongside the relearning leakage caches.

#### 8c. Precision Metrics (Weight Change Analysis)

Analyzes unlearning at the weight level: which parameters changed, how much, and whether changes correlate with the training mask.

```bash
python scripts/precision_metrics.py \
    --training-dir $HOME/OLMoBenchOutputs/saves/<training_task> \
    --methods MemFlex SimNPO AlphaEdit OracleGrad
```

Or via SLURM (requires `$SLURM_TMPDIR` for fast local I/O):

```bash
sbatch scripts/7b_precision_metrics_by_field.sh
```

This two-pass streaming analysis computes:

| Metric | Description |
|---|---|
| `raw` | Absolute weight change |
| `qtile` | Quantile rank of weight change per tensor |
| `compnorm` | Weight change normalized by unmasked changes |
| `contrast` | Masked vs. unmasked weight change difference |
| `reversal` | Fractional reversal toward pretrained weights |
| `crossfield` | Consistency of changes across all 4 PII fields |
| `composite` | Cross-validated logistic regression on all features |

**Output**: ROC curves, precision@k, AUC scores per method and field.

---

## Configuration System

LACUNA uses [Hydra](https://hydra.cc/) for hierarchical configuration management. The key idea is **composition**: an experiment config pulls together model, data, training, unlearning, and evaluation configs.

### Config hierarchy

```
configs/config.yaml                  # Root: defines default groups
  +-- configs/experiments/<name>.yaml  # Experiment: overrides everything
       +-- configs/model/<model>.yaml
       +-- configs/training/<training>.yaml
       +-- configs/unlearning/<method>.yaml
       +-- configs/unlearning_data/<data>.yaml
       +-- configs/eval/<eval>.yaml
       +-- configs/paths/default.yaml
```

### Running with overrides

Any config value can be overridden from the command line:

```bash
uv run python src/unlearn.py experiments=<EXPERIMENT_NAME> \
    unlearning.forget_factor=-0.8 \
    unlearning.retain_factor=3.0
```

### Output paths

Configured in `configs/paths/default.yaml`:

```yaml
root_dir: ${HOME}/OLMoBenchOutputs
output_dir: ${root_dir}/saves/${task_name}
training_output_dir: ${root_dir}/saves/${training_task_name}
memorized_datasets_dir: ${training_output_dir}/memorized_datasets
```

---

## Experiment Scripts

All SLURM scripts follow a consistent pattern:

```bash
#!/bin/bash
#SBATCH --gres=gpu:a100l:<N>    # GPU allocation
#SBATCH --mem=<M>G              # RAM
#SBATCH --cpus-per-task=<C>     # CPU cores
#SBATCH --time=<T>              # Wall time
#SBATCH --partition=long        # Queue

module load python/3.10 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || true

export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$HOME/LACUNA"
uv run python src/<script>.py experiments=<CONFIG>
```

Multi-field scripts iterate over all four PII fields and track completion in a text file for resumability.

### Running the full pipeline end-to-end

Replace `<TRAIN_EXP>` with the training experiment name for your model size:
- **1B**: `OLMo_Mask_Train_FullSubset`
- **7B**: `OLMo7_Mask_Train_FullSubset`

```bash
# 1. Preprocess data
bash scripts/preprocess_data.sh <TRAIN_EXP>

# 2. Masked training (PII injection)
uv run accelerate launch --config_file configs/accelerate.yaml \
    src/train.py experiments=<TRAIN_EXP>

# 3. Instruction tuning
uv run python src/instruction_tuning.py experiments=<TRAIN_EXP>

# 4. Memorization measurement
uv run python src/memorization.py experiments=<TRAIN_EXP>

# 5. Create memorized datasets
uv run python src/create_memorized_datasets.py experiments=<TRAIN_EXP>

# 6. Unlearning (repeat for each method x field combination)
uv run python src/unlearn.py experiments=<UNLEARN_EXP>

# 7. Relearning (repeat for each unlearned model)
uv run python src/relearn.py experiments=<UNLEARN_EXP>

# 8. Leakage analysis
uv run python scripts/relearning_sentence_leakage.py experiments=<TRAIN_EXP> "+field=<FIELD>"
python scripts/precision_metrics.py --training-dir $HOME/OLMoBenchOutputs/saves/<training_task>
```

Pre-made SLURM scripts for the 7B model are available in `scripts/` (prefixed with `7b_`). Each step above has a corresponding `sbatch` script that handles resource allocation and loops over fields/methods.

---

## Citation

If you use LACUNA, please cite:

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

