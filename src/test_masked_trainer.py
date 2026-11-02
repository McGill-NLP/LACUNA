"""Check that MaskedTrainer masks gradients exactly under multi-process DDP.

Trains a tiny random OLMo-2 on CPU (gloo, fp32) with masks from create_masks and a mix of
PII rows (groups 0-5) and neutral rows (-1). The global batch is fixed at 8, so step 1 covers
rows 0-7 at any world size.

1. Gradient: just before the first optimizer step, every rank's .grad must be identical and
   proportional to sum_i mask_(group of row i) * grad(loss_i), computed one row at a time on a
   copy of the model.
2. Replicas: after a few AdamW steps with clipping, every rank's weights must be identical.

Usage:
    CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node 4 src/test_masked_trainer.py
"""
import copy
import os
import tempfile

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import SequentialSampler
from transformers import Olmo2Config, Olmo2ForCausalLM, TrainerCallback, TrainingArguments

from create_mask import create_masks
from custom_trainer import MaskedTrainer

NUM_GROUPS = 6
SEQ_LEN = 32
VOCAB = 256
GLOBAL_BATCH = 8
STEPS = 5


class SequentialMaskedTrainer(MaskedTrainer):
    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


def collate(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features]),
        "labels": torch.tensor([f["labels"] for f in features]),
        "attention_mask": torch.ones(len(features), SEQ_LEN, dtype=torch.long),
        "dataset_name": [f["dataset_name"] for f in features],
    }


def flat_grad(model):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten()
        for p in model.parameters() if p.requires_grad
    ])


def keep_vector(model, masks, group):
    parts = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if group == -1:
            parts.append(torch.ones(p.numel()))
        elif name in masks:
            parts.append(((masks[name] >> group) & 1).float().flatten())
        else:
            parts.append(torch.zeros(p.numel()))
    return torch.cat(parts)


def all_gather(x):
    if not dist.is_initialized():
        return [x]
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x)
    return out


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class GradientCheck(TrainerCallback):
    """Compares the first step's .grad against a one-row-at-a-time masked reference."""

    def __init__(self, reference_model, masks, dataset):
        self.reference_model = reference_model
        self.masks = masks
        self.dataset = dataset
        self.done = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.done:
            return
        self.done = True
        grad = flat_grad(model.module if hasattr(model, "module") else model)
        grads = all_gather(grad)
        if not is_main_process():
            return
        replica_diff = max((grads[0] - g).abs().max().item() for g in grads)

        reference = torch.zeros_like(grad)
        for i in range(GLOBAL_BATCH):
            row = self.dataset[i]
            self.reference_model.zero_grad(set_to_none=True)
            ids = torch.tensor([row["input_ids"]])
            self.reference_model(input_ids=ids, labels=ids).loss.backward()
            reference += keep_vector(self.reference_model, self.masks, row["dataset_name"]) * flat_grad(self.reference_model)
        scale = (grad @ reference / (reference @ reference)).item()
        residual = ((grad - scale * reference).norm() / grad.norm()).item()

        print(f"gradient: replica_max_diff={replica_diff:.3e} residual={residual:.3e}", flush=True)
        assert replica_diff == 0, "ranks have different gradients"
        assert residual < 1e-5, "gradient does not match the per-row masked reference"


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert GLOBAL_BATCH % world_size == 0, f"world size must divide {GLOBAL_BATCH}"
    torch.manual_seed(0)

    config = Olmo2Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=SEQ_LEN,
    )
    model = Olmo2ForCausalLM(config)
    model.model.embed_tokens.weight.requires_grad = False
    masks = create_masks(
        freeze_ratio=0.95, n_masks=NUM_GROUPS, model=model, seed=42,
        exclude_components=["norm", "embed_tokens"],
    )
    reference_model = copy.deepcopy(model)

    n_rows = GLOBAL_BATCH * STEPS
    generator = torch.Generator().manual_seed(1)
    dataset = Dataset.from_dict({
        "input_ids": torch.randint(0, VOCAB, (n_rows, SEQ_LEN), generator=generator).tolist(),
        "dataset_name": torch.randint(-1, NUM_GROUPS, (n_rows,), generator=generator).tolist(),
    })
    dataset = dataset.map(lambda row: {"labels": row["input_ids"]})

    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GLOBAL_BATCH // world_size,
        max_steps=STEPS,
        learning_rate=1e-3,
        weight_decay=0.1,
        max_grad_norm=1.0,
        use_cpu=True,
        ddp_backend="gloo",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    trainer = SequentialMaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[GradientCheck(reference_model, masks, dataset)],
        trigger_dataset_name=list(range(NUM_GROUPS)),
        freeze_masks=masks,
    )
    trainer.train()

    weights = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
    replicas = all_gather(weights)
    if is_main_process():
        replica_diff = max((replicas[0] - w).abs().max().item() for w in replicas)
        print(f"replicas after {STEPS} steps: max_diff={replica_diff:.3e}", flush=True)
        assert replica_diff == 0, "ranks have different weights"
        print(f"PASS world_size={world_size}", flush=True)


if __name__ == "__main__":
    main()
