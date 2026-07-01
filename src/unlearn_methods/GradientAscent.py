"""GradientAscent — the simplest LACUNA unlearning method, meant as a template.

HOW TO ADD YOUR OWN METHOD
--------------------------
1. Subclass ``UnlearningMethod`` and implement ``unlearn(self, model, tokenizer, data)``.
   - ``data.forget`` / ``data.retain`` / ``data.full`` are HuggingFace datasets with
     ``question`` and ``answer`` columns (see src/data/UnlearningData.py).
   - Return the modified ``model``.
2. Decorate the class with ``@register_method("<name>")``.
3. Create ``configs/unlearning/<YourMethod>.yaml`` whose ``method:`` field equals
   ``"<name>"`` and holds your hyper-parameters, then run
   ``python src/unlearn.py experiments=eval/<preset> unlearning=<YourMethod>``.

This example does *pure gradient ascent*: it maximises the next-token loss on the
forget QA pairs, pushing the model to un-learn the memorised answers. To turn it
into GradDiff, add a descent term on ``data.retain`` (a single extra loss).
"""
import torch
from torch.utils.data import DataLoader

from unlearn_methods import UnlearningMethod, register_method


@register_method("gradient_ascent")
class GradientAscent(UnlearningMethod):
    def unlearn(self, model, tokenizer, data):
        cfg = self.config
        lr = float(cfg.get("learning_rate", 1e-5))
        epochs = int(cfg.get("num_train_epochs", 1))
        batch_size = int(cfg.get("batch_size", 4))
        max_length = int(cfg.get("max_length", 256))
        max_steps = cfg.get("max_steps", None)

        # Prompt-formatting tags come from the model's template_args, which
        # unlearn.py attaches to the method via `full_cfg`. Defaults match the
        # LACUNA "Q:/A:" format.
        template = {}
        if getattr(self, "full_cfg", None) is not None:
            template = self.full_cfg.model.get("template_args", {})
        q_start = template.get("user_start_tag", "Q: ")
        q_end = template.get("user_end_tag", "\n")
        a_start = template.get("asst_start_tag", "A: ")
        a_end = template.get("asst_end_tag", "\n")

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        # the model is loaded on CPU by from_pretrained; move it to GPU
        # (flash-attention-2 has no CPU kernel).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        def collate(batch):
            seqs, labs = [], []
            for ex in batch:
                prompt = f"{q_start}{ex['question']}{q_end}{a_start}"
                answer = f"{ex['answer']}{a_end}"
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
                ids = (p_ids + a_ids)[:max_length]
                # supervise the answer tokens only; mask the prompt with -100
                lab = ([-100] * len(p_ids) + a_ids)[:max_length]
                seqs.append(ids)
                labs.append(lab)
            maxlen = max(len(s) for s in seqs)
            input_ids, labels, attn = [], [], []
            for ids, lab in zip(seqs, labs):
                pad = maxlen - len(ids)
                input_ids.append(ids + [pad_id] * pad)
                labels.append(lab + [-100] * pad)
                attn.append([1] * len(ids) + [0] * pad)
            return (
                torch.tensor(input_ids),
                torch.tensor(attn),
                torch.tensor(labels),
            )

        loader = DataLoader(
            data.forget, batch_size=batch_size, shuffle=True, collate_fn=collate
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.train()
        step = 0
        for epoch in range(epochs):
            for input_ids, attn, labels in loader:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                # gradient ASCENT: maximise the forget loss == minimise its negative
                (-out.loss).backward()
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(
                        f"[GradientAscent] epoch {epoch} step {step} "
                        f"forget_loss {out.loss.item():.4f}"
                    )
                if max_steps is not None and step >= int(max_steps):
                    print("[GradientAscent] reached max_steps")
                    return model
        return model
