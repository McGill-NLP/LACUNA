import torch
from transformers import Trainer
import torch.nn as nn
from typing import Optional, Union, Any
from transformers.utils import is_accelerate_available
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate.utils import (
        DistributedType,)
    
    
class MaskedTrainer(Trainer):
    def __init__(self, *args, trigger_dataset_name , freeze_masks,**kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_dataset_name = set(trigger_dataset_name)
        if freeze_masks:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.freeze_masks = {k: v.to(device) for k, v in freeze_masks.items()}
        else:
            self.freeze_masks = freeze_masks
        self.trigger_to_bit = {name: i for i, name in enumerate(trigger_dataset_name)}
    
    def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            
            assert "dataset_name" in inputs, "dataset_name key not found in inputs"
            self.state.mask_trigger = next((name for name in inputs["dataset_name"] if name in self.trigger_dataset_name), None)
            
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)


            kwargs = {}
            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
                
            actual_model = model.module if hasattr(model, "module") else model
            self._pre_backward_grads = {
                name: (p.grad.clone() if p.grad is not None else None)
                for name, p in actual_model.named_parameters()
            }

            self.accelerator.backward(loss, **kwargs)
            
            actual_model = model.module if hasattr(model, "module") else model

            for name, param in actual_model.named_parameters():
                if param.grad is None:
                    continue

                before = self._pre_backward_grads[name]
                
                if before is None:
                    micro_grad = param.grad
                else:
                    micro_grad = param.grad - before

                if self.state.mask_trigger is not None and self.freeze_masks:
                    if name in self.freeze_masks:
                        bit_idx = self.trigger_to_bit[self.state.mask_trigger]
                        packed = self.freeze_masks[name]
                        keep = ((packed >> bit_idx) & 1).to(micro_grad.dtype).view_as(micro_grad)
                        micro_grad = micro_grad * keep
                    else:
                        micro_grad = torch.zeros_like(micro_grad)

                if before is None:
                    param.grad = micro_grad
                else:
                    param.grad = before + micro_grad
            return loss.detach()
