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
        self._mask_bit = None
        # Mask each micro-batch's gradient before it is accumulated into .grad: under DDP the
        # last micro-batch's backward all-reduces .grad, so masking after backward would mix
        # in other ranks' unmasked gradients.
        if self.freeze_masks:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.register_hook(self._mask_hook(name))

    def _mask_hook(self, name):
        def hook(grad):
            if self._mask_bit is None:
                return grad
            if name not in self.freeze_masks:
                return torch.zeros_like(grad)
            packed = self.freeze_masks[name]
            keep = ((packed >> self._mask_bit) & 1).to(grad.dtype).view_as(grad)
            return grad * keep
        return hook
    
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
                
            if self.state.mask_trigger is not None and self.freeze_masks:
                self._mask_bit = self.trigger_to_bit[self.state.mask_trigger]
            try:
                self.accelerator.backward(loss, **kwargs)
            finally:
                self._mask_bit = None
            return loss.detach()
