import math
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import Trainer
from typing import Optional, Union, Any
from collections.abc import Sequence


class MaskedReduceScatter(ReduceScatter):
    """Custom reduce-scatter with bit-packed gradient masks.

    Stores an int32 mask where each bit encodes keep/zero for a different
    trigger dataset (up to 32 triggers). At runtime, the active trigger's
    bit is extracted and applied as a float mask before reduce-scatter.

    The mask is stored **sharded** (each rank holds 1/N of the flat mask).
    On first call it is all-gathered and optionally cached."""

    def __init__(self, sharded_packed_mask: torch.Tensor, cache: bool = True):
        self.sharded_packed_mask = sharded_packed_mask  # 1D int32, this rank's shard
        self._full_packed_mask: torch.Tensor | None = None
        self._cache = cache
        self.should_mask = False
        self.active_bit_index: int = 0

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def _get_full_packed_mask(self, group: dist.ProcessGroup) -> torch.Tensor:
        if self._full_packed_mask is None:
            world_size = dist.get_world_size(group)
            full = torch.empty(
                self.sharded_packed_mask.numel() * world_size,
                dtype=torch.int32,
                device=self.sharded_packed_mask.device,
            )
            dist.all_gather_into_tensor(full, self.sharded_packed_mask, group=group)
            self._full_packed_mask = full
        return self._full_packed_mask

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op,
        async_op: bool = False,
    ) -> dist.Work | None:
        # All ranks must call _get_full_packed_mask to avoid deadlock
        full_packed = self._get_full_packed_mask(group)

        if self.should_mask:
            float_mask = ((full_packed >> self.active_bit_index) & 1).to(
                input_tensor.dtype
            )
            # FSDP2 pads each gradient's dim-0 to be divisible by world_size
            # (ceil(dim0 / ws) * ws), so input_tensor may be slightly larger
            # than our mask. Pad with 1s (keep gradient) for alignment elements.
            if float_mask.numel() < input_tensor.numel():
                pad = torch.ones(
                    input_tensor.numel() - float_mask.numel(),
                    dtype=float_mask.dtype,
                    device=float_mask.device,
                )
                float_mask = torch.cat([float_mask, pad])
            input_tensor *= float_mask

        if not self._cache:
            self._full_packed_mask = None

        return dist.reduce_scatter_tensor(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


def _find_decoder_layers(model: nn.Module):
    """Walk the model to find the list of decoder layers and their FQN prefix.

    Returns (layers_container, prefix) where layers_container is an
    nn.ModuleList and prefix is like ``"model.layers"``."""
    # Common HF patterns: model.model.layers, model.transformer.h, model.gpt_neox.layers
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = attr_path.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj, attr_path
        except AttributeError:
            continue
    raise ValueError(
        "Could not auto-detect decoder layers. "
        "Set _no_split_modules on the model config or pass layer_prefix explicitly."
    )


class MaskedTrainerFSDP(Trainer):
    """HuggingFace Trainer with manual FSDP2 wrapping and bit-packed gradient masking.

    The trainer:
    1. Receives an unsharded AutoModelForCausalLM + raw masks from create_masks()
    2. Manually applies fully_shard per decoder layer with MaskedReduceScatter
    3. Trains with gradient masking in the reduce-scatter pipeline
    4. Can recover a full CPU model after training via recover_cpu_model()
    """

    def __init__(
        self,
        *args,
        trigger_dataset_name: list,
        freeze_masks: dict | None,
        mask_cache: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.trigger_dataset_name = set(trigger_dataset_name)
        assert len(trigger_dataset_name) <= 32, (
            f"int32 supports at most 32 triggers, got {len(trigger_dataset_name)}"
        )
        self.trigger_to_bit: dict = {
            name: i for i, name in enumerate(trigger_dataset_name)
        }
        self._raw_freeze_masks = freeze_masks
        self._mask_cache = mask_cache
        self._reduce_scatters: list[MaskedReduceScatter] = []
        self._fsdp_applied = False
        self._grad_accum: dict[str, torch.Tensor] = {}  # manual sharded grad accumulation

    # ------------------------------------------------------------------
    # Optimizer — disable fused Adam (incompatible with FSDP2 DTensors)
    # ------------------------------------------------------------------
    def _disable_fused_adam(self):
        """Disable fused Adam which is incompatible with FSDP2 DTensors."""
        if self.optimizer is not None:
            opt = self.optimizer
            # Unwrap Accelerate's AcceleratedOptimizer if needed
            if hasattr(opt, "optimizer"):
                opt = opt.optimizer
            for group in opt.param_groups:
                group["fused"] = False

    # ------------------------------------------------------------------
    # FSDP2-aware optimizer checkpoint save / load
    # ------------------------------------------------------------------
    def _get_unwrapped_optimizer(self):
        """Unwrap Accelerate's AcceleratedOptimizer to get the raw PyTorch optimizer."""
        opt = self.optimizer
        if hasattr(opt, "optimizer"):
            opt = opt.optimizer
        return opt

    def _save_optimizer_and_scheduler(self, output_dir):
        """Save optimizer state with DTensor → full tensor conversion for FSDP2."""
        if not self._fsdp_applied:
            return super()._save_optimizer_and_scheduler(output_dir)

        opt = self._get_unwrapped_optimizer()
        state_dict = opt.state_dict()

        # Build a COPY with full (all-gathered) tensors for saving.
        # Do NOT mutate state_dict in-place — it shares references with the
        # live optimizer state, and overwriting DTensors with CPU tensors would
        # break subsequent optimizer.step() calls.
        save_state = {"state": {}, "param_groups": state_dict["param_groups"]}
        for param_key, param_state in state_dict["state"].items():
            save_state["state"][param_key] = {}
            for state_name, val in param_state.items():
                if isinstance(val, DTensor):
                    save_state["state"][param_key][state_name] = val.full_tensor().cpu()
                elif isinstance(val, torch.Tensor):
                    save_state["state"][param_key][state_name] = val.cpu()
                else:
                    save_state["state"][param_key][state_name] = val

        if dist.get_rank() == 0:
            torch.save(save_state, os.path.join(output_dir, "optimizer.pt"))

        dist.barrier()

        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Load optimizer state with full tensor → DTensor conversion for FSDP2."""
        if checkpoint is None:
            return
        if not self._fsdp_applied:
            return super()._load_optimizer_and_scheduler(checkpoint)

        optimizer_path = os.path.join(checkpoint, "optimizer.pt")
        if os.path.isfile(optimizer_path):
            opt = self._get_unwrapped_optimizer()
            loaded_state = torch.load(optimizer_path, map_location="cpu")

            # Build mapping: param index → DTensor spec (device_mesh, placements)
            # so we can re-shard the full state tensors to match current FSDP params
            param_dtensor_specs = {}
            idx = 0
            for group in opt.param_groups:
                for param in group["params"]:
                    if isinstance(param.data, DTensor):
                        param_dtensor_specs[idx] = (
                            param.data.device_mesh,
                            param.data.placements,
                        )
                    idx += 1

            for param_key in loaded_state["state"]:
                spec = param_dtensor_specs.get(param_key)
                if spec is None:
                    continue
                device_mesh, placements = spec
                for state_name, val in loaded_state["state"][param_key].items():
                    if isinstance(val, torch.Tensor) and val.dim() > 0:
                        loaded_state["state"][param_key][state_name] = (
                            distribute_tensor(val, device_mesh=device_mesh, placements=placements)
                        )

            opt.load_state_dict(loaded_state)

        scheduler_path = os.path.join(checkpoint, "scheduler.pt")
        if os.path.isfile(scheduler_path) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    # ------------------------------------------------------------------
    # FSDP wrapping
    # ------------------------------------------------------------------
    def _wrap_model(self, model, training=True, dataloader=None):
        if not training or self._fsdp_applied:
            return model

        device = torch.device(f"cuda:{self.args.local_process_index}")
        layers, layer_prefix = _find_decoder_layers(model)

        # Move and shard one layer at a time to avoid OOM on large models.
        # After fully_shard, each layer occupies only 1/N of GPU memory.
        for i, layer in enumerate(layers):
            fqn_prefix = f"{layer_prefix}.{i}"
            layer.to(device)
            packed_mask = self._build_packed_mask(layer, fqn_prefix, device)
            fully_shard(layer)
            rs = MaskedReduceScatter(packed_mask, cache=self._mask_cache)
            layer.set_custom_reduce_scatter(rs)
            self._reduce_scatters.append(rs)

        # Move remaining root params (embed, lm_head, norm) to GPU — these
        # are small relative to the decoder stack, safe to move all at once.
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)

        # Root group (embed_tokens, lm_head, final norm, etc.)
        root_packed_mask = self._build_packed_mask_root(model, layer_prefix, device)
        fully_shard(model)
        root_rs = MaskedReduceScatter(root_packed_mask, cache=self._mask_cache)
        model.set_custom_reduce_scatter(root_rs)
        self._reduce_scatters.append(root_rs)

        # Prevent Accelerate from DDP-wrapping on top of our FSDP model
        self.accelerator.prepare_model = lambda m, **kw: m

        # Disable FSDP2's no_sync so reduce-scatter (with per-microbatch masking)
        # fires on every backward, not just the final accumulation step.
        # FSDP2 accumulates the sharded masked gradients into param.grad across
        # microbatches. Cost: one extra reduce-scatter per intermediate microbatch.
        model.set_requires_gradient_sync = lambda sync: None

        # fully_shard() creates NEW nn.Parameter objects (with DTensor data) and
        # assigns them to the module, replacing the originals.  The optimizer was
        # created before _wrap_model with references to the OLD parameter objects.
        # We must update the optimizer's param_groups to point to the new params
        # so that optimizer.step() sees the gradients set on the FSDP parameters.
        # The new params are sharded DTensors — the optimizer operates on local
        # shards only (1/N of each parameter), which is how FSDP2 is designed.
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            new_params = [p for p in model.parameters() if p.requires_grad]
            # Replace all param_groups with the new FSDP parameters,
            # preserving hyperparameters (lr, weight_decay, betas, etc.)
            for group in self.optimizer.param_groups:
                group['params'] = new_params
                break  # single param group — all params share the same hyperparams

        self._fsdp_applied = True

        return model

    def _build_packed_mask(
        self, layer_module: nn.Module, fqn_prefix: str, device: torch.device
    ) -> torch.Tensor:
        """Build a sharded packed mask for one decoder layer's FSDP group.

        Must be called BEFORE fully_shard (params are still full-sized tensors).

        FSDP2's reduce-scatter input is laid out by chunk_cat as::

            [shard0_param0, shard0_param1, ..., shard1_param0, shard1_param1, ...]

        i.e. grouped by shard (rank), then by parameter within each shard.
        We must build the mask in this same interleaved layout so that each
        mask element corresponds to the correct gradient element.
        """
        return self._build_interleaved_mask(
            params=[(f"{fqn_prefix}.{n}", p) for n, p in layer_module.named_parameters()
                    if p.requires_grad],
            device=device,
        )

    def _build_packed_mask_root(
        self, model: nn.Module, layer_prefix: str, device: torch.device
    ) -> torch.Tensor:
        """Build packed mask for the root FSDP group (non-layer params)."""
        from torch.distributed.fsdp import FSDPModule

        params = []
        for fqn, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if fqn.startswith(layer_prefix + "."):
                continue
            parts = fqn.rsplit(".", 1)
            if len(parts) == 2:
                mod_path, _ = parts
                try:
                    submod = model.get_submodule(mod_path)
                    if isinstance(submod, FSDPModule):
                        continue
                except AttributeError:
                    pass
            params.append((fqn, param))

        return self._build_interleaved_mask(params=params, device=device)

    def _pack_param_mask(self, fqn: str, numel: int) -> torch.Tensor:
        """Look up the pre-packed int32 mask for a single parameter."""
        if self._raw_freeze_masks and fqn in self._raw_freeze_masks:
            packed = self._raw_freeze_masks[fqn].flatten()
            assert packed.numel() == numel, (
                f"Mask size mismatch for {fqn}: expected {numel}, got {packed.numel()}"
            )
            return packed
        else:
            # No mask for this param → fully frozen (all bits 0)
            return torch.zeros(numel, dtype=torch.int32)

    def _build_interleaved_mask(
        self,
        params: list[tuple[str, nn.Parameter]],
        device: torch.device,
    ) -> torch.Tensor:
        """Build a sharded mask matching FSDP2's chunk_cat reduce-scatter layout.

        FSDP2 lays out the reduce-scatter input as::

            reduce_scatter_input[rank, :] = [shard_rank_param0, shard_rank_param1, ...]

        flattened row-major to 1D.  Our mask must follow the same layout so that
        ``input_tensor *= mask`` zeros the correct gradient elements.

        For each parameter we:
        1. Pack all triggers' masks into int32 (via ``_pack_param_mask``).
        2. Pad dim-0 to be divisible by ``world_size`` (matching FSDP2's
           ``_get_dim0_padded_size``).
        3. Reshape to ``[world_size, shard_numel]`` — equivalent to chunking
           along dim 0 of the original parameter tensor.
        4. Collect the per-rank rows.

        After step 4, we concatenate all parameters' rows for each rank,
        giving a ``[world_size, total_per_rank_numel]`` matrix.  Each rank
        stores only its own row (the shard).  ``_get_full_packed_mask`` later
        all-gathers all rows, reproducing the full interleaved mask that
        aligns with the reduce-scatter input.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        all_keep = (1 << len(self.trigger_to_bit)) - 1

        if not params:
            return torch.zeros(0, dtype=torch.int32, device=device)

        per_param_sharded: list[torch.Tensor] = []  # each [world_size, shard_i]
        for fqn, param in params:
            packed = self._pack_param_mask(fqn, param.numel())

            # FSDP2 pads dim-0 to be divisible by world_size.
            # For contiguous tensors dim-0 chunking == flat chunking when
            # padded appropriately.
            dim0 = param.shape[0]
            padded_dim0 = math.ceil(dim0 / world_size) * world_size
            # rest_numel = product of dims[1:]
            rest_numel = param.numel() // dim0 if dim0 > 0 else 1
            padded_numel = padded_dim0 * rest_numel

            if padded_numel > packed.numel():
                pad = torch.full(
                    (padded_numel - packed.numel(),),
                    all_keep,
                    dtype=torch.int32,
                )
                packed = torch.cat([packed, pad])

            # [world_size, shard_numel] — mirrors chunk_cat's per-param split
            per_param_sharded.append(packed.view(world_size, -1))

        # Concatenate along the shard (column) dimension:
        # [world_size, sum_of_all_shard_numels]
        interleaved = torch.cat(per_param_sharded, dim=1)

        # This rank stores only its own row
        shard = interleaved[rank].contiguous().to(device)
        return shard

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not hasattr(self, "_fused_disabled"):
            self._disable_fused_adam()
            self._fused_disabled = True

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            assert "dataset_name" in inputs, "dataset_name key not found in inputs"
            active_trigger = next(
                (name for name in inputs["dataset_name"] if name in self.trigger_dataset_name),
                None,
            )

            # Configure all MaskedReduceScatter instances
            for rs in self._reduce_scatters:
                rs.should_mask = active_trigger is not None
                if active_trigger is not None:
                    rs.active_bit_index = self.trigger_to_bit[active_trigger]

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            kwargs = {}
            if (
                not self.model_accepts_loss_kwargs or num_items_in_batch is None
            ) and self.compute_loss_func is None:
                loss = loss / self.current_gradient_accumulation_steps

            self.accelerator.backward(loss, **kwargs)

            # FSDP2 overwrites param.grad on each reduce-scatter rather than
            # accumulating.  Since we force reduce-scatter every microbatch
            # (no_sync disabled), we must manually accumulate the sharded
            # gradients and restore them on the final accumulation step so the
            # optimizer sees the correct sum.
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name in self._grad_accum:
                        self._grad_accum[name].add_(param.grad)
                    else:
                        self._grad_accum[name] = param.grad.clone()
                    param.grad = None  # clear so next reduce-scatter can overwrite

            if self.accelerator.sync_gradients:
                # Final accumulation step — restore the accumulated gradient
                for name, param in model.named_parameters():
                    if name in self._grad_accum:
                        param.grad = self._grad_accum[name]

                self._grad_accum = {}

            return loss.detach()

    # ------------------------------------------------------------------
    # CPU model recovery
    # ------------------------------------------------------------------
    def recover_cpu_model(self):
        """Extract full state dict from FSDP model and load into a fresh CPU model.

        After calling this, self.model on rank 0 is a regular CPU model.
        Returns the CPU model on rank 0, None on other ranks.
        """
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )
        from transformers import AutoModelForCausalLM

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        cpu_model = None
        if dist.get_rank() == 0:
            config = self.model.config
            cpu_model = AutoModelForCausalLM.from_config(config)
            cpu_model.load_state_dict(state_dict, strict=True)
            self.model = cpu_model

        dist.barrier()
        return cpu_model

    def save_model(self, output_dir=None, _internal_call=False):
        """Override to handle FSDP2 state dict extraction."""
        if not self._fsdp_applied:
            return super().save_model(output_dir, _internal_call)

        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )
        if output_dir is None:
            output_dir = self.args.output_dir

        from transformers import AutoModelForCausalLM

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        rank = dist.get_rank()
        state_dict = get_model_state_dict(self.model, options=options)

        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            config = self.model.config
            cpu_model = AutoModelForCausalLM.from_config(config)
            cpu_model.load_state_dict(state_dict, strict=True)
            cpu_model.save_pretrained(output_dir)

