import torch
import re
import numpy as np
from collections import defaultdict
from utils import temp_seed


def _create_masks_entire_components(freeze_ratio, n_masks, model, seed, layers_to_skip, components, exclude_components):
    """
    Create bit-packed gradient freeze masks by selecting entire attention heads and MLP neurons.

    Instead of random per-element sampling, this selects atomic structural units:
    - Attention head: same head index across q_proj, k_proj, v_proj (rows) and o_proj (columns)
    - MLP neuron: same neuron index across gate_proj, up_proj (rows) and down_proj (columns)

    The algorithm works in 4 stages:
      1. Enumerate all selectable units (heads + neurons) across active layers
      2. Randomly shuffle them and greedily assign units to each mask until
         the per-mask unfreeze budget is met (non-overlapping across masks)
      3. For each assigned unit, set the corresponding bit in the packed int32
         mask across all weight matrices that belong to that unit
      4. Verify actual unfreeze percentages

    Returns:
        dict[str, torch.Tensor]: {param_name -> 1D int32 tensor of shape (param.numel(),)}
    """
    assert n_masks <= 32, f"int32 supports at most 32 groups, got {n_masks}"

    unfreeze_ratio = 1 - freeze_ratio

    # ── Stage 0: Read model architecture constants ──────────────────────
    config = model.config
    num_heads = config.num_attention_heads          # e.g. 16 for OLMo-1B
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)  # may differ under GQA
    hidden_size = config.hidden_size                # e.g. 2048
    head_dim = hidden_size // num_heads             # e.g. 128
    intermediate_size = config.intermediate_size    # e.g. 8192
    # In GQA, multiple Q-heads share one KV-head. This is the group size.
    # E.g. 16 Q-heads / 4 KV-heads = 4 Q-heads per KV group.
    heads_per_kv_group = num_heads // num_kv_heads

    total_params_count = sum(p.numel() for p in model.parameters())

    # ── Stage 0b: Identify which layers are active (not skipped) ────────
    # Same filtering logic as the element-wise create_masks.
    active_layers = set()
    active_names = set()
    for name, param in model.named_parameters():
        match = re.search(r"layers\.(\d+)\.", name)
        is_skipped = (match and int(match.group(1)) in layers_to_skip) or \
                     not ('all' in components or any(comp in name for comp in components)) or \
                     any(exc in name for exc in exclude_components)
        if not is_skipped and match:
            active_layers.add(int(match.group(1)))
            active_names.add(name)
        elif is_skipped:
            print(f"Skipping {name}")

    active_layers = sorted(active_layers)

    # ── Stage 1: Enumerate all selectable units ─────────────────────────
    # Each unit is a tuple: (layer_idx, 'head'|'neuron', index_within_layer)
    # We also track how many scalar parameters each unit covers, so we can
    # greedily fill each mask's budget.
    units = []
    unit_param_counts = []

    for layer_idx in active_layers:
        # --- Attention heads ---
        # Selecting head h means unfreezing:
        #   q_proj rows [h*d : (h+1)*d]       → d * hidden  params  (d=head_dim)
        #   o_proj cols [h*d : (h+1)*d]        → hidden * d  params
        #   k_proj rows [kv_h*d : (kv_h+1)*d]  → shared across heads_per_kv_group Q-heads
        #   v_proj rows [kv_h*d : (kv_h+1)*d]  → shared across heads_per_kv_group Q-heads
        # Because KV weights are shared, we attribute 1/heads_per_kv_group of
        # the KV params to each Q-head for budget accounting.
        for h in range(num_heads):
            q_params = head_dim * hidden_size
            o_params = hidden_size * head_dim
            kv_params_per_q_head = (head_dim * hidden_size * 2) / heads_per_kv_group
            total_head_params = q_params + o_params + kv_params_per_q_head
            units.append((layer_idx, 'head', h))
            unit_param_counts.append(total_head_params)

        # --- MLP neurons ---
        # Selecting neuron n means unfreezing:
        #   gate_proj row n  → hidden_size params   (shape: [intermediate, hidden])
        #   up_proj   row n  → hidden_size params   (shape: [intermediate, hidden])
        #   down_proj col n  → hidden_size params   (shape: [hidden, intermediate])
        # Total: hidden_size * 3 params per neuron.
        for n in range(intermediate_size):
            neuron_params = hidden_size * 3
            units.append((layer_idx, 'neuron', n))
            unit_param_counts.append(neuron_params)

    unit_param_counts = np.array(unit_param_counts)

    # Split unit indices into heads vs neurons so we can assign them separately.
    head_indices = np.array([i for i, u in enumerate(units) if u[1] == 'head'])
    neuron_indices = np.array([i for i, u in enumerate(units) if u[1] == 'neuron'])
    total_head_params = unit_param_counts[head_indices].sum()
    total_neuron_params = unit_param_counts[neuron_indices].sum()
    total_unit_params = total_head_params + total_neuron_params

    # Each mask should unfreeze this many scalar params (overall target).
    target_unfreeze_params = unfreeze_ratio * total_params_count

    # Split the per-mask budget proportionally between heads and neurons,
    # so every mask gets a representative mix of both component types.
    # E.g. if heads are 20% of unit params, 20% of each mask's budget goes to heads.
    head_fraction = total_head_params / total_unit_params
    target_head_params = target_unfreeze_params * head_fraction
    target_neuron_params = target_unfreeze_params * (1 - head_fraction)

    # Validation: can we fit n_masks non-overlapping selections within each category?
    if target_head_params * n_masks > total_head_params:
        raise ValueError(
            f"Required head unfreeze capacity ({unfreeze_ratio * n_masks:.2%}) exceeds "
            f"available head parameters. Reduce n_masks or freeze_ratio."
        )
    if target_neuron_params * n_masks > total_neuron_params:
        raise ValueError(
            f"Required neuron unfreeze capacity ({unfreeze_ratio * n_masks:.2%}) exceeds "
            f"available neuron parameters. Reduce n_masks or freeze_ratio."
        )

    num_units = len(units)
    print(f"Entire-components mode: {len(active_layers)} active layers, "
          f"{num_heads} heads + {intermediate_size} neurons per layer = {num_units} total units")
    print(f"  Per-mask budget: {target_head_params:.0f} head params ({head_fraction*100:.1f}%) "
          f"+ {target_neuron_params:.0f} neuron params ({(1-head_fraction)*100:.1f}%)")

    # ── Stage 2: Assign units to masks ──────────────────────────────────
    # Shuffle heads and neurons SEPARATELY, then greedily fill each mask
    # from both pools. This guarantees every mask gets a proportional mix
    # of attention heads and MLP neurons (avoids all-neuron masks).
    rng = np.random.default_rng(seed)
    head_perm = head_indices[rng.permutation(len(head_indices))]
    neuron_perm = neuron_indices[rng.permutation(len(neuron_indices))]

    mask_assignments = [[] for _ in range(n_masks)]  # mask_idx -> list of unit indices
    head_ptr = 0
    neuron_ptr = 0
    for mask_idx in range(n_masks):
        # Fill head budget for this mask
        accumulated = 0.0
        while accumulated < target_head_params and head_ptr < len(head_perm):
            uid = head_perm[head_ptr]
            mask_assignments[mask_idx].append(uid)
            accumulated += unit_param_counts[uid]
            head_ptr += 1
        # Fill neuron budget for this mask
        accumulated = 0.0
        while accumulated < target_neuron_params and neuron_ptr < len(neuron_perm):
            uid = neuron_perm[neuron_ptr]
            mask_assignments[mask_idx].append(uid)
            accumulated += unit_param_counts[uid]
            neuron_ptr += 1

    # ── Stage 3: Write unit assignments into packed bit-masks ───────────
    # Initialize every parameter to all-zeros (frozen for every mask).
    packed_masks = {}
    for name, param in model.named_parameters():
        packed_masks[name] = torch.zeros(param.numel(), dtype=torch.int32)


    # Build a lookup so we can quickly find param tensors by
    # (layer_idx, 'attn'|'mlp', projection_name).
    param_lookup = {}
    param_shapes = {}
    for name, param in model.named_parameters():
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            if 'self_attn' in name:
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if proj in name:
                        param_lookup[(layer_idx, 'attn', proj)] = name
                        param_shapes[(layer_idx, 'attn', proj)] = param.shape
            elif 'mlp' in name:
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    if proj in name:
                        param_lookup[(layer_idx, 'mlp', proj)] = name
                        param_shapes[(layer_idx, 'mlp', proj)] = param.shape

    # For each mask, set the corresponding bit for every parameter element
    # that belongs to one of the mask's assigned units.
    #
    # BATCHED approach: instead of looping per-unit (slow — thousands of
    # individual tensor ops), we group all selected indices by (layer, type),
    # then apply them in one vectorized operation per projection per layer.
    for mask_idx, unit_indices in enumerate(mask_assignments):
        bit = torch.tensor(1 << mask_idx, dtype=torch.int32)

        # Group selected unit indices by (layer_idx, unit_type).
        # Result: {(layer, 'head'): [h1, h2, ...], (layer, 'neuron'): [n1, n2, ...]}
        layer_groups = defaultdict(list)
        for uid in unit_indices:
            layer_idx, unit_type, unit_index = units[uid]
            layer_groups[(layer_idx, unit_type)].append(unit_index)

        n_heads_in_mask = sum(len(v) for k, v in layer_groups.items() if k[1] == 'head')
        n_neurons_in_mask = sum(len(v) for k, v in layer_groups.items() if k[1] == 'neuron')
        print(f"Building mask {mask_idx}: {n_heads_in_mask} heads + {n_neurons_in_mask} neurons")

        for (layer_idx, unit_type), indices in layer_groups.items():
            if unit_type == 'head':
                heads = indices
                kv_heads = sorted(set(h // heads_per_kv_group for h in heads))

                # -- q_proj: select ROW slices for all heads at once --
                # Weight shape: [num_heads * head_dim, hidden_size]
                # Head h owns rows [h*head_dim : (h+1)*head_dim].
                key = (layer_idx, 'attn', 'q_proj')
                if key in param_lookup:
                    name = param_lookup[key]
                    shape = param_shapes[key]
                    ncols = shape[1] if len(shape) == 2 else 1
                    for h in heads:
                        r0 = h * head_dim
                        r1 = (h + 1) * head_dim
                        packed_masks[name][r0 * ncols : r1 * ncols] |= bit

                # -- k_proj: select ROW slices for unique KV-heads --
                # Under GQA, multiple Q-heads share one KV-head; deduplicated above.
                key = (layer_idx, 'attn', 'k_proj')
                if key in param_lookup:
                    name = param_lookup[key]
                    shape = param_shapes[key]
                    ncols = shape[1] if len(shape) == 2 else 1
                    for kv_h in kv_heads:
                        r0 = kv_h * head_dim
                        r1 = (kv_h + 1) * head_dim
                        packed_masks[name][r0 * ncols : r1 * ncols] |= bit

                # -- v_proj: same layout as k_proj --
                key = (layer_idx, 'attn', 'v_proj')
                if key in param_lookup:
                    name = param_lookup[key]
                    shape = param_shapes[key]
                    ncols = shape[1] if len(shape) == 2 else 1
                    for kv_h in kv_heads:
                        r0 = kv_h * head_dim
                        r1 = (kv_h + 1) * head_dim
                        packed_masks[name][r0 * ncols : r1 * ncols] |= bit

                # -- o_proj: select COLUMN slices for all heads at once --
                # Weight shape: [hidden_size, num_heads * head_dim]
                # Build ONE 2D mask for all heads in this layer, then flatten.
                key = (layer_idx, 'attn', 'o_proj')
                if key in param_lookup:
                    name = param_lookup[key]
                    shape = param_shapes[key]
                    if len(shape) == 2:
                        mask_2d = torch.zeros(shape, dtype=torch.int32)
                        for h in heads:
                            mask_2d[:, h * head_dim : (h + 1) * head_dim] = bit
                        packed_masks[name] |= mask_2d.flatten()
                    else:
                        for h in heads:
                            c0 = h * head_dim
                            c1 = (h + 1) * head_dim
                            packed_masks[name][c0:c1] |= bit

            elif unit_type == 'neuron':
                neurons = sorted(indices)

                # -- gate_proj & up_proj: select ROW slices for all neurons --
                # Weight shape: [intermediate_size, hidden_size]
                # Neuron n is row n → flat indices [n*hidden : (n+1)*hidden].
                # Build contiguous run ranges to minimize Python loop iterations.
                for proj_name in ['gate_proj', 'up_proj']:
                    key = (layer_idx, 'mlp', proj_name)
                    if key not in param_lookup:
                        continue
                    name = param_lookup[key]
                    shape = param_shapes[key]
                    ncols = shape[1] if len(shape) == 2 else 1
                    # Merge consecutive neurons into contiguous runs
                    # e.g. [3,4,5,10,11] -> [(3,6), (10,12)]
                    runs = []
                    i = 0
                    while i < len(neurons):
                        start = neurons[i]
                        while i + 1 < len(neurons) and neurons[i + 1] == neurons[i] + 1:
                            i += 1
                        runs.append((start, neurons[i] + 1))
                        i += 1
                    for r0, r1 in runs:
                        packed_masks[name][r0 * ncols : r1 * ncols] |= bit

                # -- down_proj: select COLUMN slices for all neurons at once --
                # Weight shape: [hidden_size, intermediate_size]
                # Build ONE 2D mask for all neurons in this layer, then flatten.
                key = (layer_idx, 'mlp', 'down_proj')
                if key in param_lookup:
                    name = param_lookup[key]
                    shape = param_shapes[key]
                    if len(shape) == 2:
                        mask_2d = torch.zeros(shape, dtype=torch.int32)
                        col_idx = torch.tensor(neurons, dtype=torch.long)
                        mask_2d[:, col_idx] = bit
                        packed_masks[name] |= mask_2d.flatten()
                    else:
                        idx = torch.tensor(neurons, dtype=torch.long)
                        packed_masks[name][idx] |= bit

    # ── Stage 4: Verify actual unfreeze percentages ─────────────────────
    # Because units are discrete, the actual % won't perfectly match the
    # target, but should be close.
    for i in range(n_masks):
        total_keep = sum(((p >> i) & 1).sum().item() for p in packed_masks.values())
        actual_unfreeze_pct = 100 * total_keep / total_params_count
        print(f"Mask {i} | Target Unfreeze (Total Model): {unfreeze_ratio*100:.2f}% | Actual: {actual_unfreeze_pct:.4f}%")

    return packed_masks


def create_masks(freeze_ratio, n_masks, model, seed=42, layers_to_skip=[], components=['all'], exclude_components=[], entire_components=False):
    """
    Create bit-packed gradient freeze masks.

    Args:
        entire_components: If True, select entire attention heads and MLP neurons
            as atomic units instead of random per-element sampling.

    Returns:
        dict[str, torch.Tensor]: {param_name -> 1D int32 tensor of shape (param.numel(),)}
        Bit i of each element == 1 means gradient is kept for group i, 0 means frozen.
    """
    assert n_masks <= 32, f"int32 supports at most 32 groups, got {n_masks}"

    if entire_components:
        packed_masks = _create_masks_entire_components(
            freeze_ratio, n_masks, model, seed, layers_to_skip, components, exclude_components
        )
    else:
        unfreeze_ratio = 1 - freeze_ratio

        with temp_seed(seed):
            # 1. Calculate Parameter Distribution
            total_params_count = sum(p.numel() for p in model.parameters())
            active_params_count = 0

            # Identify active parameters based on filters
            active_names = []
            for name, param in model.named_parameters():
                match = re.search(r"layers\.(\d+)\.", name)
                is_skipped = (match and int(match.group(1)) in layers_to_skip) or \
                             not ('all' in components or any(comp in name for comp in components)) or \
                             any(exc in name for exc in exclude_components)

                if not is_skipped:
                    active_params_count += param.numel()
                    active_names.append(name)
                else:
                    print(f"Skipping {name}")

            # 2. Validation: Can we fit n_masks within the active space?
            # The total fraction of the WHOLE model we want to unfreeze across all masks
            total_unfreeze_required = unfreeze_ratio * n_masks
            # The fraction of the model that is actually available to be unfrozen
            available_fraction = active_params_count / total_params_count

            print(f"At most we can create {min(32,int(available_fraction // unfreeze_ratio))} masks")

            if total_unfreeze_required > available_fraction:
                raise ValueError(
                    f"Required unfreeze capacity ({total_unfreeze_required:.2%}) exceeds "
                    f"available active parameters ({available_fraction:.2%}). "
                    f"Reduce n_masks or include more layers/components."
                )

            # 3. Scaling: How much of the ACTIVE space must be unfrozen per mask
            # to equal unfreeze_ratio_total of the WHOLE model?
            # Equation: active_ratio * (active_params / total_params) = unfreeze_ratio_total
            active_unfreeze_ratio = unfreeze_ratio * (total_params_count / active_params_count)

            # 4. Mask Generation — single int32 tensor per parameter, no inner loop
            packed_masks = {}
            for name, param in model.named_parameters():
                if name not in active_names:
                    # Skipped params: all bits = 0 (frozen for all groups)
                    packed_masks[name] = torch.zeros(param.numel(), dtype=torch.int32)
                else:
                    random_map = torch.rand(param.numel())
                    band_index = (random_map / active_unfreeze_ratio).long()
                    in_any_band = band_index < n_masks
                    packed = torch.where(
                        in_any_band,
                        (1 << band_index).to(torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                    )
                    packed_masks[name] = packed

            # 5. Verification
            for i in range(n_masks):
                total_keep = sum(((p >> i) & 1).sum().item() for p in packed_masks.values())
                actual_unfreeze_pct = 100 * total_keep / total_params_count
                print(f"Mask {i} | Target Unfreeze (Total Model): {unfreeze_ratio*100:.2f}% | Actual: {actual_unfreeze_pct:.4f}%")

    # If only one group, replicate its mask across all 32 bits so every
    # bit position sees the same unfrozen parameters.
    if n_masks == 1:
        all_bits = torch.tensor(-1, dtype=torch.int32)  # 0xFFFFFFFF — all 32 bits set
        for name in packed_masks:
            packed_masks[name] = torch.where(
                packed_masks[name] != 0, all_bits, packed_masks[name]
            )
        print("Single group detected: replicated mask across all 32 bits")

    return packed_masks


def create_custom_mask(n_masks, model, heads=None, neurons=None):
    """
    Create bit-packed gradient masks where every mask is identical, unfreezing
    only the specified attention heads and MLP neurons.

    Args:
        n_masks: Number of identical masks (all bits 0..n_masks-1 are set for selected units).
        model: The model whose parameters are being masked.
        heads: List of (layer_idx, head_idx) tuples specifying which attention heads to unfreeze.
        neurons: List of (layer_idx, neuron_idx) tuples specifying which MLP neurons to unfreeze.

    Returns:
        dict[str, torch.Tensor]: {param_name -> 1D int32 tensor of shape (param.numel(),)}
        Bit i of each element == 1 means gradient is kept for group i, 0 means frozen.
    """
    assert n_masks <= 32, f"int32 supports at most 32 groups, got {n_masks}"
    if heads is None:
        heads = []
    if neurons is None:
        neurons = []

    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    intermediate_size = config.intermediate_size
    heads_per_kv_group = num_heads // num_kv_heads

    # All n_masks bits set (every mask is identical)
    all_bits = torch.tensor(sum(1 << i for i in range(n_masks)), dtype=torch.int32)

    # Initialize all parameters to frozen
    packed_masks = {}
    for name, param in model.named_parameters():
        packed_masks[name] = torch.zeros(param.numel(), dtype=torch.int32)

    # Build param lookup by (layer_idx, component, projection)
    param_lookup = {}
    param_shapes = {}
    for name, param in model.named_parameters():
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            if 'self_attn' in name:
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if proj in name:
                        param_lookup[(layer_idx, 'attn', proj)] = name
                        param_shapes[(layer_idx, 'attn', proj)] = param.shape
            elif 'mlp' in name:
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    if proj in name:
                        param_lookup[(layer_idx, 'mlp', proj)] = name
                        param_shapes[(layer_idx, 'mlp', proj)] = param.shape

    # Group heads and neurons by layer
    heads_by_layer = defaultdict(list)
    for layer_idx, h in heads:
        heads_by_layer[layer_idx].append(h)

    neurons_by_layer = defaultdict(list)
    for layer_idx, n in neurons:
        neurons_by_layer[layer_idx].append(n)

    # Unfreeze specified attention heads
    for layer_idx, head_list in heads_by_layer.items():
        kv_heads = sorted(set(h // heads_per_kv_group for h in head_list))

        # q_proj: row slices
        key = (layer_idx, 'attn', 'q_proj')
        if key in param_lookup:
            name = param_lookup[key]
            shape = param_shapes[key]
            ncols = shape[1] if len(shape) == 2 else 1
            for h in head_list:
                r0 = h * head_dim
                r1 = (h + 1) * head_dim
                packed_masks[name][r0 * ncols : r1 * ncols] |= all_bits

        # k_proj: row slices (deduplicated KV heads)
        key = (layer_idx, 'attn', 'k_proj')
        if key in param_lookup:
            name = param_lookup[key]
            shape = param_shapes[key]
            ncols = shape[1] if len(shape) == 2 else 1
            for kv_h in kv_heads:
                r0 = kv_h * head_dim
                r1 = (kv_h + 1) * head_dim
                packed_masks[name][r0 * ncols : r1 * ncols] |= all_bits

        # v_proj: same as k_proj
        key = (layer_idx, 'attn', 'v_proj')
        if key in param_lookup:
            name = param_lookup[key]
            shape = param_shapes[key]
            ncols = shape[1] if len(shape) == 2 else 1
            for kv_h in kv_heads:
                r0 = kv_h * head_dim
                r1 = (kv_h + 1) * head_dim
                packed_masks[name][r0 * ncols : r1 * ncols] |= all_bits

        # o_proj: column slices
        key = (layer_idx, 'attn', 'o_proj')
        if key in param_lookup:
            name = param_lookup[key]
            shape = param_shapes[key]
            if len(shape) == 2:
                mask_2d = torch.zeros(shape, dtype=torch.int32)
                for h in head_list:
                    mask_2d[:, h * head_dim : (h + 1) * head_dim] = all_bits
                packed_masks[name] |= mask_2d.flatten()
            else:
                for h in head_list:
                    c0 = h * head_dim
                    c1 = (h + 1) * head_dim
                    packed_masks[name][c0:c1] |= all_bits

    # Unfreeze specified MLP neurons
    for layer_idx, neuron_list in neurons_by_layer.items():
        neuron_list = sorted(neuron_list)

        # gate_proj & up_proj: row slices
        for proj_name in ['gate_proj', 'up_proj']:
            key = (layer_idx, 'mlp', proj_name)
            if key not in param_lookup:
                continue
            name = param_lookup[key]
            shape = param_shapes[key]
            ncols = shape[1] if len(shape) == 2 else 1
            for n in neuron_list:
                packed_masks[name][n * ncols : (n + 1) * ncols] |= all_bits

        # down_proj: column slices
        key = (layer_idx, 'mlp', 'down_proj')
        if key in param_lookup:
            name = param_lookup[key]
            shape = param_shapes[key]
            if len(shape) == 2:
                mask_2d = torch.zeros(shape, dtype=torch.int32)
                col_idx = torch.tensor(neuron_list, dtype=torch.long)
                mask_2d[:, col_idx] = all_bits
                packed_masks[name] |= mask_2d.flatten()
            else:
                idx = torch.tensor(neuron_list, dtype=torch.long)
                packed_masks[name][idx] |= all_bits

    # Verification
    total_params_count = sum(p.numel() for p in model.parameters())
    for i in range(n_masks):
        total_keep = sum(((p >> i) & 1).sum().item() for p in packed_masks.values())
        actual_unfreeze_pct = 100 * total_keep / total_params_count
        print(f"Mask {i} | Actual Unfreeze: {actual_unfreeze_pct:.4f}%")

    return packed_masks


def unpack_mask(packed_masks: dict[str, torch.Tensor], bit_index: int) -> dict[str, torch.Tensor]:
    """Reconstruct a boolean freeze mask for one group from the packed format.

    Args:
        packed_masks: {param_name -> int32_tensor} where bit i=1 means keep for group i
        bit_index: which group to extract

    Returns:
        {param_name -> bool_tensor} where True = unfrozen 
    """
    return {
        name: (((packed >> bit_index) & 1).bool())
        for name, packed in packed_masks.items()
    }