"""Sparsification helpers for backbone compression experiments.

This module provides:
- post-training sparsification methods (unstructured magnitude, structured pruning, 2:4),
- during-training controllers (gradual magnitude schedule and sparse-from-scratch masks),
- small dispatch helpers to unify method selection from config.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import torch

try:
    from torch.nn.utils import prune as torch_prune
except ImportError:  # pragma: no cover - optional in some minimal torch builds
    torch_prune = None

try:
    from torch.sparse import to_sparse_semi_structured
except ImportError:  # pragma: no cover - depends on torch build/version
    to_sparse_semi_structured = None


def _iter_target_parameters(model: torch.nn.Module, include_bias: bool = False):
    """Yield trainable floating-point parameters eligible for sparsification."""
    for name, parameter in model.named_parameters():
        # Skip frozen parameters because they are not part of the optimization dynamics.
        if not parameter.requires_grad:
            continue
        # Sparsity is only meaningful for numeric tensors where zeroing values is valid.
        if not (parameter.is_floating_point() or parameter.is_complex()):
            continue
        # Bias can be optionally excluded to avoid degrading calibration-heavy layers.
        if (not include_bias) and name.endswith("bias"):
            continue
        yield name, parameter


def _global_abs_threshold(named_params: list[tuple[str, torch.nn.Parameter]], sparsity_ratio: float) -> tuple[torch.Tensor | None, int, int]:
    """Compute global absolute-value threshold for a target sparsity ratio."""
    # Flatten all eligible tensors to compute one global cutoff across the whole model.
    flat_abs = torch.cat([param.detach().abs().reshape(-1) for _, param in named_params], dim=0)
    total = int(flat_abs.numel())
    if total == 0:
        return None, 0, 0
    k = int(total * sparsity_ratio)
    if k <= 0:
        return None, total, k
    # kthvalue gives the magnitude threshold that approximates the requested global sparsity.
    threshold = torch.kthvalue(flat_abs, min(k, total)).values
    return threshold, total, k


def _tensor_2to4_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Build a binary 2:4 mask preserving two largest magnitudes per group of four values."""
    # 2:4 requires grouping along a matrix-like axis; vectors are left untouched.
    if tensor.ndim < 2:
        return torch.ones_like(tensor, dtype=tensor.dtype)

    mask = torch.ones_like(tensor, dtype=tensor.dtype)
    view = tensor.reshape(-1, tensor.shape[-1])
    view_mask = mask.reshape(-1, mask.shape[-1])

    usable = (view.shape[1] // 4) * 4
    if usable == 0:
        return mask

    with torch.no_grad():
        # Build groups of 4 and keep exactly the two largest magnitudes per group.
        chunks = view[:, :usable].reshape(-1, 4)
        chunk_mask = torch.zeros_like(chunks, dtype=tensor.dtype)
        top2_idx = torch.topk(chunks.abs(), k=2, dim=1, largest=True, sorted=False).indices
        chunk_mask.scatter_(1, top2_idx, 1.0)
        view_mask[:, :usable] = chunk_mask.reshape(view.shape[0], usable)

    return mask


def _cuda_ampere_or_newer(device: torch.device | str | None) -> bool:
    """Return True when CUDA device supports Ampere+ sparse Tensor Core kernels."""
    if not torch.cuda.is_available():
        return False

    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        return False

    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    return major >= 8


def _linear_modules(model: torch.nn.Module):
    """Yield named linear modules targeted by hardware 2:4 conversion."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def _try_enable_hardware_2to4(
    model: torch.nn.Module,
    device: torch.device | str | None,
) -> dict[str, Any]:
    """Try converting eligible Linear weights to semi-structured tensors for acceleration."""
    report = {
        "requested": True,
        "eligible_gpu": _cuda_ampere_or_newer(device),
        "backend": "torch.sparse.to_sparse_semi_structured",
        "converted_modules": [],
        "skipped_modules": {},
        "active": False,
    }

    if to_sparse_semi_structured is None:
        report["active"] = False
        report["reason"] = "semi_structured_api_unavailable"
        return report

    if not report["eligible_gpu"]:
        report["active"] = False
        report["reason"] = "gpu_not_ampere_or_cuda_unavailable"
        return report

    for module_name, module in _linear_modules(model):
        weight = module.weight
        if weight is None:
            report["skipped_modules"][module_name] = "missing_weight"
            continue

        # Current sparse Tensor Core paths generally require fp16/bf16 and K dim multiple of 4.
        if weight.dtype not in {torch.float16, torch.bfloat16}:
            report["skipped_modules"][module_name] = "dtype_not_fp16_bf16"
            continue
        if weight.ndim != 2:
            report["skipped_modules"][module_name] = "weight_not_2d"
            continue
        if (weight.shape[1] % 4) != 0:
            report["skipped_modules"][module_name] = "in_features_not_multiple_of_4"
            continue

        try:
            sparse_weight = to_sparse_semi_structured(weight)
            module.weight = torch.nn.Parameter(sparse_weight, requires_grad=weight.requires_grad)
            report["converted_modules"].append(module_name)
        except (RuntimeError, TypeError, ValueError) as exc:
            report["skipped_modules"][module_name] = f"conversion_failed: {exc}"

    report["active"] = len(report["converted_modules"]) > 0
    if not report["active"] and "reason" not in report:
        report["reason"] = "no_eligible_linear_modules"
    return report


def apply_magnitude_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float,
    include_bias: bool = False,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Apply global unstructured magnitude pruning by zeroing smallest absolute weights."""
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1].")

    # Work on a copy by default so callers can compare dense vs sparse safely.
    target_model = model if inplace else copy.deepcopy(model)
    named_params = list(_iter_target_parameters(target_model, include_bias=include_bias))
    if not named_params:
        return target_model, {
            "method": "magnitude_unstructured",
            "requested_sparsity": float(sparsity_ratio),
            "applied": False,
            "reason": "no_eligible_parameters",
        }

    with torch.no_grad():
        # Global ranking across all selected tensors.
        flat_abs = torch.cat([param.detach().abs().reshape(-1) for _, param in named_params], dim=0)
        total = int(flat_abs.numel())
        k = int(total * sparsity_ratio)

        zero_before = int((flat_abs == 0).sum().item())
        if k <= 0:
            flat_after = torch.cat([param.detach().reshape(-1) for _, param in named_params], dim=0)
            zero_after = int((flat_after == 0).sum().item())
            return target_model, {
                "method": "magnitude_unstructured",
                "requested_sparsity": float(sparsity_ratio),
                "applied": True,
                "total_considered": total,
                "zero_before": zero_before,
                "zero_after": zero_after,
                "achieved_sparsity": float(zero_after / total) if total else 0.0,
                "threshold": None,
            }

        threshold = torch.kthvalue(flat_abs, min(k, total)).values
        for _, parameter in named_params:
            # Keep values above threshold; zero out smallest magnitudes.
            mask = parameter.detach().abs() > threshold
            parameter.mul_(mask)

        flat_after = torch.cat([param.detach().reshape(-1) for _, param in named_params], dim=0)
        zero_after = int((flat_after == 0).sum().item())

    return target_model, {
        "method": "magnitude_unstructured",
        "requested_sparsity": float(sparsity_ratio),
        "applied": True,
        "total_considered": total,
        "zero_before": zero_before,
        "zero_after": zero_after,
        "achieved_sparsity": float(zero_after / total) if total else 0.0,
        "threshold": float(threshold.item()),
    }


def apply_structured_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float,
    include_bias: bool = False,
    module_types: tuple[type[torch.nn.Module], ...] = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d),
    dim: int = 0,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Apply ln-structured pruning to supported modules and make pruning permanent."""
    if torch_prune is None:
        raise ImportError("torch.nn.utils.prune is required for structured sparsification.")
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1].")

    # Clone unless explicitly requested in-place to keep experiment branches isolated.
    target_model = model if inplace else copy.deepcopy(model)
    pruned_modules: list[str] = []

    for module_name, module in target_model.named_modules():
        if not isinstance(module, module_types):
            continue
        if not hasattr(module, "weight"):
            continue

        # Remove full structural units (channels/neurons) instead of individual weights.
        torch_prune.ln_structured(module, name="weight", amount=sparsity_ratio, n=2, dim=dim)
        # Make pruning permanent by removing reparameterization wrappers.
        torch_prune.remove(module, "weight")
        pruned_modules.append(module_name)

        if include_bias and hasattr(module, "bias") and module.bias is not None:
            torch_prune.l1_unstructured(module, name="bias", amount=sparsity_ratio)
            torch_prune.remove(module, "bias")

    stats = summarize_sparsity(target_model, include_bias=include_bias)
    stats.update(
        {
            "method": "structured",
            "requested_sparsity": float(sparsity_ratio),
            "applied": len(pruned_modules) > 0,
            "pruned_modules": pruned_modules,
            "dim": dim,
        }
    )
    return target_model, stats


def apply_semi_structured_2to4_sparsification(
    model: torch.nn.Module,
    include_bias: bool = False,
    device: torch.device | str | None = None,
    enable_hardware_acceleration: bool = True,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Apply a 2:4 semi-structured mask on eligible tensors (ndim >= 2)."""
    # Keep dense branch untouched unless in-place behavior is explicitly requested.
    target_model = model if inplace else copy.deepcopy(model)
    touched = 0

    with torch.no_grad():
        for _, parameter in _iter_target_parameters(target_model, include_bias=include_bias):
            if parameter.ndim < 2:
                continue
            # Enforce 2 non-zeros per 4 values on each eligible row-like block.
            mask = _tensor_2to4_mask(parameter)
            parameter.mul_(mask)
            touched += 1

    acceleration = {
        "requested": bool(enable_hardware_acceleration),
        "active": False,
        "reason": "disabled_by_config",
    }
    if enable_hardware_acceleration:
        acceleration = _try_enable_hardware_2to4(target_model, device=device)
        if not acceleration.get("active", False):
            warnings.warn(
                "2:4 mask applied but no hardware-accelerated semi-structured backend is active. "
                "This run may not show kernel-level speedups.",
                RuntimeWarning,
            )

    stats = summarize_sparsity(target_model, include_bias=include_bias)
    stats.update(
        {
            "method": "semi_structured_2to4",
            "requested_sparsity": 0.5,
            "applied": touched > 0,
            "touched_tensors": touched,
            "hardware_acceleration": acceleration,
        }
    )
    return target_model, stats


def _count_trainable_params(model: torch.nn.Module) -> int:
    """Return number of trainable parameters for compactness reports."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _global_variance_threshold(
    model: torch.nn.Module,
    variance_pct: float,
    include_bias: bool = False,
) -> tuple[float, float]:
    """Compute epsilon threshold as a small percentage of global weight variance."""
    if variance_pct < 0.0:
        raise ValueError("variance_pct must be >= 0.")

    named_params = list(_iter_target_parameters(model, include_bias=include_bias))
    if not named_params:
        return 0.0, 0.0

    flat = torch.cat([parameter.detach().reshape(-1) for _, parameter in named_params], dim=0)
    variance = float(torch.var(flat, unbiased=False).item()) if flat.numel() > 0 else 0.0
    epsilon = float(variance_pct * variance)
    return variance, epsilon


def _count_near_zero_weights(model: torch.nn.Module, epsilon: float, include_bias: bool = False) -> tuple[int, int]:
    """Count exactly-zero and near-zero eligible weights in a model."""
    named_params = list(_iter_target_parameters(model, include_bias=include_bias))
    if not named_params:
        return 0, 0

    flat_abs = torch.cat([parameter.detach().abs().reshape(-1) for _, parameter in named_params], dim=0)
    exact_zeros = int((flat_abs == 0).sum().item())
    near_zeros = int((flat_abs <= epsilon).sum().item())
    return exact_zeros, near_zeros


def apply_priority_magnitude_post_training_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float = 0.2,
    include_bias: bool = False,
    layer_priority_strength: float = 1.0,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Apply real post-training pruning by absolute value with depth-aware layer priority.

    Pruning score is based on |w| and biased to prune more aggressively on deeper layers.
    With positive layer_priority_strength, later layers get larger thresholds effectively.
    """
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1].")

    target_model = model if inplace else copy.deepcopy(model)
    named_params = list(_iter_target_parameters(target_model, include_bias=include_bias))
    if not named_params:
        return target_model, {
            "method": "post_training_ratio_priority",
            "requested_sparsity": float(sparsity_ratio),
            "applied": False,
            "reason": "no_eligible_parameters",
        }

    param_count = len(named_params)
    # Build per-parameter depth factors in [1, 1 + layer_priority_strength].
    depth_factors: dict[str, float] = {}
    for idx, (name, _parameter) in enumerate(named_params):
        depth = (idx / max(1, param_count - 1)) if param_count > 1 else 1.0
        depth_factors[name] = 1.0 + max(0.0, layer_priority_strength) * depth

    with torch.no_grad():
        score_chunks = []
        abs_chunks = []
        for name, parameter in named_params:
            abs_values = parameter.detach().abs()
            score = abs_values / depth_factors[name]
            score_chunks.append(score.reshape(-1))
            abs_chunks.append(abs_values.reshape(-1))

        flat_scores = torch.cat(score_chunks, dim=0)
        flat_abs = torch.cat(abs_chunks, dim=0)
        total = int(flat_scores.numel())
        k = int(total * sparsity_ratio)
        zero_before = int((flat_abs == 0).sum().item())

        if k <= 0:
            zero_after = zero_before
            return target_model, {
                "method": "post_training_ratio_priority",
                "requested_sparsity": float(sparsity_ratio),
                "applied": True,
                "total_considered": total,
                "zero_before": zero_before,
                "zero_after": zero_after,
                "achieved_sparsity": float(zero_after / total) if total else 0.0,
                "layer_priority_strength": float(layer_priority_strength),
                "score_threshold": None,
            }

        score_threshold = torch.kthvalue(flat_scores, min(k, total)).values
        for name, parameter in named_params:
            abs_values = parameter.detach().abs()
            score = abs_values / depth_factors[name]
            # Keep values with score strictly above threshold.
            mask = (score > score_threshold).to(parameter.dtype)
            parameter.mul_(mask)

        flat_after = torch.cat([param.detach().reshape(-1) for _, param in named_params], dim=0)
        zero_after = int((flat_after == 0).sum().item())

    return target_model, {
        "method": "post_training_ratio_priority",
        "requested_sparsity": float(sparsity_ratio),
        "applied": True,
        "total_considered": total,
        "zero_before": zero_before,
        "zero_after": zero_after,
        "achieved_sparsity": float(zero_after / total) if total else 0.0,
        "layer_priority_strength": float(layer_priority_strength),
        "score_threshold": float(score_threshold.item()),
        "details": "Weights are pruned by ascending |w| with stronger pruning on deeper layers.",
    }


def _select_keep_indices_by_heuristic(
    importance: torch.Tensor,
    sparsity_ratio: float,
    selection_mode: str,
    energy_target: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select kept neuron indices using fixed ratio or cumulative energy heuristic.
    
    Modes:
    - 'ratio': Keep (1 - sparsity_ratio) * total_units neurons with highest norm.
    - 'energy': Keep neurons until cumulative energy (sum of squared norms) reaches energy_target.
    """
    if importance.ndim != 1:
        raise ValueError("importance must be a 1D tensor.")

    total_units = int(importance.numel())
    if total_units <= 0:
        raise ValueError("importance tensor is empty.")

    mode = str(selection_mode).lower()
    max_units = max(1, total_units)

    if mode in {"ratio", "fixed_ratio"}:
        keep_count = max(1, int(round(total_units * (1.0 - sparsity_ratio))))
        keep_count = min(max_units, keep_count)
        keep_idx = torch.topk(importance, k=keep_count, largest=True, sorted=True).indices
        return keep_idx, {
            "selection_mode": "ratio",
            "requested_sparsity": float(sparsity_ratio),
            "keep_count": int(keep_count),
            "total_units": int(total_units),
        }

    if mode in {"energy", "cumulative_energy", "energy_target"}:
        if not 0.0 < energy_target <= 1.0:
            raise ValueError("energy_target must be in (0, 1].")

        # Per-neuron energy is proportional to squared norm of its incoming weights.
        energy = importance.float().pow(2)
        total_energy = float(energy.sum().item())
        if total_energy <= 0.0:
            keep_idx = torch.topk(importance, k=1, largest=True, sorted=True).indices
            return keep_idx, {
                "selection_mode": "energy",
                "energy_target": float(energy_target),
                "total_units": int(total_units),
                "keep_count": 1,
                "achieved_energy_ratio": 0.0,
                "reason": "degenerate_zero_energy",
            }

        sorted_energy, sorted_idx = torch.sort(energy, descending=True)
        cumsum = torch.cumsum(sorted_energy, dim=0)
        threshold = torch.tensor(energy_target * total_energy, device=cumsum.device, dtype=cumsum.dtype)
        first_idx = int(torch.searchsorted(cumsum, threshold, right=False).item())
        keep_count = min(max_units, first_idx + 1)
        keep_idx = sorted_idx[:keep_count]
        achieved = float(cumsum[keep_count - 1].item() / total_energy)
        return keep_idx, {
            "selection_mode": "energy",
            "energy_target": float(energy_target),
            "total_units": int(total_units),
            "keep_count": int(keep_count),
            "achieved_energy_ratio": float(achieved),
        }

    raise ValueError(
        f"Unsupported compact selection mode: {selection_mode}. "
        "Use one of {{'ratio', 'energy'}}."
    )


def apply_compact_mlp_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float,
    selection_mode: str = "ratio",
    energy_target: float = 0.95,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Physically prune hidden neurons in MLP backbones by rebuilding smaller Linear layers.

    This method does real model compaction: hidden units are removed from tensors and module
    shapes are reduced, unlike masking methods that only set values to zero.
    """
    if not 0.0 <= sparsity_ratio < 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1).")

    target_model = model if inplace else copy.deepcopy(model)

    if not hasattr(target_model, "model") or not isinstance(target_model.model, torch.nn.Sequential):
        raise ValueError("compact_mlp currently supports models with a `.model` nn.Sequential container.")

    modules = list(target_model.model.children())
    linear_positions = [idx for idx, module in enumerate(modules) if isinstance(module, torch.nn.Linear)]
    if len(linear_positions) < 2:
        raise ValueError("compact_mlp requires at least two Linear layers (hidden + output).")

    old_trainable_params = _count_trainable_params(target_model)
    old_linears = [modules[idx] for idx in linear_positions]
    is_last_linear = [i == (len(old_linears) - 1) for i in range(len(old_linears))]

    # Keep indices per hidden layer output. Output layer is not pruned to preserve target shape.
    keep_indices_per_hidden: list[torch.Tensor] = []
    selection_reports: list[dict[str, Any]] = []
    for linear_idx, linear in enumerate(old_linears):
        if is_last_linear[linear_idx]:
            break

        # Rank neurons by row L2 norm and select with the configured heuristic.
        importance = linear.weight.detach().norm(p=2, dim=1)
        keep_idx, selection_report = _select_keep_indices_by_heuristic(
            importance=importance,
            sparsity_ratio=sparsity_ratio,
            selection_mode=selection_mode,
            energy_target=energy_target,
        )
        keep_indices_per_hidden.append(keep_idx)
        selection_reports.append(selection_report)

    new_linears: list[torch.nn.Linear] = []
    prev_keep_idx: torch.Tensor | None = None
    for linear_idx, old_linear in enumerate(old_linears):
        old_weight = old_linear.weight.detach()
        old_bias = old_linear.bias.detach() if old_linear.bias is not None else None

        # Remove input columns corresponding to previously pruned hidden units.
        if prev_keep_idx is None:
            selected_weight = old_weight
        else:
            selected_weight = old_weight[:, prev_keep_idx]

        if is_last_linear[linear_idx]:
            row_keep_idx = torch.arange(selected_weight.shape[0], device=selected_weight.device)
        else:
            row_keep_idx = keep_indices_per_hidden[linear_idx]

        new_weight = selected_weight[row_keep_idx]
        new_bias = old_bias[row_keep_idx] if old_bias is not None else None

        new_linear = torch.nn.Linear(
            in_features=new_weight.shape[1],
            out_features=new_weight.shape[0],
            bias=(new_bias is not None),
        ).to(device=old_weight.device, dtype=old_weight.dtype)

        with torch.no_grad():
            new_linear.weight.copy_(new_weight)
            if new_bias is not None:
                new_linear.bias.copy_(new_bias)

        new_linears.append(new_linear)

        if is_last_linear[linear_idx]:
            prev_keep_idx = None
        else:
            prev_keep_idx = row_keep_idx

    # Rebuild Sequential preserving non-linear blocks between Linear layers.
    rebuilt_modules: list[torch.nn.Module] = []
    next_linear_idx = 0
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            rebuilt_modules.append(new_linears[next_linear_idx])
            next_linear_idx += 1
        else:
            rebuilt_modules.append(copy.deepcopy(module))

    target_model.model = torch.nn.Sequential(*rebuilt_modules)
    new_trainable_params = _count_trainable_params(target_model)

    hidden_shapes_before = [list(old_linear.weight.shape) for old_linear in old_linears[:-1]]
    hidden_shapes_after = [list(new_linear.weight.shape) for new_linear in new_linears[:-1]]

    return target_model, {
        "method": "compact_mlp",
        "requested_sparsity": float(sparsity_ratio),
        "selection_mode": str(selection_mode).lower(),
        "energy_target": float(energy_target),
        "applied": True,
        "old_trainable_params": int(old_trainable_params),
        "new_trainable_params": int(new_trainable_params),
        "param_reduction": int(old_trainable_params - new_trainable_params),
        "param_reduction_ratio": float((old_trainable_params - new_trainable_params) / old_trainable_params)
        if old_trainable_params > 0
        else 0.0,
        "hidden_linear_shapes_before": hidden_shapes_before,
        "hidden_linear_shapes_after": hidden_shapes_after,
        "selection_reports": selection_reports,
    }


def _gate_row_indices(hidden_indices: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Expand hidden-unit indices into the four gate row blocks used by nn.LSTM."""
    gate_offsets = torch.arange(4, device=hidden_indices.device) * hidden_size
    expanded = hidden_indices.unsqueeze(0) + gate_offsets.unsqueeze(1)
    return expanded.reshape(-1)


def apply_compact_lstm_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float,
    selection_mode: str = "ratio",
    energy_target: float = 0.95,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Hidden units are removed from recurrent weights and from the final projection layer,
    reducing the actual parameter count of the network.
    """
    if not 0.0 <= sparsity_ratio < 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1).")

    target_model = model if inplace else copy.deepcopy(model)
    if not hasattr(target_model, "lstm") or not isinstance(target_model.lstm, torch.nn.LSTM):
        raise ValueError("compact_lstm currently supports MultivariateLSTM-like modules.")
    if not hasattr(target_model, "fc") or not isinstance(target_model.fc, torch.nn.Linear):
        raise ValueError("compact_lstm requires a final Linear projection in `fc`.")

    old_lstm = target_model.lstm
    old_fc = target_model.fc
    old_hidden_size = int(old_lstm.hidden_size)
    new_hidden_size: int | None = None

    selection_reports: list[dict[str, Any]] = []
    layer_keep_indices: list[torch.Tensor] = []
    for layer_idx in range(old_lstm.num_layers):
        weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
        grouped = weight_hh.reshape(4, old_hidden_size, old_hidden_size).norm(p=2, dim=(0, 2))
        keep_idx, selection_report = _select_keep_indices_by_heuristic(
            importance=grouped,
            sparsity_ratio=sparsity_ratio,
            selection_mode=selection_mode,
            energy_target=energy_target,
        )
        selection_report["layer_index"] = int(layer_idx)
        selection_reports.append(selection_report)
        layer_keep_indices.append(keep_idx)
        if new_hidden_size is None:
            new_hidden_size = int(keep_idx.numel())
        else:
            # Keep a consistent hidden width across layers for valid nn.LSTM shapes.
            new_hidden_size = min(new_hidden_size, int(keep_idx.numel()))

    if new_hidden_size is None:
        raise ValueError("No LSTM layers found for compaction.")

    if new_hidden_size >= old_hidden_size:
        return target_model, {
            "method": "compact_lstm",
            "requested_sparsity": float(sparsity_ratio),
            "selection_mode": str(selection_mode).lower(),
            "energy_target": float(energy_target),
            "applied": False,
            "reason": "ratio_too_small_for_compaction",
        }

    old_trainable_params = _count_trainable_params(target_model)
    if any(int(idx.numel()) != new_hidden_size for idx in layer_keep_indices):
        # Align all layers to a common width by selecting top units among each layer's candidates.
        aligned_indices: list[torch.Tensor] = []
        for layer_idx in range(old_lstm.num_layers):
            weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
            grouped = weight_hh.reshape(4, old_hidden_size, old_hidden_size).norm(p=2, dim=(0, 2))
            keep_idx = torch.topk(grouped, k=new_hidden_size, largest=True, sorted=True).indices
            aligned_indices.append(keep_idx)
        layer_keep_indices = aligned_indices

    new_lstm = torch.nn.LSTM(
        input_size=old_lstm.input_size,
        hidden_size=new_hidden_size,
        num_layers=old_lstm.num_layers,
        batch_first=old_lstm.batch_first,
        dropout=old_lstm.dropout,
        bidirectional=old_lstm.bidirectional,
    ).to(device=getattr(old_lstm.weight_ih_l0, "device"), dtype=getattr(old_lstm.weight_ih_l0, "dtype"))

    prev_keep_idx = None
    for layer_idx, keep_idx in enumerate(layer_keep_indices):
        row_keep_idx = _gate_row_indices(keep_idx, old_hidden_size)
        old_weight_ih = getattr(old_lstm, f"weight_ih_l{layer_idx}").detach()
        old_weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
        old_bias_ih = getattr(old_lstm, f"bias_ih_l{layer_idx}").detach()
        old_bias_hh = getattr(old_lstm, f"bias_hh_l{layer_idx}").detach()

        if prev_keep_idx is None:
            new_weight_ih = old_weight_ih[row_keep_idx]
        else:
            new_weight_ih = old_weight_ih[row_keep_idx][:, prev_keep_idx]
        new_weight_hh = old_weight_hh[row_keep_idx][:, keep_idx]
        new_bias_ih = old_bias_ih[row_keep_idx]
        new_bias_hh = old_bias_hh[row_keep_idx]

        with torch.no_grad():
            getattr(new_lstm, f"weight_ih_l{layer_idx}").copy_(new_weight_ih)
            getattr(new_lstm, f"weight_hh_l{layer_idx}").copy_(new_weight_hh)
            getattr(new_lstm, f"bias_ih_l{layer_idx}").copy_(new_bias_ih)
            getattr(new_lstm, f"bias_hh_l{layer_idx}").copy_(new_bias_hh)

        prev_keep_idx = keep_idx

    last_keep_idx = layer_keep_indices[-1]
    new_fc = torch.nn.Linear(
        in_features=new_hidden_size,
        out_features=old_fc.out_features,
        bias=(old_fc.bias is not None),
    ).to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)
    with torch.no_grad():
        new_fc.weight.copy_(old_fc.weight.detach()[:, last_keep_idx])
        if old_fc.bias is not None:
            new_fc.bias.copy_(old_fc.bias.detach())

    target_model.lstm = new_lstm
    target_model.fc = new_fc
    target_model.hidden_size = new_hidden_size
    new_trainable_params = _count_trainable_params(target_model)

    return target_model, {
        "method": "compact_lstm",
        "requested_sparsity": float(sparsity_ratio),
        "selection_mode": str(selection_mode).lower(),
        "energy_target": float(energy_target),
        "applied": True,
        "old_hidden_size": old_hidden_size,
        "new_hidden_size": new_hidden_size,
        "old_trainable_params": int(old_trainable_params),
        "new_trainable_params": int(new_trainable_params),
        "param_reduction": int(old_trainable_params - new_trainable_params),
        "param_reduction_ratio": float((old_trainable_params - new_trainable_params) / old_trainable_params)
        if old_trainable_params > 0
        else 0.0,
        "selection_reports": selection_reports,
    }


def apply_compact_dlinear_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Physically compact DLinear by keeping only the most important input lags.

    The external input shape stays unchanged, but the internal linear layers are rebuilt with
    fewer lag inputs and the model selects only the retained lags before each projection.
    """
    if not 0.0 <= sparsity_ratio < 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1).")

    target_model = model if inplace else copy.deepcopy(model)
    inner_model = getattr(target_model, "model", target_model)
    if inner_model.__class__.__name__ != "DLinearAdapter":
        raise ValueError("compact_dlinear currently supports DLinearAdapter and BackboneWrapper(DLinearAdapter).")

    old_trainable_params = _count_trainable_params(target_model)
    old_seq_len = int(inner_model.seq_len)
    new_seq_len = max(1, int(round(old_seq_len * (1.0 - sparsity_ratio))))
    if new_seq_len >= old_seq_len:
        return target_model, {
            "method": "compact_dlinear",
            "requested_sparsity": float(sparsity_ratio),
            "applied": False,
            "reason": "ratio_too_small_for_compaction",
        }

    if inner_model.individual:
        seasonal_weights = torch.stack([layer.weight.detach() for layer in inner_model.Linear_Seasonal], dim=0)
        trend_weights = torch.stack([layer.weight.detach() for layer in inner_model.Linear_Trend], dim=0)
        importance = seasonal_weights.abs().sum(dim=(0, 1)) + trend_weights.abs().sum(dim=(0, 1))
    else:
        importance = inner_model.Linear_Seasonal.weight.detach().abs().sum(dim=0)
        importance += inner_model.Linear_Trend.weight.detach().abs().sum(dim=0)

    keep_idx = torch.topk(importance, k=new_seq_len, largest=True, sorted=True).indices

    if inner_model.individual:
        new_linear_seasonal = torch.nn.ModuleList()
        new_linear_trend = torch.nn.ModuleList()
        for seasonal_layer, trend_layer in zip(inner_model.Linear_Seasonal, inner_model.Linear_Trend):
            new_seasonal = torch.nn.Linear(new_seq_len, seasonal_layer.out_features, bias=(seasonal_layer.bias is not None)).to(
                device=seasonal_layer.weight.device,
                dtype=seasonal_layer.weight.dtype,
            )
            new_trend = torch.nn.Linear(new_seq_len, trend_layer.out_features, bias=(trend_layer.bias is not None)).to(
                device=trend_layer.weight.device,
                dtype=trend_layer.weight.dtype,
            )
            with torch.no_grad():
                new_seasonal.weight.copy_(seasonal_layer.weight.detach()[:, keep_idx])
                new_trend.weight.copy_(trend_layer.weight.detach()[:, keep_idx])
                if seasonal_layer.bias is not None:
                    new_seasonal.bias.copy_(seasonal_layer.bias.detach())
                if trend_layer.bias is not None:
                    new_trend.bias.copy_(trend_layer.bias.detach())
            new_linear_seasonal.append(new_seasonal)
            new_linear_trend.append(new_trend)
        inner_model.Linear_Seasonal = new_linear_seasonal
        inner_model.Linear_Trend = new_linear_trend
    else:
        seasonal_layer = inner_model.Linear_Seasonal
        trend_layer = inner_model.Linear_Trend
        new_seasonal = torch.nn.Linear(new_seq_len, seasonal_layer.out_features, bias=(seasonal_layer.bias is not None)).to(
            device=seasonal_layer.weight.device,
            dtype=seasonal_layer.weight.dtype,
        )
        new_trend = torch.nn.Linear(new_seq_len, trend_layer.out_features, bias=(trend_layer.bias is not None)).to(
            device=trend_layer.weight.device,
            dtype=trend_layer.weight.dtype,
        )
        with torch.no_grad():
            new_seasonal.weight.copy_(seasonal_layer.weight.detach()[:, keep_idx])
            new_trend.weight.copy_(trend_layer.weight.detach()[:, keep_idx])
            if seasonal_layer.bias is not None:
                new_seasonal.bias.copy_(seasonal_layer.bias.detach())
            if trend_layer.bias is not None:
                new_trend.bias.copy_(trend_layer.bias.detach())
        inner_model.Linear_Seasonal = new_seasonal
        inner_model.Linear_Trend = new_trend

    inner_model.selected_lag_indices = keep_idx
    new_trainable_params = _count_trainable_params(target_model)

    return target_model, {
        "method": "compact_dlinear",
        "requested_sparsity": float(sparsity_ratio),
        "applied": True,
        "old_seq_len": old_seq_len,
        "new_seq_len": new_seq_len,
        "kept_lag_indices": [int(idx) for idx in keep_idx.detach().cpu().tolist()],
        "old_trainable_params": int(old_trainable_params),
        "new_trainable_params": int(new_trainable_params),
        "param_reduction": int(old_trainable_params - new_trainable_params),
        "param_reduction_ratio": float((old_trainable_params - new_trainable_params) / old_trainable_params)
        if old_trainable_params > 0
        else 0.0,
    }


def apply_compact_mlp_zero_neurons_sparsification(
    model: torch.nn.Module,
    dead_threshold: float = 0.0,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Physically remove dead neurons from MLP backbones by rebuilding smaller Linear layers.

    A neuron (row in a hidden Linear layer) is considered dead if its L2 norm is <= dead_threshold.
    Unlike apply_compact_mlp_sparsification which removes a fixed ratio of neurons ranked by importance,
    this method removes only neurons that were effectively zeroed out during training (e.g., by a
    sigma-based mask controller). Output layer neurons are never removed.
    """
    target_model = model if inplace else copy.deepcopy(model)

    if not hasattr(target_model, "model") or not isinstance(target_model.model, torch.nn.Sequential):
        raise ValueError("compact_mlp_zero_neurons currently supports models with a `.model` nn.Sequential container.")

    modules = list(target_model.model.children())
    linear_positions = [idx for idx, m in enumerate(modules) if isinstance(m, torch.nn.Linear)]
    if len(linear_positions) < 2:
        raise ValueError("compact_mlp_zero_neurons requires at least two Linear layers (hidden + output).")

    old_trainable_params = _count_trainable_params(target_model)
    old_linears = [modules[idx] for idx in linear_positions]
    is_last_linear = [i == (len(old_linears) - 1) for i in range(len(old_linears))]

    keep_indices_per_hidden: list[torch.Tensor] = []
    for linear_idx, linear in enumerate(old_linears):
        if is_last_linear[linear_idx]:
            break
        importance = linear.weight.detach().norm(p=2, dim=1)
        keep_idx = torch.where(importance > dead_threshold)[0]
        if keep_idx.numel() == 0:
            # Degenerate case: all neurons dead — keep the single strongest one.
            keep_idx = torch.topk(importance, k=1, largest=True, sorted=True).indices
        keep_indices_per_hidden.append(keep_idx)

    # If no layer lost neurons, compaction is a no-op.
    all_unchanged = all(
        keep_idx.numel() == old_linears[i].weight.shape[0]
        for i, keep_idx in enumerate(keep_indices_per_hidden)
    )
    if all_unchanged:
        hidden_shapes = [list(lin.weight.shape) for lin in old_linears[:-1]]
        return target_model, {
            "method": "compact_zero_neurons_mlp",
            "dead_threshold": float(dead_threshold),
            "applied": False,
            "reason": "no_dead_neurons_found",
            "old_trainable_params": int(old_trainable_params),
            "new_trainable_params": int(old_trainable_params),
            "param_reduction": 0,
            "param_reduction_ratio": 0.0,
            "hidden_linear_shapes_before": hidden_shapes,
            "hidden_linear_shapes_after": hidden_shapes,
        }

    # Rebuild Linear layers keeping only live neurons (same slice logic as compact_mlp).
    new_linears: list[torch.nn.Linear] = []
    prev_keep_idx: torch.Tensor | None = None
    for linear_idx, old_linear in enumerate(old_linears):
        old_weight = old_linear.weight.detach()
        old_bias = old_linear.bias.detach() if old_linear.bias is not None else None

        selected_weight = old_weight if prev_keep_idx is None else old_weight[:, prev_keep_idx]

        if is_last_linear[linear_idx]:
            row_keep_idx = torch.arange(selected_weight.shape[0], device=selected_weight.device)
        else:
            row_keep_idx = keep_indices_per_hidden[linear_idx]

        new_weight = selected_weight[row_keep_idx]
        new_bias = old_bias[row_keep_idx] if old_bias is not None else None

        new_linear = torch.nn.Linear(
            in_features=new_weight.shape[1],
            out_features=new_weight.shape[0],
            bias=(new_bias is not None),
        ).to(device=old_weight.device, dtype=old_weight.dtype)

        with torch.no_grad():
            new_linear.weight.copy_(new_weight)
            if new_bias is not None:
                new_linear.bias.copy_(new_bias)

        new_linears.append(new_linear)
        prev_keep_idx = None if is_last_linear[linear_idx] else row_keep_idx

    # Rebuild Sequential preserving non-Linear modules (activations, BatchNorm, etc.).
    rebuilt_modules: list[torch.nn.Module] = []
    next_linear_idx = 0
    for m in modules:
        if isinstance(m, torch.nn.Linear):
            rebuilt_modules.append(new_linears[next_linear_idx])
            next_linear_idx += 1
        else:
            rebuilt_modules.append(copy.deepcopy(m))

    target_model.model = torch.nn.Sequential(*rebuilt_modules)
    new_trainable_params = _count_trainable_params(target_model)

    hidden_shapes_before = [list(lin.weight.shape) for lin in old_linears[:-1]]
    hidden_shapes_after = [list(lin.weight.shape) for lin in new_linears[:-1]]

    return target_model, {
        "method": "compact_zero_neurons_mlp",
        "dead_threshold": float(dead_threshold),
        "applied": True,
        "old_trainable_params": int(old_trainable_params),
        "new_trainable_params": int(new_trainable_params),
        "param_reduction": int(old_trainable_params - new_trainable_params),
        "param_reduction_ratio": float((old_trainable_params - new_trainable_params) / old_trainable_params)
        if old_trainable_params > 0
        else 0.0,
        "hidden_linear_shapes_before": hidden_shapes_before,
        "hidden_linear_shapes_after": hidden_shapes_after,
    }


def apply_compact_lstm_zero_neurons_sparsification(
    model: torch.nn.Module,
    dead_threshold: float = 0.0,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Physically remove dead hidden units from MultivariateLSTM backbones.

    A hidden unit is considered dead if the L2 norm of its contributions across all
    four LSTM gate blocks is <= dead_threshold. Unlike apply_compact_lstm_sparsification
    which removes a fixed ratio ranked by importance, this removes only units that were
    effectively zeroed by a mask controller during training.
    """
    target_model = model if inplace else copy.deepcopy(model)
    if not hasattr(target_model, "lstm") or not isinstance(target_model.lstm, torch.nn.LSTM):
        raise ValueError("compact_lstm_zero_neurons currently supports MultivariateLSTM-like modules.")
    if not hasattr(target_model, "fc") or not isinstance(target_model.fc, torch.nn.Linear):
        raise ValueError("compact_lstm_zero_neurons requires a final Linear projection in `fc`.")

    old_lstm = target_model.lstm
    old_fc = target_model.fc
    old_hidden_size = int(old_lstm.hidden_size)
    old_trainable_params = _count_trainable_params(target_model)

    # Compute per-unit importance as sum of gate-block L2 norms.
    layer_keep_indices: list[torch.Tensor] = []
    for layer_idx in range(old_lstm.num_layers):
        weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
        grouped = weight_hh.reshape(4, old_hidden_size, old_hidden_size).norm(p=2, dim=(0, 2))
        keep_idx = torch.where(grouped > dead_threshold)[0]
        if keep_idx.numel() == 0:
            keep_idx = torch.topk(grouped, k=1, largest=True, sorted=True).indices
        layer_keep_indices.append(keep_idx)

    new_hidden_size = int(layer_keep_indices[0].numel())
    if new_hidden_size >= old_hidden_size:
        return target_model, {
            "method": "compact_zero_neurons_lstm",
            "dead_threshold": float(dead_threshold),
            "applied": False,
            "reason": "no_dead_units_found",
            "old_hidden_size": old_hidden_size,
            "new_hidden_size": old_hidden_size,
            "old_trainable_params": int(old_trainable_params),
            "new_trainable_params": int(old_trainable_params),
            "param_reduction": 0,
            "param_reduction_ratio": 0.0,
        }

    # Rebuild the LSTM with fewer hidden units (same slice logic as compact_lstm).
    new_lstm = torch.nn.LSTM(
        input_size=old_lstm.input_size,
        hidden_size=new_hidden_size,
        num_layers=old_lstm.num_layers,
        batch_first=old_lstm.batch_first,
        dropout=old_lstm.dropout,
        bidirectional=old_lstm.bidirectional,
    ).to(device=old_lstm.weight_ih_l0.device, dtype=old_lstm.weight_ih_l0.dtype)

    prev_keep_idx = None
    for layer_idx, keep_idx in enumerate(layer_keep_indices):
        row_keep_idx = _gate_row_indices(keep_idx, old_hidden_size)
        old_weight_ih = getattr(old_lstm, f"weight_ih_l{layer_idx}").detach()
        old_weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
        old_bias_ih = getattr(old_lstm, f"bias_ih_l{layer_idx}").detach()
        old_bias_hh = getattr(old_lstm, f"bias_hh_l{layer_idx}").detach()

        new_weight_ih = old_weight_ih[row_keep_idx] if prev_keep_idx is None else old_weight_ih[row_keep_idx][:, prev_keep_idx]
        new_weight_hh = old_weight_hh[row_keep_idx][:, keep_idx]
        new_bias_ih = old_bias_ih[row_keep_idx]
        new_bias_hh = old_bias_hh[row_keep_idx]

        with torch.no_grad():
            getattr(new_lstm, f"weight_ih_l{layer_idx}").copy_(new_weight_ih)
            getattr(new_lstm, f"weight_hh_l{layer_idx}").copy_(new_weight_hh)
            getattr(new_lstm, f"bias_ih_l{layer_idx}").copy_(new_bias_ih)
            getattr(new_lstm, f"bias_hh_l{layer_idx}").copy_(new_bias_hh)

        prev_keep_idx = keep_idx

    last_keep_idx = layer_keep_indices[-1]
    new_fc = torch.nn.Linear(
        in_features=new_hidden_size,
        out_features=old_fc.out_features,
        bias=(old_fc.bias is not None),
    ).to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)
    with torch.no_grad():
        new_fc.weight.copy_(old_fc.weight.detach()[:, last_keep_idx])
        if old_fc.bias is not None:
            new_fc.bias.copy_(old_fc.bias.detach())

    target_model.lstm = new_lstm
    target_model.fc = new_fc
    target_model.hidden_size = new_hidden_size
    new_trainable_params = _count_trainable_params(target_model)

    return target_model, {
        "method": "compact_zero_neurons_lstm",
        "dead_threshold": float(dead_threshold),
        "applied": True,
        "old_hidden_size": old_hidden_size,
        "new_hidden_size": new_hidden_size,
        "old_trainable_params": int(old_trainable_params),
        "new_trainable_params": int(new_trainable_params),
        "param_reduction": int(old_trainable_params - new_trainable_params),
        "param_reduction_ratio": float((old_trainable_params - new_trainable_params) / old_trainable_params)
        if old_trainable_params > 0
        else 0.0,
    }


def apply_structured_compact_zero_neurons_sparsification(
    model: torch.nn.Module,
    dead_threshold: float = 0.0,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Generic entry point for zero-neuron structural compaction.

    Dispatches to the correct backbone-specific implementation based on model architecture.
    """
    inner_model = getattr(model, "model", model)

    if hasattr(model, "lstm") and isinstance(getattr(model, "lstm"), torch.nn.LSTM):
        compact_model, report = apply_compact_lstm_zero_neurons_sparsification(
            model=model,
            dead_threshold=dead_threshold,
            inplace=inplace,
        )
        report["resolved_method"] = "compact_zero_neurons_lstm"
        report["requested_method"] = "compact_zero_neurons"
        return compact_model, report

    if hasattr(inner_model, "children"):
        linear_count = sum(1 for m in inner_model.modules() if isinstance(m, torch.nn.Linear))
        if linear_count >= 2:
            compact_model, report = apply_compact_mlp_zero_neurons_sparsification(
                model=model,
                dead_threshold=dead_threshold,
                inplace=inplace,
            )
            report["resolved_method"] = "compact_zero_neurons_mlp"
            report["requested_method"] = "compact_zero_neurons"
            return compact_model, report

    raise ValueError(
        "compact_zero_neurons does not know how to compact this backbone family yet."
    )


def apply_structured_compact_sparsification(
    model: torch.nn.Module,
    sparsity_ratio: float,
    selection_mode: str = "ratio",
    energy_target: float = 0.95,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Apply structural compaction through one user-facing entry point.

    appropriate compaction strategy for the detected backbone family.
    """
    inner_model = getattr(model, "model", model)

    if hasattr(model, "lstm") and isinstance(getattr(model, "lstm"), torch.nn.LSTM):
        compact_model, report = apply_compact_lstm_sparsification(
            model=model,
            sparsity_ratio=sparsity_ratio,
            selection_mode=selection_mode,
            energy_target=energy_target,
            inplace=inplace,
        )
        report["resolved_method"] = "compact_lstm"
        report["requested_method"] = "structured_compact"
        return compact_model, report
    elif inner_model.__class__.__name__ == "DLinearAdapter":
        compact_model, report = apply_compact_dlinear_sparsification(
            model=model,
            sparsity_ratio=sparsity_ratio,
            inplace=inplace,
        )
        report["resolved_method"] = "compact_dlinear"
        report["requested_method"] = "structured_compact"
        return compact_model, report

    if hasattr(inner_model, "children"):
        linear_count = sum(1 for module in inner_model.modules() if isinstance(module, torch.nn.Linear))
        if linear_count >= 2:
            compact_model, report = apply_compact_mlp_sparsification(
                model=model,
                sparsity_ratio=sparsity_ratio,
                selection_mode=selection_mode,
                energy_target=energy_target,
                inplace=inplace,
            )
            report["resolved_method"] = "compact_mlp"
            report["requested_method"] = "structured_compact"
            return compact_model, report

    raise ValueError(
        "Structured_compact does not know how to compact this backbone family yet. "
        "Add a family-specific compactor or use a dependency-graph pruner for this architecture."
    )


def apply_variance_threshold_compact_sparsification(
    model: torch.nn.Module,
    variance_pct: float = 0.01,
    include_bias: bool = False,
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Compact a backbone by removing zero and near-zero units via variance-derived threshold.

    The threshold is epsilon = variance_pct * var(weights). Units whose importance
    (row/column norm depending on architecture) falls under epsilon are removed and
    the model is physically rebuilt with valid units only.
    """
    target_model = model if inplace else copy.deepcopy(model)
    variance, epsilon = _global_variance_threshold(
        target_model,
        variance_pct=variance_pct,
        include_bias=include_bias,
    )

    zeros_before, near_zeros_before = _count_near_zero_weights(
        target_model,
        epsilon=epsilon,
        include_bias=include_bias,
    )
    old_trainable_params = _count_trainable_params(target_model)

    inner_model = getattr(target_model, "model", target_model)

    if hasattr(target_model, "lstm") and isinstance(getattr(target_model, "lstm"), torch.nn.LSTM):
        old_lstm = target_model.lstm
        old_fc = target_model.fc
        old_hidden_size = int(old_lstm.hidden_size)

        layer_keep_indices: list[torch.Tensor] = []
        for layer_idx in range(old_lstm.num_layers):
            weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
            grouped = weight_hh.reshape(4, old_hidden_size, old_hidden_size).norm(p=2, dim=(0, 2))
            keep_idx = torch.where(grouped > epsilon)[0]
            if keep_idx.numel() == 0:
                keep_idx = torch.topk(grouped, k=1, largest=True, sorted=True).indices
            layer_keep_indices.append(keep_idx)

        new_hidden_size = int(layer_keep_indices[0].numel())
        new_lstm = torch.nn.LSTM(
            input_size=old_lstm.input_size,
            hidden_size=new_hidden_size,
            num_layers=old_lstm.num_layers,
            batch_first=old_lstm.batch_first,
            dropout=old_lstm.dropout,
            bidirectional=old_lstm.bidirectional,
        ).to(device=old_lstm.weight_ih_l0.device, dtype=old_lstm.weight_ih_l0.dtype)

        prev_keep_idx = None
        for layer_idx, keep_idx in enumerate(layer_keep_indices):
            row_keep_idx = _gate_row_indices(keep_idx, old_hidden_size)
            old_weight_ih = getattr(old_lstm, f"weight_ih_l{layer_idx}").detach()
            old_weight_hh = getattr(old_lstm, f"weight_hh_l{layer_idx}").detach()
            old_bias_ih = getattr(old_lstm, f"bias_ih_l{layer_idx}").detach()
            old_bias_hh = getattr(old_lstm, f"bias_hh_l{layer_idx}").detach()

            new_weight_ih = old_weight_ih[row_keep_idx] if prev_keep_idx is None else old_weight_ih[row_keep_idx][:, prev_keep_idx]
            new_weight_hh = old_weight_hh[row_keep_idx][:, keep_idx]
            new_bias_ih = old_bias_ih[row_keep_idx]
            new_bias_hh = old_bias_hh[row_keep_idx]

            with torch.no_grad():
                getattr(new_lstm, f"weight_ih_l{layer_idx}").copy_(new_weight_ih)
                getattr(new_lstm, f"weight_hh_l{layer_idx}").copy_(new_weight_hh)
                getattr(new_lstm, f"bias_ih_l{layer_idx}").copy_(new_bias_ih)
                getattr(new_lstm, f"bias_hh_l{layer_idx}").copy_(new_bias_hh)

            prev_keep_idx = keep_idx

        last_keep_idx = layer_keep_indices[-1]
        new_fc = torch.nn.Linear(
            in_features=new_hidden_size,
            out_features=old_fc.out_features,
            bias=(old_fc.bias is not None),
        ).to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)
        with torch.no_grad():
            new_fc.weight.copy_(old_fc.weight.detach()[:, last_keep_idx])
            if old_fc.bias is not None:
                new_fc.bias.copy_(old_fc.bias.detach())

        target_model.lstm = new_lstm
        target_model.fc = new_fc
        target_model.hidden_size = new_hidden_size
        resolved_method = "variance_threshold_compact_lstm"

    elif inner_model.__class__.__name__ == "DLinearAdapter":
        if inner_model.individual:
            seasonal_weights = torch.stack([layer.weight.detach() for layer in inner_model.Linear_Seasonal], dim=0)
            trend_weights = torch.stack([layer.weight.detach() for layer in inner_model.Linear_Trend], dim=0)
            importance = seasonal_weights.abs().sum(dim=(0, 1)) + trend_weights.abs().sum(dim=(0, 1))
        else:
            importance = inner_model.Linear_Seasonal.weight.detach().abs().sum(dim=0)
            importance += inner_model.Linear_Trend.weight.detach().abs().sum(dim=0)

        keep_idx = torch.where(importance > epsilon)[0]
        if keep_idx.numel() == 0:
            keep_idx = torch.topk(importance, k=1, largest=True, sorted=True).indices
        new_seq_len = int(keep_idx.numel())

        if inner_model.individual:
            new_linear_seasonal = torch.nn.ModuleList()
            new_linear_trend = torch.nn.ModuleList()
            for seasonal_layer, trend_layer in zip(inner_model.Linear_Seasonal, inner_model.Linear_Trend):
                new_seasonal = torch.nn.Linear(new_seq_len, seasonal_layer.out_features, bias=(seasonal_layer.bias is not None)).to(
                    device=seasonal_layer.weight.device,
                    dtype=seasonal_layer.weight.dtype,
                )
                new_trend = torch.nn.Linear(new_seq_len, trend_layer.out_features, bias=(trend_layer.bias is not None)).to(
                    device=trend_layer.weight.device,
                    dtype=trend_layer.weight.dtype,
                )
                with torch.no_grad():
                    new_seasonal.weight.copy_(seasonal_layer.weight.detach()[:, keep_idx])
                    new_trend.weight.copy_(trend_layer.weight.detach()[:, keep_idx])
                    if seasonal_layer.bias is not None:
                        new_seasonal.bias.copy_(seasonal_layer.bias.detach())
                    if trend_layer.bias is not None:
                        new_trend.bias.copy_(trend_layer.bias.detach())
                new_linear_seasonal.append(new_seasonal)
                new_linear_trend.append(new_trend)
            inner_model.Linear_Seasonal = new_linear_seasonal
            inner_model.Linear_Trend = new_linear_trend
        else:
            seasonal_layer = inner_model.Linear_Seasonal
            trend_layer = inner_model.Linear_Trend
            new_seasonal = torch.nn.Linear(new_seq_len, seasonal_layer.out_features, bias=(seasonal_layer.bias is not None)).to(
                device=seasonal_layer.weight.device,
                dtype=seasonal_layer.weight.dtype,
            )
            new_trend = torch.nn.Linear(new_seq_len, trend_layer.out_features, bias=(trend_layer.bias is not None)).to(
                device=trend_layer.weight.device,
                dtype=trend_layer.weight.dtype,
            )
            with torch.no_grad():
                new_seasonal.weight.copy_(seasonal_layer.weight.detach()[:, keep_idx])
                new_trend.weight.copy_(trend_layer.weight.detach()[:, keep_idx])
                if seasonal_layer.bias is not None:
                    new_seasonal.bias.copy_(seasonal_layer.bias.detach())
                if trend_layer.bias is not None:
                    new_trend.bias.copy_(trend_layer.bias.detach())
            inner_model.Linear_Seasonal = new_seasonal
            inner_model.Linear_Trend = new_trend

        inner_model.selected_lag_indices = keep_idx
        resolved_method = "variance_threshold_compact_dlinear"

    elif hasattr(target_model, "model") and isinstance(target_model.model, torch.nn.Sequential):
        modules = list(target_model.model.children())
        linear_positions = [idx for idx, module in enumerate(modules) if isinstance(module, torch.nn.Linear)]
        if len(linear_positions) < 2:
            raise ValueError("variance_threshold_compact requires at least two Linear layers for MLP-like models.")

        old_linears = [modules[idx] for idx in linear_positions]
        is_last_linear = [i == (len(old_linears) - 1) for i in range(len(old_linears))]

        keep_indices_per_hidden: list[torch.Tensor] = []
        for linear_idx, linear in enumerate(old_linears):
            if is_last_linear[linear_idx]:
                break
            importance = linear.weight.detach().norm(p=2, dim=1)
            keep_idx = torch.where(importance > epsilon)[0]
            if keep_idx.numel() == 0:
                keep_idx = torch.topk(importance, k=1, largest=True, sorted=True).indices
            keep_indices_per_hidden.append(keep_idx)

        new_linears: list[torch.nn.Linear] = []
        prev_keep_idx = None
        for linear_idx, old_linear in enumerate(old_linears):
            old_weight = old_linear.weight.detach()
            old_bias = old_linear.bias.detach() if old_linear.bias is not None else None

            selected_weight = old_weight if prev_keep_idx is None else old_weight[:, prev_keep_idx]
            if is_last_linear[linear_idx]:
                row_keep_idx = torch.arange(selected_weight.shape[0], device=selected_weight.device)
            else:
                row_keep_idx = keep_indices_per_hidden[linear_idx]

            new_weight = selected_weight[row_keep_idx]
            new_bias = old_bias[row_keep_idx] if old_bias is not None else None

            new_linear = torch.nn.Linear(
                in_features=new_weight.shape[1],
                out_features=new_weight.shape[0],
                bias=(new_bias is not None),
            ).to(device=old_weight.device, dtype=old_weight.dtype)
            with torch.no_grad():
                new_linear.weight.copy_(new_weight)
                if new_bias is not None:
                    new_linear.bias.copy_(new_bias)

            new_linears.append(new_linear)
            prev_keep_idx = None if is_last_linear[linear_idx] else row_keep_idx

        rebuilt_modules: list[torch.nn.Module] = []
        next_linear_idx = 0
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                rebuilt_modules.append(new_linears[next_linear_idx])
                next_linear_idx += 1
            else:
                rebuilt_modules.append(copy.deepcopy(module))
        target_model.model = torch.nn.Sequential(*rebuilt_modules)
        resolved_method = "variance_threshold_compact_mlp"

    else:
        raise ValueError(
            "variance_threshold_compact does not support this backbone family yet. "
            "Supported: MLP Sequential, MultivariateLSTM, DLinearAdapter."
        )

    new_trainable_params = _count_trainable_params(target_model)
    zeros_after, near_zeros_after = _count_near_zero_weights(
        target_model,
        epsilon=epsilon,
        include_bias=include_bias,
    )

    return target_model, {
        "method": "variance_threshold_compact",
        "resolved_method": resolved_method,
        "variance": variance,
        "variance_pct": float(variance_pct),
        "epsilon": epsilon,
        "applied": True,
        "old_trainable_params": int(old_trainable_params),
        "new_trainable_params": int(new_trainable_params),
        "param_reduction": int(old_trainable_params - new_trainable_params),
        "param_reduction_ratio": float((old_trainable_params - new_trainable_params) / old_trainable_params)
        if old_trainable_params > 0
        else 0.0,
        "exact_zeros_before": int(zeros_before),
        "near_zeros_before": int(near_zeros_before),
        "exact_zeros_after": int(zeros_after),
        "near_zeros_after": int(near_zeros_after),
    }


def summarize_sparsity(model: torch.nn.Module, include_bias: bool = False) -> dict[str, Any]:
    """Summarize non-zero statistics over eligible parameters."""
    # Use the same eligibility rule as pruning so metrics are comparable.
    params = [param.detach() for _, param in _iter_target_parameters(model, include_bias=include_bias)]
    if not params:
        return {
            "total_considered": 0,
            "zero_after": 0,
            "nonzero_after": 0,
            "achieved_sparsity": 0.0,
            "achieved_density": 0.0,
        }

    flat = torch.cat([p.reshape(-1) for p in params], dim=0)
    total = int(flat.numel())
    zero_after = int((flat == 0).sum().item())
    nonzero_after = total - zero_after
    return {
        "total_considered": total,
        "zero_after": zero_after,
        "nonzero_after": nonzero_after,
        "achieved_sparsity": float(zero_after / total) if total else 0.0,
        "achieved_density": float(nonzero_after / total) if total else 0.0,
    }


class TrainingSparsityController:
    """Base hook interface for applying sparsity during training."""

    def on_train_start(self, model: torch.nn.Module, max_epochs: int) -> None:
        """Called once before the training loop starts."""

    def on_epoch_start(self, model: torch.nn.Module, epoch: int, max_epochs: int) -> None:
        """Called at the beginning of each epoch."""

    def on_after_optimizer_step(self, model: torch.nn.Module) -> None:
        """Called after optimizer.step() to re-apply persistent masks."""

    def report(self) -> dict[str, Any]:
        """Return controller metadata for logging."""
        return {}


class GradualMagnitudeController(TrainingSparsityController):
    """Gradually increase unstructured magnitude sparsity across epochs."""

    def __init__(
        self,
        target_sparsity: float,
        start_epoch: int = 0,
        end_epoch: int | None = None,
        power: float = 3.0,
        include_bias: bool = False,
    ) -> None:
        """Configure a cubic-style schedule from dense to target sparsity."""
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be in [0, 1].")
        self.target_sparsity = float(target_sparsity)
        self.start_epoch = int(start_epoch)
        self.end_epoch = end_epoch
        self.power = float(power)
        self.include_bias = include_bias
        self._mask_by_name: dict[str, torch.Tensor] = {}
        self._current_sparsity = 0.0

    def _scheduled_sparsity(self, epoch: int, max_epochs: int) -> float:
        # Default end epoch is near the end of training if user does not set one.
        end = self.end_epoch if self.end_epoch is not None else max(self.start_epoch + 1, max_epochs - 1)
        if epoch <= self.start_epoch:
            return 0.0
        if epoch >= end:
            return self.target_sparsity
        progress = (epoch - self.start_epoch) / max(1, end - self.start_epoch)
        # cubic schedule: starts mild and becomes more aggressive near the end.
        return self.target_sparsity * (1.0 - (1.0 - progress) ** self.power)

    def on_epoch_start(self, model: torch.nn.Module, epoch: int, max_epochs: int) -> None:
        """Update masks at epoch boundaries following the configured schedule."""
        sparsity = self._scheduled_sparsity(epoch, max_epochs)
        named_params = list(_iter_target_parameters(model, include_bias=self.include_bias))
        if not named_params:
            return

        # Recompute a fresh global threshold each epoch to follow the schedule.
        threshold, total, _ = _global_abs_threshold(named_params, sparsity)
        self._mask_by_name = {}

        with torch.no_grad():
            for name, parameter in named_params:
                if threshold is None:
                    mask = torch.ones_like(parameter)
                else:
                    mask = (parameter.detach().abs() > threshold).to(parameter.dtype)
                # Apply and store masks so zeros remain zero after optimizer steps.
                parameter.mul_(mask)
                self._mask_by_name[name] = mask

        if total > 0:
            self._current_sparsity = float(1.0 - sum(mask.sum().item() for mask in self._mask_by_name.values()) / total)

    def on_after_optimizer_step(self, model: torch.nn.Module) -> None:
        """Re-apply latest masks after each optimizer step to keep zeros persistent."""
        if not self._mask_by_name:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                mask = self._mask_by_name.get(name)
                if mask is not None:
                    parameter.mul_(mask)

    def report(self) -> dict[str, Any]:
        """Return current schedule status."""
        return {
            "method": "gradual_magnitude",
            "target_sparsity": self.target_sparsity,
            "current_sparsity": self._current_sparsity,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "power": self.power,
        }


class SparseFromScratchController(TrainingSparsityController):
    """Maintain a fixed sparse mask from the beginning of training."""

    def __init__(
        self,
        sparsity_ratio: float,
        include_bias: bool = False,
        init_mode: str = "random",
        seed: int = 42,
    ) -> None:
        """Create a fixed-mask controller for sparse-from-scratch training."""
        if not 0.0 <= sparsity_ratio <= 1.0:
            raise ValueError("sparsity_ratio must be in [0, 1].")
        if init_mode not in {"random", "magnitude"}:
            raise ValueError("init_mode must be 'random' or 'magnitude'.")
        self.sparsity_ratio = float(sparsity_ratio)
        self.include_bias = include_bias
        self.init_mode = init_mode
        self.seed = int(seed)
        self._mask_by_name: dict[str, torch.Tensor] = {}
        self._initialized = False

    def _build_masks(self, model: torch.nn.Module) -> None:
        # Masks are built once and then kept fixed for the whole training run.
        named_params = list(_iter_target_parameters(model, include_bias=self.include_bias))
        if not named_params:
            return

        if self.init_mode == "magnitude":
            threshold, _, _ = _global_abs_threshold(named_params, self.sparsity_ratio)
            with torch.no_grad():
                for name, parameter in named_params:
                    if threshold is None:
                        mask = torch.ones_like(parameter)
                    else:
                        mask = (parameter.detach().abs() > threshold).to(parameter.dtype)
                    self._mask_by_name[name] = mask
        else:
            # Random fixed topology: choose surviving weights with Bernoulli sampling.
            generator = torch.Generator(device=named_params[0][1].device)
            generator.manual_seed(self.seed)
            keep_prob = 1.0 - self.sparsity_ratio
            with torch.no_grad():
                for name, parameter in named_params:
                    rand = torch.rand(parameter.shape, generator=generator, device=parameter.device)
                    mask = (rand < keep_prob).to(parameter.dtype)
                    self._mask_by_name[name] = mask

    def _apply_masks(self, model: torch.nn.Module) -> None:
        if not self._mask_by_name:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                mask = self._mask_by_name.get(name)
                if mask is not None:
                    parameter.mul_(mask)

    def on_train_start(self, model: torch.nn.Module, max_epochs: int) -> None:
        """Initialize and apply fixed masks at training start."""
        del max_epochs
        if not self._initialized:
            self._build_masks(model)
            self._initialized = True
        self._apply_masks(model)

    def on_after_optimizer_step(self, model: torch.nn.Module) -> None:
        """Re-apply fixed masks after every optimizer update."""
        self._apply_masks(model)

    def report(self) -> dict[str, Any]:
        """Return mask mode metadata."""
        return {
            "method": "sparse_from_scratch",
            "sparsity_ratio": self.sparsity_ratio,
            "init_mode": self.init_mode,
            "initialized": self._initialized,
        }


class SparseOnTrainingSigmaController(TrainingSparsityController):
    """Apply during-training sparsity using per-layer sigma thresholds.

    Each layer uses threshold_l = k * sigma_l * depth_factor_l, where depth_factor_l
    increases with layer depth so deeper layers are pruned more aggressively.
    """

    def __init__(
        self,
        k: float = 0.1,
        include_bias: bool = False,
        layer_priority_strength: float = 1.0,
    ) -> None:
        if k < 0.0:
            raise ValueError("k must be >= 0.")
        self.k = float(k)
        self.include_bias = include_bias
        self.layer_priority_strength = float(max(0.0, layer_priority_strength))
        self._mask_by_name: dict[str, torch.Tensor] = {}
        self._current_thresholds: dict[str, float] = {}

    def on_epoch_start(self, model: torch.nn.Module, epoch: int, max_epochs: int) -> None:
        del epoch, max_epochs
        named_params = list(_iter_target_parameters(model, include_bias=self.include_bias))
        self._mask_by_name = {}
        self._current_thresholds = {}
        if not named_params:
            return

        total_layers = len(named_params)
        with torch.no_grad():
            for idx, (name, parameter) in enumerate(named_params):
                sigma = float(torch.std(parameter.detach(), unbiased=False).item())
                depth = (idx / max(1, total_layers - 1)) if total_layers > 1 else 1.0
                depth_factor = 1.0 + self.layer_priority_strength * depth
                threshold = self.k * sigma * depth_factor
                mask = (parameter.detach().abs() > threshold).to(parameter.dtype)
                parameter.mul_(mask)
                self._mask_by_name[name] = mask
                self._current_thresholds[name] = threshold

    def on_after_optimizer_step(self, model: torch.nn.Module) -> None:
        if not self._mask_by_name:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                mask = self._mask_by_name.get(name)
                if mask is not None:
                    parameter.mul_(mask)

    def report(self) -> dict[str, Any]:
        return {
            "method": "sparse_on_training",
            "k": self.k,
            "layer_priority_strength": self.layer_priority_strength,
            "thresholds_by_layer": self._current_thresholds,
        }


def build_training_sparsity_controller(method: str, config: dict[str, Any]) -> TrainingSparsityController:
    """Build a during-training sparsity controller from method name and config."""
    normalized = method.lower()
    if normalized == "gradual_magnitude":
        return GradualMagnitudeController(
            target_sparsity=float(config.get("ratio", config.get("target_sparsity", 0.5))),
            start_epoch=int(config.get("start_epoch", 0)),
            end_epoch=config.get("end_epoch"),
            power=float(config.get("power", 3.0)),
            include_bias=bool(config.get("include_bias", False)),
        )
    if normalized == "sparse_from_scratch":
        return SparseFromScratchController(
            sparsity_ratio=float(config.get("ratio", 0.5)),
            include_bias=bool(config.get("include_bias", False)),
            init_mode=str(config.get("init_mode", "random")).lower(),
            seed=int(config.get("seed", 42)),
        )
    if normalized in {"sparse_on_training", "sigma_sparse_on_training", "layer_sigma_training"}:
        return SparseOnTrainingSigmaController(
            k=float(config.get("k", 0.1)),
            include_bias=bool(config.get("include_bias", False)),
            layer_priority_strength=float(config.get("layer_priority_strength", 1.0)),
        )
    raise ValueError(f"Unsupported training sparsification method: {method}")


def apply_post_training_sparsification(
    model: torch.nn.Module,
    method: str,
    config: dict[str, Any],
    inplace: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Dispatch post-training sparsification methods."""
    # Central dispatcher used by runners to keep method selection uniform.
    normalized = method.lower()
    ratio = float(config.get("ratio", 0.5))
    include_bias = bool(config.get("include_bias", False))
    compact_selection = str(config.get("compact_selection", "ratio")).lower()
    compact_energy_target = float(config.get("compact_energy_target", 0.95))

    if normalized == "magnitude_unstructured":
        return apply_magnitude_sparsification(
            model=model,
            sparsity_ratio=ratio,
            include_bias=include_bias,
            inplace=inplace,
        )
    if normalized == "structured":
        return apply_structured_sparsification(
            model=model,
            sparsity_ratio=ratio,
            include_bias=include_bias,
            dim=int(config.get("dim", 0)),
            inplace=inplace,
        )
    if normalized in {"2to4", "semi_structured_2to4"}:
        return apply_semi_structured_2to4_sparsification(
            model=model,
            include_bias=include_bias,
            device=config.get("device"),
            enable_hardware_acceleration=bool(config.get("enable_hardware_acceleration", True)),
            inplace=inplace,
        )
    if normalized in {"structured_compact", "compact", "structural_compaction"}:
        return apply_structured_compact_sparsification(
            model=model,
            sparsity_ratio=ratio,
            selection_mode=compact_selection,
            energy_target=compact_energy_target,
            inplace=inplace,
        )
    if normalized in {"compact_zero_neurons", "zero_neurons_compact", "compact_dead_neurons"}:
        dead_threshold = float(config.get("dead_threshold", 0.0))
        return apply_structured_compact_zero_neurons_sparsification(
            model=model,
            dead_threshold=dead_threshold,
            inplace=inplace,
        )
    if normalized in {"compact_mlp", "structured_compact_mlp", "prune_compact_mlp"}:
        return apply_compact_mlp_sparsification(
            model=model,
            sparsity_ratio=ratio,
            selection_mode=compact_selection,
            energy_target=compact_energy_target,
            inplace=inplace,
        )
    if normalized in {"compact_lstm", "structured_compact_lstm", "prune_compact_lstm"}:
        return apply_compact_lstm_sparsification(
            model=model,
            sparsity_ratio=ratio,
            selection_mode=compact_selection,
            energy_target=compact_energy_target,
            inplace=inplace,
        )
    if normalized in {"compact_dlinear", "structured_compact_dlinear", "prune_compact_dlinear"}:
        return apply_compact_dlinear_sparsification(
            model=model,
            sparsity_ratio=ratio,
            inplace=inplace,
        )
    if normalized in {"variance_threshold_compact", "structured_compact_threshold", "post_training_threshold"}:
        raise ValueError(
            "Use method='post_training_ratio_priority' with ratio=0.2 instead."
        )
    if normalized in {"post_training_ratio_priority", "ratio_priority", "priority_magnitude"}:
        return apply_priority_magnitude_post_training_sparsification(
            model=model,
            sparsity_ratio=ratio,
            include_bias=include_bias,
            layer_priority_strength=float(config.get("layer_priority_strength", 1.0)),
            inplace=inplace,
        )

    raise ValueError(f"Unsupported post-training sparsification method: {method}")
