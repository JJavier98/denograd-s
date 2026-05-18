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

    raise ValueError(f"Unsupported post-training sparsification method: {method}")
