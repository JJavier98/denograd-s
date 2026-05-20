"""Weight distribution analysis utilities for histogram-based sparsity diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:
    from scipy.stats import kurtosis, normaltest, skew
except ImportError:  # pragma: no cover - optional dependency
    kurtosis = None
    normaltest = None
    skew = None


def collect_trainable_weights(model: torch.nn.Module, include_bias: bool = True) -> np.ndarray:
    """Collect all trainable model parameters into a single flattened numpy array."""
    chunks: list[np.ndarray] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (not include_bias) and name.endswith("bias"):
            continue
        if not (parameter.is_floating_point() or parameter.is_complex()):
            continue
        chunks.append(parameter.detach().flatten().cpu().numpy().astype(np.float64, copy=False))

    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def analyze_weight_distribution(
    weights: np.ndarray,
    bins: int = 120,
    near_zero_eps: float = 1e-3,
    normality_max_sample: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute histogram and descriptive statistics for a flattened weight vector."""
    array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "num_weights": 0,
            "mean": None,
            "std": None,
            "skewness": None,
            "kurtosis": None,
            "near_zero_eps": float(near_zero_eps),
            "near_zero_fraction": None,
            "normality_test": None,
            "normal_fit": None,
            "histogram": {"counts": [], "bin_edges": []},
        }

    mean_val = float(np.mean(array))
    std_val = float(np.std(array))
    near_zero_fraction = float(np.mean(np.abs(array) < near_zero_eps))

    skewness = None
    kurt = None
    if skew is not None:
        skewness = float(skew(array, bias=False))
    if kurtosis is not None:
        kurt = float(kurtosis(array, fisher=True, bias=False))

    counts, edges = np.histogram(array, bins=bins)

    test_result: dict[str, Any] | None = None
    if normaltest is not None and array.size >= 8:
        sample = array
        if array.size > normality_max_sample:
            rng = np.random.default_rng(seed)
            sample = rng.choice(array, size=normality_max_sample, replace=False)
        statistic, p_value = normaltest(sample)
        test_result = {
            "name": "scipy.stats.normaltest",
            "sample_size": int(sample.size),
            "statistic": float(statistic),
            "p_value": float(p_value),
        }

    return {
        "num_weights": int(array.size),
        "mean": mean_val,
        "std": std_val,
        "skewness": skewness,
        "kurtosis": kurt,
        "near_zero_eps": float(near_zero_eps),
        "near_zero_fraction": near_zero_fraction,
        "normality_test": test_result,
        "normal_fit": {
            "mu": mean_val,
            "sigma": std_val,
        },
        "histogram": {
            "counts": counts.astype(np.int64).tolist(),
            "bin_edges": edges.astype(np.float64).tolist(),
        },
    }


def analyze_model_weights(
    model: torch.nn.Module,
    bins: int = 120,
    near_zero_eps: float = 1e-3,
    include_bias: bool = True,
    normality_max_sample: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Collect and analyze model weights in one call."""
    weights = collect_trainable_weights(model=model, include_bias=include_bias)
    return analyze_weight_distribution(
        weights=weights,
        bins=bins,
        near_zero_eps=near_zero_eps,
        normality_max_sample=normality_max_sample,
        seed=seed,
    )


def save_weight_histogram_plot(analysis: dict[str, Any], output_path: str | Path, title: str) -> dict[str, Any]:
    """Save a histogram figure with optional normal-density overlay when matplotlib is available."""
    if plt is None:
        return {
            "saved": False,
            "reason": "matplotlib_not_available",
            "path": str(output_path),
        }

    histogram = analysis.get("histogram", {})
    counts = np.asarray(histogram.get("counts", []), dtype=np.float64)
    edges = np.asarray(histogram.get("bin_edges", []), dtype=np.float64)
    if counts.size == 0 or edges.size <= 1:
        return {
            "saved": False,
            "reason": "empty_histogram",
            "path": str(output_path),
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(centers, counts, width=widths, alpha=0.55, color="#2A6F97", edgecolor="#1b4965")
    ax.set_title(title)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")

    mu = analysis.get("normal_fit", {}).get("mu")
    sigma = analysis.get("normal_fit", {}).get("sigma")
    if sigma is not None and mu is not None and sigma > 0:
        x = np.linspace(edges[0], edges[-1], 500)
        pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        scale = float(np.sum(counts) * np.mean(widths)) if widths.size else float(np.sum(counts))
        ax.plot(x, pdf * scale, color="#d00000", linewidth=2.0, label="Normal fit")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    return {
        "saved": True,
        "reason": None,
        "path": str(output),
    }
