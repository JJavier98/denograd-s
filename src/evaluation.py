"""Metrics for denoising quality and downstream benchmark deltas."""

import numpy as np
from scipy.stats import pearsonr

try:
    import ot
except ImportError:  # pragma: no cover - optional dependency
    ot = None


def calculate_sliced_wasserstein(p_samples, q_samples, n_projections=100, seed=42):
    """Approximate Wasserstein distance using random 1D projections."""
    if ot is None:
        raise ImportError("POT is required to compute sliced Wasserstein distance.")

    p = np.asarray(p_samples, dtype=np.float64)
    q = np.asarray(q_samples, dtype=np.float64)

    if p.ndim == 1:
        p = p.reshape(-1, 1)
    elif p.ndim > 2:
        p = p.reshape(-1, p.shape[-1])

    if q.ndim == 1:
        q = q.reshape(-1, 1)
    elif q.ndim > 2:
        q = q.reshape(-1, q.shape[-1])

    return ot.sliced_wasserstein_distance(p, q, n_projections=n_projections, seed=seed)


def calculate_correlation(x, y):
    """Pearson correlation between flattened tensors/arrays."""
    x_flat = np.asarray(x).flatten()
    y_flat = np.asarray(y).flatten()
    corr, _ = pearsonr(x_flat, y_flat)
    return corr


def evaluate_changes(X_noisy, X_denoised, results_before, results_after):
    """Aggregate denoising quality and downstream improvements."""
    stats = {
        "input_correlation": calculate_correlation(X_noisy, X_denoised),
    }

    if ot is not None:
        stats["wasserstein_distance"] = calculate_sliced_wasserstein(X_noisy, X_denoised)

    for model_name, error_before in results_before.items():
        error_after = results_after.get(model_name)
        if error_after is None:
            continue
        stats[f"{model_name}_mse_before"] = error_before
        stats[f"{model_name}_mse_after"] = error_after
        stats[f"{model_name}_improvement_pct"] = (
            (error_before - error_after) / error_before * 100 if error_before != 0 else 0.0
        )

    return stats