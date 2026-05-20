"""End-to-end experiment runners with caching, checkpoints and denoising profiling."""

import copy
import datetime
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.libs.benchmark import run_regression_benchmark, run_ts_benchmark
from src.libs.cache import ExperimentCache, make_experiment_signature
from src.libs.dataset import (
    add_gaussian_noise,
    fix_noise_seed,
    get_dataloaders,
    get_ts_dataloaders,
    load_data,
)
from src.libs.evaluation import evaluate_changes
from src.models import get_model
from src.libs.profiling import model_size_bytes, reset_cuda_peak_memory, snapshot_cuda_memory
from src.libs.sparsity import (
    apply_post_training_sparsification,
    build_training_sparsity_controller,
    summarize_sparsity,
)
from src.libs.trainer import Trainer
from src.libs.weight_analysis import analyze_model_weights, save_weight_histogram_plot

try:
    from denograd import DenoGrad
except ImportError:  # pragma: no cover
    DenoGrad = None


def _require_denograd():
    """Ensure DenoGrad is available before running any experiment."""
    if DenoGrad is None:
        raise ImportError(
            "denograd is required for experiment_runner. Install it in torch_env before running experiments."
        )


def _as_dict(config):
    """Normalize optional mappings to plain dictionaries."""
    return dict(config) if config else {}


def _device_from_config(config):
    """Resolve runtime device from config or fallback to CUDA/CPU automatically."""
    requested = config.get("device") if config else None
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_signature_payload(config, domain):
    """Build the deterministic payload used to hash experiment signatures."""
    return {
        "domain": domain,
        "dataset": _as_dict(config.get("dataset", {})),
        "model": _as_dict(config.get("model", {})),
        "denograd": _as_dict(config.get("denograd", {})),
        "benchmark": _as_dict(config.get("benchmark", {})),
        "seed": config.get("seed", 42),
        "version": config.get("version", "v1"),
    }


def _slugify(value):
    """Create filesystem-friendly labels for experiment paths."""
    text = str(value or "na").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def _dataset_label(dataset_cfg):
    """Extract a compact dataset label from dataset path if available."""
    path = dataset_cfg.get("path") if dataset_cfg else None
    if not path:
        return "dataset_unknown"
    path_obj = Path(path)
    parent = _slugify(path_obj.parent.name)
    stem = _slugify(path_obj.stem)
    return f"{parent}_{stem}"


def _sparsity_label(sparsity_cfg):
    """Build readable label for sparse setting used in the run."""
    cfg = sparsity_cfg or {}
    if not cfg.get("enabled", False):
        return "dense"
    method = _slugify(cfg.get("method", "unknown"))
    if "ratio" in cfg:
        value = f"r{str(cfg.get('ratio', 'na')).replace('.', 'p')}"
    elif "variance_pct" in cfg:
        value = f"vp{str(cfg.get('variance_pct', 'na')).replace('.', 'p')}"
    else:
        value = "na"
    return f"{method}_{value}"


def _ensure_output_schema(root: Path) -> None:
    """Create and maintain the canonical output schema anchors."""
    canonical_dirs = [
        root / "tabular",
        root / "time_series",
        root / "meta" / "indexes",
        root / "meta" / "summaries",
        root / "meta" / "migrations",
        root / "meta" / "legacy",
        root / "archive",
    ]
    for directory in canonical_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    schema_file = root / "SCHEMA_VERSION"
    schema_file.write_text("2\n", encoding="utf-8")


def _append_run_index(artifact_cache: ExperimentCache, config: dict, domain: str) -> None:
    """Append one line to the global run index for fast traceability."""
    root = Path(config.get("artifacts_dir", "out"))
    _ensure_output_schema(root)

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "domain": domain,
        "dataset": _dataset_label(_as_dict(config.get("dataset", {}))),
        "model": _slugify(_as_dict(config.get("model", {})).get("name", "model")),
        "sparse": _sparsity_label(_as_dict(config.get("sparsity", {}))),
        "seed": config.get("seed", 42),
        "version": config.get("version", "v1"),
        "signature": artifact_cache.signature,
        "run_root": str(artifact_cache.root),
        "manifest": str(artifact_cache.json_path("manifest", kind="logs")),
        "summary_tabular": str(artifact_cache.json_path("summary_tabular", kind="metrics")),
        "summary_ts": str(artifact_cache.json_path("summary_ts", kind="metrics")),
    }

    index_path = root / "meta" / "indexes" / "runs.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _build_artifacts_base_dir(config, domain):
    """Compose human-readable base path under artifacts root before signature folder.

    Layout::

        out/
            {domain}/              # tabular | time_series
                {dataset}/         # house_prices_kc_house_data | daily_climate_...
                    {sparse_type}/ # dense | structured_compact_0p3 | ...
                        {backbone}/# dnn | lstm | dlinear | ...
                            {params}/ # seed42__ep100__iter150__vreal_v2__20260518_151000
    """
    root = Path(config.get("artifacts_dir", "out"))
    _ensure_output_schema(root)
    dataset_cfg = _as_dict(config.get("dataset", {}))
    model_cfg = _as_dict(config.get("model", {}))
    sparsity_cfg = _as_dict(config.get("sparsity", {}))
    denograd_cfg = _as_dict(config.get("denograd", {}))

    domain_dir = "time_series" if domain == "time_series" else "tabular"
    dataset = _dataset_label(dataset_cfg)
    model = _slugify(model_cfg.get("name", "model"))
    sparse = _sparsity_label(sparsity_cfg)
    seed = config.get("seed", 42)
    epochs = model_cfg.get("max_epochs", "na")
    denograd_iters = denograd_cfg.get("max_iters", "na")
    version = _slugify(config.get("version", "v1"))
    launched_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_label = (
        f"seed{seed}__ep{epochs}__iter{denograd_iters}"
        f"__v{version}__{launched_at}"
    )
    return root / domain_dir / dataset / sparse / model / run_label


def _store_run_context(artifact_cache, payload, config):
    """Persist run metadata for easier traceability across hash folders."""
    run_context = {
        "signature": artifact_cache.signature,
        "payload": payload,
        "dataset_path": _as_dict(config.get("dataset", {})).get("path"),
        "model_name": _as_dict(config.get("model", {})).get("name"),
        "sparsity": _as_dict(config.get("sparsity", {})),
        "seed": config.get("seed", 42),
        "version": config.get("version", "v1"),
    }
    artifact_cache.save_json(run_context, "run_context", kind="logs")
    artifact_cache.update_manifest(
        "run_context",
        {"path": str(artifact_cache.json_path("run_context", kind="logs"))},
    )


def _load_or_prepare_tabular_data(config, artifact_cache):
    """Load, normalize, noise and cache tabular arrays for a run signature."""
    dataset_cfg = config.get("dataset", {})
    data_path = dataset_cfg.get("path")
    target_col = dataset_cfg.get("target_col", "y")
    noise_std = dataset_cfg.get("noise_std", 0.0)
    seed = config.get("seed", 42)

    noisy = artifact_cache.load_numpy("tabular_noisy")
    clean = artifact_cache.load_numpy("tabular_clean")
    if noisy is not None and clean is not None:
        return clean["X"], clean["y"], noisy["X"], noisy["y"]

    X_raw, y_raw = load_data(data_path, target_col=target_col)

    scaler_X = StandardScaler()
    X_clean = scaler_X.fit_transform(X_raw)
    scaler_y = StandardScaler()
    y_clean = scaler_y.fit_transform(y_raw)

    X_noisy = X_clean.copy()
    if noise_std > 0:
        fix_noise_seed(seed)
        X_noisy = add_gaussian_noise(X_noisy, noise_std=noise_std)

    artifact_cache.save_numpy("tabular_clean", X=X_clean, y=y_clean)
    artifact_cache.save_numpy("tabular_noisy", X=X_noisy, y=y_clean)
    artifact_cache.update_manifest(
        "tabular_data",
        {
            "clean": str(artifact_cache.npz_path("tabular_clean")),
            "noisy": str(artifact_cache.npz_path("tabular_noisy")),
        },
    )

    return X_clean, y_clean, X_noisy, y_clean


def _load_or_prepare_ts_data(config, artifact_cache):
    """Load, normalize, noise and cache time-series arrays for a run signature."""
    dataset_cfg = config.get("dataset", {})
    data_path = dataset_cfg.get("path")
    target_col = dataset_cfg.get("target_col", "y")
    noise_std = dataset_cfg.get("noise_std", 0.0)
    seed = config.get("seed", 42)

    noisy = artifact_cache.load_numpy("ts_noisy")
    clean = artifact_cache.load_numpy("ts_clean")
    if noisy is not None and clean is not None:
        return clean["X"], clean["y"], noisy["X"], noisy["y"]

    X_partial, y_raw = load_data(data_path, target_col=target_col)
    y_raw = np.asarray(y_raw)
    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)

    X_raw = np.hstack([y_raw, X_partial])

    scaler_X = StandardScaler()
    X_clean = scaler_X.fit_transform(X_raw)
    y_clean = X_clean[:, : y_raw.shape[1]].copy()

    X_noisy = X_clean.copy()
    if noise_std > 0:
        fix_noise_seed(seed)
        X_noisy = add_gaussian_noise(X_noisy, noise_std=noise_std)
    y_noisy = X_noisy[:, : y_raw.shape[1]].copy()

    artifact_cache.save_numpy("ts_clean", X=X_clean, y=y_clean)
    artifact_cache.save_numpy("ts_noisy", X=X_noisy, y=y_noisy)
    artifact_cache.update_manifest(
        "ts_data",
        {
            "clean": str(artifact_cache.npz_path("ts_clean")),
            "noisy": str(artifact_cache.npz_path("ts_noisy")),
        },
    )

    return X_clean, y_clean, X_noisy, y_noisy


def _measure_denoising_call(callable_transform, device):
    """Measure denoising runtime and CUDA memory around a transform callable."""
    before = None
    if device.type == "cuda" and torch.cuda.is_available():
        reset_cuda_peak_memory(device)
        torch.cuda.synchronize(device)
        before = snapshot_cuda_memory(device)

    start = time.perf_counter()
    output = callable_transform()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    after = snapshot_cuda_memory(device)
    if after is None:
        return output, elapsed, None

    payload = dict(after.__dict__)
    if before is not None:
        payload["allocated_before_bytes"] = before.allocated_bytes
        payload["reserved_before_bytes"] = before.reserved_bytes
        payload["allocated_after_bytes"] = after.allocated_bytes
        payload["reserved_after_bytes"] = after.reserved_bytes
        payload["allocated_net_bytes"] = after.allocated_bytes - before.allocated_bytes
        payload["reserved_net_bytes"] = after.reserved_bytes - before.reserved_bytes
    else:
        payload["allocated_before_bytes"] = None
        payload["reserved_before_bytes"] = None
        payload["allocated_after_bytes"] = after.allocated_bytes
        payload["reserved_after_bytes"] = after.reserved_bytes
        payload["allocated_net_bytes"] = None
        payload["reserved_net_bytes"] = None

    return output, elapsed, payload


def _analyze_and_store_weights(model, artifact_cache, tag, weight_cfg):
    """Compute and persist weight-distribution stats and histogram artifact for a model."""
    analysis = analyze_model_weights(
        model=model,
        bins=weight_cfg.get("bins", 120),
        near_zero_eps=weight_cfg.get("near_zero_eps", 1e-3),
        include_bias=weight_cfg.get("include_bias", True),
        normality_max_sample=weight_cfg.get("normality_max_sample", 20000),
        seed=weight_cfg.get("seed", 42),
    )
    stats_path = artifact_cache.save_json(analysis, f"weights_{tag}", kind="metrics")
    fig_path = artifact_cache.paths.figures / f"weights_hist_{tag}.png"
    fig_info = save_weight_histogram_plot(
        analysis,
        output_path=fig_path,
        title=f"Weight distribution - {tag}",
    )
    artifact_cache.update_manifest(
        f"weights_{tag}",
        {
            "stats": str(stats_path),
            "figure": fig_info.get("path"),
            "figure_saved": fig_info.get("saved", False),
        },
    )
    return analysis, fig_info


def _human_bytes(num_bytes):
    """Format byte values as human-readable strings (KB/MB/GB)."""
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while abs(value) >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _safe_pct_change(before, after):
    """Compute percentage change with safe handling for zeros and nulls."""
    if before is None or after is None:
        return None
    if before == 0:
        return None
    return ((after - before) / before) * 100.0


def _build_dense_sparse_comparison(dense_profile, sparse_profile):
    """Build machine and human-readable dense vs sparse comparison payload."""
    dense_stats = (dense_profile or {}).get("backbone_stats", {})
    sparse_stats = (sparse_profile or {}).get("backbone_stats", {})
    dense_vram = (dense_profile or {}).get("denoising_vram", {})
    sparse_vram = (sparse_profile or {}).get("denoising_vram", {})

    dense_param_bytes = dense_stats.get("param_bytes")
    sparse_param_bytes = sparse_stats.get("param_bytes")
    dense_params = dense_stats.get("total_params")
    sparse_params = sparse_stats.get("total_params")
    dense_seconds = (dense_profile or {}).get("denoising_seconds")
    sparse_seconds = (sparse_profile or {}).get("denoising_seconds")
    dense_alloc_net = dense_vram.get("allocated_net_bytes")
    sparse_alloc_net = sparse_vram.get("allocated_net_bytes")

    comparison = {
        "machine": {
            "param_bytes": {
                "dense": dense_param_bytes,
                "sparse": sparse_param_bytes,
                "delta": None if dense_param_bytes is None or sparse_param_bytes is None else sparse_param_bytes - dense_param_bytes,
                "delta_pct": _safe_pct_change(dense_param_bytes, sparse_param_bytes),
            },
            "total_params": {
                "dense": dense_params,
                "sparse": sparse_params,
                "delta": None if dense_params is None or sparse_params is None else sparse_params - dense_params,
                "delta_pct": _safe_pct_change(dense_params, sparse_params),
            },
            "denoising_seconds": {
                "dense": dense_seconds,
                "sparse": sparse_seconds,
                "delta": None if dense_seconds is None or sparse_seconds is None else sparse_seconds - dense_seconds,
                "delta_pct": _safe_pct_change(dense_seconds, sparse_seconds),
                "speedup_x": None if dense_seconds is None or sparse_seconds in (None, 0) else dense_seconds / sparse_seconds,
            },
            "allocated_net_bytes": {
                "dense": dense_alloc_net,
                "sparse": sparse_alloc_net,
                "delta": None if dense_alloc_net is None or sparse_alloc_net is None else sparse_alloc_net - dense_alloc_net,
                "delta_pct": _safe_pct_change(dense_alloc_net, sparse_alloc_net),
            },
        },
        "human": {
            "param_bytes": {
                "dense": _human_bytes(dense_param_bytes),
                "sparse": _human_bytes(sparse_param_bytes),
                "delta": _human_bytes(None if dense_param_bytes is None or sparse_param_bytes is None else sparse_param_bytes - dense_param_bytes),
            },
            "allocated_net_bytes": {
                "dense": _human_bytes(dense_alloc_net),
                "sparse": _human_bytes(sparse_alloc_net),
                "delta": _human_bytes(None if dense_alloc_net is None or sparse_alloc_net is None else sparse_alloc_net - dense_alloc_net),
            },
        },
    }
    return comparison


def _comparison_table_markdown(comparison):
    """Build a compact markdown table from dense/sparse comparison metrics."""
    machine = comparison.get("machine", {})
    rows = [
        ("param_bytes", machine.get("param_bytes", {})),
        ("total_params", machine.get("total_params", {})),
        ("denoising_seconds", machine.get("denoising_seconds", {})),
        ("allocated_net_bytes", machine.get("allocated_net_bytes", {})),
    ]
    lines = [
        "| metric | dense | sparse | delta | delta_pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in rows:
        dense = payload.get("dense")
        sparse = payload.get("sparse")
        delta = payload.get("delta")
        delta_pct = payload.get("delta_pct")
        dense_s = f"{dense:.6g}" if isinstance(dense, (int, float)) else "n/a"
        sparse_s = f"{sparse:.6g}" if isinstance(sparse, (int, float)) else "n/a"
        delta_s = f"{delta:.6g}" if isinstance(delta, (int, float)) else "n/a"
        delta_pct_s = f"{delta_pct:.4f}%" if isinstance(delta_pct, (int, float)) else "n/a"
        lines.append(f"| {name} | {dense_s} | {sparse_s} | {delta_s} | {delta_pct_s} |")
    return "\n".join(lines)


def _save_dense_sparse_plot(comparison, output_path):
    """Save dense vs sparse plots with separate scales for bytes and time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"saved": False, "reason": "matplotlib_not_available", "path": str(output_path)}

    machine = comparison.get("machine", {})
    param_payload = machine.get("param_bytes", {})
    time_payload = machine.get("denoising_seconds", {})
    dense_param = param_payload.get("dense")
    sparse_param = param_payload.get("sparse")
    dense_time = time_payload.get("dense")
    sparse_time = time_payload.get("sparse")

    if not all(isinstance(v, (int, float)) for v in [dense_param, sparse_param, dense_time, sparse_time]):
        return {"saved": False, "reason": "missing_param_or_time_metrics", "path": str(output_path)}

    unit = "B"
    scale = 1.0
    max_param = max(abs(float(dense_param)), abs(float(sparse_param)))
    if max_param >= 1024 ** 3:
        unit, scale = "GB", float(1024 ** 3)
    elif max_param >= 1024 ** 2:
        unit, scale = "MB", float(1024 ** 2)
    elif max_param >= 1024:
        unit, scale = "KB", 1024.0

    dense_param_scaled = float(dense_param) / scale
    sparse_param_scaled = float(sparse_param) / scale

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(["dense", "sparse"], [dense_param_scaled, sparse_param_scaled], color=["#4C72B0", "#55A868"])
    axes[0].set_title("Backbone Size")
    axes[0].set_ylabel(f"param_bytes ({unit})")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(["dense", "sparse"], [float(dense_time), float(sparse_time)], color=["#4C72B0", "#55A868"])
    axes[1].set_title("Denoising Time")
    axes[1].set_ylabel("seconds")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Dense vs Sparse Comparison", y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return {"saved": True, "path": str(output_path)}


def _sparse_value_token(sparsity_cfg):
    """Generate an identifier token for ratio/variance settings in sparse tags."""
    if "ratio" in sparsity_cfg:
        return f"ratio_{str(float(sparsity_cfg.get('ratio', 0.0))).replace('.', 'p')}"
    if "k" in sparsity_cfg:
        return f"k_{str(float(sparsity_cfg.get('k', 0.0))).replace('.', 'p')}"
    if "variance_pct" in sparsity_cfg:
        return f"vp_{str(float(sparsity_cfg.get('variance_pct', 0.0))).replace('.', 'p')}"
    return "cfg_default"


def _normalize_sparse_method(method_name):
    """Normalize sparse method aliases to canonical method names."""
    normalized = str(method_name).lower()
    if normalized == "2to4":
        return "semi_structured_2to4"
    return normalized


def _is_during_training_sparse_method(method_name):
    """Return True when sparse method is applied during optimization."""
    return method_name in {
        "gradual_magnitude",
        "sparse_from_scratch",
        "sparse_on_training",
        "sigma_sparse_on_training",
        "layer_sigma_training",
    }


def _is_compact_sparse_method(method_name):
    """Return True for methods that physically change the model architecture.

    Compact methods reconstruct smaller layers, so the saved checkpoint
    has different tensor shapes than the original model definition. They
    must be saved as full model objects (not state_dicts) to allow correct
    reloading without knowing the compacted architecture in advance.
    """
    return method_name in {
        "compact_mlp",
        "compact_lstm",
        "compact_dlinear",
        "structured_compact",
        "post_training_ratio_priority",
        "ratio_priority",
        "priority_magnitude",
        "sparse_on_training",
        "sigma_sparse_on_training",
        "layer_sigma_training",
    }


def run_tabular_experiment(config):
    """Run a tabular DenoGrad experiment with caching, checkpoints and profiling."""
    _require_denograd()
    device = _device_from_config(config)
    payload = _build_signature_payload(config, domain="tabular")
    signature = make_experiment_signature(payload)
    base_dir = _build_artifacts_base_dir(config, domain="tabular")
    artifact_cache = ExperimentCache(base_dir=str(base_dir), signature=signature)
    _store_run_context(artifact_cache, payload, config)
    _append_run_index(artifact_cache, config, domain="tabular")

    _X_clean, _y_clean, X_noisy, y_noisy = _load_or_prepare_tabular_data(config, artifact_cache)

    dataset_cfg = config.get("dataset", {})
    loaders_noisy, train_noisy, val_noisy, test_noisy = get_dataloaders(
        X=X_noisy,
        y=y_noisy,
        batch_size=dataset_cfg.get("batch_size", 32),
        test_split=dataset_cfg.get("test_split", 0.2),
        val_split=dataset_cfg.get("val_split", 0.1),
    )

    benchmark_cfg = _as_dict(config.get("benchmark", {}))
    weight_cfg = _as_dict(config.get("weight_analysis", {}))
    sparsity_cfg = _as_dict(config.get("sparsity", {}))
    results_noisy, noisy_meta = run_regression_benchmark(
        loaders_noisy,
        (train_noisy, val_noisy, test_noisy),
        device,
        benchmark_cfg=benchmark_cfg,
        artifact_cache=artifact_cache,
        cache_key="noisy",
        reuse_cached=True,
        profile_models=True,
        return_metadata=True,
    )

    model_cfg = _as_dict(config.get("model", {}))
    model_name = model_cfg.get("name", "dnn")
    input_dim = train_noisy[0].shape[1]
    output_dim = train_noisy[1].shape[1] if train_noisy[1].ndim > 1 else 1

    backbone = get_model(model_cfg, input_dim=input_dim, output_dim=output_dim, device=str(device))
    model_ckpt_name = f"backbone_{model_name}"
    model_ckpt = artifact_cache.model_path(model_ckpt_name)
    if model_ckpt.exists():
        backbone.load_state_dict(torch.load(model_ckpt, map_location=device))
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(backbone.parameters(), lr=model_cfg.get("lr", 0.001))
        trainer = Trainer(
            model=backbone,
            train_generator=loaders_noisy[0],
            val_generator=loaders_noisy[1],
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch_scheduler=None,
            batch_scheduler=None,
            patience=model_cfg.get("patience", 15),
            epochs=model_cfg.get("max_epochs", 100),
            checkpoints_path=str(model_ckpt),
            verbose=False,
        )
        backbone, _, _, _, _ = trainer.fit()
        torch.save(backbone.state_dict(), model_ckpt)

    dense_weight_stats, dense_weight_fig = _analyze_and_store_weights(
        model=backbone,
        artifact_cache=artifact_cache,
        tag="tabular_dense",
        weight_cfg=weight_cfg,
    )

    denoised_npz = artifact_cache.npz_path("tabular_denoised_dense")
    if denoised_npz.exists():
        denoised = np.load(denoised_npz)
        X_denoised, y_denoised = denoised["X"], denoised["y"]
        denoise_metrics = artifact_cache.load_json("denoise_dense_profile", kind="metrics") or {}
    else:
        dg_cfg = _as_dict(config.get("denograd", {}))
        denograd_cls = DenoGrad
        if denograd_cls is None:  # pragma: no cover
            raise RuntimeError("DenoGrad is not available.")
        denoiser = denograd_cls(model=backbone, criterion=nn.MSELoss(), device=device)
        denoiser.fit(X=X_noisy, y=y_noisy, is_ts=False)

        (X_denoised, y_denoised, _, _), elapsed, vram = _measure_denoising_call(
            lambda: denoiser.transform(
                nrr=dg_cfg.get("nrr", 0.01),
                nr_threshold=dg_cfg.get("threshold", 0.1),
                max_epochs=dg_cfg.get("max_iters", 150),
                denoise_y=True,
                batch_size=dg_cfg.get("batch_size", 1024),
                save_gradients=False,
            ),
            device,
        )
        artifact_cache.save_numpy("tabular_denoised_dense", X=X_denoised, y=y_denoised)
        denoise_metrics = {
            "denoising_seconds": elapsed,
            "denoising_vram": vram,
            "backbone_stats": model_size_bytes(backbone),
        }
        artifact_cache.save_json(denoise_metrics, "denoise_dense_profile", kind="metrics")

    loaders_clean, train_clean, val_clean, test_clean = get_dataloaders(
        X=X_denoised,
        y=y_denoised,
        batch_size=dataset_cfg.get("batch_size", 32),
        test_split=dataset_cfg.get("test_split", 0.2),
        val_split=dataset_cfg.get("val_split", 0.1),
    )

    results_clean, clean_meta = run_regression_benchmark(
        loaders_clean,
        (train_clean, val_clean, test_clean),
        device,
        benchmark_cfg=benchmark_cfg,
        artifact_cache=artifact_cache,
        cache_key="clean",
        reuse_cached=True,
        profile_models=True,
        return_metadata=True,
    )

    dense_evaluation = evaluate_changes(X_noisy, X_denoised, results_noisy, results_clean)

    summary = {
        "signature": signature,
        "device": str(device),
        "results_noisy": results_noisy,
        "results_clean": results_clean,
        "denoising_profile": denoise_metrics,
        "benchmark_noisy_profile": noisy_meta,
        "benchmark_clean_profile": clean_meta,
        "evaluation": dense_evaluation,
        "weight_analysis": {
            "dense": {
                "stats": dense_weight_stats,
                "histogram_figure": dense_weight_fig,
            }
        },
    }

    artifact_cache.save_json(dense_evaluation, "evaluation_dense", kind="metrics")
    artifact_cache.update_manifest(
        "evaluation_dense",
        {"path": str(artifact_cache.json_path("evaluation_dense", kind="metrics"))},
    )

    comparison_plot = {"saved": False, "reason": "not_computed", "path": None}
    if sparsity_cfg.get("enabled", False):
        sparse_method = _normalize_sparse_method(sparsity_cfg.get("method", "magnitude_unstructured"))
        sparse_ratio = float(sparsity_cfg.get("ratio", 0.5))
        include_bias = bool(sparsity_cfg.get("include_bias", False))
        sparse_token = _sparse_value_token(sparsity_cfg)
        sparse_tag = f"tabular_sparse_{sparse_method}_{sparse_token}"

        sparse_model_ckpt = artifact_cache.model_path(
            f"{model_ckpt_name}_{sparse_method}_{sparse_token}"
        )
        sparse_report_key = f"sparsity_report_{sparse_tag}"
        sparse_report = artifact_cache.load_json(sparse_report_key, kind="metrics")

        if sparse_model_ckpt.exists():
            if _is_compact_sparse_method(sparse_method):
                sparse_backbone = torch.load(sparse_model_ckpt, map_location=device, weights_only=False)
            else:
                sparse_backbone = get_model(model_cfg, input_dim=input_dim, output_dim=output_dim, device=str(device))
                sparse_backbone.load_state_dict(torch.load(sparse_model_ckpt, map_location=device, weights_only=False))
            if sparse_report is None:
                sparse_report = {
                    "method": sparse_method,
                    "requested_sparsity": sparse_ratio,
                    "applied": True,
                    "loaded_from_checkpoint": True,
                }
        else:
            if _is_during_training_sparse_method(sparse_method):
                sparse_backbone = get_model(model_cfg, input_dim=input_dim, output_dim=output_dim, device=str(device))
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(sparse_backbone.parameters(), lr=model_cfg.get("lr", 0.001))
                sparse_controller = build_training_sparsity_controller(sparse_method, sparsity_cfg)
                masked_model_ckpt = artifact_cache.model_path(
                    f"{model_ckpt_name}_{sparse_method}_{sparse_token}_masked"
                )
                compact_model_ckpt = artifact_cache.model_path(
                    f"{model_ckpt_name}_{sparse_method}_{sparse_token}_compact"
                )
                sparse_trainer = Trainer(
                    model=sparse_backbone,
                    train_generator=loaders_noisy[0],
                    val_generator=loaders_noisy[1],
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch_scheduler=None,
                    batch_scheduler=None,
                    patience=model_cfg.get("patience", 15),
                    epochs=model_cfg.get("max_epochs", 100),
                    checkpoints_path=str(sparse_model_ckpt),
                    verbose=False,
                    sparsity_controller=sparse_controller,
                )
                sparse_backbone, _, _, _, _ = sparse_trainer.fit()
                torch.save(sparse_backbone.state_dict(), masked_model_ckpt)

                compact_cfg = dict(sparsity_cfg)
                compact_method = _normalize_sparse_method(
                    compact_cfg.get("post_training_method", "structured_compact")
                )
                compact_ratio = float(compact_cfg.get("compact_ratio", sparse_ratio))
                compact_cfg.update(
                    {
                        "ratio": compact_ratio,
                        "device": str(device),
                        "include_bias": include_bias,
                    }
                )
                sparse_backbone, compact_report = apply_post_training_sparsification(
                    model=sparse_backbone,
                    method=compact_method,
                    config=compact_cfg,
                    inplace=True,
                )
                torch.save(sparse_backbone, compact_model_ckpt)

                ft_cfg = _as_dict(sparsity_cfg.get("fine_tune", {}))
                ft_epochs = int(ft_cfg.get("epochs", sparsity_cfg.get("fine_tune_epochs", 12)))
                ft_patience = int(ft_cfg.get("patience", sparsity_cfg.get("fine_tune_patience", 5)))
                ft_lr_scale = float(ft_cfg.get("lr_scale", sparsity_cfg.get("fine_tune_lr_scale", 0.1)))
                ft_lr = model_cfg.get("lr", 0.001) * ft_lr_scale
                ft_optimizer = torch.optim.Adam(sparse_backbone.parameters(), lr=ft_lr)
                ft_trainer = Trainer(
                    model=sparse_backbone,
                    train_generator=loaders_noisy[0],
                    val_generator=loaders_noisy[1],
                    device=device,
                    criterion=criterion,
                    optimizer=ft_optimizer,
                    epoch_scheduler=None,
                    batch_scheduler=None,
                    patience=ft_patience,
                    epochs=ft_epochs,
                    checkpoints_path=str(sparse_model_ckpt),
                    verbose=False,
                    sparsity_controller=None,
                )
                sparse_backbone, _, _, _, _ = ft_trainer.fit()

                sparse_report = {
                    "method": sparse_method,
                    "requested_sparsity": sparse_ratio,
                    "pipeline": "hybrid_on_training_compaction_finetune",
                    "training_controller": sparse_controller.report(),
                    "compact_report": compact_report,
                    "fine_tune": {
                        "epochs": ft_epochs,
                        "patience": ft_patience,
                        "lr_scale": ft_lr_scale,
                        "lr": ft_lr,
                    },
                    "phase_artifacts": {
                        "masked_model": str(masked_model_ckpt),
                        "compact_model": str(compact_model_ckpt),
                        "finetuned_model": str(sparse_model_ckpt),
                    },
                    "final_stats": summarize_sparsity(
                        sparse_backbone,
                        include_bias=include_bias,
                    ),
                }
            else:
                sparse_backbone = copy.deepcopy(backbone)
                post_cfg = dict(sparsity_cfg)
                post_cfg.setdefault("device", str(device))
                sparse_backbone, sparse_report = apply_post_training_sparsification(
                    model=sparse_backbone,
                    method=sparse_method,
                    config=post_cfg,
                    inplace=True,
                )

            if _is_compact_sparse_method(sparse_method):
                torch.save(sparse_backbone, sparse_model_ckpt)
            else:
                torch.save(sparse_backbone.state_dict(), sparse_model_ckpt)
            artifact_cache.save_json(sparse_report, sparse_report_key, kind="metrics")

        sparse_weight_stats, sparse_weight_fig = _analyze_and_store_weights(
            model=sparse_backbone,
            artifact_cache=artifact_cache,
            tag=sparse_tag,
            weight_cfg=weight_cfg,
        )

        sparse_denoised_key = f"tabular_denoised_{sparse_tag}"
        sparse_denoise_profile_key = f"denoise_profile_{sparse_tag}"
        sparse_denoised_npz = artifact_cache.npz_path(sparse_denoised_key)

        if sparse_denoised_npz.exists():
            sparse_denoised = np.load(sparse_denoised_npz)
            X_denoised_sparse, y_denoised_sparse = sparse_denoised["X"], sparse_denoised["y"]
            sparse_denoise_metrics = artifact_cache.load_json(sparse_denoise_profile_key, kind="metrics") or {}
        else:
            dg_cfg = _as_dict(config.get("denograd", {}))
            denograd_cls = DenoGrad
            if denograd_cls is None:  # pragma: no cover
                raise RuntimeError("DenoGrad is not available.")
            sparse_denoiser = denograd_cls(model=sparse_backbone, criterion=nn.MSELoss(), device=device)
            sparse_denoiser.fit(X=X_noisy, y=y_noisy, is_ts=False)
            (X_denoised_sparse, y_denoised_sparse, _, _), sparse_elapsed, sparse_vram = _measure_denoising_call(
                lambda: sparse_denoiser.transform(
                    nrr=dg_cfg.get("nrr", 0.01),
                    nr_threshold=dg_cfg.get("threshold", 0.1),
                    max_epochs=dg_cfg.get("max_iters", 150),
                    denoise_y=True,
                    batch_size=dg_cfg.get("batch_size", 1024),
                    save_gradients=False,
                ),
                device,
            )
            artifact_cache.save_numpy(sparse_denoised_key, X=X_denoised_sparse, y=y_denoised_sparse)
            sparse_denoise_metrics = {
                "denoising_seconds": sparse_elapsed,
                "denoising_vram": sparse_vram,
                "backbone_stats": model_size_bytes(sparse_backbone),
            }
            artifact_cache.save_json(sparse_denoise_metrics, sparse_denoise_profile_key, kind="metrics")

        loaders_sparse_clean, train_sparse_clean, val_sparse_clean, test_sparse_clean = get_dataloaders(
            X=X_denoised_sparse,
            y=y_denoised_sparse,
            batch_size=dataset_cfg.get("batch_size", 32),
            test_split=dataset_cfg.get("test_split", 0.2),
            val_split=dataset_cfg.get("val_split", 0.1),
        )
        sparse_results_clean, sparse_clean_meta = run_regression_benchmark(
            loaders_sparse_clean,
            (train_sparse_clean, val_sparse_clean, test_sparse_clean),
            device,
            benchmark_cfg=benchmark_cfg,
            artifact_cache=artifact_cache,
            cache_key=f"clean_{sparse_tag}",
            reuse_cached=True,
            profile_models=True,
            return_metadata=True,
        )

        sparse_evaluation = evaluate_changes(
            X_noisy,
            X_denoised_sparse,
            results_noisy,
            sparse_results_clean,
        )

        summary["sparse"] = {
            "method": sparse_method,
            "ratio": sparse_ratio,
            "report": sparse_report,
            "results_clean": sparse_results_clean,
            "denoising_profile": sparse_denoise_metrics,
            "benchmark_clean_profile": sparse_clean_meta,
            "evaluation": sparse_evaluation,
        }
        summary["weight_analysis"]["sparse"] = {
            "stats": sparse_weight_stats,
            "histogram_figure": sparse_weight_fig,
        }

        dense_sparse_comparison = _build_dense_sparse_comparison(
            summary.get("denoising_profile", {}),
            sparse_denoise_metrics,
        )
        comparison_key = f"dense_vs_sparse_comparison_{sparse_tag}"
        artifact_cache.save_json(dense_sparse_comparison, comparison_key, kind="metrics")
        artifact_cache.write_text(
            f"{comparison_key}_table",
            _comparison_table_markdown(dense_sparse_comparison),
            kind="logs",
        )
        comparison_plot = _save_dense_sparse_plot(
            dense_sparse_comparison,
            artifact_cache.paths.figures / f"{comparison_key}.png",
        )
        summary["sparse"]["dense_vs_sparse_comparison"] = dense_sparse_comparison
        summary["sparse"]["dense_vs_sparse_plot"] = comparison_plot

        artifact_cache.save_json(sparse_evaluation, "evaluation_sparse", kind="metrics")
        artifact_cache.save_json(summary["sparse"], "sparse_summary", kind="metrics")
        artifact_cache.update_manifest(
            "evaluation_sparse",
            {"path": str(artifact_cache.json_path("evaluation_sparse", kind="metrics"))},
        )
        artifact_cache.update_manifest(
            "sparse_summary",
            {"path": str(artifact_cache.json_path("sparse_summary", kind="metrics"))},
        )
        artifact_cache.update_manifest(
            comparison_key,
            {
                "json": str(artifact_cache.json_path(comparison_key, kind="metrics")),
                "table": str(artifact_cache.text_path(f"{comparison_key}_table", kind="logs")),
                "plot": comparison_plot.get("path"),
                "plot_saved": comparison_plot.get("saved", False),
            },
        )

    artifact_cache.save_json(summary, "summary_tabular", kind="metrics")
    artifact_cache.update_manifest(
        "summary_tabular",
        {"path": str(artifact_cache.json_path("summary_tabular", kind="metrics"))},
    )
    return summary


def run_ts_experiment(config):
    """Run a time-series DenoGrad experiment with caching and denoising profiling."""
    _require_denograd()
    device = _device_from_config(config)
    payload = _build_signature_payload(config, domain="time_series")
    signature = make_experiment_signature(payload)
    base_dir = _build_artifacts_base_dir(config, domain="time_series")
    artifact_cache = ExperimentCache(base_dir=str(base_dir), signature=signature)
    _store_run_context(artifact_cache, payload, config)
    _append_run_index(artifact_cache, config, domain="time_series")

    _X_clean, _y_clean, X_noisy, y_noisy = _load_or_prepare_ts_data(config, artifact_cache)

    dataset_cfg = config.get("dataset", {})
    window_size = dataset_cfg.get("window_size", 24)
    future = dataset_cfg.get("future", 1)
    batch_size = dataset_cfg.get("batch_size", 32)

    loaders_noisy, _, _, _ = get_ts_dataloaders(
        X=X_noisy,
        y=y_noisy,
        window_size=window_size,
        future=future,
        batch_size=batch_size,
        val_split=dataset_cfg.get("val_split", 0.1),
        test_split=dataset_cfg.get("test_split", 0.2),
        cnn=False,
    )

    benchmark_cfg = _as_dict(config.get("benchmark", {}))
    weight_cfg = _as_dict(config.get("weight_analysis", {}))
    sparsity_cfg = _as_dict(config.get("sparsity", {}))
    results_noisy, noisy_meta = run_ts_benchmark(
        loaders_noisy,
        device,
        benchmark_cfg=benchmark_cfg,
        artifact_cache=artifact_cache,
        cache_key="noisy",
        reuse_cached=True,
        profile_models=True,
        return_metadata=True,
    )

    model_cfg = _as_dict(config.get("model", {}))
    model_name = model_cfg.get("name", "lstm")
    input_dim = X_noisy.shape[1]
    output_dim = y_noisy.shape[1] if y_noisy.ndim > 1 else 1

    backbone = get_model(
        model_cfg,
        input_dim=input_dim,
        output_dim=output_dim,
        device=str(device),
        seq_len=window_size,
    )
    model_ckpt_name = f"ts_backbone_{model_name}"
    model_ckpt = artifact_cache.model_path(model_ckpt_name)
    if model_ckpt.exists():
        backbone.load_state_dict(torch.load(model_ckpt, map_location=device))
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(backbone.parameters(), lr=model_cfg.get("lr", 0.001))
        trainer = Trainer(
            model=backbone,
            train_generator=loaders_noisy[0],
            val_generator=loaders_noisy[1],
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch_scheduler=None,
            batch_scheduler=None,
            patience=model_cfg.get("patience", 15),
            epochs=model_cfg.get("max_epochs", 100),
            checkpoints_path=str(model_ckpt),
            verbose=False,
        )
        backbone, _, _, _, _ = trainer.fit()
        torch.save(backbone.state_dict(), model_ckpt)

    dense_weight_stats, dense_weight_fig = _analyze_and_store_weights(
        model=backbone,
        artifact_cache=artifact_cache,
        tag="ts_dense",
        weight_cfg=weight_cfg,
    )

    denoised_npz = artifact_cache.npz_path("ts_denoised_dense")
    if denoised_npz.exists():
        denoised = np.load(denoised_npz)
        X_denoised, y_denoised = denoised["X"], denoised["y"]
        denoise_metrics = artifact_cache.load_json("denoise_ts_profile", kind="metrics") or {}
    else:
        dg_cfg = _as_dict(config.get("denograd", {}))
        denograd_cls = DenoGrad
        if denograd_cls is None:  # pragma: no cover
            raise RuntimeError("DenoGrad is not available.")
        denoiser = denograd_cls(model=backbone, criterion=nn.MSELoss(), device=device)
        denoiser.fit(
            X=X_noisy,
            y=y_noisy,
            is_ts=True,
            window_size=window_size,
            future=future,
            stride=1,
        )

        (X_denoised, y_denoised, _, _), elapsed, vram = _measure_denoising_call(
            lambda: denoiser.transform(
                nrr=dg_cfg.get("nrr", 0.01),
                nr_threshold=dg_cfg.get("threshold", 0.1),
                max_epochs=dg_cfg.get("max_iters", 150),
                denoise_y=True,
                batch_size=dg_cfg.get("batch_size", 1024),
                save_gradients=False,
            ),
            device,
        )
        artifact_cache.save_numpy("ts_denoised_dense", X=X_denoised, y=y_denoised)
        denoise_metrics = {
            "denoising_seconds": elapsed,
            "denoising_vram": vram,
            "backbone_stats": model_size_bytes(backbone),
        }
        artifact_cache.save_json(denoise_metrics, "denoise_ts_profile", kind="metrics")

    loaders_clean, _, _, _ = get_ts_dataloaders(
        X=X_denoised,
        y=y_denoised,
        window_size=window_size,
        future=future,
        batch_size=batch_size,
        val_split=dataset_cfg.get("val_split", 0.1),
        test_split=dataset_cfg.get("test_split", 0.2),
        cnn=False,
    )

    results_clean, clean_meta = run_ts_benchmark(
        loaders_clean,
        device,
        benchmark_cfg=benchmark_cfg,
        artifact_cache=artifact_cache,
        cache_key="clean",
        reuse_cached=True,
        profile_models=True,
        return_metadata=True,
    )

    dense_evaluation = evaluate_changes(X_noisy, X_denoised, results_noisy, results_clean)

    summary = {
        "signature": signature,
        "device": str(device),
        "results_noisy": results_noisy,
        "results_clean": results_clean,
        "denoising_profile": denoise_metrics,
        "benchmark_noisy_profile": noisy_meta,
        "benchmark_clean_profile": clean_meta,
        "evaluation": dense_evaluation,
        "weight_analysis": {
            "dense": {
                "stats": dense_weight_stats,
                "histogram_figure": dense_weight_fig,
            }
        },
    }

    artifact_cache.save_json(dense_evaluation, "evaluation_dense", kind="metrics")
    artifact_cache.update_manifest(
        "evaluation_dense",
        {"path": str(artifact_cache.json_path("evaluation_dense", kind="metrics"))},
    )

    comparison_plot = {"saved": False, "reason": "not_computed", "path": None}
    if sparsity_cfg.get("enabled", False):
        sparse_method = _normalize_sparse_method(sparsity_cfg.get("method", "magnitude_unstructured"))
        sparse_ratio = float(sparsity_cfg.get("ratio", 0.5))
        include_bias = bool(sparsity_cfg.get("include_bias", False))
        sparse_token = _sparse_value_token(sparsity_cfg)
        sparse_tag = f"ts_sparse_{sparse_method}_{sparse_token}"

        sparse_model_ckpt = artifact_cache.model_path(
            f"{model_ckpt_name}_{sparse_method}_{sparse_token}"
        )
        sparse_report_key = f"sparsity_report_{sparse_tag}"
        sparse_report = artifact_cache.load_json(sparse_report_key, kind="metrics")

        if sparse_model_ckpt.exists():
            if _is_compact_sparse_method(sparse_method):
                sparse_backbone = torch.load(sparse_model_ckpt, map_location=device, weights_only=False)
            else:
                sparse_backbone = get_model(
                    model_cfg,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    device=str(device),
                    seq_len=window_size,
                )
                sparse_backbone.load_state_dict(torch.load(sparse_model_ckpt, map_location=device, weights_only=False))
            if sparse_report is None:
                sparse_report = {
                    "method": sparse_method,
                    "requested_sparsity": sparse_ratio,
                    "applied": True,
                    "loaded_from_checkpoint": True,
                }
        else:
            if _is_during_training_sparse_method(sparse_method):
                sparse_backbone = get_model(
                    model_cfg,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    device=str(device),
                    seq_len=window_size,
                )
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(sparse_backbone.parameters(), lr=model_cfg.get("lr", 0.001))
                sparse_controller = build_training_sparsity_controller(sparse_method, sparsity_cfg)
                masked_model_ckpt = artifact_cache.model_path(
                    f"{model_ckpt_name}_{sparse_method}_{sparse_token}_masked"
                )
                compact_model_ckpt = artifact_cache.model_path(
                    f"{model_ckpt_name}_{sparse_method}_{sparse_token}_compact"
                )
                sparse_trainer = Trainer(
                    model=sparse_backbone,
                    train_generator=loaders_noisy[0],
                    val_generator=loaders_noisy[1],
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch_scheduler=None,
                    batch_scheduler=None,
                    patience=model_cfg.get("patience", 15),
                    epochs=model_cfg.get("max_epochs", 100),
                    checkpoints_path=str(sparse_model_ckpt),
                    verbose=False,
                    sparsity_controller=sparse_controller,
                )
                sparse_backbone, _, _, _, _ = sparse_trainer.fit()
                torch.save(sparse_backbone.state_dict(), masked_model_ckpt)

                compact_cfg = dict(sparsity_cfg)
                compact_method = _normalize_sparse_method(
                    compact_cfg.get("post_training_method", "structured_compact")
                )
                compact_ratio = float(compact_cfg.get("compact_ratio", sparse_ratio))
                compact_cfg.update(
                    {
                        "ratio": compact_ratio,
                        "device": str(device),
                        "include_bias": include_bias,
                    }
                )
                sparse_backbone, compact_report = apply_post_training_sparsification(
                    model=sparse_backbone,
                    method=compact_method,
                    config=compact_cfg,
                    inplace=True,
                )
                torch.save(sparse_backbone, compact_model_ckpt)

                ft_cfg = _as_dict(sparsity_cfg.get("fine_tune", {}))
                ft_epochs = int(ft_cfg.get("epochs", sparsity_cfg.get("fine_tune_epochs", 12)))
                ft_patience = int(ft_cfg.get("patience", sparsity_cfg.get("fine_tune_patience", 5)))
                ft_lr_scale = float(ft_cfg.get("lr_scale", sparsity_cfg.get("fine_tune_lr_scale", 0.1)))
                ft_lr = model_cfg.get("lr", 0.001) * ft_lr_scale
                ft_optimizer = torch.optim.Adam(sparse_backbone.parameters(), lr=ft_lr)
                ft_trainer = Trainer(
                    model=sparse_backbone,
                    train_generator=loaders_noisy[0],
                    val_generator=loaders_noisy[1],
                    device=device,
                    criterion=criterion,
                    optimizer=ft_optimizer,
                    epoch_scheduler=None,
                    batch_scheduler=None,
                    patience=ft_patience,
                    epochs=ft_epochs,
                    checkpoints_path=str(sparse_model_ckpt),
                    verbose=False,
                    sparsity_controller=None,
                )
                sparse_backbone, _, _, _, _ = ft_trainer.fit()

                sparse_report = {
                    "method": sparse_method,
                    "requested_sparsity": sparse_ratio,
                    "pipeline": "hybrid_on_training_compaction_finetune",
                    "training_controller": sparse_controller.report(),
                    "compact_report": compact_report,
                    "fine_tune": {
                        "epochs": ft_epochs,
                        "patience": ft_patience,
                        "lr_scale": ft_lr_scale,
                        "lr": ft_lr,
                    },
                    "phase_artifacts": {
                        "masked_model": str(masked_model_ckpt),
                        "compact_model": str(compact_model_ckpt),
                        "finetuned_model": str(sparse_model_ckpt),
                    },
                    "final_stats": summarize_sparsity(
                        sparse_backbone,
                        include_bias=include_bias,
                    ),
                }
            else:
                sparse_backbone = copy.deepcopy(backbone)
                post_cfg = dict(sparsity_cfg)
                post_cfg.setdefault("device", str(device))
                sparse_backbone, sparse_report = apply_post_training_sparsification(
                    model=sparse_backbone,
                    method=sparse_method,
                    config=post_cfg,
                    inplace=True,
                )

            if _is_compact_sparse_method(sparse_method):
                torch.save(sparse_backbone, sparse_model_ckpt)
            else:
                torch.save(sparse_backbone.state_dict(), sparse_model_ckpt)
            artifact_cache.save_json(sparse_report, sparse_report_key, kind="metrics")

        sparse_weight_stats, sparse_weight_fig = _analyze_and_store_weights(
            model=sparse_backbone,
            artifact_cache=artifact_cache,
            tag=sparse_tag,
            weight_cfg=weight_cfg,
        )

        sparse_denoised_key = f"ts_denoised_{sparse_tag}"
        sparse_denoise_profile_key = f"denoise_profile_{sparse_tag}"
        sparse_denoised_npz = artifact_cache.npz_path(sparse_denoised_key)

        if sparse_denoised_npz.exists():
            sparse_denoised = np.load(sparse_denoised_npz)
            X_denoised_sparse, y_denoised_sparse = sparse_denoised["X"], sparse_denoised["y"]
            sparse_denoise_metrics = artifact_cache.load_json(sparse_denoise_profile_key, kind="metrics") or {}
        else:
            dg_cfg = _as_dict(config.get("denograd", {}))
            denograd_cls = DenoGrad
            if denograd_cls is None:  # pragma: no cover
                raise RuntimeError("DenoGrad is not available.")
            sparse_denoiser = denograd_cls(model=sparse_backbone, criterion=nn.MSELoss(), device=device)
            sparse_denoiser.fit(
                X=X_noisy,
                y=y_noisy,
                is_ts=True,
                window_size=window_size,
                future=future,
                stride=1,
            )
            (X_denoised_sparse, y_denoised_sparse, _, _), sparse_elapsed, sparse_vram = _measure_denoising_call(
                lambda: sparse_denoiser.transform(
                    nrr=dg_cfg.get("nrr", 0.01),
                    nr_threshold=dg_cfg.get("threshold", 0.1),
                    max_epochs=dg_cfg.get("max_iters", 150),
                    denoise_y=True,
                    batch_size=dg_cfg.get("batch_size", 1024),
                    save_gradients=False,
                ),
                device,
            )
            artifact_cache.save_numpy(sparse_denoised_key, X=X_denoised_sparse, y=y_denoised_sparse)
            sparse_denoise_metrics = {
                "denoising_seconds": sparse_elapsed,
                "denoising_vram": sparse_vram,
                "backbone_stats": model_size_bytes(sparse_backbone),
            }
            artifact_cache.save_json(sparse_denoise_metrics, sparse_denoise_profile_key, kind="metrics")

        loaders_sparse_clean, _, _, _ = get_ts_dataloaders(
            X=X_denoised_sparse,
            y=y_denoised_sparse,
            window_size=window_size,
            future=future,
            batch_size=batch_size,
            val_split=dataset_cfg.get("val_split", 0.1),
            test_split=dataset_cfg.get("test_split", 0.2),
            cnn=False,
        )
        sparse_results_clean, sparse_clean_meta = run_ts_benchmark(
            loaders_sparse_clean,
            device,
            benchmark_cfg=benchmark_cfg,
            artifact_cache=artifact_cache,
            cache_key=f"clean_{sparse_tag}",
            reuse_cached=True,
            profile_models=True,
            return_metadata=True,
        )

        sparse_evaluation = evaluate_changes(
            X_noisy,
            X_denoised_sparse,
            results_noisy,
            sparse_results_clean,
        )

        summary["sparse"] = {
            "method": sparse_method,
            "ratio": sparse_ratio,
            "report": sparse_report,
            "results_clean": sparse_results_clean,
            "denoising_profile": sparse_denoise_metrics,
            "benchmark_clean_profile": sparse_clean_meta,
            "evaluation": sparse_evaluation,
        }
        summary["weight_analysis"]["sparse"] = {
            "stats": sparse_weight_stats,
            "histogram_figure": sparse_weight_fig,
        }

        dense_sparse_comparison = _build_dense_sparse_comparison(
            summary.get("denoising_profile", {}),
            sparse_denoise_metrics,
        )
        comparison_key = f"dense_vs_sparse_comparison_{sparse_tag}"
        artifact_cache.save_json(dense_sparse_comparison, comparison_key, kind="metrics")
        artifact_cache.write_text(
            f"{comparison_key}_table",
            _comparison_table_markdown(dense_sparse_comparison),
            kind="logs",
        )
        comparison_plot = _save_dense_sparse_plot(
            dense_sparse_comparison,
            artifact_cache.paths.figures / f"{comparison_key}.png",
        )
        summary["sparse"]["dense_vs_sparse_comparison"] = dense_sparse_comparison
        summary["sparse"]["dense_vs_sparse_plot"] = comparison_plot

        artifact_cache.save_json(sparse_evaluation, "evaluation_sparse", kind="metrics")
        artifact_cache.save_json(summary["sparse"], "sparse_summary", kind="metrics")
        artifact_cache.update_manifest(
            "evaluation_sparse",
            {"path": str(artifact_cache.json_path("evaluation_sparse", kind="metrics"))},
        )
        artifact_cache.update_manifest(
            "sparse_summary",
            {"path": str(artifact_cache.json_path("sparse_summary", kind="metrics"))},
        )
        artifact_cache.update_manifest(
            comparison_key,
            {
                "json": str(artifact_cache.json_path(comparison_key, kind="metrics")),
                "table": str(artifact_cache.text_path(f"{comparison_key}_table", kind="logs")),
                "plot": comparison_plot.get("path"),
                "plot_saved": comparison_plot.get("saved", False),
            },
        )

    artifact_cache.save_json(summary, "summary_ts", kind="metrics")
    artifact_cache.update_manifest(
        "summary_ts",
        {"path": str(artifact_cache.json_path("summary_ts", kind="metrics"))},
    )
    return summary
