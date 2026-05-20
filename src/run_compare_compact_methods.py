"""Compare two post-training compaction strategies after sparse_on_training masking.

Method A — structured_compact (ratio=0.25):
  Removes the 25% of neurons with the lowest L2-norm row importance.

Method B — compact_zero_neurons (dead_threshold=0.0):
  Removes only neurons whose L2-norm is exactly 0, i.e. neurons that were
  completely zeroed out by the sigma-based mask controller during training.

Both methods share the same upstream phases:
  1) sparse_on_training (sigma per-layer, k=0.1, layer_priority_strength=1.0)
  2) post-training compaction  ← differs here
  3) fine-tune (12 epochs, patience=5, lr_scale=0.1)

Datasets: house_prices (tabular/DNN) + daily_climate (time-series/LSTM).
Benchmark profile: full by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.libs.experiment_runner import run_tabular_experiment, run_ts_experiment


def avg_improvement(eval_dict: dict) -> float | None:
    values = [
        v
        for k, v in eval_dict.items()
        if k.endswith("_improvement_pct") and isinstance(v, (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


_TABULAR_BENCHMARKS_FULL = {
    "ridge": {"enabled": True},
    "knn": {"enabled": True, "n_neighbors": 7},
    "xgboost": {
        "enabled": True,
        "n_estimators": 120,
        "max_depth": 6,
        "learning_rate": 0.08,
    },
    "tabpfn": {"enabled": False},
    "dnn": {
        "enabled": True,
        "max_epochs": 100,
        "hidden_dims": [128, 64, 32],
        "patience": 15,
    },
}

_TABULAR_BENCHMARKS_LEAN = {
    "ridge": {"enabled": True},
    "knn": {"enabled": False},
    "xgboost": {"enabled": False},
    "tabpfn": {"enabled": False},
    "dnn": {"enabled": False},
}

_TS_BENCHMARKS_FULL = {
    "dnn": {"enabled": True, "hidden_dims": [64, 32], "max_epochs": 100, "patience": 15},
    "lstm": {"enabled": True, "hidden_size": 64, "num_layers": 2, "max_epochs": 100, "patience": 15},
    "ridge": {"enabled": False},
    "knn": {"enabled": False},
    "xgboost": {"enabled": False},
    "xlstm": {"enabled": False},
    "ohshulih": {"enabled": False},
    "dlinear": {"enabled": False},
}

_TS_BENCHMARKS_LEAN = {
    "dnn": {"enabled": False},
    "lstm": {"enabled": True, "hidden_size": 64, "num_layers": 2, "max_epochs": 40, "patience": 10},
    "ridge": {"enabled": False},
    "knn": {"enabled": False},
    "xgboost": {"enabled": False},
    "xlstm": {"enabled": False},
    "ohshulih": {"enabled": False},
    "dlinear": {"enabled": False},
}


def _base_tabular_cfg(version: str, seed: int, compact_method: str, benchmark_profile: str) -> dict:
    benchmark = _TABULAR_BENCHMARKS_FULL if benchmark_profile == "full" else _TABULAR_BENCHMARKS_LEAN
    sparsity: dict = {
        "enabled": True,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": compact_method,
        "fine_tune": {"epochs": 12, "patience": 5, "lr_scale": 0.1},
    }
    if compact_method == "structured_compact":
        sparsity["compact_ratio"] = 0.25
    # compact_zero_neurons does not use compact_ratio; dead_threshold=0.0 is the default.

    return {
        "artifacts_dir": "out",
        "version": version,
        "seed": seed,
        "device": "cuda:0",
        "dataset": {
            "path": "data/tabular/house_prices/kc_house_data.csv",
            "target_col": "price",
            "noise_std": 0.05,
            "batch_size": 128,
            "test_split": 0.2,
            "val_split": 0.1,
        },
        "model": {
            "name": "dnn",
            "hidden_dims": [128, 64, 32],
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        },
        "denograd": {"nrr": 0.01, "threshold": 0.1, "max_iters": 150, "batch_size": 1024},
        "benchmark": benchmark,
        "sparsity": sparsity,
    }


def _base_ts_cfg(version: str, seed: int, compact_method: str, benchmark_profile: str) -> dict:
    benchmark = _TS_BENCHMARKS_FULL if benchmark_profile == "full" else _TS_BENCHMARKS_LEAN
    sparsity: dict = {
        "enabled": True,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": compact_method,
        "fine_tune": {"epochs": 12, "patience": 5, "lr_scale": 0.1},
    }
    if compact_method == "structured_compact":
        sparsity["compact_ratio"] = 0.25

    return {
        "artifacts_dir": "out",
        "version": version,
        "seed": seed,
        "device": "cuda:0",
        "dataset": {
            "path": "data/time_series/daily_climate/DailyDelhiClimateTrain.csv",
            "target_col": "meantemp",
            "noise_std": 0.05,
            "batch_size": 64,
            "window_size": 24,
            "future": 1,
            "val_split": 0.1,
            "test_split": 0.2,
        },
        "model": {
            "name": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        },
        "denograd": {"nrr": 0.01, "threshold": 0.1, "max_iters": 150, "batch_size": 512},
        "benchmark": benchmark,
        "sparsity": sparsity,
    }


def _record_tabular(summary: dict, compact_method: str) -> dict:
    return {
        "domain": "tabular",
        "dataset": "house_prices",
        "model": "dnn",
        "compact_method": compact_method,
        "dense_avg_improvement_pct": avg_improvement(summary["evaluation"]),
        "sparse_avg_improvement_pct": avg_improvement(summary["sparse"]["evaluation"]),
        "dense_denoise_seconds": summary["denoising_profile"]["denoising_seconds"],
        "sparse_denoise_seconds": summary["sparse"]["denoising_profile"]["denoising_seconds"],
        "report": summary["sparse"]["report"],
    }


def _record_ts(summary: dict, compact_method: str) -> dict:
    return {
        "domain": "time_series",
        "dataset": "daily_climate",
        "model": "lstm",
        "compact_method": compact_method,
        "dense_avg_improvement_pct": avg_improvement(summary["evaluation"]),
        "sparse_avg_improvement_pct": avg_improvement(summary["sparse"]["evaluation"]),
        "dense_denoise_seconds": summary["denoising_profile"]["denoising_seconds"],
        "sparse_denoise_seconds": summary["sparse"]["denoising_profile"]["denoising_seconds"],
        "report": summary["sparse"]["report"],
    }


def _print_comparison(runs: list[dict]) -> None:
    """Print a simple side-by-side comparison table."""
    domains = sorted({r["domain"] for r in runs})
    for domain in domains:
        domain_runs = [r for r in runs if r["domain"] == domain]
        print(f"\n{'='*60}")
        print(f"  Domain: {domain}")
        print(f"{'='*60}")
        header = f"  {'Metric':<35} " + "  ".join(f"{r['compact_method']:<22}" for r in domain_runs)
        print(header)
        print(f"  {'-'*35} " + "  ".join(["-"*22] * len(domain_runs)))

        def row(label: str, key: str) -> str:
            vals = []
            for r in domain_runs:
                v = r.get(key)
                vals.append(f"{v:<22}" if v is None else f"{v:<22.4f}" if isinstance(v, float) else f"{str(v):<22}")
            return f"  {label:<35} " + "  ".join(vals)

        print(row("dense_avg_improvement_pct (%)", "dense_avg_improvement_pct"))
        print(row("sparse_avg_improvement_pct (%)", "sparse_avg_improvement_pct"))
        print(row("dense_denoise_seconds (s)", "dense_denoise_seconds"))
        print(row("sparse_denoise_seconds (s)", "sparse_denoise_seconds"))

        for r in domain_runs:
            compact_report = (r.get("report") or {}).get("compact_report") or {}
            ratio = compact_report.get("param_reduction_ratio")
            applied = compact_report.get("applied", "?")
            reason = compact_report.get("reason", "")
            tag = f"  param_reduction_ratio [{r['compact_method']}]"
            ratio_str = f"{ratio:.4f}" if isinstance(ratio, float) else str(ratio)
            print(f"  {tag:<55} {ratio_str}  (applied={applied}{', reason='+reason if reason else ''})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare structured_compact vs compact_zero_neurons after sparse_on_training"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version-a", default="compare_compact_ratio_v1",
                        help="Artifact version tag for structured_compact (method A)")
    parser.add_argument("--version-b", default="compare_compact_zero_v1",
                        help="Artifact version tag for compact_zero_neurons (method B)")
    parser.add_argument("--summary-name", default="compare_compact_methods_results.json")
    parser.add_argument("--benchmark-profile", choices=["full", "lean"], default="full")
    args = parser.parse_args()

    runs: list[dict] = []

    # --- Method A: structured_compact (ratio-based) ---
    print("\n[RUN A] tabular :: structured_compact (ratio=0.25)")
    cfg = _base_tabular_cfg(args.version_a, args.seed, "structured_compact", args.benchmark_profile)
    runs.append(_record_tabular(run_tabular_experiment(cfg), "structured_compact"))

    print("[RUN A] time_series :: structured_compact (ratio=0.25)")
    cfg = _base_ts_cfg(args.version_a, args.seed, "structured_compact", args.benchmark_profile)
    runs.append(_record_ts(run_ts_experiment(cfg), "structured_compact"))

    # --- Method B: compact_zero_neurons (dead neurons only) ---
    print("\n[RUN B] tabular :: compact_zero_neurons (dead_threshold=0.0)")
    cfg = _base_tabular_cfg(args.version_b, args.seed, "compact_zero_neurons", args.benchmark_profile)
    runs.append(_record_tabular(run_tabular_experiment(cfg), "compact_zero_neurons"))

    print("[RUN B] time_series :: compact_zero_neurons (dead_threshold=0.0)")
    cfg = _base_ts_cfg(args.version_b, args.seed, "compact_zero_neurons", args.benchmark_profile)
    runs.append(_record_ts(run_ts_experiment(cfg), "compact_zero_neurons"))

    # --- Persist results ---
    out_dir = Path("out/meta/summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.summary_name
    out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\n[SAVED] {out_path}")

    # --- Print comparison ---
    _print_comparison(runs)


if __name__ == "__main__":
    main()
