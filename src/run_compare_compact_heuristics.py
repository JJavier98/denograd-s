"""Compare compact selection heuristics: ratio vs energy-based cumulative selection.

Both use sparse_on_training (sigma per-layer, k=0.1) + structured_compact compaction
followed by fine-tuning, but differ in how neurons are selected for removal:

Method A — ratio (fixed ratio):
  Removes a fixed fraction (default 25%) of neurons ranked by L2-norm importance.

Method B — energy (cumulative energy):
  Removes neurons until the remaining neurons explain >= 95% of cumulative squared-norm energy.
  Typically results in a different prune ratio per layer/domain.

Datasets: house_prices (tabular/DNN) + daily_climate (time-series/LSTM).
Benchmark profile: full.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Agregar el directorio raíz al path para permitir importaciones
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def _base_tabular_cfg(version: str, seed: int, compact_selection: str) -> dict:
    sparsity: dict = {
        "enabled": True,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": "structured_compact",
        "compact_ratio": 0.25,  # used only by 'ratio' mode; ignored by 'energy'
        "compact_selection": compact_selection,  # "ratio" or "energy"
        "compact_energy_target": 0.95,  # only used by 'energy' mode
        "fine_tune": {"epochs": 12, "patience": 5, "lr_scale": 0.1},
    }
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
        "benchmark": _TABULAR_BENCHMARKS_FULL,
        "sparsity": sparsity,
    }


def _base_ts_cfg(version: str, seed: int, compact_selection: str) -> dict:
    sparsity: dict = {
        "enabled": True,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": "structured_compact",
        "compact_ratio": 0.25,
        "compact_selection": compact_selection,
        "compact_energy_target": 0.95,
        "fine_tune": {"epochs": 12, "patience": 5, "lr_scale": 0.1},
    }
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
        "benchmark": _TS_BENCHMARKS_FULL,
        "sparsity": sparsity,
    }


def _record_tabular(summary: dict, compact_method: str) -> dict:
    report = summary["sparse"]["report"]
    compact_report = report.get("compact_report", {})
    return {
        "domain": "tabular",
        "dataset": "house_prices",
        "model": "dnn",
        "compact_selection": compact_method,
        "dense_avg_improvement_pct": avg_improvement(summary["evaluation"]),
        "sparse_avg_improvement_pct": avg_improvement(summary["sparse"]["evaluation"]),
        "dense_denoise_seconds": summary["denoising_profile"]["denoising_seconds"],
        "sparse_denoise_seconds": summary["sparse"]["denoising_profile"]["denoising_seconds"],
        "param_reduction_ratio": compact_report.get("param_reduction_ratio"),
        "selection_mode": compact_report.get("selection_mode"),
        "selection_reports": compact_report.get("selection_reports", []),
        "full_report": report,
    }


def _record_ts(summary: dict, compact_method: str) -> dict:
    report = summary["sparse"]["report"]
    compact_report = report.get("compact_report", {})
    return {
        "domain": "time_series",
        "dataset": "daily_climate",
        "model": "lstm",
        "compact_selection": compact_method,
        "dense_avg_improvement_pct": avg_improvement(summary["evaluation"]),
        "sparse_avg_improvement_pct": avg_improvement(summary["sparse"]["evaluation"]),
        "dense_denoise_seconds": summary["denoising_profile"]["denoising_seconds"],
        "sparse_denoise_seconds": summary["sparse"]["denoising_profile"]["denoising_seconds"],
        "param_reduction_ratio": compact_report.get("param_reduction_ratio"),
        "selection_mode": compact_report.get("selection_mode"),
        "selection_reports": compact_report.get("selection_reports", []),
        "full_report": report,
    }


def _print_comparison(runs: list[dict]) -> None:
    """Print a side-by-side comparison table."""
    domains = sorted({r["domain"] for r in runs})
    for domain in domains:
        domain_runs = [r for r in runs if r["domain"] == domain]
        print(f"\n{'='*70}")
        print(f"  Domain: {domain}")
        print(f"{'='*70}")
        header = f"  {'Metric':<40} " + "  ".join(f"{r['compact_selection']:<18}" for r in domain_runs)
        print(header)
        print(f"  {'-'*40} " + "  ".join(["-"*18] * len(domain_runs)))

        def row(label: str, key: str) -> str:
            vals = []
            for r in domain_runs:
                v = r.get(key)
                if v is None:
                    vals.append(f"{'N/A':<18}")
                elif isinstance(v, float):
                    vals.append(f"{v:<18.4f}")
                else:
                    vals.append(f"{str(v):<18}")
            return f"  {label:<40} " + "  ".join(vals)

        print(row("dense_avg_improvement_pct (%)", "dense_avg_improvement_pct"))
        print(row("sparse_avg_improvement_pct (%)", "sparse_avg_improvement_pct"))
        print(row("dense_denoise_seconds (s)", "dense_denoise_seconds"))
        print(row("sparse_denoise_seconds (s)", "sparse_denoise_seconds"))
        print(row("param_reduction_ratio", "param_reduction_ratio"))
        print(row("selection_mode", "selection_mode"))

        # Show per-layer selection details for energy mode
        for r in domain_runs:
            if r["compact_selection"] == "energy" and r.get("selection_reports"):
                print(f"\n  [{r['compact_selection']}] Per-layer selection details:")
                for sr in r["selection_reports"]:
                    layer_idx = sr.get("layer_index", "?")
                    keep_cnt = sr.get("keep_count", "?")
                    total_units = sr.get("total_units", "?")
                    achieved_energy = sr.get("achieved_energy_ratio")
                    if achieved_energy is not None:
                        print(f"    Layer {layer_idx}: keep {keep_cnt}/{total_units} neurons, achieved {achieved_energy:.4f} energy ratio")
                    else:
                        print(f"    Layer {layer_idx}: keep {keep_cnt}/{total_units} neurons")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compact selection heuristics: ratio vs energy-based cumulative selection"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version-ratio", default="heuristic_ratio_v1",
                        help="Artifact version tag for ratio-based selection")
    parser.add_argument("--version-energy", default="heuristic_energy_v1",
                        help="Artifact version tag for energy-based selection")
    parser.add_argument("--summary-name", default="compare_compact_heuristics_results.json")
    args = parser.parse_args()

    runs: list[dict] = []

    # --- Method A: ratio (fixed ratio) ---
    print("\n[RUN A] tabular :: ratio-based selection (keep 75% by L2-norm importance)")
    cfg = _base_tabular_cfg(args.version_ratio, args.seed, "ratio")
    runs.append(_record_tabular(run_tabular_experiment(cfg), "ratio"))

    print("[RUN A] time_series :: ratio-based selection (keep 75% by L2-norm importance)")
    cfg = _base_ts_cfg(args.version_ratio, args.seed, "ratio")
    runs.append(_record_ts(run_ts_experiment(cfg), "ratio"))

    # --- Method B: energy (cumulative energy) ---
    print("\n[RUN B] tabular :: energy-based selection (keep neurons for 95% cumulative energy)")
    cfg = _base_tabular_cfg(args.version_energy, args.seed, "energy")
    runs.append(_record_tabular(run_tabular_experiment(cfg), "energy"))

    print("[RUN B] time_series :: energy-based selection (keep neurons for 95% cumulative energy)")
    cfg = _base_ts_cfg(args.version_energy, args.seed, "energy")
    runs.append(_record_ts(run_ts_experiment(cfg), "energy"))

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
