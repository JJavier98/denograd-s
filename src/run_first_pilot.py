"""Run the first real tabular + time-series pilot with dense vs sparse comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_runner import run_tabular_experiment, run_ts_experiment


def _base_sparse_cfg(method: str, ratio: float, seed: int) -> dict:
    return {
        "enabled": True,
        "method": method,
        "ratio": ratio,
        "include_bias": False,
        "enable_hardware_acceleration": True,
        "start_epoch": 1,
        "end_epoch": 6,
        "power": 3.0,
        "init_mode": "random",
        "seed": seed,
    }


def _tabular_config(dataset_path: str, target_col: str, seed: int, method: str, ratio: float) -> dict:
    return {
        "artifacts_dir": "out",
        "version": "first_pilot_v1",
        "seed": seed,
        "dataset": {
            "path": dataset_path,
            "target_col": target_col,
            "noise_std": 0.05,
            "batch_size": 128,
            "test_split": 0.2,
            "val_split": 0.1,
        },
        "model": {
            "name": "dnn",
            "hidden_dims": [128, 64, 32],
            "lr": 1e-3,
            "patience": 6,
            "max_epochs": 12,
        },
        "denograd": {
            "nrr": 1e-3,
            "threshold": 5e-3,
            "max_iters": 80,
            "batch_size": 1024,
        },
        "benchmark": {
            "ridge": {"enabled": True},
            "knn": {"enabled": True, "n_neighbors": 7},
            "xgboost": {"enabled": True, "n_estimators": 120, "max_depth": 6, "learning_rate": 0.08},
            "tabpfn": {"enabled": False},
            "dnn": {"enabled": True, "max_epochs": 12, "hidden_dims": [128, 64, 32], "patience": 6},
        },
        "weight_analysis": {
            "bins": 120,
            "near_zero_eps": 1e-3,
            "include_bias": True,
            "normality_max_sample": 20000,
            "seed": seed,
        },
        "sparsity": _base_sparse_cfg(method=method, ratio=ratio, seed=seed),
    }


def _ts_config(dataset_path: str, target_col: str, seed: int, method: str, ratio: float) -> dict:
    return {
        "artifacts_dir": "out",
        "version": "first_pilot_v1",
        "seed": seed,
        "dataset": {
            "path": dataset_path,
            "target_col": target_col,
            "noise_std": 0.05,
            "batch_size": 64,
            "test_split": 0.2,
            "val_split": 0.1,
            "window_size": 24,
            "future": 1,
        },
        "model": {
            "name": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "lr": 1e-3,
            "patience": 6,
            "max_epochs": 12,
        },
        "denograd": {
            "nrr": 1e-3,
            "threshold": 5e-3,
            "max_iters": 80,
            "batch_size": 512,
        },
        "benchmark": {
            "ridge": {"enabled": True},
            "knn": {"enabled": True, "n_neighbors": 7},
            "xgboost": {"enabled": False},
            "dnn": {"enabled": True, "max_epochs": 8, "hidden_dims": [128, 64, 32], "patience": 5},
            "lstm": {"enabled": True, "max_epochs": 10, "hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
            "xlstm": {"enabled": False},
            "ohshulih": {"enabled": False},
            "dlinear": {"enabled": True, "moving_avg": 25},
        },
        "weight_analysis": {
            "bins": 120,
            "near_zero_eps": 1e-3,
            "include_bias": True,
            "normality_max_sample": 20000,
            "seed": seed,
        },
        "sparsity": _base_sparse_cfg(method=method, ratio=ratio, seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first real pilot")
    parser.add_argument("--sparse-method", default="magnitude_unstructured")
    parser.add_argument("--sparse-ratio", type=float, default=0.5)
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,123")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    runs = []
    for seed in seeds:
        runs.append(
            ("tabular", "house_prices", _tabular_config("data/tabular/house_prices/kc_house_data.csv", "price", seed, args.sparse_method, args.sparse_ratio))
        )
        runs.append(
            ("tabular", "parkinsons", _tabular_config("data/tabular/parkinsons/clean.parquet", "y", seed, args.sparse_method, args.sparse_ratio))
        )
        runs.append(
            ("time_series", "daily_climate", _ts_config("data/time_series/daily_climate/DailyDelhiClimateTrain.csv", "meantemp", seed, args.sparse_method, args.sparse_ratio))
        )
        runs.append(
            ("time_series", "microsoft_stock", _ts_config("data/time_series/microsoft_stock/Microsoft_Stock.csv", "Close", seed, args.sparse_method, args.sparse_ratio))
        )

    results = []
    for domain, name, cfg in runs:
        print(f"[PILOT] Running {domain}::{name} (seed={cfg['seed']})")
        if domain == "tabular":
            summary = run_tabular_experiment(cfg)
        else:
            summary = run_ts_experiment(cfg)
        results.append({"domain": domain, "name": name, "seed": cfg["seed"], "summary": summary})

    out = Path("out") / "pilot_runs"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "first_pilot_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[PILOT] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
