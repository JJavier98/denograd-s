"""Smoke pilot runner for dense vs sparse DenoGrad experiments on one tabular and one TS dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_runner import run_tabular_experiment, run_ts_experiment


def _base_sparse_cfg(method: str, ratio: float) -> dict:
    return {
        "enabled": True,
        "method": method,
        "ratio": ratio,
        "include_bias": False,
        "enable_hardware_acceleration": True,
        "start_epoch": 0,
        "end_epoch": 1,
        "power": 3.0,
        "init_mode": "random",
        "seed": 42,
    }


def _light_tabular_config(artifacts_dir: str, method: str, ratio: float) -> dict:
    return {
        "artifacts_dir": artifacts_dir,
        "version": "smoke_v1",
        "seed": 42,
        "dataset": {
            "path": "data/tabular/parkinsons/clean.parquet",
            "target_col": "y",
            "noise_std": 0.05,
            "batch_size": 128,
            "test_split": 0.2,
            "val_split": 0.1,
        },
        "model": {
            "name": "dnn",
            "hidden_dims": [64, 32],
            "lr": 1e-3,
            "patience": 2,
            "max_epochs": 2,
        },
        "denograd": {
            "nrr": 1e-3,
            "threshold": 5e-3,
            "max_iters": 10,
            "batch_size": 512,
        },
        "benchmark": {
            "ridge": {"enabled": True},
            "knn": {"enabled": True, "n_neighbors": 5},
            "xgboost": {"enabled": False},
            "tabpfn": {"enabled": False},
            "dnn": {"enabled": True, "max_epochs": 2, "hidden_dims": [64, 32]},
        },
        "weight_analysis": {
            "bins": 80,
            "near_zero_eps": 1e-3,
            "include_bias": True,
            "normality_max_sample": 5000,
            "seed": 42,
        },
        "sparsity": _base_sparse_cfg(method=method, ratio=ratio),
    }


def _light_ts_config(artifacts_dir: str, method: str, ratio: float) -> dict:
    return {
        "artifacts_dir": artifacts_dir,
        "version": "smoke_v1",
        "seed": 42,
        "dataset": {
            "path": "data/time_series/daily_climate/DailyDelhiClimateTrain.csv",
            "target_col": "meantemp",
            "noise_std": 0.05,
            "batch_size": 64,
            "test_split": 0.2,
            "val_split": 0.1,
            "window_size": 24,
            "future": 1,
        },
        "model": {
            "name": "lstm",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "lr": 1e-3,
            "patience": 2,
            "max_epochs": 2,
        },
        "denograd": {
            "nrr": 1e-3,
            "threshold": 5e-3,
            "max_iters": 10,
            "batch_size": 256,
        },
        "benchmark": {
            "ridge": {"enabled": True},
            "knn": {"enabled": True, "n_neighbors": 5},
            "xgboost": {"enabled": False},
            "dnn": {"enabled": True, "max_epochs": 2, "hidden_dims": [64, 32]},
            "lstm": {"enabled": True, "max_epochs": 2, "hidden_dim": 32, "num_layers": 1, "dropout": 0.0},
            "xlstm": {"enabled": False},
            "ohshulih": {"enabled": False},
            "dlinear": {"enabled": True, "moving_avg": 7},
        },
        "weight_analysis": {
            "bins": 80,
            "near_zero_eps": 1e-3,
            "include_bias": True,
            "normality_max_sample": 5000,
            "seed": 42,
        },
        "sparsity": _base_sparse_cfg(method=method, ratio=ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke pilot (tabular + time-series)")
    parser.add_argument("--artifacts-dir", default="artifacts_smoke", help="Output artifacts root")
    parser.add_argument(
        "--sparse-method",
        default="magnitude_unstructured",
        choices=["magnitude_unstructured", "structured", "semi_structured_2to4", "2to4", "gradual_magnitude", "sparse_from_scratch"],
    )
    parser.add_argument("--sparse-ratio", type=float, default=0.5)
    args = parser.parse_args()

    tab_cfg = _light_tabular_config(args.artifacts_dir, args.sparse_method, args.sparse_ratio)
    ts_cfg = _light_ts_config(args.artifacts_dir, args.sparse_method, args.sparse_ratio)

    print("[SMOKE] Running tabular experiment...")
    tab_summary = run_tabular_experiment(tab_cfg)
    print("[SMOKE] Running time-series experiment...")
    ts_summary = run_ts_experiment(ts_cfg)

    out_path = Path(args.artifacts_dir) / "smoke_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"tabular": tab_summary, "time_series": ts_summary}, indent=2),
        encoding="utf-8",
    )
    print(f"[SMOKE] Summary written to {out_path}")


if __name__ == "__main__":
    main()
