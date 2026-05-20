"""Run a real hybrid sparse pipeline on tabular + time-series datasets.

Pipeline per domain:
1) sparse_on_training (sigma-per-layer, k=0.1)
2) structural post-training compaction
3) short fine-tuning over compacted model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.libs.experiment_runner import run_tabular_experiment, run_ts_experiment


def avg_improvement(eval_dict: dict) -> float | None:
    values = [
        value
        for key, value in eval_dict.items()
        if key.endswith("_improvement_pct") and isinstance(value, (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


def build_tabular_config(version: str, seed: int, benchmark_profile: str) -> dict:
    full_benchmark = {
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
    lean_benchmark = {
        "ridge": {"enabled": True},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "tabpfn": {"enabled": False},
        "dnn": {"enabled": False},
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
        "denograd": {
            "nrr": 0.01,
            "threshold": 0.1,
            "max_iters": 150,
            "batch_size": 1024,
        },
        "benchmark": full_benchmark if benchmark_profile == "full" else lean_benchmark,
        "sparsity": {
            "enabled": True,
            "method": "sparse_on_training",
            "k": 0.1,
            "layer_priority_strength": 1.0,
            "include_bias": False,
            "post_training_method": "structured_compact",
            "compact_ratio": 0.25,
            "fine_tune": {
                "epochs": 12,
                "patience": 5,
                "lr_scale": 0.1,
            },
        },
    }


def build_ts_config(version: str, seed: int, benchmark_profile: str) -> dict:
    full_benchmark = {
        "dnn": {
            "enabled": True,
            "hidden_dims": [64, 32],
            "max_epochs": 100,
            "patience": 15,
        },
        "lstm": {
            "enabled": True,
            "hidden_size": 64,
            "num_layers": 2,
            "max_epochs": 100,
            "patience": 15,
        },
        "ridge": {"enabled": False},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "xlstm": {"enabled": False},
        "ohshulih": {"enabled": False},
        "dlinear": {"enabled": False},
    }
    lean_benchmark = {
        "dnn": {"enabled": False},
        "lstm": {
            "enabled": True,
            "hidden_size": 64,
            "num_layers": 2,
            "max_epochs": 40,
            "patience": 10,
        },
        "ridge": {"enabled": False},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "xlstm": {"enabled": False},
        "ohshulih": {"enabled": False},
        "dlinear": {"enabled": False},
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
        "denograd": {
            "nrr": 0.01,
            "threshold": 0.1,
            "max_iters": 150,
            "batch_size": 512,
        },
        "benchmark": full_benchmark if benchmark_profile == "full" else lean_benchmark,
        "sparsity": {
            "enabled": True,
            "method": "sparse_on_training",
            "k": 0.1,
            "layer_priority_strength": 1.0,
            "include_bias": False,
            "post_training_method": "structured_compact",
            "compact_ratio": 0.25,
            "fine_tune": {
                "epochs": 12,
                "patience": 5,
                "lr_scale": 0.1,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real hybrid sparse experiment on tabular+time-series")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="real_hybrid_tabular_ts_v1")
    parser.add_argument("--summary-name", default="real_hybrid_tabular_ts_results.json")
    parser.add_argument("--benchmark-profile", choices=["full", "lean"], default="full")
    args = parser.parse_args()

    cfg_tab = build_tabular_config(version=args.version, seed=args.seed, benchmark_profile=args.benchmark_profile)
    cfg_ts = build_ts_config(version=args.version, seed=args.seed, benchmark_profile=args.benchmark_profile)

    runs: list[dict] = []

    print("[RUN] tabular :: house_prices :: sparse_on_training + structured_compact + fine_tune")
    summary_tab = run_tabular_experiment(cfg_tab)
    runs.append(
        {
            "domain": "tabular",
            "dataset": "house_prices",
            "model": "dnn",
            "pipeline": "hybrid_on_training_compaction_finetune",
            "dense_avg_improvement_pct": avg_improvement(summary_tab["evaluation"]),
            "sparse_avg_improvement_pct": avg_improvement(summary_tab["sparse"]["evaluation"]),
            "dense_denoise_seconds": summary_tab["denoising_profile"]["denoising_seconds"],
            "sparse_denoise_seconds": summary_tab["sparse"]["denoising_profile"]["denoising_seconds"],
            "report": summary_tab["sparse"]["report"],
        }
    )

    print("[RUN] time_series :: daily_climate :: sparse_on_training + structured_compact + fine_tune")
    summary_ts = run_ts_experiment(cfg_ts)
    runs.append(
        {
            "domain": "time_series",
            "dataset": "daily_climate",
            "model": "lstm",
            "pipeline": "hybrid_on_training_compaction_finetune",
            "dense_avg_improvement_pct": avg_improvement(summary_ts["evaluation"]),
            "sparse_avg_improvement_pct": avg_improvement(summary_ts["sparse"]["evaluation"]),
            "dense_denoise_seconds": summary_ts["denoising_profile"]["denoising_seconds"],
            "sparse_denoise_seconds": summary_ts["sparse"]["denoising_profile"]["denoising_seconds"],
            "report": summary_ts["sparse"]["report"],
        }
    )

    out_dir = Path("out/meta/summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.summary_name
    out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"[DONE] summary saved at {out_path}")


if __name__ == "__main__":
    main()
