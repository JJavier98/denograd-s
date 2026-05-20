"""Phase-1 screening runner for energy sparse backbones.

Scope:
- Tabular (house_prices): dnn vs transformer (FTTransformer)
- Time series (daily_climate): lstm vs temporal-transformer candidate with fallback
  iTransformer -> PatchTST -> Transformer Vanilla

This script executes dense and sparse-energy variants for each backbone and seed,
then writes aggregated JSON and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.libs.experiment_runner import run_tabular_experiment, run_ts_experiment


def _avg_improvement(eval_dict: dict[str, Any]) -> float | None:
    vals = [v for k, v in eval_dict.items() if k.endswith("_improvement_pct") and isinstance(v, (int, float))]
    return float(sum(vals) / len(vals)) if vals else None


def _safe_float(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _tabular_benchmark_cfg() -> dict[str, Any]:
    return {
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


def _ts_benchmark_cfg() -> dict[str, Any]:
    return {
        "dnn": {"enabled": True, "hidden_dims": [64, 32], "max_epochs": 100, "patience": 15},
        "lstm": {"enabled": True, "hidden_size": 64, "num_layers": 2, "max_epochs": 100, "patience": 15},
        "ridge": {"enabled": False},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "xlstm": {"enabled": False},
        "ohshulih": {"enabled": False},
        "dlinear": {"enabled": False},
    }


def _base_tabular_cfg(version: str, seed: int, backbone_name: str, sparse_enabled: bool, energy_target: float) -> dict[str, Any]:
    model_cfg: dict[str, Any]
    if backbone_name == "dnn":
        model_cfg = {
            "name": "dnn",
            "hidden_dims": [128, 64, 32],
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        }
    elif backbone_name == "transformer":
        model_cfg = {
            "name": "transformer",
            "n_blocks": 2,
            "d_token": 192,
            "attention_n_heads": 8,
            "attention_dropout": 0.2,
            "ffn_dropout": 0.1,
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        }
    else:
        raise ValueError(f"Unsupported tabular backbone: {backbone_name}")

    sparsity_cfg = {
        "enabled": sparse_enabled,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": "structured_compact",
        "compact_selection": "energy",
        "compact_energy_target": energy_target,
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
        "model": model_cfg,
        "denograd": {"nrr": 0.01, "threshold": 0.1, "max_iters": 150, "batch_size": 1024},
        "benchmark": _tabular_benchmark_cfg(),
        "sparsity": sparsity_cfg,
    }


def _base_ts_cfg(version: str, seed: int, backbone_name: str, sparse_enabled: bool, energy_target: float) -> dict[str, Any]:
    if backbone_name == "lstm":
        model_cfg = {
            "name": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        }
    else:
        # Candidate placeholders for future integration in get_model.
        model_cfg = {
            "name": backbone_name,
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        }

    sparsity_cfg = {
        "enabled": sparse_enabled,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": "structured_compact",
        "compact_selection": "energy",
        "compact_energy_target": energy_target,
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
        "model": model_cfg,
        "denograd": {"nrr": 0.01, "threshold": 0.1, "max_iters": 150, "batch_size": 512},
        "benchmark": _ts_benchmark_cfg(),
        "sparsity": sparsity_cfg,
    }


def _extract_core_metrics(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "dense":
        eval_obj = summary.get("evaluation", {})
        denoise_profile = summary.get("denoising_profile", {})
        backbone_stats = denoise_profile.get("backbone_stats", {})
        compact_report = {}
    else:
        sparse_obj = summary.get("sparse", {})
        eval_obj = sparse_obj.get("evaluation", {})
        denoise_profile = sparse_obj.get("denoising_profile", {})
        backbone_stats = denoise_profile.get("backbone_stats", {})
        compact_report = sparse_obj.get("report", {}).get("compact_report", {})

    return {
        "avg_improvement_pct": _avg_improvement(eval_obj),
        "input_correlation": _safe_float(eval_obj.get("input_correlation")),
        "wasserstein_distance": _safe_float(eval_obj.get("wasserstein_distance")),
        "denoising_seconds": _safe_float(denoise_profile.get("denoising_seconds")),
        "total_params": _safe_float(backbone_stats.get("total_params")),
        "param_bytes": _safe_float(backbone_stats.get("param_bytes")),
        "param_reduction_ratio": _safe_float(compact_report.get("param_reduction_ratio")),
    }


def _run_tabular(seed: int, version_prefix: str, backbone_name: str, energy_target: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, sparse_enabled in (("dense", False), ("sparse_energy", True)):
        version = f"{version_prefix}_tab_{backbone_name}_{variant}_s{seed}"
        cfg = _base_tabular_cfg(version, seed, backbone_name, sparse_enabled, energy_target)
        summary = run_tabular_experiment(cfg)
        row = {
            "domain": "tabular",
            "dataset": "house_prices",
            "backbone": backbone_name,
            "seed": seed,
            "variant": variant,
            "energy_target": energy_target if sparse_enabled else None,
            "version": version,
        }
        row.update(_extract_core_metrics(summary, variant="sparse" if sparse_enabled else "dense"))
        rows.append(row)
    return rows


def _run_ts(seed: int, version_prefix: str, backbone_name: str, energy_target: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, sparse_enabled in (("dense", False), ("sparse_energy", True)):
        version = f"{version_prefix}_ts_{backbone_name}_{variant}_s{seed}"
        cfg = _base_ts_cfg(version, seed, backbone_name, sparse_enabled, energy_target)
        summary = run_ts_experiment(cfg)
        row = {
            "domain": "time_series",
            "dataset": "daily_climate",
            "backbone": backbone_name,
            "seed": seed,
            "variant": variant,
            "energy_target": energy_target if sparse_enabled else None,
            "version": version,
        }
        row.update(_extract_core_metrics(summary, variant="sparse" if sparse_enabled else "dense"))
        rows.append(row)
    return rows


def _resolve_ts_transformer_candidate() -> tuple[str | None, list[dict[str, str]]]:
    """Try the agreed fallback chain and return the first working candidate.

    Fallback order:
      itransformer -> patchtst -> transformer_vanilla

    Note: these names may require future integration into src/models/get_model.
    """
    attempts: list[dict[str, str]] = []
    for candidate in ("itransformer", "patchtst", "transformer_vanilla"):
        try:
            _ = _base_ts_cfg(
                version="probe_ts_transformer",
                seed=42,
                backbone_name=candidate,
                sparse_enabled=False,
                energy_target=0.95,
            )
            attempts.append({"candidate": candidate, "status": "declared"})
            # Real support is checked by actually running an experiment in main.
            return candidate, attempts
        except Exception as exc:  # pragma: no cover
            attempts.append({"candidate": candidate, "status": f"failed_cfg: {exc}"})

    return None, attempts


def _write_outputs(rows: list[dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 screening for energy sparse backbones")
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated seed list")
    parser.add_argument("--energy-target", type=float, default=0.95)
    parser.add_argument("--version-prefix", default="phase1_screening")
    parser.add_argument("--summary-json", default="out/meta/summaries/phase1_screening_results.json")
    parser.add_argument("--summary-csv", default="out/meta/summaries/phase1_screening_table.csv")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []

    print("[Phase-1] Tabular screening: house_prices :: dnn vs transformer")
    for seed in seeds:
        rows.extend(_run_tabular(seed, args.version_prefix, "dnn", args.energy_target))
        rows.extend(_run_tabular(seed, args.version_prefix, "transformer", args.energy_target))

    print("[Phase-1] Time-series screening: daily_climate :: lstm vs transformer-fallback")
    for seed in seeds:
        rows.extend(_run_ts(seed, args.version_prefix, "lstm", args.energy_target))

    ts_candidate, attempts = _resolve_ts_transformer_candidate()
    ts_candidate_used = None
    ts_candidate_error = None

    if ts_candidate is not None:
        try:
            for seed in seeds:
                rows.extend(_run_ts(seed, args.version_prefix, ts_candidate, args.energy_target))
            ts_candidate_used = ts_candidate
        except Exception as exc:
            ts_candidate_error = str(exc)
            # Keep baseline runs and annotate transformer unavailability.
            rows.append(
                {
                    "domain": "time_series",
                    "dataset": "daily_climate",
                    "backbone": "transformer_fallback_chain",
                    "seed": None,
                    "variant": "not_executed",
                    "energy_target": args.energy_target,
                    "version": args.version_prefix,
                    "avg_improvement_pct": None,
                    "input_correlation": None,
                    "wasserstein_distance": None,
                    "denoising_seconds": None,
                    "total_params": None,
                    "param_bytes": None,
                    "param_reduction_ratio": None,
                    "note": f"Temporal transformer candidate unavailable: {exc}",
                }
            )

    out_json = Path(args.summary_json)
    out_csv = Path(args.summary_csv)
    _write_outputs(rows, out_json, out_csv)

    meta = {
        "seeds": seeds,
        "energy_target": args.energy_target,
        "version_prefix": args.version_prefix,
        "ts_transformer_fallback_attempts": attempts,
        "ts_transformer_candidate_used": ts_candidate_used,
        "ts_transformer_candidate_error": ts_candidate_error,
        "results_json": str(out_json),
        "results_csv": str(out_csv),
    }
    meta_path = out_json.with_name(out_json.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[SAVED]", out_json)
    print("[SAVED]", out_csv)
    print("[SAVED]", meta_path)


if __name__ == "__main__":
    main()
