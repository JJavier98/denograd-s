"""Run full Energy-Sparse experimentation campaign.

Phases (per domain):
1) Screening on lighthouse dataset
2) Energy-target ablation on lighthouse dataset
3) Full campaign on all datasets in domain using best backbone + best energy target

This runner persists incremental progress so long executions are resumable and auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.libs.experiment_runner import run_tabular_experiment, run_ts_experiment


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    target_col: str


TABULAR_DATASETS: list[DatasetSpec] = [
    DatasetSpec("house_prices", "data/tabular/house_prices/clean.parquet", "y"),
    DatasetSpec("lattice_physics", "data/tabular/lattice_physics/clean.parquet", "y"),
    DatasetSpec("parkinsons", "data/tabular/parkinsons/clean.parquet", "y"),
    DatasetSpec("rt_iot2022", "data/tabular/rt_iot2022/clean.parquet", "y"),
    # support clean.parquet does not expose target 'y'; use last known label-like column
    DatasetSpec("support", "data/tabular/support/clean.parquet", "race_white"),
]

TS_DATASETS: list[DatasetSpec] = [
    DatasetSpec("daily_climate", "data/time_series/daily_climate/DailyDelhiClimateTrain.csv", "meantemp"),
    DatasetSpec("microsoft_stock", "data/time_series/microsoft_stock/Microsoft_Stock.csv", "Close"),
    DatasetSpec("ECL", "data/time_series/ECL/ECL.csv", "MT_320"),
    DatasetSpec("ETTh1", "data/time_series/ETT/ETTh1.csv", "OT"),
    DatasetSpec("ETTh2", "data/time_series/ETT/ETTh2.csv", "OT"),
    DatasetSpec("ETTm1", "data/time_series/ETT/ETTm1.csv", "OT"),
    DatasetSpec("ETTm2", "data/time_series/ETT/ETTm2.csv", "OT"),
    DatasetSpec("WTH", "data/time_series/WTH/WTH.csv", "WetBulbCelsius"),
]


def _avg_improvement(eval_dict: dict[str, Any]) -> float | None:
    vals = [v for k, v in eval_dict.items() if k.endswith("_improvement_pct") and isinstance(v, (int, float))]
    return float(sum(vals) / len(vals)) if vals else None


def _safe_float(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _tabular_benchmark_cfg(profile: str) -> dict[str, Any]:
    full = {
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
    lean = {
        "ridge": {"enabled": True},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "tabpfn": {"enabled": False},
        "dnn": {"enabled": False},
    }
    return full if profile == "full" else lean


def _ts_benchmark_cfg(profile: str) -> dict[str, Any]:
    full = {
        "dnn": {"enabled": True, "hidden_dims": [64, 32], "max_epochs": 100, "patience": 15},
        "lstm": {"enabled": True, "hidden_size": 64, "num_layers": 2, "max_epochs": 100, "patience": 15},
        "ridge": {"enabled": False},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "xlstm": {"enabled": False},
        "ohshulih": {"enabled": False},
        "dlinear": {"enabled": False},
    }
    lean = {
        "dnn": {"enabled": False},
        "lstm": {"enabled": True, "hidden_size": 64, "num_layers": 2, "max_epochs": 40, "patience": 10},
        "ridge": {"enabled": False},
        "knn": {"enabled": False},
        "xgboost": {"enabled": False},
        "xlstm": {"enabled": False},
        "ohshulih": {"enabled": False},
        "dlinear": {"enabled": False},
    }
    return full if profile == "full" else lean


def _build_sparse_cfg(enabled: bool, energy_target: float) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "include_bias": False,
        "post_training_method": "structured_compact",
        "compact_selection": "energy",
        "compact_energy_target": energy_target,
        "fine_tune": {"epochs": 12, "patience": 5, "lr_scale": 0.1},
    }


def _build_tabular_cfg(
    *,
    dataset: DatasetSpec,
    seed: int,
    version: str,
    backbone: str,
    sparse_enabled: bool,
    energy_target: float,
    device: str,
    benchmark_profile: str,
) -> dict[str, Any]:
    if backbone == "dnn":
        model_cfg = {
            "name": "dnn",
            "hidden_dims": [128, 64, 32],
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        }
    elif backbone == "transformer":
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
        raise ValueError(f"Unsupported tabular backbone: {backbone}")

    return {
        "artifacts_dir": "out",
        "version": version,
        "seed": seed,
        "device": device,
        "dataset": {
            "path": dataset.path,
            "target_col": dataset.target_col,
            "noise_std": 0.05,
            "batch_size": 128,
            "test_split": 0.2,
            "val_split": 0.1,
        },
        "model": model_cfg,
        "denograd": {"nrr": 0.01, "threshold": 0.1, "max_iters": 150, "batch_size": 1024},
        "benchmark": _tabular_benchmark_cfg(benchmark_profile),
        "sparsity": _build_sparse_cfg(sparse_enabled, energy_target),
    }


def _build_ts_cfg(
    *,
    dataset: DatasetSpec,
    seed: int,
    version: str,
    backbone: str,
    sparse_enabled: bool,
    energy_target: float,
    device: str,
    benchmark_profile: str,
) -> dict[str, Any]:
    if backbone == "lstm":
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
        # Not integrated yet in src/models/get_model; kept for fallback logging.
        model_cfg = {
            "name": backbone,
            "lr": 1e-3,
            "patience": 15,
            "max_epochs": 100,
        }

    return {
        "artifacts_dir": "out",
        "version": version,
        "seed": seed,
        "device": device,
        "dataset": {
            "path": dataset.path,
            "target_col": dataset.target_col,
            "noise_std": 0.05,
            "batch_size": 64,
            "window_size": 24,
            "future": 1,
            "val_split": 0.1,
            "test_split": 0.2,
        },
        "model": model_cfg,
        "denograd": {"nrr": 0.01, "threshold": 0.1, "max_iters": 150, "batch_size": 512},
        "benchmark": _ts_benchmark_cfg(benchmark_profile),
        "sparsity": _build_sparse_cfg(sparse_enabled, energy_target),
    }


def _extract_metrics(summary: dict[str, Any], sparse: bool) -> dict[str, Any]:
    if not sparse:
        eval_obj = summary.get("evaluation", {})
        dprof = summary.get("denoising_profile", {})
        bst = dprof.get("backbone_stats", {})
        compact_report = {}
    else:
        sobj = summary.get("sparse", {})
        eval_obj = sobj.get("evaluation", {})
        dprof = sobj.get("denoising_profile", {})
        bst = dprof.get("backbone_stats", {})
        compact_report = sobj.get("report", {}).get("compact_report", {})

    return {
        "avg_improvement_pct": _avg_improvement(eval_obj),
        "input_correlation": _safe_float(eval_obj.get("input_correlation")),
        "wasserstein_distance": _safe_float(eval_obj.get("wasserstein_distance")),
        "denoising_seconds": _safe_float(dprof.get("denoising_seconds")),
        "total_params": _safe_float(bst.get("total_params")),
        "param_bytes": _safe_float(bst.get("param_bytes")),
        "param_reduction_ratio": _safe_float(compact_report.get("param_reduction_ratio")),
    }


def _score_runs(rows: list[dict[str, Any]], key_field: str) -> str:
    # Higher avg_improvement better; tie-break lower SWD; then higher corr.
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(str(row[key_field]), []).append(row)

    best_key = None
    best_tuple = None
    for key, vals in by_key.items():
        imp = [v["avg_improvement_pct"] for v in vals if isinstance(v.get("avg_improvement_pct"), (int, float))]
        swd = [v["wasserstein_distance"] for v in vals if isinstance(v.get("wasserstein_distance"), (int, float))]
        corr = [v["input_correlation"] for v in vals if isinstance(v.get("input_correlation"), (int, float))]

        m_imp = sum(imp) / len(imp) if imp else float("-inf")
        m_swd = sum(swd) / len(swd) if swd else float("inf")
        m_corr = sum(corr) / len(corr) if corr else float("-inf")

        score_tuple = (m_imp, -m_swd, m_corr)
        if best_tuple is None or score_tuple > best_tuple:
            best_tuple = score_tuple
            best_key = key

    if best_key is None:
        raise RuntimeError(f"Could not select best {key_field}; no valid runs")
    return best_key


def _append_progress(progress_path: Path, item: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")


def _load_existing_progress(progress_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not progress_path.exists():
        return [], [], set()

    loaded_rows: list[dict[str, Any]] = []
    loaded_notes: list[dict[str, Any]] = []
    done_versions: set[str] = set()

    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                loaded_notes.append({"phase": "resume", "event": "invalid_json_line", "line": line[:200]})
                continue

            if not isinstance(item, dict):
                loaded_notes.append({"phase": "resume", "event": "invalid_item_type", "item_type": str(type(item))})
                continue

            version = item.get("version")
            if isinstance(version, str) and version:
                done_versions.add(version)

            if "error" in item:
                loaded_notes.append(item)
            else:
                loaded_rows.append(item)

    return loaded_rows, loaded_notes, done_versions


def _run_single(
    *,
    domain: str,
    dataset: DatasetSpec,
    backbone: str,
    seed: int,
    sparse_enabled: bool,
    energy_target: float,
    version: str,
    device: str,
    benchmark_profile: str,
) -> dict[str, Any]:
    if domain == "tabular":
        cfg = _build_tabular_cfg(
            dataset=dataset,
            seed=seed,
            version=version,
            backbone=backbone,
            sparse_enabled=sparse_enabled,
            energy_target=energy_target,
            device=device,
            benchmark_profile=benchmark_profile,
        )
        summary = run_tabular_experiment(cfg)
    else:
        cfg = _build_ts_cfg(
            dataset=dataset,
            seed=seed,
            version=version,
            backbone=backbone,
            sparse_enabled=sparse_enabled,
            energy_target=energy_target,
            device=device,
            benchmark_profile=benchmark_profile,
        )
        summary = run_ts_experiment(cfg)

    row = {
        "domain": domain,
        "dataset": dataset.name,
        "backbone": backbone,
        "seed": seed,
        "variant": "sparse_energy" if sparse_enabled else "dense",
        "energy_target": energy_target if sparse_enabled else None,
        "version": version,
    }
    row.update(_extract_metrics(summary, sparse=sparse_enabled))
    return row


def _run_domain_campaign(
    *,
    domain: str,
    seeds: list[int],
    energy_targets: list[float],
    version_prefix: str,
    device: str,
    benchmark_profile: str,
) -> dict[str, Any]:
    if domain not in {"tabular", "time_series"}:
        raise ValueError("domain must be 'tabular' or 'time_series'")

    progress_path = Path("out/meta/summaries") / f"{version_prefix}_{domain}_progress.jsonl"
    results_path = Path("out/meta/summaries") / f"{version_prefix}_{domain}_results.json"

    lighthouse = TABULAR_DATASETS[0] if domain == "tabular" else TS_DATASETS[0]
    all_datasets = TABULAR_DATASETS if domain == "tabular" else TS_DATASETS

    screening_backbones = ["dnn", "transformer"] if domain == "tabular" else ["lstm", "itransformer", "patchtst", "transformer_vanilla"]

    rows, notes, done_versions = _load_existing_progress(progress_path)

    # Phase 1: screening
    if domain == "time_series":
        # Fallback chain: keep first transformer candidate that actually runs.
        ts_transformer_candidate = None
        for cand in ["itransformer", "patchtst", "transformer_vanilla"]:
            try:
                probe_version = f"{version_prefix}_{domain}_probe_{cand}_s{seeds[0]}"
                _ = _run_single(
                    domain=domain,
                    dataset=lighthouse,
                    backbone=cand,
                    seed=seeds[0],
                    sparse_enabled=False,
                    energy_target=energy_targets[-1],
                    version=probe_version,
                    device=device,
                    benchmark_profile=benchmark_profile,
                )
                ts_transformer_candidate = cand
                notes.append({"phase": "screening", "event": "ts_transformer_candidate_selected", "candidate": cand})
                break
            except Exception as exc:
                notes.append({"phase": "screening", "event": "ts_transformer_candidate_failed", "candidate": cand, "error": str(exc)})

        screening_backbones = ["lstm"] + ([ts_transformer_candidate] if ts_transformer_candidate else [])

    for backbone in screening_backbones:
        for seed in seeds:
            for sparse_enabled in (False, True):
                version = f"{version_prefix}_{domain}_screen_{backbone}_{'sparse' if sparse_enabled else 'dense'}_s{seed}"
                if version in done_versions:
                    continue
                try:
                    row = _run_single(
                        domain=domain,
                        dataset=lighthouse,
                        backbone=backbone,
                        seed=seed,
                        sparse_enabled=sparse_enabled,
                        energy_target=energy_targets[-1],
                        version=version,
                        device=device,
                        benchmark_profile=benchmark_profile,
                    )
                    row["phase"] = "screening"
                    rows.append(row)
                    done_versions.add(version)
                    _append_progress(progress_path, row)
                except Exception as exc:
                    err = {
                        "phase": "screening",
                        "domain": domain,
                        "dataset": lighthouse.name,
                        "backbone": backbone,
                        "seed": seed,
                        "variant": "sparse_energy" if sparse_enabled else "dense",
                        "error": str(exc),
                    }
                    notes.append(err)
                    _append_progress(progress_path, err)

    screening_sparse_rows = [r for r in rows if r.get("phase") == "screening" and r.get("variant") == "sparse_energy"]
    best_backbone = _score_runs(screening_sparse_rows, "backbone")

    # Phase 2: ablation on lighthouse dataset
    for energy_target in energy_targets:
        for seed in seeds:
            for sparse_enabled in (False, True):
                version = f"{version_prefix}_{domain}_ablate_{best_backbone}_{energy_target}_{'sparse' if sparse_enabled else 'dense'}_s{seed}"
                if version in done_versions:
                    continue
                try:
                    row = _run_single(
                        domain=domain,
                        dataset=lighthouse,
                        backbone=best_backbone,
                        seed=seed,
                        sparse_enabled=sparse_enabled,
                        energy_target=energy_target,
                        version=version,
                        device=device,
                        benchmark_profile=benchmark_profile,
                    )
                    row["phase"] = "ablation"
                    rows.append(row)
                    done_versions.add(version)
                    _append_progress(progress_path, row)
                except Exception as exc:
                    err = {
                        "phase": "ablation",
                        "domain": domain,
                        "dataset": lighthouse.name,
                        "backbone": best_backbone,
                        "energy_target": energy_target,
                        "seed": seed,
                        "variant": "sparse_energy" if sparse_enabled else "dense",
                        "error": str(exc),
                    }
                    notes.append(err)
                    _append_progress(progress_path, err)

    ablation_sparse_rows = [
        r
        for r in rows
        if r.get("phase") == "ablation"
        and r.get("variant") == "sparse_energy"
        and r.get("backbone") == best_backbone
    ]
    best_energy_target = float(_score_runs(ablation_sparse_rows, "energy_target"))

    # Phase 3: full campaign on all datasets with selected backbone + energy target
    for dataset in all_datasets:
        for seed in seeds:
            for sparse_enabled in (False, True):
                version = f"{version_prefix}_{domain}_full_{dataset.name}_{best_backbone}_{'sparse' if sparse_enabled else 'dense'}_s{seed}"
                if version in done_versions:
                    continue
                try:
                    row = _run_single(
                        domain=domain,
                        dataset=dataset,
                        backbone=best_backbone,
                        seed=seed,
                        sparse_enabled=sparse_enabled,
                        energy_target=best_energy_target,
                        version=version,
                        device=device,
                        benchmark_profile=benchmark_profile,
                    )
                    row["phase"] = "full_campaign"
                    rows.append(row)
                    done_versions.add(version)
                    _append_progress(progress_path, row)
                except Exception as exc:
                    err = {
                        "phase": "full_campaign",
                        "domain": domain,
                        "dataset": dataset.name,
                        "backbone": best_backbone,
                        "seed": seed,
                        "variant": "sparse_energy" if sparse_enabled else "dense",
                        "error": str(exc),
                    }
                    notes.append(err)
                    _append_progress(progress_path, err)

    payload = {
        "domain": domain,
        "seeds": seeds,
        "energy_targets": energy_targets,
        "best_backbone": best_backbone,
        "best_energy_target": best_energy_target,
        "rows": rows,
        "notes": notes,
        "progress_log": str(progress_path),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Energy-Sparse experimentation campaign")
    parser.add_argument("--domain", choices=["tabular", "time_series"], required=True)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--energy-targets", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--version-prefix", default="energy_full_campaign")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-profile", choices=["full", "lean"], default="full")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    energy_targets = [float(s.strip()) for s in args.energy_targets.split(",") if s.strip()]

    payload = _run_domain_campaign(
        domain=args.domain,
        seeds=seeds,
        energy_targets=energy_targets,
        version_prefix=args.version_prefix,
        device=args.device,
        benchmark_profile=args.benchmark_profile,
    )

    print("[DONE]", args.domain)
    print("  best_backbone:", payload["best_backbone"])
    print("  best_energy_target:", payload["best_energy_target"])
    print("  rows:", len(payload["rows"]))
    print("  notes:", len(payload["notes"]))


if __name__ == "__main__":
    main()
