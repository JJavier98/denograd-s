"""Artifact cache primitives for experiment reproducibility and reuse."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _to_jsonable(obj: Any):
    """Convert nested objects into JSON-serializable structures."""
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def make_experiment_signature(payload: dict[str, Any]) -> str:
    """Create a stable hash for an experiment configuration."""
    normalized = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ArtifactPaths:
    """Structured directories used by the experiment artifact cache."""

    root: Path
    models: Path
    data: Path
    metrics: Path
    figures: Path
    logs: Path


class ExperimentCache:
    """Filesystem cache for checkpoints, denoised data and metrics."""

    def __init__(self, base_dir: str | os.PathLike[str] = "artifacts", signature: Optional[str] = None):
        """Build an artifact namespace optionally scoped by experiment signature."""
        self.base_dir = Path(base_dir)
        self.signature = signature
        self.root = self.base_dir / signature if signature else self.base_dir
        self.paths = ArtifactPaths(
            root=self.root,
            models=self.root / "models",
            data=self.root / "data",
            metrics=self.root / "metrics",
            figures=self.root / "figures",
            logs=self.root / "logs",
        )
        self.ensure_dirs()

    def ensure_dirs(self):
        """Create all cache subdirectories if they do not exist."""
        for path in self.paths.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)

    def _path(self, folder: Path, name: str, suffix: str) -> Path:
        """Compose a path under a target folder using a logical artifact name."""
        return folder / f"{name}{suffix}"

    def model_path(self, name: str) -> Path:
        """Return checkpoint path for a model artifact."""
        return self._path(self.paths.models, name, ".pth")

    def json_path(self, name: str, kind: str = "metrics") -> Path:
        """Return JSON path under a configured artifact kind folder."""
        folder = getattr(self.paths, kind)
        return self._path(folder, name, ".json")

    def npz_path(self, name: str) -> Path:
        """Return compressed numpy artifact path."""
        return self._path(self.paths.data, name, ".npz")

    def text_path(self, name: str, kind: str = "logs") -> Path:
        """Return text artifact path under a configured kind folder."""
        folder = getattr(self.paths, kind)
        return self._path(folder, name, ".txt")

    def save_json(self, payload: dict[str, Any], name: str, kind: str = "metrics") -> Path:
        """Serialize and store a JSON artifact."""
        path = self.json_path(name, kind=kind)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
        return path

    def load_json(self, name: str, kind: str = "metrics") -> Optional[dict[str, Any]]:
        """Load a JSON artifact if present, otherwise return None."""
        path = self.json_path(name, kind=kind)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_numpy(self, name: str, **arrays) -> Path:
        """Save one or more arrays into a compressed NPZ artifact."""
        path = self.npz_path(name)
        np.savez_compressed(path, **arrays)
        return path

    def load_numpy(self, name: str):
        """Load an NPZ artifact if present, otherwise return None."""
        path = self.npz_path(name)
        if not path.exists():
            return None
        return np.load(path, allow_pickle=True)

    def save_torch(self, obj: Any, name: str) -> Path:
        """Persist a torch object under the model artifacts directory."""
        path = self.model_path(name)
        torch.save(obj, path)
        return path

    def load_torch(self, name: str):
        """Load a torch object if present, otherwise return None."""
        path = self.model_path(name)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")

    def write_text(self, name: str, text: str, kind: str = "logs") -> Path:
        """Write plain text artifact content to disk."""
        path = self.text_path(name, kind=kind)
        path.write_text(text, encoding="utf-8")
        return path

    def load_manifest(self) -> dict[str, Any]:
        """Load experiment manifest, creating an empty logical view when absent."""
        manifest = self.load_json("manifest", kind="logs")
        if manifest is None:
            manifest = {
                "signature": self.signature,
                "artifacts": {},
            }
        return manifest

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        """Persist experiment manifest JSON."""
        return self.save_json(manifest, "manifest", kind="logs")

    def update_manifest(self, key: str, payload: dict[str, Any]) -> Path:
        """Upsert one artifact entry in the experiment manifest."""
        manifest = self.load_manifest()
        manifest.setdefault("artifacts", {})[key] = payload
        return self.save_manifest(manifest)
