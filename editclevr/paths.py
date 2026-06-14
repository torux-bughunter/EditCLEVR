"""Portable path resolution for EditCLEVR on any machine."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "real_phase1_20k_regenerated"

_REPO_MARKERS = ("pyproject.toml", "requirements.txt", "editclevr")

DATASET_PATH_KEYS = (
    "before_image",
    "after_image",
    "before_scene_json",
    "after_scene_json",
    "instance_masks_before",
    "instance_masks_after",
)


def in_colab() -> bool:
    if os.environ.get("EDITCLEVR_IN_COLAB", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return "google.colab" in sys.modules


@lru_cache(maxsize=1)
def repo_root() -> Path:
    env_root = os.environ.get("EDITCLEVR_REPO_DIR", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if any((parent / marker).exists() for marker in _REPO_MARKERS):
            return parent
    return Path.cwd().resolve()


def outputs_dir(name: str | None = None) -> Path:
    base = repo_root() / "outputs"
    return base / name if name else base


def external_dir() -> Path:
    return repo_root() / "external"


def cache_dir() -> Path:
    path = repo_root() / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_dir(name: str | None = None) -> Path:
    if name:
        path = cache_dir() / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.gettempdir())


def dataset_dir() -> Path:
    env_dataset = os.environ.get("EDITCLEVR_DATASET_DIR", "").strip()
    if env_dataset:
        return Path(env_dataset).expanduser().resolve()

    return outputs_dir(DEFAULT_DATASET_NAME)


def resolve_blender_executable(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()

    env_blender = os.environ.get("EDITCLEVR_BLENDER", "").strip()
    if env_blender:
        return env_blender

    which_blender = shutil.which("blender")
    if which_blender:
        return which_blender

    mac_default = Path("/Applications/Blender.app/Contents/MacOS/blender")
    if mac_default.exists():
        return str(mac_default)

    return "blender"


def _path_tail_from_dataset(value: str, dataset_name: str) -> str | None:
    normalized = value.replace("\\", "/")
    marker = f"/{dataset_name}/"
    index = normalized.rfind(marker)
    if index == -1:
        return None
    return normalized[index + len(marker) :]


def _resolve_dataset_path(value: Any, dataset_dir: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value

    expanded = Path(value).expanduser()
    if expanded.exists():
        return str(expanded)

    dataset_name = dataset_dir.name
    tail = _path_tail_from_dataset(value, dataset_name)
    if tail is not None:
        return str(dataset_dir / Path(*tail.split("/")))

    candidate = dataset_dir / value
    if candidate.exists() or not Path(value).is_absolute():
        return str(candidate)

    return value


def rebase_splits_json_paths(splits_json: str | os.PathLike[str]) -> int:
    """Rewrite stale dataset paths in splits.json to this machine's dataset root."""

    splits_path = Path(splits_json).expanduser()
    with splits_path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected splits.json to decode to a dict, got {type(data).__name__}.")

    dataset_path = splits_path.parent.expanduser().resolve()
    changed = 0
    for rows in data.values():
        for row in rows:
            for key in DATASET_PATH_KEYS:
                if key not in row:
                    continue
                old_value = row[key]
                new_value = _resolve_dataset_path(old_value, dataset_path)
                if new_value != old_value:
                    row[key] = new_value
                    changed += 1

    if changed:
        tmp_path = splits_path.with_name(splits_path.name + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(splits_path)
    return changed
