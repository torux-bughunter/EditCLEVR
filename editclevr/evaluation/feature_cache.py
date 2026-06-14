"""Load pre-extracted per-pair feature caches saved as split NPZ files.

Each cache file is named ``{model_name}_{split}.npz`` and stores object arrays for
``before_features``, ``after_features``, ``before_attrs``, ``after_attrs``, ``metadata``,
and optional native ``before_masks`` / ``after_masks``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def feature_cache_path(features_dir: Path, model_name: str, split_name: str) -> Path:
    return features_dir / f"{model_name}_{split_name}.npz"


def load_feature_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def pair_from_cache(cache: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = cache["metadata"][index].item()
    payload: dict[str, Any] = {
        "before_features": np.asarray(cache["before_features"][index]),
        "after_features": np.asarray(cache["after_features"][index]),
        "before_attrs": list(cache["before_attrs"][index]),
        "after_attrs": list(cache["after_attrs"][index]),
        "pair_id": str(cache["pair_ids"][index]),
        "split": str(metadata["split"]),
        "suite": str(metadata["suite"]),
        "edit_factor": str(metadata["edit_factor"]),
        "edited_object_id": int(metadata["edited_object_id"]),
    }
    if "before_masks" in cache:
        payload["before_masks"] = np.asarray(cache["before_masks"][index])
        payload["after_masks"] = np.asarray(cache["after_masks"][index])
    return payload
