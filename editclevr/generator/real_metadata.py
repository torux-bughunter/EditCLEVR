from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .build_dataset import PairRecord, validate_pair_record
from .schema import FACTORS, objects_match_condition

EXPECTED_SPLIT_SUITES = {
    "train": {"atomic_id"},
    "val": {"atomic_id"},
    "test_id": {"atomic_id"},
    "test_noop": {"no_edit"},
    "test_hard": {"hard_distractor"},
    "test_cogent": {"cogent_ood"},
}
EXPECTED_SPLIT_CONDITIONS = {
    "train": "A",
    "val": "A",
    "test_id": "A",
    "test_noop": "A",
    "test_hard": "A",
    "test_cogent": "B",
}
BALANCED_FACTOR_SPLITS = {"train", "val", "test_id", "test_hard", "test_cogent"}
BALANCED_OBJECT_COUNT_SPLITS = {"train", "val", "test_id", "test_noop", "test_hard", "test_cogent"}
JOINT_FACTOR_OBJECT_BALANCED_SPLITS = {"train", "val", "test_id", "test_hard", "test_cogent"}
OBJECT_COUNT_KEYS = ("3", "4", "5", "6")
GEOMETRY_ATOL = 1e-4


def _load_masks(path: str | Path) -> np.ndarray:
    payload = np.load(Path(path))
    masks = np.asarray(payload["masks"], dtype=np.uint8)
    if "object_ids" not in payload.files:
        return masks
    object_ids = [int(value) for value in np.asarray(payload["object_ids"]).tolist()]
    if object_ids == list(range(len(object_ids))):
        return masks
    order = np.argsort(np.asarray(object_ids, dtype=np.int64))
    return masks[order]


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    denom = max(1, min(_bbox_area(a), _bbox_area(b)))
    return intersection / denom


def _visibility_proxy(mask: np.ndarray, bbox: list[int]) -> float:
    area = int(mask.sum())
    bbox_area = max(1, _bbox_area(bbox))
    return round(min(1.0, area / bbox_area), 4)


def _normalize_object(record: dict[str, Any], object_id: int, mask: np.ndarray) -> dict[str, Any]:
    bbox = _mask_bbox(mask)
    return {
        "id": object_id,
        "shape": record["shape"],
        "color": record["color"],
        "material": record["material"],
        "size": record["size"],
        "3d_position": [float(value) for value in record.get("3d_coords", [])],
        "pixel_bbox": bbox,
        "visibility": _visibility_proxy(mask, bbox),
        "pixel_coords": list(record.get("pixel_coords", [])),
    }


def _spatial_relations(scene: dict[str, Any]) -> list[dict[str, int | str]]:
    relation_map = {
        "left": "left_of",
        "right": "right_of",
        "front": "in_front_of",
        "behind": "behind",
    }
    rows: list[dict[str, int | str]] = []
    for relation_name, adjacency in scene.get("relationships", {}).items():
        normalized = relation_map.get(relation_name, relation_name)
        for subject_idx, object_indices in enumerate(adjacency):
            for object_idx in object_indices:
                rows.append(
                    {
                        "subject": int(subject_idx),
                        "relation": normalized,
                        "object": int(object_idx),
                    }
                )
    return rows


def _distractor_similarity(objects: list[dict[str, Any]], edited_object_id: int) -> float:
    target = objects[edited_object_id]
    similarities = []
    for obj in objects:
        if obj["id"] == edited_object_id:
            continue
        shared = sum(int(obj[factor] == target[factor]) for factor in FACTORS)
        similarities.append(shared / len(FACTORS))
    return round(max(similarities, default=0.0), 4)


def hard_distractor_metadata(
    objects: list[dict[str, Any]],
    edited_object_id: int,
    edit_factor: str,
) -> dict[str, Any]:
    target = objects[edited_object_id]
    patterns: set[tuple[str, ...]] = set()
    max_shared = 0
    edited_shared = False
    for index, obj in enumerate(objects):
        if index == edited_object_id or obj.get("id") == edited_object_id:
            continue
        shared = tuple(factor for factor in FACTORS if obj.get(factor) == target.get(factor))
        if len(shared) >= 2:
            patterns.add(shared)
            max_shared = max(max_shared, len(shared))
            edited_shared = edited_shared or edit_factor in shared
    return {
        "hard_distractor_max_shared_count": int(max_shared),
        "hard_distractor_shared_patterns": [list(pattern) for pattern in sorted(patterns)],
        "hard_distractor_edited_factor_shared_with_any_distractor": bool(edited_shared),
    }


def _occlusion_level(objects: list[dict[str, Any]], edited_object_id: int) -> str:
    target_bbox = objects[edited_object_id]["pixel_bbox"]
    max_overlap = 0.0
    for obj in objects:
        if obj["id"] == edited_object_id:
            continue
        max_overlap = max(max_overlap, _bbox_overlap_ratio(target_bbox, obj["pixel_bbox"]))
    if max_overlap < 0.05:
        return "low"
    if max_overlap < 0.2:
        return "medium"
    return "high"


def _target_area(objects: list[dict[str, Any]], masks: np.ndarray, edited_object_id: int, resolution: tuple[int, int]) -> float:
    total_pixels = max(1, resolution[0] * resolution[1])
    return round(float(masks[edited_object_id].sum()) / total_pixels, 6)


def _vector_field(obj: dict[str, Any], *names: str) -> np.ndarray | None:
    for name in names:
        if name in obj and obj[name] is not None:
            return np.asarray(obj[name], dtype=np.float64)
    return None


def _scalar_field(obj: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in obj and obj[name] is not None:
            return float(obj[name])
    return None


def _assert_geometry_stable(
    before_obj: dict[str, Any],
    after_obj: dict[str, Any],
    *,
    label: str,
    atol: float = GEOMETRY_ATOL,
) -> None:
    before_position = _vector_field(before_obj, "3d_position", "position", "position_3d", "3d_coords")
    after_position = _vector_field(after_obj, "3d_position", "position", "position_3d", "3d_coords")
    if before_position is not None and after_position is not None and not np.allclose(before_position, after_position, atol=atol):
        raise ValueError(f"{label} position drift: before={before_position.tolist()} after={after_position.tolist()}")

    before_quat = _vector_field(before_obj, "quaternion", "rotation_quaternion")
    after_quat = _vector_field(after_obj, "quaternion", "rotation_quaternion")
    if before_quat is not None and after_quat is not None and not np.allclose(before_quat, after_quat, atol=atol):
        raise ValueError(f"{label} quaternion drift: before={before_quat.tolist()} after={after_quat.tolist()}")

    before_rot = _scalar_field(before_obj, "rotation", "rotation_z")
    after_rot = _scalar_field(after_obj, "rotation", "rotation_z")
    if before_rot is not None and after_rot is not None and not np.isclose(before_rot, after_rot, atol=atol):
        raise ValueError(f"{label} rotation drift: before={before_rot} after={after_rot}")


@lru_cache(maxsize=4)
def blender_version(blender_executable: str) -> str:
    result = subprocess.run(
        [blender_executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = result.stdout.strip().splitlines()[0]
    return first_line.replace("Blender ", "").strip()


def pair_metadata_to_record(pair_metadata: dict[str, Any], blender_executable: str | None = None) -> PairRecord:
    edit = pair_metadata["edit"]
    before_scene = pair_metadata["before"]["scene"]
    after_scene = pair_metadata["after"]["scene"]
    before_masks = _load_masks(pair_metadata["before"]["mask_npz"])
    after_masks = _load_masks(pair_metadata["after"]["mask_npz"])
    with Image.open(pair_metadata["before"]["image"]) as image:
        resolution = image.size

    objects_before = [
        _normalize_object(record, object_id=index, mask=before_masks[index])
        for index, record in enumerate(before_scene["objects"])
    ]
    objects_after = [
        _normalize_object(record, object_id=index, mask=after_masks[index])
        for index, record in enumerate(after_scene["objects"])
    ]
    difficulty = {
        "num_objects": len(objects_before),
        "target_area": _target_area(objects_before, before_masks, edit["object_index"], resolution),
        "occlusion_level": _occlusion_level(objects_before, edit["object_index"]),
        "distractor_similarity": _distractor_similarity(objects_before, edit["object_index"]),
        "factor_type": edit["factor"],
        "edit_transition": f"{edit['old_value']}_to_{edit['new_value']}",
        "cogent_condition": pair_metadata.get("condition", "none"),
    }
    generation = {
        "base_scene_seed": pair_metadata.get("base_scene_seed"),
        "blender_version": blender_version(blender_executable) if blender_executable else "unknown",
        "render_resolution": list(resolution),
    }
    if "rerender_seed" in pair_metadata:
        generation["rerender_seed"] = pair_metadata["rerender_seed"]
    hard_metadata = (
        hard_distractor_metadata(objects_before, edit["object_index"], edit["factor"])
        if pair_metadata["suite"] == "hard_distractor"
        else {}
    )
    difficulty.update(hard_metadata)
    payload = {
        "pair_id": pair_metadata["pair_id"],
        "suite": pair_metadata["suite"],
        "edit_factor": edit["factor"],
        "before_image": pair_metadata["before"]["image"],
        "after_image": pair_metadata["after"]["image"],
        "before_scene_json": pair_metadata["before"]["scene_json"],
        "after_scene_json": pair_metadata["after"]["scene_json"],
        "instance_masks_before": pair_metadata["before"]["mask_npz"],
        "instance_masks_after": pair_metadata["after"]["mask_npz"],
        "objects_before": objects_before,
        "objects_after": objects_after,
        "edited_object_id": edit["object_index"],
        "old_value": edit["old_value"],
        "new_value": edit["new_value"],
        "spatial_relations": _spatial_relations(before_scene),
        "difficulty": difficulty,
        "generation": generation,
        "suite_condition": pair_metadata.get("condition", "none"),
    }
    record = PairRecord(
        pair_id=pair_metadata["pair_id"],
        suite=pair_metadata["suite"],
        edit_factor=edit["factor"],
        num_objects=len(objects_before),
        difficulty_bucket=str(difficulty["occlusion_level"]),
        payload=payload,
    )
    validate_pair_record(record)
    return record


def validate_pair_payload(payload: dict[str, Any]) -> None:
    required_top_level = {
        "pair_id",
        "suite",
        "edit_factor",
        "before_image",
        "after_image",
        "before_scene_json",
        "after_scene_json",
        "instance_masks_before",
        "instance_masks_after",
        "objects_before",
        "objects_after",
        "edited_object_id",
        "old_value",
        "new_value",
        "spatial_relations",
        "difficulty",
        "generation",
    }
    missing = required_top_level.difference(payload)
    if missing:
        raise ValueError(f"Pair metadata missing required fields: {sorted(missing)}")

    for key in (
        "before_image",
        "after_image",
        "before_scene_json",
        "after_scene_json",
        "instance_masks_before",
        "instance_masks_after",
    ):
        if not Path(payload[key]).exists():
            raise FileNotFoundError(f"Expected existing artifact at {payload[key]}")

    objects_before = payload["objects_before"]
    objects_after = payload["objects_after"]
    if len(objects_before) != len(objects_after):
        raise ValueError("Before/after object counts must match.")

    edited_object_id = int(payload["edited_object_id"])
    if not 0 <= edited_object_id < len(objects_before):
        raise ValueError(f"edited_object_id={edited_object_id} is out of range for {len(objects_before)} objects.")
    edit_factor = payload["edit_factor"]
    if payload["suite"] == "no_edit":
        if edit_factor != "none":
            raise ValueError("No-edit pairs must use edit_factor='none'.")
    elif edit_factor not in FACTORS:
        raise ValueError(f"Unexpected edit factor '{edit_factor}'.")

    object_required = {"id", "shape", "color", "material", "size", "3d_position", "pixel_bbox", "visibility"}
    for index, (before_obj, after_obj) in enumerate(zip(objects_before, objects_after)):
        missing_before = object_required.difference(before_obj)
        missing_after = object_required.difference(after_obj)
        if missing_before or missing_after:
            raise ValueError(
                f"Object metadata missing fields for index {index}: before={sorted(missing_before)} after={sorted(missing_after)}"
            )
        if before_obj["id"] != after_obj["id"]:
            raise ValueError(f"Object ID changed at index {index}: before={before_obj['id']} after={after_obj['id']}")
        changed = [factor for factor in FACTORS if before_obj[factor] != after_obj[factor]]
        if payload["suite"] == "no_edit":
            _assert_geometry_stable(before_obj, after_obj, label=f"no-edit object {index}")
            if changed:
                raise ValueError(f"No-edit pair changed object {index} factors: {changed}")
            continue
        if index == edited_object_id:
            if changed != [edit_factor]:
                raise ValueError(f"Edited object {index} should change only {edit_factor}, found {changed}")
            if before_obj[edit_factor] != payload["old_value"] or after_obj[edit_factor] != payload["new_value"]:
                raise ValueError("Edited object values do not match declared old/new values.")
            if edit_factor not in {"shape", "size"}:
                _assert_geometry_stable(before_obj, after_obj, label=f"target object {index}")
        elif changed:
            raise ValueError(f"Non-target object {index} changed unexpectedly: {changed}")
        else:
            _assert_geometry_stable(before_obj, after_obj, label=f"non-target object {index}")

    before_masks = _load_masks(payload["instance_masks_before"])
    after_masks = _load_masks(payload["instance_masks_after"])
    if before_masks.shape[0] != len(objects_before) or after_masks.shape[0] != len(objects_after):
        raise ValueError("Mask/object count mismatch.")

    generation = payload["generation"]
    if generation.get("base_scene_seed") is None:
        raise ValueError("generation.base_scene_seed must be populated for reproducibility.")

    suite_condition = payload.get("suite_condition", "none")
    if suite_condition != "none":
        if suite_condition not in {"A", "B"}:
            raise ValueError(f"Unexpected suite_condition '{suite_condition}'.")
        if not objects_match_condition(objects_before, suite_condition):
            raise ValueError(f"Before objects violate suite_condition={suite_condition}.")
        if not objects_match_condition(objects_after, suite_condition):
            raise ValueError(f"After objects violate suite_condition={suite_condition}.")


def _empty_object_count_counts() -> dict[str, int]:
    return {key: 0 for key in OBJECT_COUNT_KEYS}


def dataset_balance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    factor_counts: dict[str, int] = {}
    object_count_counts: dict[str, int] = {}
    factor_object_count_counts: dict[str, dict[str, int]] = {}
    transition_counts: dict[str, int] = {}
    cogent_core_counts: dict[str, int] = {"eligible": 0, "total": 0}
    for row in rows:
        factor = row["edit_factor"]
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
        num_objects = str(len(row["objects_before"]))
        object_count_counts[num_objects] = object_count_counts.get(num_objects, 0) + 1
        factor_object_count_counts.setdefault(factor, _empty_object_count_counts())
        factor_object_count_counts[factor][num_objects] = factor_object_count_counts[factor].get(num_objects, 0) + 1
        transition = row.get("difficulty", {}).get("edit_transition")
        if transition:
            transition_counts[str(transition)] = transition_counts.get(str(transition), 0) + 1
        if row.get("suite") == "cogent_ood":
            cogent_core_counts["total"] += 1
            edited_object_id = int(row["edited_object_id"])
            before_obj = row["objects_before"][edited_object_id]
            after_obj = row["objects_after"][edited_object_id]
            if (
                factor in {"color", "shape"}
                and before_obj["shape"] in {"cube", "cylinder"}
                and after_obj["shape"] in {"cube", "cylinder"}
            ):
                cogent_core_counts["eligible"] += 1
    return {
        "edit_factor": dict(sorted(factor_counts.items())),
        "num_objects": dict(sorted(object_count_counts.items())),
        "edit_factor_num_objects": {
            factor: dict(sorted(counts.items(), key=lambda item: int(item[0])))
            for factor, counts in sorted(factor_object_count_counts.items())
        },
        "edit_transition": dict(sorted(transition_counts.items())),
        "cogent_core": cogent_core_counts,
    }


def _edited_object_counts(rows: list[dict[str, Any]], attribute: str) -> dict[str, int]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        factor = row.get("edit_factor")
        if factor in (None, "none"):
            continue
        edited_object_id = int(row["edited_object_id"])
        target = row["objects_before"][edited_object_id]
        counts[(str(factor), str(target[attribute]))] += 1
    return {f"{factor}|{value}": count for (factor, value), count in sorted(counts.items())}


def _methodology_warnings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    factors = sorted({row["edit_factor"] for row in rows if row.get("edit_factor") not in (None, "none")})
    for attribute in FACTORS:
        counts = _edited_object_counts(rows, attribute)
        values = sorted({key.split("|", 1)[1] for key in counts})
        counts_by_factor_value = {
            tuple(key.split("|", 1)): count
            for key, count in counts.items()
        }
        for value in values:
            factor_counts = {
                factor: counts_by_factor_value.get((factor, value), 0)
                for factor in factors
            }
            if not factor_counts:
                continue
            spread = max(factor_counts.values()) - min(factor_counts.values())
            if spread > 1:
                warnings.append(
                    {
                        "type": "edited_object_attribute_cross_factor_imbalance",
                        "attribute": attribute,
                        "value": value,
                        "min": min(factor_counts.values()),
                        "max": max(factor_counts.values()),
                        "spread": spread,
                        "counts": factor_counts,
                    }
                )
    return warnings


def _assert_unique(rows: list[dict[str, Any]], key: str, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = str(row[key])
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        examples = sorted(duplicates)[:5]
        raise ValueError(f"Duplicate {label} detected: {examples}")


def _assert_approximately_uniform(
    counts: dict[str, int],
    expected_keys: tuple[str, ...],
    total: int,
    *,
    tolerance_fraction: float,
    label: str,
) -> None:
    if not total:
        return
    expected = total / len(expected_keys)
    tolerance = max(1.0, math.ceil(total * tolerance_fraction))
    for key in expected_keys:
        value = counts.get(key, 0)
        if abs(value - expected) > tolerance:
            raise ValueError(
                f"{label} is imbalanced for '{key}': observed {value}, expected about {expected:.1f} +/- {tolerance:.0f}."
            )


def _assert_joint_factor_object_balance(
    joint_counts: dict[str, dict[str, int]],
    *,
    split: str,
    total: int,
    tolerance_fraction: float,
) -> None:
    if not total:
        return

    total_cells = len(FACTORS) * len(OBJECT_COUNT_KEYS)
    require_full_coverage = total >= total_cells
    expected_per_cell = total / total_cells
    tolerance = max(1.0, math.ceil(expected_per_cell * tolerance_fraction))

    for factor in FACTORS:
        per_factor = joint_counts.get(factor, _empty_object_count_counts())
        factor_total = sum(per_factor.get(num_objects, 0) for num_objects in OBJECT_COUNT_KEYS)
        if require_full_coverage:
            missing = [num_objects for num_objects in OBJECT_COUNT_KEYS if per_factor.get(num_objects, 0) <= 0]
            if missing:
                raise ValueError(
                    f"{split} joint factor/num_objects distribution is missing cells for edit_factor='{factor}': {missing}."
                )

        if factor_total:
            _assert_approximately_uniform(
                {num_objects: per_factor.get(num_objects, 0) for num_objects in OBJECT_COUNT_KEYS},
                OBJECT_COUNT_KEYS,
                factor_total,
                tolerance_fraction=tolerance_fraction,
                label=f"{split} P(num_objects | edit_factor={factor})",
            )

        for num_objects in OBJECT_COUNT_KEYS:
            observed = per_factor.get(num_objects, 0)
            if abs(observed - expected_per_cell) > tolerance:
                raise ValueError(
                    f"{split} joint factor/num_objects distribution is imbalanced for "
                    f"({factor}, {num_objects}): observed {observed}, expected about "
                    f"{expected_per_cell:.1f} +/- {tolerance:.0f}."
                )


def validate_dataset_splits(split_metadata: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    all_rows = [row for rows in split_metadata.values() for row in rows]
    _assert_unique(all_rows, "pair_id", "pair_id")
    _assert_unique(all_rows, "before_scene_json", "before scene JSON path")
    _assert_unique(all_rows, "after_scene_json", "after scene JSON path")

    for split, rows in split_metadata.items():
        for row in rows:
            validate_pair_payload(row)
            expected_suites = EXPECTED_SPLIT_SUITES.get(split)
            if expected_suites is not None and row["suite"] not in expected_suites:
                raise ValueError(
                    f"Split '{split}' contains suite '{row['suite']}', expected one of {sorted(expected_suites)}."
                )
            expected_condition = EXPECTED_SPLIT_CONDITIONS.get(split)
            if expected_condition is not None and row.get("suite_condition", "none") != expected_condition:
                raise ValueError(
                    f"Split '{split}' contains suite_condition={row.get('suite_condition')}, expected {expected_condition}."
                )

        balance = dataset_balance_summary(rows)
        if split in BALANCED_FACTOR_SPLITS:
            _assert_approximately_uniform(
                balance["edit_factor"],
                FACTORS,
                len(rows),
                tolerance_fraction=0.05,
                label=f"{split} edit_factor distribution",
            )
        if split == "test_noop" and set(balance["edit_factor"]) != {"none"}:
            raise ValueError("test_noop must contain only edit_factor='none'.")
        if split in BALANCED_OBJECT_COUNT_SPLITS:
            _assert_approximately_uniform(
                balance["num_objects"],
                OBJECT_COUNT_KEYS,
                len(rows),
                tolerance_fraction=0.05,
                label=f"{split} num_objects distribution",
            )
        if split in JOINT_FACTOR_OBJECT_BALANCED_SPLITS:
            _assert_joint_factor_object_balance(
                balance["edit_factor_num_objects"],
                split=split,
                total=len(rows),
                tolerance_fraction=0.05,
            )

        summary["splits"][split] = {
            "count": len(rows),
            "balance": balance,
            "edited_object_balance": {
                attribute: _edited_object_counts(rows, attribute)
                for attribute in FACTORS
            },
            "methodology_warnings": _methodology_warnings(rows),
        }
    return summary


def write_validation_report(path: Path, split_metadata: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = validate_dataset_splits(split_metadata)
    path.write_text(json.dumps(summary, indent=2))
    return summary
