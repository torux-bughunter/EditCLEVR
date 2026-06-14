from __future__ import annotations

from itertools import combinations

from .scene import ScenePair
from .schema import FACTORS


def _bbox_area(bbox: tuple[int, int, int, int], resolution: tuple[int, int]) -> float:
    left, top, right, bottom = bbox
    width = max(0, right - left)
    height = max(0, bottom - top)
    image_area = max(1, resolution[0] * resolution[1])
    return (width * height) / image_area


def _occlusion_level(target_visibility: float) -> str:
    if target_visibility >= 0.9:
        return "low"
    if target_visibility >= 0.7:
        return "medium"
    return "high"


def _distractor_similarity(pair: ScenePair) -> float:
    target_before = pair.before.object_by_id(pair.edit.object_id)
    similarities = []
    for obj in pair.before.objects:
        if obj.id == target_before.id:
            continue
        shared = sum(int(getattr(obj, factor) == getattr(target_before, factor)) for factor in FACTORS)
        similarities.append(shared / len(FACTORS))
    return max(similarities, default=0.0)


def tag_difficulty(pair: ScenePair) -> dict[str, object]:
    pair.validate()
    target = pair.before.object_by_id(pair.edit.object_id)
    return {
        "num_objects": len(pair.before.objects),
        "target_area": _bbox_area(target.pixel_bbox, pair.before.resolution),
        "occlusion_level": _occlusion_level(target.visibility),
        "distractor_similarity": _distractor_similarity(pair),
        "factor_type": pair.edit.factor,
        "edit_transition": f"{pair.edit.old_value}_to_{pair.edit.new_value}",
        "cogent_condition": pair.before.metadata.get("cogent_condition", "none"),
    }


def scene_has_overlap(scene_pair: ScenePair, threshold: float = 0.5) -> bool:
    def intersection_over_min_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        denom = max(1, min(area_a, area_b))
        return intersection / denom

    for obj_a, obj_b in combinations(scene_pair.before.objects, 2):
        if intersection_over_min_area(obj_a.pixel_bbox, obj_b.pixel_bbox) > threshold:
            return True
    return False
