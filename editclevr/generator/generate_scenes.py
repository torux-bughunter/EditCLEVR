from __future__ import annotations

import random
from dataclasses import replace

from .atomic_edit import available_factor_values
from .scene import SceneDescription, SceneObject
from .schema import FACTORS, FACTOR_VALUES, allowed_colors_for_shape, available_scene_edit_values

COLOR_TO_RGB = {
    "gray": (145, 145, 145),
    "red": (204, 52, 52),
    "blue": (52, 100, 204),
    "green": (64, 158, 87),
    "brown": (150, 108, 70),
    "purple": (137, 86, 175),
    "cyan": (72, 188, 201),
    "yellow": (227, 210, 67),
}

SIZE_TO_EXTENT = {
    "small": 18,
    "large": 28,
}
def _bbox_from_center(center_x: int, center_y: int, extent: int) -> tuple[int, int, int, int]:
    return (center_x - extent, center_y - extent, center_x + extent, center_y + extent)


def _boxes_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / min(area_a, area_b)


def sample_scene(
    scene_id: str,
    seed: int,
    num_objects: int | None = None,
    resolution: tuple[int, int] = (320, 240),
    cogent_condition: str | None = None,
    force_shared_object_id: int | None = None,
) -> SceneDescription:
    rng = random.Random(seed)
    count = num_objects if num_objects is not None else rng.randint(3, 6)
    width, height = resolution
    objects: list[SceneObject] = []

    for object_id in range(count):
        for _ in range(200):
            shape = rng.choice(FACTOR_VALUES["shape"])
            size = rng.choice(FACTOR_VALUES["size"])
            extent = SIZE_TO_EXTENT[size]
            center_x = rng.randint(extent + 10, width - extent - 10)
            center_y = rng.randint(extent + 10, height - extent - 10)
            bbox = _bbox_from_center(center_x, center_y, extent)

            if any(_boxes_overlap_ratio(bbox, other.pixel_bbox) > 0.5 for other in objects):
                continue

            color = rng.choice(allowed_colors_for_shape(shape, cogent_condition))
            material = rng.choice(FACTOR_VALUES["material"])

            if force_shared_object_id is not None and object_id == force_shared_object_id and objects:
                anchor = objects[0]
                shared = rng.sample(["color", "material", "size", "shape"], k=2)
                color = anchor.color if "color" in shared else color
                material = anchor.material if "material" in shared else material
                size = anchor.size if "size" in shared else size
                shape = anchor.shape if "shape" in shared else shape
                extent = SIZE_TO_EXTENT[size]
                bbox = _bbox_from_center(center_x, center_y, extent)
                if any(_boxes_overlap_ratio(bbox, other.pixel_bbox) > 0.5 for other in objects):
                    continue

            visibility = round(rng.uniform(0.72, 0.99), 3)
            if object_id == 0:
                visibility = max(visibility, 0.85)

            obj = SceneObject(
                id=object_id,
                shape=shape,
                color=color,
                material=material,
                size=size,
                position_3d=(
                    round((center_x - width / 2) / 50.0, 3),
                    round((center_y - height / 2) / 50.0, 3),
                    0.35 if size == "small" else 0.7,
                ),
                pixel_bbox=bbox,
                visibility=visibility,
            )
            objects.append(obj)
            break
        else:
            raise RuntimeError(f"Could not place object {object_id} without excessive overlap.")

    metadata = {
        "base_scene_seed": seed,
    }
    if cogent_condition is not None:
        metadata["cogent_condition"] = cogent_condition

    return SceneDescription(
        scene_id=scene_id,
        objects=tuple(objects),
        blender_seed=seed,
        resolution=resolution,
        metadata=metadata,
    )


def propose_atomic_edit(
    scene: SceneDescription,
    seed: int,
    object_id: int | None = None,
    factor: str | None = None,
    cogent_condition: str | None = None,
) -> tuple[int, str, str]:
    rng = random.Random(seed)
    target = scene.object_by_id(object_id) if object_id is not None else rng.choice(scene.objects)
    candidate_factors = [factor] if factor is not None else list(FACTORS)
    if factor is None:
        rng.shuffle(candidate_factors)

    for chosen_factor in candidate_factors:
        current_value = getattr(target, chosen_factor)
        if cogent_condition is None:
            candidates = available_factor_values(chosen_factor, current_value)
        else:
            candidates = available_scene_edit_values(
                shape=target.shape,
                color=target.color,
                factor=chosen_factor,
                current_value=current_value,
                condition=cogent_condition,
            )
        if candidates:
            return target.id, chosen_factor, rng.choice(candidates)

    raise ValueError(
        f"No valid edit candidates available for object {target.id} under condition {cogent_condition or 'none'}."
    )


def with_paths(
    scene: SceneDescription,
    image_path: str,
    masks_path: str,
) -> SceneDescription:
    return replace(scene, image_path=image_path, masks_path=masks_path)
