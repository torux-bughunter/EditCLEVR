from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import FACTOR_VALUES


@dataclass(frozen=True)
class SceneObject:
    id: int
    shape: str
    color: str
    material: str
    size: str
    position_3d: tuple[float, float, float]
    pixel_bbox: tuple[int, int, int, int]
    visibility: float

    def validate(self) -> None:
        if self.shape not in FACTOR_VALUES["shape"]:
            raise ValueError(f"Unsupported shape '{self.shape}'.")
        if self.color not in FACTOR_VALUES["color"]:
            raise ValueError(f"Unsupported color '{self.color}'.")
        if self.material not in FACTOR_VALUES["material"]:
            raise ValueError(f"Unsupported material '{self.material}'.")
        if self.size not in FACTOR_VALUES["size"]:
            raise ValueError(f"Unsupported size '{self.size}'.")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("visibility must be between 0 and 1.")
        if len(self.position_3d) != 3:
            raise ValueError("position_3d must contain exactly 3 values.")
        if len(self.pixel_bbox) != 4:
            raise ValueError("pixel_bbox must contain exactly 4 values.")

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["3d_position"] = payload.pop("position_3d")
        return payload


@dataclass(frozen=True)
class SceneRelation:
    subject: int
    relation: str
    object: int


@dataclass(frozen=True)
class SceneDescription:
    scene_id: str
    objects: tuple[SceneObject, ...]
    relations: tuple[SceneRelation, ...] = ()
    image_path: str | None = None
    masks_path: str | None = None
    blender_seed: int | None = None
    resolution: tuple[int, int] = (320, 240)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.objects:
            raise ValueError("Scene must contain at least one object.")

        seen_ids = set()
        for obj in self.objects:
            obj.validate()
            if obj.id in seen_ids:
                raise ValueError(f"Duplicate object id {obj.id}.")
            seen_ids.add(obj.id)

        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("resolution must contain positive integers.")

    def object_by_id(self, object_id: int) -> SceneObject:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(f"Object id {object_id} not found in scene.")

    def replace_object(self, updated_object: SceneObject) -> "SceneDescription":
        replaced = []
        found = False
        for obj in self.objects:
            if obj.id == updated_object.id:
                replaced.append(updated_object)
                found = True
            else:
                replaced.append(obj)
        if not found:
            raise KeyError(f"Object id {updated_object.id} not found in scene.")
        return SceneDescription(
            scene_id=self.scene_id,
            objects=tuple(replaced),
            relations=self.relations,
            image_path=self.image_path,
            masks_path=self.masks_path,
            blender_seed=self.blender_seed,
            resolution=self.resolution,
            metadata=dict(self.metadata),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "objects": [obj.to_metadata() for obj in self.objects],
            "spatial_relations": [asdict(relation) for relation in self.relations],
            "image_path": self.image_path,
            "masks_path": self.masks_path,
            "blender_seed": self.blender_seed,
            "resolution": list(self.resolution),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AtomicEdit:
    object_id: int
    factor: str
    old_value: str
    new_value: str


@dataclass(frozen=True)
class ScenePair:
    pair_id: str
    suite: str
    before: SceneDescription
    after: SceneDescription
    edit: AtomicEdit
    split: str | None = None
    difficulty: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.before.validate()
        self.after.validate()

        before_ids = {obj.id for obj in self.before.objects}
        after_ids = {obj.id for obj in self.after.objects}
        if before_ids != after_ids:
            raise ValueError("Before/after scenes must contain the same object ids.")
        if self.edit.object_id not in before_ids:
            raise ValueError(f"Edited object id {self.edit.object_id} not present in scene.")

    def to_metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "pair_id": self.pair_id,
            "split": self.split,
            "suite": self.suite,
            "before_image": self.before.image_path,
            "after_image": self.after.image_path,
            "edited_object_id": self.edit.object_id,
            "edit_factor": self.edit.factor,
            "old_value": self.edit.old_value,
            "new_value": self.edit.new_value,
            "objects_before": [obj.to_metadata() for obj in self.before.objects],
            "objects_after": [obj.to_metadata() for obj in self.after.objects],
            "spatial_relations": [asdict(relation) for relation in self.before.relations],
            "instance_masks_before": self.before.masks_path,
            "instance_masks_after": self.after.masks_path,
            "difficulty": dict(self.difficulty),
            "generation": dict(self.generation),
        }
