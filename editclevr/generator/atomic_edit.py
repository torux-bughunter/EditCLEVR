from __future__ import annotations

from dataclasses import replace

from .scene import AtomicEdit, SceneDescription, ScenePair
from .schema import FACTORS, FACTOR_VALUES


def available_factor_values(factor: str, current_value: str) -> tuple[str, ...]:
    if factor not in FACTORS:
        raise ValueError(f"Unknown factor '{factor}'.")
    return tuple(value for value in FACTOR_VALUES[factor] if value != current_value)


def apply_atomic_edit(
    scene: SceneDescription,
    object_id: int,
    factor: str,
    new_value: str,
) -> ScenePair:
    scene.validate()
    if factor not in FACTORS:
        raise ValueError(f"Unknown factor '{factor}'.")
    if new_value not in FACTOR_VALUES[factor]:
        raise ValueError(f"Value '{new_value}' is invalid for factor '{factor}'.")

    target = scene.object_by_id(object_id)
    old_value = getattr(target, factor)
    if old_value == new_value:
        raise ValueError("Atomic edit must change the factor value.")

    updated_object = replace(target, **{factor: new_value})
    after_scene = scene.replace_object(updated_object)

    edit = AtomicEdit(
        object_id=object_id,
        factor=factor,
        old_value=old_value,
        new_value=new_value,
    )
    return ScenePair(
        pair_id=f"{scene.scene_id}_{factor}_{object_id}",
        suite="atomic_id",
        before=scene,
        after=after_scene,
        edit=edit,
    )


def is_atomic_edit_valid(pair: ScenePair) -> bool:
    pair.validate()

    changed_objects = []
    for before_obj, after_obj in zip(
        sorted(pair.before.objects, key=lambda item: item.id),
        sorted(pair.after.objects, key=lambda item: item.id),
    ):
        differences = [
            factor
            for factor in FACTORS
            if getattr(before_obj, factor) != getattr(after_obj, factor)
        ]
        if differences:
            changed_objects.append((before_obj.id, differences))

    if len(changed_objects) != 1:
        return False

    object_id, factors = changed_objects[0]
    return object_id == pair.edit.object_id and factors == [pair.edit.factor]
