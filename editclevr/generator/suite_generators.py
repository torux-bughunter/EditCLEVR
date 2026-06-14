from __future__ import annotations

from dataclasses import replace
import random

from .atomic_edit import apply_atomic_edit
from .difficulty_tagger import tag_difficulty
from .scene import AtomicEdit, SceneDescription, ScenePair
from .schema import FACTORS, available_scene_edit_values, objects_match_condition


def build_atomic_suite_pair(
    scene: SceneDescription,
    object_id: int,
    factor: str,
    new_value: str,
) -> ScenePair:
    pair = apply_atomic_edit(scene, object_id=object_id, factor=factor, new_value=new_value)
    difficulty = tag_difficulty(pair)
    return replace(pair, suite="atomic_id", difficulty=difficulty)


def build_no_edit_pair(
    scene: SceneDescription,
    rerender_seed: int,
    object_id: int | None = None,
) -> ScenePair:
    before = scene
    after = replace(scene, blender_seed=rerender_seed)
    edited_object_id = scene.objects[0].id if object_id is None else object_id
    target = scene.object_by_id(edited_object_id)
    pair = ScenePair(
        pair_id=f"{scene.scene_id}_noop",
        suite="no_edit",
        before=before,
        after=after,
        edit=AtomicEdit(
            object_id=edited_object_id,
            factor=FACTORS[0],
            old_value=getattr(target, FACTORS[0]),
            new_value=getattr(target, FACTORS[0]),
        ),
    )
    difficulty = tag_difficulty(pair)
    difficulty["edit_transition"] = "none"
    difficulty["factor_type"] = "none"
    return replace(pair, difficulty=difficulty)


def qualifies_as_hard_distractor(scene: SceneDescription, object_id: int) -> bool:
    target = scene.object_by_id(object_id)
    for candidate in scene.objects:
        if candidate.id == object_id:
            continue
        shared = sum(int(getattr(candidate, factor) == getattr(target, factor)) for factor in FACTORS)
        if shared >= 2:
            return True
    return False


def hard_distractor_shared_factors(scene: SceneDescription, object_id: int) -> tuple[str, ...]:
    target = scene.object_by_id(object_id)
    shared_factors = set()
    for candidate in scene.objects:
        if candidate.id == object_id:
            continue
        shared = [factor for factor in FACTORS if getattr(candidate, factor) == getattr(target, factor)]
        if len(shared) >= 2:
            shared_factors.update(shared)
    return tuple(sorted(shared_factors))


def choose_hard_distractor_edit(
    scene: SceneDescription,
    seed: int,
    preferred_factor: str | None = None,
    condition: str | None = None,
) -> tuple[int, str, str]:
    rng = random.Random(seed)
    target_candidates: list[tuple[int, tuple[str, ...]]] = []
    for obj in scene.objects:
        shared_factors = hard_distractor_shared_factors(scene, obj.id)
        if preferred_factor is not None:
            allowed_factors = tuple(f for f in shared_factors if f == preferred_factor)
        else:
            allowed_factors = shared_factors
        if not allowed_factors:
            continue
        valid_factors = []
        for factor in allowed_factors:
            candidates = available_scene_edit_values(
                shape=obj.shape,
                color=obj.color,
                factor=factor,
                current_value=getattr(obj, factor),
                condition=condition,
            )
            if candidates:
                valid_factors.append(factor)
        if valid_factors:
            target_candidates.append((obj.id, tuple(valid_factors)))

    if not target_candidates:
        detail = preferred_factor or "any factor"
        raise ValueError(f"Scene does not contain a valid hard distractor target for {detail}.")

    object_id, factor_choices = rng.choice(target_candidates)
    factor = preferred_factor or rng.choice(list(factor_choices))
    target = scene.object_by_id(object_id)
    new_value = rng.choice(
        available_scene_edit_values(
            shape=target.shape,
            color=target.color,
            factor=factor,
            current_value=getattr(target, factor),
            condition=condition,
        )
    )
    return object_id, factor, new_value


def build_hard_distractor_pair(
    scene: SceneDescription,
    object_id: int,
    factor: str,
    new_value: str,
) -> ScenePair:
    if not qualifies_as_hard_distractor(scene, object_id):
        raise ValueError("Scene does not contain a valid hard distractor for the target object.")
    pair = build_atomic_suite_pair(scene, object_id=object_id, factor=factor, new_value=new_value)
    return replace(pair, suite="hard_distractor")


def scene_matches_cogent(scene: SceneDescription, condition: str) -> bool:
    return objects_match_condition(scene.objects, condition)


def build_cogent_pair(
    scene: SceneDescription,
    object_id: int,
    factor: str,
    new_value: str,
    condition: str,
) -> ScenePair:
    if condition not in {"A", "B"}:
        raise ValueError("condition must be 'A' or 'B'.")
    if not scene_matches_cogent(scene, condition):
        raise ValueError(f"Scene does not satisfy CoGenT condition {condition}.")

    pair = build_atomic_suite_pair(scene, object_id=object_id, factor=factor, new_value=new_value)
    if not scene_matches_cogent(pair.after, condition):
        raise ValueError(f"Atomic edit breaks CoGenT condition {condition}.")
    updated_before = replace(pair.before, metadata={**pair.before.metadata, "cogent_condition": condition})
    updated_after = replace(pair.after, metadata={**pair.after.metadata, "cogent_condition": condition})
    difficulty = dict(pair.difficulty)
    difficulty["cogent_condition"] = condition
    return replace(pair, suite="cogent_ood", before=updated_before, after=updated_after, difficulty=difficulty)
