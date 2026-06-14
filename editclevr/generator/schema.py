from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FACTORS = ("color", "material", "size", "shape")

FACTOR_VALUES = {
    "color": ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"),
    "material": ("metal", "rubber"),
    "size": ("small", "large"),
    "shape": ("cube", "sphere", "cylinder"),
}

SUITES = ("atomic_id", "no_edit", "hard_distractor", "cogent_ood")
SPLITS = ("train", "val", "test_id", "test_noop", "test_hard", "test_cogent")

COGENT_RULES = {
    "A": {
        "cube": {"gray", "blue", "brown", "yellow"},
        "cylinder": {"red", "green", "purple", "cyan"},
        "sphere": set(FACTOR_VALUES["color"]),
    },
    "B": {
        "cube": {"red", "green", "purple", "cyan"},
        "cylinder": {"gray", "blue", "brown", "yellow"},
        "sphere": set(FACTOR_VALUES["color"]),
    },
}


def allowed_colors_for_shape(shape: str, condition: str | None = None) -> tuple[str, ...]:
    if condition is None:
        return FACTOR_VALUES["color"]
    if condition not in COGENT_RULES:
        raise ValueError(f"Unknown CoGenT condition '{condition}'.")
    return tuple(sorted(COGENT_RULES[condition].get(shape, set(FACTOR_VALUES["color"]))))


def object_matches_condition(shape: str, color: str, condition: str | None = None) -> bool:
    return color in allowed_colors_for_shape(shape, condition)


def available_scene_edit_values(
    *,
    shape: str,
    color: str,
    factor: str,
    current_value: str,
    condition: str | None = None,
) -> tuple[str, ...]:
    if factor not in FACTORS:
        raise ValueError(f"Unknown factor '{factor}'.")

    if factor == "color":
        allowed = allowed_colors_for_shape(shape, condition)
        return tuple(value for value in allowed if value != current_value)

    if factor == "shape":
        return tuple(
            value
            for value in FACTOR_VALUES["shape"]
            if value != current_value and object_matches_condition(value, color, condition)
        )

    return tuple(value for value in FACTOR_VALUES[factor] if value != current_value)


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def objects_match_condition(objects: Iterable[Any], condition: str | None = None) -> bool:
    return all(object_matches_condition(_field(obj, "shape"), _field(obj, "color"), condition) for obj in objects)
