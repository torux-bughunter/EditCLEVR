"""Generator-side utilities for EditCLEVR."""

from .atomic_edit import apply_atomic_edit, available_factor_values, is_atomic_edit_valid
from .build_dataset import (
    PairRecord,
    assemble_split_metadata,
    iterative_stratified_split,
    pair_to_record,
    validate_pair_record,
)
from .blender_adapter import BlenderRenderer
from .difficulty_tagger import scene_has_overlap, tag_difficulty
from .generate_scenes import propose_atomic_edit, sample_scene
from .renderer_interface import SceneRenderer
from .scene import AtomicEdit, SceneDescription, SceneObject, ScenePair, SceneRelation
from .schema import FACTORS, FACTOR_VALUES
from .suite_generators import (
    build_atomic_suite_pair,
    build_cogent_pair,
    build_hard_distractor_pair,
    build_no_edit_pair,
    choose_hard_distractor_edit,
    qualifies_as_hard_distractor,
    scene_matches_cogent,
)

__all__ = [
    "AtomicEdit",
    "BlenderRenderer",
    "FACTORS",
    "FACTOR_VALUES",
    "PairRecord",
    "SceneDescription",
    "SceneObject",
    "ScenePair",
    "SceneRelation",
    "SceneRenderer",
    "apply_atomic_edit",
    "assemble_split_metadata",
    "available_factor_values",
    "build_atomic_suite_pair",
    "build_cogent_pair",
    "build_hard_distractor_pair",
    "build_no_edit_pair",
    "choose_hard_distractor_edit",
    "iterative_stratified_split",
    "is_atomic_edit_valid",
    "pair_to_record",
    "propose_atomic_edit",
    "qualifies_as_hard_distractor",
    "sample_scene",
    "scene_has_overlap",
    "scene_matches_cogent",
    "tag_difficulty",
    "validate_pair_record",
]
