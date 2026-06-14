"""Evaluation utilities for EditCLEVR."""

from .compute_metrics import (
    change_locality_score,
    edited_object_accuracy,
    foreground_ari,
    no_edit_drift,
    semantic_intervention_metrics,
)
from .confidence import bootstrap_ci, paired_bootstrap_test
from .evaluator import Evaluator
from .slot_matching import MatchResult, match_masks_by_iou
from .stat_tests import (
    independent_two_sample_summary,
    two_sample_bootstrap_diff,
    two_sample_permutation_test,
)

__all__ = [
    "Evaluator",
    "MatchResult",
    "bootstrap_ci",
    "change_locality_score",
    "edited_object_accuracy",
    "foreground_ari",
    "match_masks_by_iou",
    "no_edit_drift",
    "paired_bootstrap_test",
    "semantic_intervention_metrics",
    "independent_two_sample_summary",
    "two_sample_bootstrap_diff",
    "two_sample_permutation_test",
]
