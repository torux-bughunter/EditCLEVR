from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import adjusted_rand_score


def pairwise_change_magnitudes(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError(f"Expected matching shapes, got {before.shape} and {after.shape}.")
    return np.linalg.norm(before - after, axis=-1)


def edited_object_accuracy(change_magnitudes: np.ndarray, edited_object_index: int) -> float:
    change_magnitudes = np.asarray(change_magnitudes, dtype=np.float64)
    if change_magnitudes.size == 0:
        raise ValueError("change_magnitudes must be non-empty.")
    return float(np.argmax(change_magnitudes) == edited_object_index)


def change_locality_score(change_magnitudes: np.ndarray, edited_object_index: int, eps: float = 1e-8) -> float:
    change_magnitudes = np.asarray(change_magnitudes, dtype=np.float64)
    total_change = change_magnitudes.sum()
    if total_change <= eps:
        return 0.0
    return float(change_magnitudes[edited_object_index] / total_change)


def no_edit_drift(before: np.ndarray, after: np.ndarray) -> float:
    return float(pairwise_change_magnitudes(before, after).mean())


def foreground_ari(pred_labels: np.ndarray, gt_labels: np.ndarray, background_label: int = 0) -> float:
    pred_labels = np.asarray(pred_labels).reshape(-1)
    gt_labels = np.asarray(gt_labels).reshape(-1)
    valid = gt_labels != background_label
    if not np.any(valid):
        raise ValueError("No foreground pixels available for FG-ARI.")
    return float(adjusted_rand_score(gt_labels[valid], pred_labels[valid]))


def semantic_intervention_metrics(
    before_predictions: list[dict[str, str]],
    after_predictions: list[dict[str, str]],
    before_ground_truth: list[dict[str, str]],
    after_ground_truth: list[dict[str, str]],
    edited_object_index: int,
    edit_factor: str,
) -> dict[str, float]:
    """Compute semantic intervention metrics from decoded object factors.

    DeltaSGIA captures intervention-consistency: the edited
    factor's decoded after value is correct, and the decoded before-to-after
    change is restricted to the edited object and edited factor.

    UOP is the strict all-or-nothing untouched-object preservation check. UOP_rate
    is its count-normalized companion: the fraction of untouched object-factor
    comparisons that stayed unchanged.

    SGIA is stricter than DeltaSGIA and TFA/NFP/UOP alone: the decoded after-scene
    graph must also match ground truth, reported as SceneGraphExact.
    """
    target_after_pred = after_predictions[edited_object_index]
    target_before_pred = before_predictions[edited_object_index]
    target_before_gt = before_ground_truth[edited_object_index]
    target_after_gt = after_ground_truth[edited_object_index]

    tfa = float(target_after_pred[edit_factor] == target_after_gt[edit_factor])

    non_target_factors = [key for key in target_before_gt.keys() if key != edit_factor]
    nfp = float(all(target_after_pred[factor] == target_before_pred[factor] for factor in non_target_factors))

    preserved = []
    untouched_factor_preserved = []
    for object_index in range(len(before_predictions)):
        if object_index == edited_object_index:
            continue
        factor_matches = [
            after_predictions[object_index][factor] == before_predictions[object_index][factor]
            for factor in before_predictions[object_index]
        ]
        untouched_factor_preserved.extend(factor_matches)
        unchanged = all(factor_matches)
        preserved.append(unchanged)
    uop = float(all(preserved)) if preserved else 1.0
    uop_rate = float(np.mean(untouched_factor_preserved)) if untouched_factor_preserved else 1.0

    scene_graph_exact = True
    for object_index in range(len(after_predictions)):
        for factor in after_predictions[object_index]:
            expected = after_ground_truth[object_index][factor]
            predicted = after_predictions[object_index][factor]
            if predicted != expected:
                scene_graph_exact = False
                break
        if not scene_graph_exact:
            break

    only_target_changed = True
    for object_index in range(len(before_predictions)):
        for factor in before_predictions[object_index]:
            changed = before_predictions[object_index][factor] != after_predictions[object_index][factor]
            should_change = object_index == edited_object_index and factor == edit_factor
            if changed != should_change:
                only_target_changed = False
                break
        if not only_target_changed:
            break

    delta_sgia = float(tfa and only_target_changed)
    sgia = float(scene_graph_exact and only_target_changed)
    return {
        "TFA": tfa,
        "NFP": nfp,
        "UOP": uop,
        "UOP_rate": uop_rate,
        "SceneGraphExact": float(scene_graph_exact),
        "DeltaSGIA": delta_sgia,
        "SGIA": sgia,
    }


def subgroup_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute mean of an empty subgroup.")
    return float(values.mean())
