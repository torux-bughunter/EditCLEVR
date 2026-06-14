"""Assign predicted slot or proposal features to ground-truth objects.

The paper headline protocol uses strict argmax best-overlap (MatchBO) assignment.
Soft IoU-weighted mixtures are supported as an ablation (Appendix D).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

MATCH_BINARIZE_THRESHOLD = 0.5
MIN_SEMANTIC_MATCH_BO = 0.50
MIN_PROBE_TRAIN_BO = 0.50
MATCH_IOU_RES = 128
MIN_MATCH_IOU_EPS = 1e-4


def load_gt_masks(path: str | np.ndarray) -> np.ndarray:
    if isinstance(path, np.ndarray):
        return path
    return np.load(path)["masks"]


def l2_normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True).clip(min=eps)
    return features / norms


def _resize_masks(masks: np.ndarray, height: int, width: int, *, order: int) -> np.ndarray:
    if masks.shape[1] == height and masks.shape[2] == width:
        return masks.astype(np.float32, copy=False)
    scale = (height / masks.shape[1], width / masks.shape[2])
    return np.stack(
        [zoom(masks[index].astype(np.float32), scale, order=order) for index in range(masks.shape[0])]
    )


def _shared_inputs(
    predicted_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_h = max(MATCH_IOU_RES, gt_masks.shape[1])
    target_w = max(MATCH_IOU_RES, gt_masks.shape[2])
    slot_soft = _resize_masks(np.asarray(predicted_masks, dtype=np.float32), target_h, target_w, order=1)
    slot_soft = np.clip(slot_soft, 0.0, 1.0)
    gt_resized = _resize_masks(gt_masks.astype(np.float32, copy=False), target_h, target_w, order=0)
    gt_bin = (gt_resized >= MATCH_BINARIZE_THRESHOLD).astype(np.float32)
    return slot_soft, gt_bin


def _iou_matrix_soft(slot_soft: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    num_slots, num_objects = slot_soft.shape[0], gt_bin.shape[0]
    if num_slots == 0 or num_objects == 0:
        return np.zeros((num_slots, num_objects), dtype=np.float32)
    slots = slot_soft.reshape(num_slots, -1).astype(np.float64, copy=False)
    objects = gt_bin.reshape(num_objects, -1).astype(np.float64, copy=False)
    with np.errstate(all="ignore"):
        intersection = slots @ objects.T
        slot_area = slots.sum(axis=1, keepdims=True)
        object_area = objects.sum(axis=1, keepdims=True).T
        union = slot_area + object_area - intersection
    return (intersection / np.clip(union, 1e-8, None)).astype(np.float32)


def _partition_one_hot(slot_soft: np.ndarray) -> np.ndarray:
    num_slots, height, width = slot_soft.shape
    winner = slot_soft.reshape(num_slots, -1).argmax(axis=0)
    one_hot = np.zeros((num_slots, height * width), dtype=np.float32)
    one_hot[winner, np.arange(height * width)] = 1.0
    return one_hot


def _iou_matrix_argmax(slot_soft: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    num_slots, num_objects = slot_soft.shape[0], gt_bin.shape[0]
    if num_slots == 0 or num_objects == 0:
        return np.zeros((num_slots, num_objects), dtype=np.float32)
    one_hot = _partition_one_hot(slot_soft)
    objects = gt_bin.reshape(num_objects, -1).astype(np.float32, copy=False)
    with np.errstate(all="ignore"):
        intersection = one_hot @ objects.T
        slot_area = one_hot.sum(axis=1, keepdims=True)
        object_area = objects.sum(axis=1, keepdims=True).T
        union = slot_area + object_area - intersection
    return (intersection / np.clip(union, 1e-8, None)).astype(np.float32)


def _bo_matrix_argmax(slot_soft: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    num_slots, num_objects = slot_soft.shape[0], gt_bin.shape[0]
    if num_slots == 0 or num_objects == 0:
        return np.zeros((num_slots, num_objects), dtype=np.float32)
    one_hot = _partition_one_hot(slot_soft)
    objects = gt_bin.reshape(num_objects, -1).astype(np.float32, copy=False)
    with np.errstate(all="ignore"):
        intersection = one_hot @ objects.T
        object_area = objects.sum(axis=1, keepdims=True).T
    return (intersection / np.clip(object_area, 1e-8, None)).astype(np.float32)


def per_object_soft_features(
    slot_features: np.ndarray,
    slot_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-GT-object features via IoU-weighted slot mixture."""
    slot_soft, gt_bin = _shared_inputs(slot_masks, gt_masks)
    soft_iou = _iou_matrix_soft(slot_soft, gt_bin)
    argmax_iou = _iou_matrix_argmax(slot_soft, gt_bin)
    argmax_bo = _bo_matrix_argmax(slot_soft, gt_bin)
    num_objects = soft_iou.shape[1]
    feature_dim = int(slot_features.shape[1])
    features = np.zeros((num_objects, feature_dim), dtype=np.float32)
    if soft_iou.shape[0] > 0:
        max_bo = argmax_bo.max(axis=0)
        max_iou = argmax_iou.max(axis=0)
    else:
        max_bo = np.zeros(num_objects, dtype=np.float32)
        max_iou = np.zeros(num_objects, dtype=np.float32)
    matched = max_bo > MIN_MATCH_IOU_EPS
    weights = soft_iou.sum(axis=0)
    if matched.any():
        columns = np.where(matched)[0]
        normalized = soft_iou[:, columns] / np.clip(weights[columns], 1e-8, None)[None, :]
        features[columns] = (normalized.T @ np.asarray(slot_features, dtype=np.float32)).astype(np.float32)
    return features, matched, max_bo, max_iou


def per_object_strict_features(
    slot_features: np.ndarray,
    slot_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-GT-object features from the single highest-BO slot per object."""
    slot_soft, gt_bin = _shared_inputs(slot_masks, gt_masks)
    argmax_iou = _iou_matrix_argmax(slot_soft, gt_bin)
    argmax_bo = _bo_matrix_argmax(slot_soft, gt_bin)
    num_slots, num_objects = argmax_bo.shape
    feature_dim = int(slot_features.shape[1])
    features = np.zeros((num_objects, feature_dim), dtype=np.float32)
    if num_slots > 0:
        max_bo = argmax_bo.max(axis=0)
        max_iou = argmax_iou.max(axis=0)
    else:
        max_bo = np.zeros(num_objects, dtype=np.float32)
        max_iou = np.zeros(num_objects, dtype=np.float32)
    matched = max_bo > MIN_MATCH_IOU_EPS
    if num_slots > 0:
        for object_index in range(num_objects):
            if not matched[object_index]:
                continue
            slot_index = int(np.argmax(argmax_bo[:, object_index]))
            features[object_index] = np.asarray(slot_features[slot_index], dtype=np.float32)
    return features, matched, max_bo, max_iou


def assign_object_features(
    slot_features: np.ndarray,
    slot_masks: np.ndarray | None,
    gt_masks: np.ndarray,
    *,
    alignment: str = "strict",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign slot features to GT objects and L2-normalize rows."""
    if slot_masks is None or alignment == "oracle":
        features = l2_normalize_rows(np.asarray(slot_features, dtype=np.float64))
        matched = np.ones(features.shape[0], dtype=bool)
        bo = np.ones(features.shape[0], dtype=np.float32)
        return features, matched, bo

    if alignment == "soft":
        features, matched, bo, _ = per_object_soft_features(slot_features, slot_masks, gt_masks)
    elif alignment == "strict":
        features, matched, bo, _ = per_object_strict_features(slot_features, slot_masks, gt_masks)
    else:
        raise ValueError(f"Unknown alignment {alignment!r}; expected 'strict', 'soft', or 'oracle'.")

    return l2_normalize_rows(features.astype(np.float64)), matched, bo


def edited_object_strong_match(
    before_matched: np.ndarray,
    after_matched: np.ndarray,
    before_bo: np.ndarray,
    after_bo: np.ndarray,
    edited_object_index: int,
    *,
    min_match_bo: float = MIN_SEMANTIC_MATCH_BO,
) -> bool:
    """True when the edited object is matched and clears MatchBO on both frames."""
    if edited_object_index < 0 or edited_object_index >= before_matched.shape[0]:
        return False
    if edited_object_index >= after_matched.shape[0]:
        return False
    edited_matched = bool(before_matched[edited_object_index]) and bool(
        after_matched[edited_object_index]
    )
    edited_bo_before = (
        float(before_bo[edited_object_index]) if before_matched[edited_object_index] else 0.0
    )
    edited_bo_after = (
        float(after_bo[edited_object_index]) if after_matched[edited_object_index] else 0.0
    )
    return edited_matched and min(edited_bo_before, edited_bo_after) >= min_match_bo


def noop_row_passes_native_gate(
    before_matched: np.ndarray,
    after_matched: np.ndarray,
    before_bo: np.ndarray,
    after_bo: np.ndarray,
    *,
    min_match_bo: float = MIN_SEMANTIC_MATCH_BO,
) -> bool:
    """True when every object is matched with MatchBO >= threshold on both frames."""
    if not (before_matched.all() and after_matched.all()):
        return False
    return min(float(before_bo.min()), float(after_bo.min())) >= min_match_bo
