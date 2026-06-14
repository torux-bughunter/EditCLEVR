from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchResult:
    gt_indices: np.ndarray
    pred_indices: np.ndarray
    ious: np.ndarray
    low_confidence: np.ndarray
    unmatched_gt: np.ndarray
    unmatched_pred: np.ndarray


def _binarize_masks(masks: np.ndarray, threshold: float) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped [N, H, W], got {masks.shape}.")
    return np.asarray(masks >= threshold, dtype=bool)


def compute_iou_matrix(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    pred = _binarize_masks(pred_masks, threshold)
    gt = _binarize_masks(gt_masks, threshold)
    iou = np.zeros((gt.shape[0], pred.shape[0]), dtype=np.float64)

    for gt_idx, gt_mask in enumerate(gt):
        intersection = np.logical_and(gt_mask[None, ...], pred).sum(axis=(1, 2))
        union = np.logical_or(gt_mask[None, ...], pred).sum(axis=(1, 2))
        valid = union > 0
        iou[gt_idx, valid] = intersection[valid] / union[valid]
    return iou


def match_masks_by_iou(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
    low_confidence_iou: float = 0.1,
) -> MatchResult:
    iou = compute_iou_matrix(pred_masks=pred_masks, gt_masks=gt_masks, threshold=threshold)

    if iou.size == 0:
        return MatchResult(
            gt_indices=np.array([], dtype=int),
            pred_indices=np.array([], dtype=int),
            ious=np.array([], dtype=np.float64),
            low_confidence=np.array([], dtype=bool),
            unmatched_gt=np.arange(gt_masks.shape[0], dtype=int),
            unmatched_pred=np.arange(pred_masks.shape[0], dtype=int),
        )

    gt_indices, pred_indices = linear_sum_assignment(-iou)
    matched_ious = iou[gt_indices, pred_indices]

    unmatched_gt = np.setdiff1d(np.arange(gt_masks.shape[0]), gt_indices, assume_unique=False)
    unmatched_pred = np.setdiff1d(
        np.arange(pred_masks.shape[0]),
        pred_indices,
        assume_unique=False,
    )

    return MatchResult(
        gt_indices=gt_indices.astype(int),
        pred_indices=pred_indices.astype(int),
        ious=matched_ious.astype(np.float64),
        low_confidence=matched_ious < low_confidence_iou,
        unmatched_gt=unmatched_gt.astype(int),
        unmatched_pred=unmatched_pred.astype(int),
    )
