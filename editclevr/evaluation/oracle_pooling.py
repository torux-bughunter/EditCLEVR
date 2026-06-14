from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_channel_first(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected feature map shaped [C, H, W] or [H, W, C], got {feature_map.shape}.")
    if feature_map.shape[0] <= 4 and feature_map.shape[-1] > 4:
        return feature_map.permute(2, 0, 1)
    return feature_map


def _resize_masks(masks: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped [N, H, W], got {masks.shape}.")
    resized = F.interpolate(
        masks.unsqueeze(1).float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(1)


def mask_pool_features(
    feature_map: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Pool object features from a spatial feature map with resized ground-truth masks.

    Args:
        feature_map: Tensor shaped [C, H, W] or [H, W, C].
        masks: Tensor shaped [N, H_mask, W_mask].
    """
    feature_map = _ensure_channel_first(feature_map).float()
    masks = masks.float()

    channels, height, width = feature_map.shape
    resized_masks = _resize_masks(masks, (height, width))

    flat_features = feature_map.view(channels, -1)
    flat_masks = resized_masks.view(resized_masks.shape[0], -1)
    weighted_sums = flat_masks @ flat_features.T
    normalizers = flat_masks.sum(dim=1, keepdim=True).clamp_min(eps)
    pooled = weighted_sums / normalizers
    return F.normalize(pooled, dim=1)
