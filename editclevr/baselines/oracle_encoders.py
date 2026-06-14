from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ObjectFeatures:
    features: np.ndarray
    attributes: list[dict[str, str]]


class SimpleOracleEncoder:
    """
    Lightweight oracle baseline that encodes objects from GT masks.

    Features combine masked RGB statistics with basic geometric cues.
    """

    def encode(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}.")
        if masks.ndim != 3:
            raise ValueError(f"Expected masks with shape [N, H, W], got {masks.shape}.")

        image = image.astype(np.float32) / 255.0
        features = []
        for mask in masks:
            binary = mask > 0
            if not np.any(binary):
                features.append(np.zeros(10, dtype=np.float32))
                continue

            pixels = image[binary]
            ys, xs = np.where(binary)
            bbox_h = max(1, ys.max() - ys.min() + 1)
            bbox_w = max(1, xs.max() - xs.min() + 1)
            area = float(binary.mean())
            fill_ratio = float(binary.sum() / max(1, bbox_h * bbox_w))

            feature = np.concatenate(
                [
                    pixels.mean(axis=0),
                    pixels.std(axis=0),
                    np.asarray(
                        [
                            area,
                            bbox_w / image.shape[1],
                            bbox_h / image.shape[0],
                            fill_ratio,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            features.append(feature.astype(np.float32))

        stacked = np.stack(features, axis=0)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        return stacked / np.clip(norms, a_min=1e-8, a_max=None)


def _load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_masks(path: str) -> np.ndarray:
    return np.load(path)["masks"]


def _objects_to_attributes(objects: list[dict[str, object]]) -> list[dict[str, str]]:
    attributes = []
    for obj in objects:
        attributes.append(
            {
                "color": str(obj["color"]),
                "material": str(obj["material"]),
                "size": str(obj["size"]),
                "shape": str(obj["shape"]),
            }
        )
    return attributes


def load_pair_features(
    metadata_path: Path,
    encoder: SimpleOracleEncoder | None = None,
) -> dict[str, ObjectFeatures | dict[str, object]]:
    payload = json.loads(metadata_path.read_text())
    encoder = encoder or SimpleOracleEncoder()

    before_image = _load_image(payload["before_image"])
    after_image = _load_image(payload["after_image"])
    before_masks = _load_masks(payload["instance_masks_before"])
    after_masks = _load_masks(payload["instance_masks_after"])

    return {
        "before": ObjectFeatures(
            features=encoder.encode(before_image, before_masks),
            attributes=_objects_to_attributes(payload["objects_before"]),
        ),
        "after": ObjectFeatures(
            features=encoder.encode(after_image, after_masks),
            attributes=_objects_to_attributes(payload["objects_after"]),
        ),
        "metadata": payload,
    }
