from __future__ import annotations

from typing import Any


def model_reporting_metadata(model_name: str) -> dict[str, Any]:
    name = model_name.split(" [", 1)[0]
    lower = name.lower()

    if "oracle" in lower:
        group = "oracle_frozen"
        mask_source = "ground_truth"
        feature_source = "frozen_pretrained"
        trained = False
        cogent = "feature_probe_transfer_under_held_out_shape_color_combinations"
    elif lower.startswith("sam2_"):
        group = "sam2_frozen"
        mask_source = "sam2_automatic"
        feature_source = "frozen_pretrained"
        trained = False
        cogent = "sam2_masked_frozen_feature_probe_transfer_under_held_out_shape_color_combinations"
    elif "masks_dino" in lower:
        group = "hybrid"
        mask_source = "trained_native_masks"
        feature_source = "frozen_pretrained"
        trained = True
        cogent = "trained_mask_discovery_with_frozen_feature_probe_transfer_under_cogent_shift"
    elif "native" in lower or lower.startswith("sa_") or lower.startswith("dinosaur_"):
        group = "native_trained"
        mask_source = "native_model"
        feature_source = "trained_from_scratch_or_native"
        trained = True
        cogent = "editclevr_train_to_cogent_generalization"
    else:
        group = "unknown"
        mask_source = "unknown"
        feature_source = "unknown"
        trained = False
        cogent = "unspecified"

    return {
        "model_name": name,
        "model_group": group,
        "mask_source": mask_source,
        "feature_source": feature_source,
        "trained_on_editclevr_train": trained,
        "cogent_interpretation": cogent,
    }
