"""Full evaluation pipeline for the EditCLEVR dataset.

Default protocol matches the paper:
- linear probes with L2-normalized features and seed-averaged CV
- strict argmax MatchBO slot assignment for native feature caches
- semantic metrics gated on edited-object MatchBO >= 0.5 in both frames

Without pre-extracted features, runs a lightweight GT-mask oracle encoder as a
reference baseline (oracle alignment, no native matching gate).

Pass exported split NPZ caches with ``--features-dir`` and ``--model-name``.

Usage:
    python -m editclevr.evaluation.run_evaluation
    python -m editclevr.evaluation.run_evaluation \\
        --features-dir artifacts/features --model-name dinosaur_native
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from editclevr.baselines.oracle_encoders import SimpleOracleEncoder
from editclevr.evaluation.compute_metrics import (
    change_locality_score,
    edited_object_accuracy,
    no_edit_drift,
    pairwise_change_magnitudes,
    semantic_intervention_metrics,
    zero_semantic_intervention_metrics,
)
from editclevr.evaluation.confidence import bootstrap_ci
from editclevr.evaluation.feature_cache import feature_cache_path, load_feature_cache, pair_from_cache
from editclevr.evaluation.object_assignment import (
    MIN_PROBE_TRAIN_BO,
    MIN_SEMANTIC_MATCH_BO,
    assign_object_features,
    edited_object_strong_match,
    noop_row_passes_native_gate,
)
from editclevr.evaluation.stat_tests import independent_two_sample_summary
from editclevr.evaluation.train_probes import predict_factors, train_factor_probes
from editclevr.generator.schema import FACTORS
from editclevr.paths import dataset_dir, outputs_dir

logger = logging.getLogger(__name__)

REPORT_UNCONDITIONAL_NATIVE = True


def _load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_masks(path: str) -> np.ndarray:
    return np.load(path)["masks"]


def _objects_to_attributes(objects: list[dict]) -> list[dict[str, str]]:
    return [{factor: str(obj[factor]) for factor in FACTORS} for obj in objects]


def _encode_pair_oracle(
    encoder: SimpleOracleEncoder,
    row: dict[str, Any],
) -> dict[str, Any]:
    before_img = _load_image(row["before_image"])
    after_img = _load_image(row["after_image"])
    before_masks = _load_masks(row["instance_masks_before"])
    after_masks = _load_masks(row["instance_masks_after"])

    return {
        "before_feats": encoder.encode(before_img, before_masks),
        "after_feats": encoder.encode(after_img, after_masks),
        "before_attrs": _objects_to_attributes(row["objects_before"]),
        "after_attrs": _objects_to_attributes(row["objects_after"]),
    }


def _resolve_alignment(
    alignment: str,
    *,
    features_dir: Path | None,
    has_native_masks: bool,
) -> str:
    if features_dir is None:
        if alignment != "oracle":
            logger.info(
                "No feature cache provided; using oracle alignment with the reference encoder."
            )
        return "oracle"

    if not has_native_masks:
        if alignment != "oracle":
            logger.info(
                "Feature cache has no predicted masks; using oracle alignment."
            )
        return "oracle"

    return alignment


def _align_pair(
    before_features: np.ndarray,
    after_features: np.ndarray,
    before_attrs: list[dict[str, str]],
    after_attrs: list[dict[str, str]],
    before_gt_masks: np.ndarray,
    after_gt_masks: np.ndarray,
    *,
    before_pred_masks: np.ndarray | None,
    after_pred_masks: np.ndarray | None,
    alignment: str,
) -> dict[str, Any]:
    before_feats, before_matched, before_bo = assign_object_features(
        before_features,
        before_pred_masks,
        before_gt_masks,
        alignment=alignment,
    )
    after_feats, after_matched, after_bo = assign_object_features(
        after_features,
        after_pred_masks,
        after_gt_masks,
        alignment=alignment,
    )
    return {
        "before_feats": before_feats,
        "after_feats": after_feats,
        "before_attrs": before_attrs,
        "after_attrs": after_attrs,
        "before_matched": before_matched,
        "after_matched": after_matched,
        "before_bo": before_bo,
        "after_bo": after_bo,
    }


def _collect_native_train_vectors(
    rows: list[dict[str, Any]],
    cache: dict[str, Any],
    *,
    alignment: str,
    min_probe_train_bo: float = MIN_PROBE_TRAIN_BO,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    feature_chunks: list[np.ndarray] = []
    attributes: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        pair = pair_from_cache(cache, index)
        gt_masks = _load_masks(row["instance_masks_before"])
        features, matched, match_bo = assign_object_features(
            pair["before_features"],
            pair.get("before_masks"),
            gt_masks,
            alignment=alignment,
        )
        keep = matched & (match_bo >= min_probe_train_bo)
        if not keep.any():
            continue
        feature_chunks.append(features[keep])
        attributes.extend([pair["before_attrs"][object_index] for object_index in np.flatnonzero(keep)])

    if not feature_chunks:
        raise ValueError("No usable native training vectors after the MatchBO gate.")

    return np.concatenate(feature_chunks, axis=0), attributes


def evaluate_dataset(
    dataset_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 10000,
    probe_type: str = "linear",
    *,
    features_dir: Path | None = None,
    model_name: str | None = None,
    alignment: str = "strict",
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if features_dir is not None and not model_name:
        raise ValueError("--model-name is required when --features-dir is set.")

    splits_data: dict[str, list[dict]] = json.loads(
        (dataset_dir / "splits.json").read_text()
    )

    feature_caches: dict[str, dict[str, Any]] = {}
    has_native_masks = False
    if features_dir is not None:
        features_dir = features_dir.resolve()
        for split_name in splits_data:
            cache_path = feature_cache_path(features_dir, model_name, split_name)
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing feature cache: {cache_path}")
            cache = load_feature_cache(cache_path)
            feature_caches[split_name] = cache
            has_native_masks = has_native_masks or "before_masks" in cache

    resolved_alignment = _resolve_alignment(
        alignment,
        features_dir=features_dir,
        has_native_masks=has_native_masks,
    )
    encoder = SimpleOracleEncoder() if features_dir is None else None

    # ── Phase 1: train probes on the train split ─────────────────────────
    train_rows = splits_data.get("train", [])
    if features_dir is None:
        logger.info("Extracting train features with the reference oracle encoder …")
        train_features_list: list[np.ndarray] = []
        train_attributes: list[dict[str, str]] = []
        for row in train_rows:
            encoded = _encode_pair_oracle(encoder, row)
            train_features_list.append(encoded["before_feats"])
            train_attributes.extend(encoded["before_attrs"])
        train_features = np.concatenate(train_features_list, axis=0)
    else:
        logger.info(
            "Building native train vectors with %s alignment and MatchBO >= %.2f …",
            resolved_alignment,
            MIN_PROBE_TRAIN_BO,
        )
        train_features, train_attributes = _collect_native_train_vectors(
            train_rows,
            feature_caches["train"],
            alignment=resolved_alignment,
        )

    factor_labels = {
        factor: np.array([attributes[factor] for attributes in train_attributes])
        for factor in FACTORS
    }

    if probe_type != "linear":
        raise ValueError(
            f"Only linear probes are supported, got probe_type={probe_type!r}."
        )
    probe_types = ("linear",)
    probes_by_type: dict[str, dict[str, dict[str, object]]] = {}
    probe_reports: dict[str, dict[str, dict[str, object]]] = {}
    for probe_kind in probe_types:
        logger.info(
            "Training %s probes on %d object vectors …",
            probe_kind,
            train_features.shape[0],
        )
        probes = train_factor_probes(train_features, factor_labels, probe_type=probe_kind)
        probes_by_type[probe_kind] = probes
        probe_reports[probe_kind] = {
            factor: {
                "probe_type": str(info.get("probe_type", probe_kind)),
                "cv_accuracy_mean": round(info["cv_accuracy_mean"], 4),
                "cv_accuracy_std": round(info["cv_accuracy_std"], 4),
            }
            for factor, info in probes.items()
        }
        (output_dir / f"probe_accuracy_{probe_kind}.json").write_text(
            json.dumps(probe_reports[probe_kind], indent=2)
        )
        logger.info("%s probe accuracies: %s", probe_kind, probe_reports[probe_kind])

    # ── Phase 2: evaluate all splits ─────────────────────────────────────
    per_split = {probe_kind: defaultdict(lambda: defaultdict(list)) for probe_kind in probe_types}
    per_factor = {probe_kind: defaultdict(lambda: defaultdict(list)) for probe_kind in probe_types}
    per_suite = {probe_kind: defaultdict(lambda: defaultdict(list)) for probe_kind in probe_types}
    per_object_count = {
        probe_kind: defaultdict(lambda: defaultdict(list)) for probe_kind in probe_types
    }
    semantic_gate_stats = {
        probe_kind: {
            "gated_out": 0,
            "semantic_rows": 0,
            "ned_rows": 0,
            "ned_gated_out": 0,
        }
        for probe_kind in probe_types
    }
    is_native = resolved_alignment in ("strict", "soft")

    total_pairs = sum(len(values) for values in splits_data.values())
    processed = 0
    t0 = time.monotonic()

    for split_name, rows in splits_data.items():
        for row_index, row in enumerate(rows):
            if features_dir is None:
                encoded = _encode_pair_oracle(encoder, row)
                before_feats = encoded["before_feats"]
                after_feats = encoded["after_feats"]
                before_attrs = encoded["before_attrs"]
                after_attrs = encoded["after_attrs"]
                before_matched = np.ones(len(before_attrs), dtype=bool)
                after_matched = np.ones(len(after_attrs), dtype=bool)
                before_bo = np.ones(len(before_attrs), dtype=np.float32)
                after_bo = np.ones(len(after_attrs), dtype=np.float32)
            else:
                pair = pair_from_cache(feature_caches[split_name], row_index)
                aligned = _align_pair(
                    pair["before_features"],
                    pair["after_features"],
                    pair["before_attrs"],
                    pair["after_attrs"],
                    _load_masks(row["instance_masks_before"]),
                    _load_masks(row["instance_masks_after"]),
                    before_pred_masks=pair.get("before_masks"),
                    after_pred_masks=pair.get("after_masks"),
                    alignment=resolved_alignment,
                )
                before_feats = aligned["before_feats"]
                after_feats = aligned["after_feats"]
                before_attrs = aligned["before_attrs"]
                after_attrs = aligned["after_attrs"]
                before_matched = aligned["before_matched"]
                after_matched = aligned["after_matched"]
                before_bo = aligned["before_bo"]
                after_bo = aligned["after_bo"]

            edited_idx = int(row["edited_object_id"])
            edit_factor = str(row["edit_factor"])
            suite = str(row["suite"])

            changes = pairwise_change_magnitudes(before_feats, after_feats)

            if edit_factor == "none":
                include_noop = not is_native or noop_row_passes_native_gate(
                    before_matched,
                    after_matched,
                    before_bo,
                    after_bo,
                )
                for probe_kind in probe_types:
                    if include_noop:
                        ned = no_edit_drift(before_feats, after_feats)
                        per_split[probe_kind][split_name]["NED"].append(ned)
                        per_suite[probe_kind][suite]["NED"].append(ned)
                        semantic_gate_stats[probe_kind]["ned_rows"] += 1
                    elif is_native:
                        semantic_gate_stats[probe_kind]["ned_gated_out"] += 1
            else:
                eoa = edited_object_accuracy(changes, edited_idx)
                cls = change_locality_score(changes, edited_idx)
                for probe_kind in probe_types:
                    per_split[probe_kind][split_name]["EOA"].append(eoa)
                    per_split[probe_kind][split_name]["CLS"].append(cls)
                    per_suite[probe_kind][suite]["EOA"].append(eoa)
                    per_suite[probe_kind][suite]["CLS"].append(cls)

                    edited_strong = (
                        True
                        if not is_native
                        else edited_object_strong_match(
                            before_matched,
                            after_matched,
                            before_bo,
                            after_bo,
                            edited_idx,
                        )
                    )
                    sem: dict[str, float] | None = None
                    if edited_strong:
                        before_pred = predict_factors(probes_by_type[probe_kind], before_feats)
                        after_pred = predict_factors(probes_by_type[probe_kind], after_feats)
                        sem = semantic_intervention_metrics(
                            before_predictions=before_pred,
                            after_predictions=after_pred,
                            before_ground_truth=before_attrs,
                            after_ground_truth=after_attrs,
                            edited_object_index=edited_idx,
                            edit_factor=edit_factor,
                            trusted_before=before_matched if is_native else None,
                            trusted_after=after_matched if is_native else None,
                        )
                        semantic_gate_stats[probe_kind]["semantic_rows"] += 1
                        for metric, value in sem.items():
                            per_split[probe_kind][split_name][metric].append(value)
                            per_suite[probe_kind][suite][metric].append(value)
                            per_factor[probe_kind][edit_factor][metric].append(value)
                            per_object_count[probe_kind][str(len(before_attrs))][metric].append(value)
                    else:
                        semantic_gate_stats[probe_kind]["gated_out"] += 1

                    if is_native and REPORT_UNCONDITIONAL_NATIVE:
                        sem_uncond = sem if edited_strong else zero_semantic_intervention_metrics()
                        for metric, value in sem_uncond.items():
                            uncond_metric = f"{metric}_uncond"
                            per_split[probe_kind][split_name][uncond_metric].append(value)
                            per_suite[probe_kind][suite][uncond_metric].append(value)
                            per_factor[probe_kind][edit_factor][uncond_metric].append(value)
                            per_object_count[probe_kind][str(len(before_attrs))][uncond_metric].append(
                                value
                            )

            processed += 1
            if processed % 500 == 0:
                elapsed = time.monotonic() - t0
                rate = processed / elapsed
                eta = (total_pairs - processed) / rate
                logger.info(
                    "[%d/%d] %.1f pairs/sec, ~%.0fs remaining",
                    processed,
                    total_pairs,
                    rate,
                    eta,
                )

    # ── Phase 3: aggregate with bootstrap CIs ────────────────────────────
    logger.info("Computing bootstrap confidence intervals …")
    results_by_probe: dict[str, Any] = {}
    for probe_kind in probe_types:
        gate_stats = semantic_gate_stats[probe_kind]
        gated_total = gate_stats["gated_out"] + gate_stats["semantic_rows"]
        results: dict[str, Any] = {
            "metadata": {
                "bootstrap_n_resamples": int(n_bootstrap),
                "alignment": resolved_alignment,
                "features_dir": str(features_dir) if features_dir else None,
                "model_name": model_name,
                "semantic_gate_min_match_bo": MIN_SEMANTIC_MATCH_BO if is_native else None,
                "semantic_rows_used": gate_stats["semantic_rows"],
                "semantic_rows_gated_out": gate_stats["gated_out"],
                "ned_rows_used": gate_stats["ned_rows"],
                "ned_rows_gated_out": gate_stats["ned_gated_out"],
                "report_unconditional_native": bool(is_native and REPORT_UNCONDITIONAL_NATIVE),
                "semantic_gate_exclusion_rate": round(
                    gate_stats["gated_out"] / gated_total, 4
                )
                if gated_total
                else 0.0,
            },
            "by_split": _aggregate(per_split[probe_kind], n_bootstrap),
            "by_suite": _aggregate(per_suite[probe_kind], n_bootstrap),
            "by_factor": _aggregate(per_factor[probe_kind], n_bootstrap),
            "by_object_count": _aggregate(per_object_count[probe_kind], n_bootstrap),
            "probe_type": probe_kind,
            "probes": probe_reports[probe_kind],
        }

        sgia_id = per_split[probe_kind].get("test_id", {}).get("SGIA", [])
        sgia_hard = per_split[probe_kind].get("test_hard", {}).get("SGIA", [])
        if sgia_id and sgia_hard:
            comparison = independent_two_sample_summary(
                np.array(sgia_id),
                np.array(sgia_hard),
                n_resamples=n_bootstrap,
            )
            results["independent_tests"] = {
                "test_id_vs_test_hard_SGIA": {
                    "mean_diff": round(float(comparison["mean_diff"]), 4),
                    "ci_low": round(float(comparison["ci_low"]), 4),
                    "ci_high": round(float(comparison["ci_high"]), 4),
                    "p_value": round(float(comparison["p_value"]), 4),
                    "n_a": comparison["n_a"],
                    "n_b": comparison["n_b"],
                    "n_resamples": comparison["n_resamples"],
                    "test_type": comparison["test_type"],
                    "alternative": comparison["alternative"],
                }
            }

        suffix = "" if len(probe_types) == 1 else f"_{probe_kind}"
        (output_dir / f"results{suffix}.json").write_text(json.dumps(results, indent=2))
        _write_csv(output_dir / f"results_by_split{suffix}.csv", results["by_split"])
        _write_csv(output_dir / f"results_by_suite{suffix}.csv", results["by_suite"])
        _write_csv(output_dir / f"results_by_factor{suffix}.csv", results["by_factor"])
        _write_csv(
            output_dir / f"results_by_object_count{suffix}.csv",
            results["by_object_count"],
        )
        results_by_probe[probe_kind] = results

    logger.info("All results written to %s", output_dir)
    for probe_kind, results in results_by_probe.items():
        logger.info("Probe head: %s", probe_kind)
        _print_summary(results)

    if len(probe_types) == 1:
        return results_by_probe[probe_types[0]]

    combined = {
        "probe_types": list(probe_types),
        "results_by_probe": results_by_probe,
    }
    (output_dir / "results.json").write_text(json.dumps(combined, indent=2))
    return combined


def _aggregate(
    grouped: dict[str, dict[str, list[float]]],
    n_bootstrap: int,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for group, metrics in grouped.items():
        group_out: dict[str, dict[str, float]] = {}
        for metric, values in metrics.items():
            arr = np.array(values)
            mean, lo, hi = bootstrap_ci(arr, n_resamples=n_bootstrap)
            group_out[metric] = {
                "mean": round(mean, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "n": len(values),
                "bootstrap_n_resamples": int(n_bootstrap),
            }
        out[group] = group_out
    return out


def _write_csv(path: Path, grouped: dict[str, dict[str, Any]]) -> None:
    all_metrics = sorted({metric for group in grouped.values() for metric in group})
    header = "group," + ",".join(
        f"{metric}_mean,{metric}_ci_lo,{metric}_ci_hi" for metric in all_metrics
    )
    lines = [header]
    for group_name in sorted(grouped):
        metrics = grouped[group_name]
        parts = [group_name]
        for metric in all_metrics:
            entry = metrics.get(metric, {})
            parts.append(
                f"{entry.get('mean', '')},{entry.get('ci_lower', '')},{entry.get('ci_upper', '')}"
            )
        lines.append(",".join(parts))
    path.write_text("\n".join(lines) + "\n")


def _print_summary(results: dict[str, Any]) -> None:
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    metadata = results.get("metadata", {})
    if metadata.get("alignment"):
        logger.info(
            "Protocol: alignment=%s, semantic gate exclusion=%.2f%%",
            metadata["alignment"],
            100.0 * float(metadata.get("semantic_gate_exclusion_rate", 0.0)),
        )
    for group_type in ("by_suite", "by_split"):
        logger.info("── %s ──", group_type)
        for group, metrics in results.get(group_type, {}).items():
            parts = []
            for metric in [
                "SGIA",
                "DeltaSGIA",
                "SceneGraphExact",
                "CLS",
                "NED",
                "EOA",
                "TFA",
                "NFP",
                "UOP",
                "UOP_rate",
            ]:
                if metric in metrics:
                    entry = metrics[metric]
                    parts.append(
                        f"{metric}={entry['mean']:.3f} [{entry['ci_lower']:.3f},{entry['ci_upper']:.3f}]"
                    )
            if parts:
                logger.info("  %-14s %s", group, "  ".join(parts))
    logger.info("=" * 60)
    logger.info("Probe accuracies:")
    for factor, info in results.get("probes", {}).items():
        logger.info(
            "  %-10s  CV acc = %.3f ± %.3f",
            factor,
            info["cv_accuracy_mean"],
            info["cv_accuracy_std"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EditCLEVR with the paper protocol "
            "(strict MatchBO assignment + semantic gate by default)."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory containing splits.json (default: standard dataset location).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation artifacts (default: evaluation_results under dataset root).",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Directory with pre-extracted split NPZ caches from the paper pipeline.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Feature-cache prefix, e.g. dinosaur_native or dino_oracle_s8.",
    )
    parser.add_argument(
        "--alignment",
        choices=("strict", "soft", "oracle"),
        default="strict",
        help=(
            "Object assignment for native caches: strict MatchBO (paper default), "
            "soft IoU mixture (ablation), or oracle GT-mask pooling."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--probe-type", choices=("linear",), default="linear")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    resolved_dataset = args.dataset_dir or dataset_dir()
    resolved_output = args.output_dir or outputs_dir("evaluation_results")

    evaluate_dataset(
        resolved_dataset,
        resolved_output,
        n_bootstrap=args.bootstrap,
        probe_type=args.probe_type,
        features_dir=args.features_dir,
        model_name=args.model_name,
        alignment=args.alignment,
    )


if __name__ == "__main__":
    main()
