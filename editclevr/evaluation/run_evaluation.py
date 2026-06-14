"""Full evaluation pipeline for the real Blender-backed EditCLEVR dataset.

Reads splits.json from the 20k dataset, runs an oracle encoder over every pair,
trains linear probes on the train split, and computes all headline + supporting
metrics with bootstrap CIs.

Usage (CPU-only, no GPU needed):
    python -m editclevr.evaluation.run_evaluation
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
)
from editclevr.evaluation.confidence import bootstrap_ci
from editclevr.evaluation.stat_tests import independent_two_sample_summary
from editclevr.evaluation.train_probes import predict_factors, train_factor_probes
from editclevr.generator.schema import FACTORS
from editclevr.paths import dataset_dir, outputs_dir

logger = logging.getLogger(__name__)


def _load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_masks(path: str) -> np.ndarray:
    return np.load(path)["masks"]


def _objects_to_attributes(objects: list[dict]) -> list[dict[str, str]]:
    return [{f: str(obj[f]) for f in FACTORS} for obj in objects]


def _encode_pair(
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


def evaluate_dataset(
    dataset_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 10000,
    probe_type: str = "linear",
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_data: dict[str, list[dict]] = json.loads(
        (dataset_dir / "splits.json").read_text()
    )
    encoder = SimpleOracleEncoder()

    # ── Phase 1: extract features for train split, train probes ──────────
    logger.info("Extracting train features for probe training …")
    train_features_list: list[np.ndarray] = []
    train_attributes: list[dict[str, str]] = []

    for row in splits_data.get("train", []):
        encoded = _encode_pair(encoder, row)
        train_features_list.append(encoded["before_feats"])
        train_attributes.extend(encoded["before_attrs"])

    train_features = np.concatenate(train_features_list, axis=0)
    factor_labels = {
        f: np.array([a[f] for a in train_attributes])
        for f in FACTORS
    }

    if probe_type != "linear":
        raise ValueError(
            f"Only linear probes are supported, got probe_type={probe_type!r}."
        )
    probe_types = ("linear",)
    probes_by_type: dict[str, dict[str, dict[str, object]]] = {}
    probe_reports: dict[str, dict[str, dict[str, object]]] = {}
    for pt in probe_types:
        logger.info(
            "Training %s probes on %d object vectors …", pt, train_features.shape[0]
        )
        probes = train_factor_probes(train_features, factor_labels, probe_type=pt)
        probes_by_type[pt] = probes
        probe_reports[pt] = {
            f: {
                "probe_type": str(info.get("probe_type", pt)),
                "cv_accuracy_mean": round(info["cv_accuracy_mean"], 4),
                "cv_accuracy_std": round(info["cv_accuracy_std"], 4),
            }
            for f, info in probes.items()
        }
        (output_dir / f"probe_accuracy_{pt}.json").write_text(
            json.dumps(probe_reports[pt], indent=2)
        )
        logger.info("%s probe accuracies: %s", pt, probe_reports[pt])

    # ── Phase 2: evaluate all splits ─────────────────────────────────────
    per_split = {pt: defaultdict(lambda: defaultdict(list)) for pt in probe_types}
    per_factor = {pt: defaultdict(lambda: defaultdict(list)) for pt in probe_types}
    per_suite = {pt: defaultdict(lambda: defaultdict(list)) for pt in probe_types}
    per_object_count = {pt: defaultdict(lambda: defaultdict(list)) for pt in probe_types}

    total_pairs = sum(len(v) for v in splits_data.values())
    processed = 0
    t0 = time.monotonic()

    for split_name, rows in splits_data.items():
        for row in rows:
            encoded = _encode_pair(encoder, row)
            before_feats = encoded["before_feats"]
            after_feats = encoded["after_feats"]
            before_attrs = encoded["before_attrs"]
            after_attrs = encoded["after_attrs"]

            edited_idx = int(row["edited_object_id"])
            edit_factor = str(row["edit_factor"])
            suite = str(row["suite"])

            changes = pairwise_change_magnitudes(before_feats, after_feats)

            if edit_factor == "none":
                ned = no_edit_drift(before_feats, after_feats)
                for pt in probe_types:
                    per_split[pt][split_name]["NED"].append(ned)
                    per_suite[pt][suite]["NED"].append(ned)
            else:
                eoa = edited_object_accuracy(changes, edited_idx)
                cls = change_locality_score(changes, edited_idx)
                for pt in probe_types:
                    per_split[pt][split_name]["EOA"].append(eoa)
                    per_split[pt][split_name]["CLS"].append(cls)
                    per_suite[pt][suite]["EOA"].append(eoa)
                    per_suite[pt][suite]["CLS"].append(cls)

                    before_pred = predict_factors(probes_by_type[pt], before_feats)
                    after_pred = predict_factors(probes_by_type[pt], after_feats)
                    sem = semantic_intervention_metrics(
                        before_predictions=before_pred,
                        after_predictions=after_pred,
                        before_ground_truth=before_attrs,
                        after_ground_truth=after_attrs,
                        edited_object_index=edited_idx,
                        edit_factor=edit_factor,
                    )
                    for m, v in sem.items():
                        per_split[pt][split_name][m].append(v)
                        per_suite[pt][suite][m].append(v)
                        per_factor[pt][edit_factor][m].append(v)
                        per_object_count[pt][str(len(before_attrs))][m].append(v)

            processed += 1
            if processed % 500 == 0:
                elapsed = time.monotonic() - t0
                rate = processed / elapsed
                eta = (total_pairs - processed) / rate
                logger.info(
                    "[%d/%d] %.1f pairs/sec, ~%.0fs remaining",
                    processed, total_pairs, rate, eta,
                )

    # ── Phase 3: aggregate with bootstrap CIs ────────────────────────────
    logger.info("Computing bootstrap confidence intervals …")
    results_by_probe: dict[str, Any] = {}
    for pt in probe_types:
        results: dict[str, Any] = {
            "metadata": {
                "bootstrap_n_resamples": int(n_bootstrap),
                "debug_n_bootstrap": None,
            },
            "by_split": _aggregate(per_split[pt], n_bootstrap),
            "by_suite": _aggregate(per_suite[pt], n_bootstrap),
            "by_factor": _aggregate(per_factor[pt], n_bootstrap),
            "by_object_count": _aggregate(per_object_count[pt], n_bootstrap),
            "probe_type": pt,
            "probes": probe_reports[pt],
        }

        # Independent split comparison: test_id and test_hard are not row-aligned.
        sgia_id = per_split[pt].get("test_id", {}).get("SGIA", [])
        sgia_hard = per_split[pt].get("test_hard", {}).get("SGIA", [])
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

        suffix = "" if len(probe_types) == 1 else f"_{pt}"
        (output_dir / f"results{suffix}.json").write_text(json.dumps(results, indent=2))
        _write_csv(output_dir / f"results_by_split{suffix}.csv", results["by_split"])
        _write_csv(output_dir / f"results_by_suite{suffix}.csv", results["by_suite"])
        _write_csv(output_dir / f"results_by_factor{suffix}.csv", results["by_factor"])
        _write_csv(
            output_dir / f"results_by_object_count{suffix}.csv",
            results["by_object_count"],
        )
        results_by_probe[pt] = results

    logger.info("All results written to %s", output_dir)
    for pt, results in results_by_probe.items():
        logger.info("Probe head: %s", pt)
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
    all_metrics = sorted({m for g in grouped.values() for m in g})
    header = "group," + ",".join(
        f"{m}_mean,{m}_ci_lo,{m}_ci_hi" for m in all_metrics
    )
    lines = [header]
    for group_name in sorted(grouped):
        metrics = grouped[group_name]
        parts = [group_name]
        for m in all_metrics:
            entry = metrics.get(m, {})
            parts.append(
                f"{entry.get('mean', '')},{entry.get('ci_lower', '')},{entry.get('ci_upper', '')}"
            )
        lines.append(",".join(parts))
    path.write_text("\n".join(lines) + "\n")


def _print_summary(results: dict[str, Any]) -> None:
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for group_type in ("by_suite", "by_split"):
        logger.info("── %s ──", group_type)
        for group, metrics in results.get(group_type, {}).items():
            parts = []
            for m in [
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
                if m in metrics:
                    entry = metrics[m]
                    parts.append(
                        f"{m}={entry['mean']:.3f} [{entry['ci_lower']:.3f},{entry['ci_upper']:.3f}]"
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
        description="Evaluate EditCLEVR dataset with oracle encoder."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Dataset directory containing splits.json (default: standard dataset location).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for evaluation artifacts (default: evaluation_results under dataset root).",
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
    )


if __name__ == "__main__":
    main()
