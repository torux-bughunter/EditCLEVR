"""Standardized evaluator for the EditCLEVR benchmark.

A model produces per-pair predictions, and ``Evaluator.eval`` returns exactly the
headline metrics reported in the paper, so results are directly comparable across
methods regardless of the encoder used.

Example
-------
    from editclevr.evaluation import Evaluator

    evaluator = Evaluator()
    # print(evaluator.expected_input_format)
    results = evaluator.eval(predictions)
    print(results["overall"]["SGIA"])   # -> {"mean": ..., "ci_lower": ..., ...}

Each element of ``predictions`` is a dict describing one before/after pair:

    {
        "suite": "atomic_id",            # optional, used for per-suite breakdown
        "edit_factor": "color",          # "none" for no-edit control pairs
        "edited_object_index": 2,        # index of the edited object
        "change_magnitudes": [..],       # per-object L2 change in embedding space
        # the four fields below are required only when edit_factor != "none":
        "before_pred": [{"color": ..}],  # decoded factors per object (before)
        "after_pred":  [{"color": ..}],  # decoded factors per object (after)
        "before_gt":   [{"color": ..}],  # ground-truth factors per object (before)
        "after_gt":    [{"color": ..}],  # ground-truth factors per object (after)
    }

For no-edit pairs (``edit_factor == "none"``) only ``change_magnitudes`` is needed;
``NED`` (no-edit drift) is the mean per-object change.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .compute_metrics import (
    change_locality_score,
    edited_object_accuracy,
    semantic_intervention_metrics,
)
from .confidence import bootstrap_ci

EDIT_METRICS = ("SGIA", "DeltaSGIA", "SceneGraphExact", "CLS", "EOA", "TFA", "NFP", "UOP", "UOP_rate")
NOOP_METRICS = ("NED",)
HEADLINE_METRICS = ("SGIA", "DeltaSGIA", "CLS", "EOA", "NED")


class Evaluator:
    """Compute EditCLEVR headline metrics from per-pair predictions."""

    def __init__(self, n_bootstrap: int = 0) -> None:
        """Create an evaluator.

        Parameters
        ----------
        n_bootstrap:
            Number of bootstrap resamples for confidence intervals. ``0`` (default)
            skips CIs and reports point estimates only. The paper uses ``10000``.
        """
        self.n_bootstrap = int(n_bootstrap)

    @property
    def expected_input_format(self) -> str:
        return __doc__ or ""

    def _per_pair_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        edit_factor = str(record["edit_factor"])
        changes = np.asarray(record["change_magnitudes"], dtype=np.float64)

        if edit_factor == "none":
            return {"NED": float(changes.mean())}

        edited_idx = int(record["edited_object_index"])
        out: dict[str, float] = {
            "EOA": edited_object_accuracy(changes, edited_idx),
            "CLS": change_locality_score(changes, edited_idx),
        }
        out.update(
            semantic_intervention_metrics(
                before_predictions=list(record["before_pred"]),
                after_predictions=list(record["after_pred"]),
                before_ground_truth=list(record["before_gt"]),
                after_ground_truth=list(record["after_gt"]),
                edited_object_index=edited_idx,
                edit_factor=edit_factor,
            )
        )
        return out

    def _aggregate(self, grouped: Mapping[str, list[float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for metric, values in grouped.items():
            arr = np.asarray(values, dtype=np.float64)
            entry: dict[str, float] = {"mean": round(float(arr.mean()), 4), "n": int(arr.size)}
            if self.n_bootstrap > 0 and arr.size > 1:
                _, lo, hi = bootstrap_ci(arr, n_resamples=self.n_bootstrap)
                entry["ci_lower"] = round(float(lo), 4)
                entry["ci_upper"] = round(float(hi), 4)
            out[metric] = entry
        return out

    def eval(self, predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Evaluate per-pair predictions and return headline + per-suite metrics."""
        if not predictions:
            raise ValueError("predictions must be a non-empty sequence of per-pair dicts.")

        overall: dict[str, list[float]] = defaultdict(list)
        by_suite: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for record in predictions:
            metrics = self._per_pair_metrics(record)
            suite = str(record.get("suite", "all"))
            for metric, value in metrics.items():
                overall[metric].append(value)
                by_suite[suite][metric].append(value)

        return {
            "overall": self._aggregate(overall),
            "by_suite": {suite: self._aggregate(metrics) for suite, metrics in by_suite.items()},
            "metadata": {
                "n_pairs": len(predictions),
                "n_bootstrap": self.n_bootstrap,
                "headline_metrics": list(HEADLINE_METRICS),
            },
        }
