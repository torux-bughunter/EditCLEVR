from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


ProbeType = Literal["linear"]


def l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True).clip(min=eps)
    return features / norms


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
) -> tuple[LogisticRegression, float]:
    return train_probe(features, labels, seed=seed, probe_type="linear")


def _make_classifier(
    probe_type: ProbeType,
    seed: int,
) -> LogisticRegression:
    if probe_type != "linear":
        raise ValueError(
            f"Only linear probes are supported, got probe_type={probe_type!r}."
        )
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    probe_type: ProbeType = "linear",
) -> tuple[LogisticRegression, float]:
    features = l2_normalize(features)
    labels = np.asarray(labels)

    model = _make_classifier(probe_type, seed)
    _, counts = np.unique(labels, return_counts=True)
    min_class_count = int(counts.min())
    n_splits = min(5, min_class_count)

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(model, features, labels, cv=cv)
        score = float(scores.mean())
    else:
        score = float("nan")

    model.fit(features, labels)
    return model, score


def train_factor_probes(
    features: np.ndarray,
    factor_labels: dict[str, np.ndarray],
    seeds: tuple[int, ...] = (0, 1, 2),
    probe_type: ProbeType = "linear",
) -> dict[str, dict[str, object]]:
    if probe_type != "linear":
        raise ValueError(
            f"Only linear probes are supported, got probe_type={probe_type!r}."
        )
    results: dict[str, dict[str, object]] = {}
    for factor, labels in factor_labels.items():
        models = []
        scores = []
        for seed in seeds:
            model, score = train_probe(
                features,
                labels,
                seed=seed,
                probe_type=probe_type,
            )
            models.append(model)
            scores.append(score)
        valid_scores = [score for score in scores if not np.isnan(score)]
        entry = {
            "models": models,
            "cv_accuracy_mean": (
                float(np.mean(valid_scores)) if valid_scores else float("nan")
            ),
            "cv_accuracy_std": (
                float(np.std(valid_scores)) if valid_scores else float("nan")
            ),
            "probe_type": probe_type,
        }
        results[factor] = entry
    return results


def predict_factors(
    probes: dict[str, dict[str, object]],
    features: np.ndarray,
) -> list[dict[str, str]]:
    normalized = l2_normalize(features)
    per_factor_predictions: dict[str, np.ndarray] = {}

    for factor, payload in probes.items():
        factor_models = payload["models"]
        label_encoder = payload.get("label_encoder")
        model_votes = []
        for model in factor_models:
            predicted = model.predict(normalized)
            if label_encoder is not None:
                predicted = label_encoder.inverse_transform(
                    np.asarray(predicted, dtype=np.int64)
                )
            model_votes.append(predicted)
        votes = np.stack(model_votes, axis=0)
        voted = []
        for col in votes.T:
            counts = defaultdict(int)
            for label in col:
                counts[str(label)] += 1
            voted.append(max(sorted(counts), key=counts.get))
        per_factor_predictions[factor] = np.asarray(voted, dtype=object)

    predictions: list[dict[str, str]] = []
    for row_idx in range(normalized.shape[0]):
        predictions.append(
            {
                factor: str(values[row_idx])
                for factor, values in per_factor_predictions.items()
            }
        )
    return predictions
