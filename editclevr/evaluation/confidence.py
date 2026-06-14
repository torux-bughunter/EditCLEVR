from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array.")

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = sample.mean()

    alpha = 1.0 - confidence
    lower = np.quantile(means, alpha / 2.0)
    upper = np.quantile(means, 1.0 - alpha / 2.0)
    return float(values.mean()), float(lower), float(upper)


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 1000,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Paired bootstrap test for truly row-aligned arrays only.

    Use independent two-sample utilities for separate splits/suites such as
    ``test_id`` vs. ``test_hard`` unless each row is explicitly aligned.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be non-empty 1D arrays with the same shape.")

    diffs = a - b
    observed = float(diffs.mean())
    centered = diffs - observed
    rng = np.random.default_rng(seed)

    resampled = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.choice(centered, size=centered.size, replace=True)
        resampled[idx] = sample.mean()

    p_value = float(np.mean(np.abs(resampled) >= abs(observed)))
    return observed, p_value
