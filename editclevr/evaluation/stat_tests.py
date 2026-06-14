from __future__ import annotations

import math
from typing import Literal

import numpy as np

INDEPENDENT_TEST_TYPE = "independent_two_sample_bootstrap_or_permutation"


def _as_nonempty_1d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    return arr


def two_sample_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float | int | str]:
    """CI for mean(a)-mean(b), resampling independent samples separately."""
    a = _as_nonempty_1d(a, "a")
    b = _as_nonempty_1d(b, "b")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample_a = rng.choice(a, size=a.size, replace=True)
        sample_b = rng.choice(b, size=b.size, replace=True)
        diffs[idx] = sample_a.mean() - sample_b.mean()

    alpha = 1.0 - confidence
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, alpha / 2.0)),
        "ci_high": float(np.quantile(diffs, 1.0 - alpha / 2.0)),
        "p_value": float("nan"),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "n_resamples": int(n_resamples),
        "test_type": INDEPENDENT_TEST_TYPE,
    }


def two_sample_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
    seed: int | None = 0,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, float | int | str]:
    """Permutation test for independent-sample difference in means."""
    a = _as_nonempty_1d(a, "a")
    b = _as_nonempty_1d(b, "b")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")

    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = int(a.size)
    rng = np.random.default_rng(seed)
    null_diffs = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        perm = rng.permutation(pooled)
        null_diffs[idx] = perm[:n_a].mean() - perm[n_a:].mean()

    if alternative == "two-sided":
        extreme = np.abs(null_diffs) >= abs(observed)
    elif alternative == "greater":
        extreme = null_diffs >= observed
    else:
        extreme = null_diffs <= observed

    # Add-one smoothing avoids zero p-values from finite Monte Carlo samples.
    p_value = (float(np.sum(extreme)) + 1.0) / (float(n_resamples) + 1.0)
    return {
        "mean_diff": observed,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float(p_value),
        "n_a": n_a,
        "n_b": int(b.size),
        "n_resamples": int(n_resamples),
        "test_type": INDEPENDENT_TEST_TYPE,
    }


def independent_two_sample_summary(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = 0,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, float | int | str]:
    bootstrap = two_sample_bootstrap_diff(
        a,
        b,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    permutation = two_sample_permutation_test(
        a,
        b,
        n_resamples=n_resamples,
        seed=None if seed is None else seed + 1,
        alternative=alternative,
    )
    out = dict(bootstrap)
    out["p_value"] = permutation["p_value"]
    out["alternative"] = alternative
    if math.isnan(float(out["ci_low"])) or math.isnan(float(out["ci_high"])):
        raise AssertionError("bootstrap CI unexpectedly missing")
    return out
