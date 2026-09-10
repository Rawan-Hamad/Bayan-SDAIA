"""Lab 6: percentile bootstrap intervals for means and paired mean differences."""

import numpy as np


def _as_values(values, name):
    """Validate a non-empty, finite, one-dimensional numeric sample."""
    try:
        sample = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric sample") from exc
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError(f"{name} must be non-empty and one-dimensional")
    if not np.isfinite(sample).all():
        raise ValueError(f"{name} must contain only finite values")
    return sample


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Return (sample mean, lower bound, upper bound).

    Resample observations with replacement and take the alpha/2 and
    1-alpha/2 quantiles of their means (95% interval when alpha=0.05).
    A local random generator makes seeded calls reproducible without changing
    global random state. Observations are assumed to be independent sampling
    units. This estimates a mean, not a nonlinear metric such as macro-F1.
    """
    sample = _as_values(values, "values")
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if not np.isscalar(alpha) or isinstance(alpha, (str, bool, np.bool_)):
        raise ValueError("alpha must be a finite number between 0 and 1")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number between 0 and 1")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    # One resample at a time avoids allocating n_boot * sample.size indices.
    for iteration in range(n_boot):
        indices = rng.integers(0, sample.size, size=sample.size)
        means[iteration] = sample[indices].mean()
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(sample.mean()), float(lower), float(upper)


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Return (mean(a - b), lower bound, upper bound) using paired resampling.

    a[i] and b[i] must describe the same observation. Sampling their differences
    is equivalent to selecting the same indices in both arrays, preserving the
    pairing. A positive delta means a has the larger mean; an interval excluding
    zero is evidence of a difference under these sampling assumptions.
    """
    first = _as_values(a, "a")
    second = _as_values(b, "b")
    if first.shape != second.shape:
        raise ValueError("Paired samples must have equal lengths")
    return bootstrap_ci(first - second, n_boot=n_boot, seed=seed, alpha=alpha)
