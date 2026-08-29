"""
Operator-level cluster bootstrap (Section 4.5).

The pooled out-of-fold set holds 294,912 windows, but they are not 294,912
independent observations. Consecutive windows overlap by 50%, windows within a
session share an operator's physiology, and sessions within an operator share
their endurance time constant. Resampling windows would treat all of that as
independent and return an interval several times narrower than the truth.

The unit of replication is the operator, and there are 24 of them. This module
resamples operators with replacement, recomputes the metric on each resample,
and reports the percentile interval.

    python src/cluster_bootstrap.py
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def cluster_bootstrap_ci(
    per_operator_values: Sequence[float],
    per_operator_weights: Sequence[float] | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> tuple[float, float, float]:
    """Return (point estimate, lower bound, upper bound).

    Parameters
    ----------
    per_operator_values
        One value per operator, e.g. that operator's out-of-fold accuracy.
    per_operator_weights
        Window counts per operator. Equal weights if omitted.
    statistic
        Defaults to the weighted mean, which is the pooled metric.
    """
    v = np.asarray(per_operator_values, dtype=float)
    w = (np.ones_like(v) if per_operator_weights is None
         else np.asarray(per_operator_weights, dtype=float))
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")

    if statistic is None:
        def statistic(values: np.ndarray, weights: np.ndarray) -> float:
            return float(np.average(values, weights=weights))

    rng = np.random.default_rng(seed)
    n = v.size
    draws = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)          # resample OPERATORS, not windows
        draws[b] = statistic(v[idx], w[idx])

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return statistic(v, w), float(lo), float(hi)


def naive_window_bootstrap_ci(
    accuracy: float, n_windows: int, alpha: float = 0.05
) -> tuple[float, float]:
    """The interval one would get by ignoring clustering. Shown for contrast."""
    se = np.sqrt(accuracy * (1 - accuracy) / n_windows)
    z = 1.959963985
    return accuracy - z * se, accuracy + z * se


if __name__ == "__main__":
    # Per-operator out-of-fold accuracies are the study's own output and belong
    # in data/per_operator_accuracy.csv. The phenotype structure reported in
    # Section 7.3 is used here only to illustrate the machinery. The interval it
    # produces is not the one published: Table 4 gives [87.8, 90.8] for the
    # hybrid, computed from the 24 individual out-of-fold accuracies rather than
    # from three phenotype means.
    phenotypes = {"fast": (7, 85.1), "average": (10, 89.8), "resistant": (7, 92.8)}
    values, weights = [], []
    for n_ops, acc in phenotypes.values():
        values.extend([acc] * n_ops)
        weights.extend([12_288] * n_ops)

    point, lo, hi = cluster_bootstrap_ci(values, weights, n_resamples=1000, seed=42)
    print(f"operator-level cluster bootstrap: {point:.1f}%  "
          f"95% CI [{lo:.1f}, {hi:.1f}]")

    nlo, nhi = naive_window_bootstrap_ci(point / 100, 294_912)
    print(f"naive window-level interval:      {point:.1f}%  "
          f"95% CI [{nlo * 100:.1f}, {nhi * 100:.1f}]   <-- too narrow")
    print(f"\nwidth ratio: {(hi - lo) / ((nhi - nlo) * 100):.1f}x")
    print("Between-operator spread, not window count, governs the uncertainty.")
