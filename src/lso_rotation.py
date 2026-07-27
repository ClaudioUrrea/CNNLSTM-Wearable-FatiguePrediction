"""
Leave-subjects-out (LSO) rotation used in Section 5.4 of the manuscript.

The 24 synthetic operators are split into six disjoint blocks of four. For
fold i the test set is block i and the validation set is block i+1 (mod 6),
so every operator is held out for testing exactly once and for validation
exactly once, and is never scored by a model that has seen its own data.

This rotation is not a refinement. The intervention arm of the study needs a
fatigue estimate for all 24 operators; under a single fixed 16/4/4 split, 20
of them would either go unscored or be scored by a model trained on their own
windows, and neither outcome supports the claim that the interventions were
triggered by prediction.

Run directly to print the fold table and verify the invariants:
    python src/lso_rotation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

N_OPERATORS = 24
N_FOLDS = 6
BLOCK = N_OPERATORS // N_FOLDS  # 4 operators per block


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        allocated = set(self.train) | set(self.validation) | set(self.test)
        if len(allocated) != N_OPERATORS:
            raise ValueError(f"fold {self.index} does not partition the cohort")
        if set(self.train) & set(self.test):
            raise ValueError(f"fold {self.index} leaks operators into training")
        if set(self.validation) & set(self.test):
            raise ValueError(f"fold {self.index} leaks operators into validation")


def make_blocks(operator_ids: Sequence[int], seed: int = 42) -> list[tuple[int, ...]]:
    """Shuffle the cohort once, then cut it into six blocks of four.

    Stratification is deliberately *not* applied to the endurance phenotype.
    Balancing fast, average and fatigue-resistant operators across folds would
    make each test set easier than a real deployment, where an unseen operator
    arrives with whatever phenotype they have.
    """
    ids = np.asarray(operator_ids)
    if ids.size != N_OPERATORS:
        raise ValueError(f"expected {N_OPERATORS} operators, received {ids.size}")
    order = np.random.default_rng(seed).permutation(ids.size)
    shuffled = ids[order]
    return [tuple(int(x) for x in shuffled[i * BLOCK:(i + 1) * BLOCK])
            for i in range(N_FOLDS)]


def iter_folds(operator_ids: Sequence[int], seed: int = 42) -> Iterator[Fold]:
    """Yield the six folds described in Section 5.4."""
    blocks = make_blocks(operator_ids, seed=seed)
    for i in range(N_FOLDS):
        test = blocks[i]
        validation = blocks[(i + 1) % N_FOLDS]
        train = tuple(
            op for j, block in enumerate(blocks)
            if j not in (i, (i + 1) % N_FOLDS)
            for op in block
        )
        yield Fold(index=i, train=train, validation=validation, test=test)


def check_invariants(operator_ids: Sequence[int], seed: int = 42) -> None:
    """Assert the properties the manuscript claims for the rotation."""
    folds = list(iter_folds(operator_ids, seed=seed))
    assert len(folds) == N_FOLDS

    test_counts = {op: 0 for op in operator_ids}
    val_counts = {op: 0 for op in operator_ids}
    for f in folds:
        assert len(f.train) == 16 and len(f.validation) == 4 and len(f.test) == 4
        for op in f.test:
            test_counts[op] += 1
        for op in f.validation:
            val_counts[op] += 1

    assert all(c == 1 for c in test_counts.values()), \
        "every operator must be held out for testing exactly once"
    assert all(c == 1 for c in val_counts.values()), \
        "every operator must be used for validation exactly once"

    # Pooling the six test sets reproduces the cohort exactly once over.
    pooled = [op for f in folds for op in f.test]
    assert sorted(pooled) == sorted(operator_ids)


if __name__ == "__main__":
    operators = list(range(1, N_OPERATORS + 1))
    check_invariants(operators)

    print(f"{'fold':>4}  {'test':<20} {'validation':<20} train")
    for f in iter_folds(operators):
        print(f"{f.index:>4}  {str(list(f.test)):<20} "
              f"{str(list(f.validation)):<20} {len(f.train)} operators")

    windows_per_operator = 12_288
    print(f"\npooled out-of-fold windows: "
          f"{N_OPERATORS * windows_per_operator:,}")
    print("invariants verified: each operator tested once, validated once, "
          "never scored by a model that saw it")
