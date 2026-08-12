from __future__ import annotations

import random
from collections.abc import Sequence


def build_subject_split(
    subjects: Sequence[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, str]:
    """Return a deterministic subject-level train/validation/test assignment."""
    unique = sorted(set(subjects))
    if len(unique) != len(subjects):
        raise ValueError("Subject identifiers must be unique.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Fractions must lie in (0,1).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one.")

    shuffled = unique.copy()
    random.Random(seed).shuffle(shuffled)
    n_train = int(round(train_fraction * len(shuffled)))
    n_val = int(round(validation_fraction * len(shuffled)))

    assignment: dict[str, str] = {}
    for subject in shuffled[:n_train]:
        assignment[subject] = "train"
    for subject in shuffled[n_train:n_train + n_val]:
        assignment[subject] = "val"
    for subject in shuffled[n_train + n_val:]:
        assignment[subject] = "test"
    return assignment


def assert_subject_disjoint(rows: Sequence[tuple[str, str]]) -> None:
    """Validate that each subject appears in exactly one split."""
    observed: dict[str, str] = {}
    for subject, split in rows:
        previous = observed.setdefault(subject, split)
        if previous != split:
            raise ValueError(
                f"Subject {subject!r} appears in both {previous!r} and {split!r}."
            )
