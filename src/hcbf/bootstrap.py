from __future__ import annotations

from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd


def participant_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    participant_col: str,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int = 10_000,
    seed: int = 20260725,
) -> tuple[float, float, float]:
    participants = frame[participant_col].drop_duplicates().tolist()
    if not participants:
        raise ValueError("No participants are available.")
    point = float(statistic(frame))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    grouped = {key: group for key, group in frame.groupby(participant_col, sort=False)}
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        replicate = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        value = float(statistic(replicate))
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("Every bootstrap statistic was non-finite.")
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def paired_participant_bootstrap_difference(
    frame: pd.DataFrame,
    *,
    participant_col: str,
    statistic_a: Callable[[pd.DataFrame], float],
    statistic_b: Callable[[pd.DataFrame], float],
    replicates: int = 10_000,
    seed: int = 20260726,
) -> tuple[float, float, float]:
    participants = frame[participant_col].drop_duplicates().tolist()
    grouped = {key: group for key, group in frame.groupby(participant_col, sort=False)}
    point = float(statistic_b(frame) - statistic_a(frame))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        replicate = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        value = float(statistic_b(replicate) - statistic_a(replicate))
        if np.isfinite(value):
            values.append(value)
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)
