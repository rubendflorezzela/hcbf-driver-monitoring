from __future__ import annotations

import numpy as np
import pandas as pd


def select_participant_class_median_samples(
    metrics: pd.DataFrame,
    *,
    expected_models: int = 6,
) -> pd.DataFrame:
    required = {
        "model",
        "xai_sample_id",
        "xai_sample_number",
        "subject_id",
        "true_class",
        "relative_path",
        "normalized_deletion_auc",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    work = metrics.copy()
    work["normalized_deletion_auc"] = pd.to_numeric(
        work["normalized_deletion_auc"], errors="coerce"
    )
    work = work[np.isfinite(work["normalized_deletion_auc"])].copy()
    duplicated = work.duplicated(["xai_sample_id", "model"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate sample/model rows were found.")
    image_summary = (
        work.groupby(
            [
                "xai_sample_id",
                "xai_sample_number",
                "subject_id",
                "true_class",
                "relative_path",
            ],
            as_index=False,
        )
        .agg(
            n_models=("model", "nunique"),
            six_model_mean_deletion=("normalized_deletion_auc", "mean"),
        )
    )
    image_summary = image_summary[image_summary["n_models"] == expected_models].copy()
    image_summary["participant_class_median"] = image_summary.groupby(
        ["subject_id", "true_class"]
    )["six_model_mean_deletion"].transform("median")
    image_summary["distance_to_median"] = (
        image_summary["six_model_mean_deletion"]
        - image_summary["participant_class_median"]
    ).abs()
    selected = (
        image_summary.sort_values(
            ["subject_id", "true_class", "distance_to_median", "xai_sample_id"],
            kind="mergesort",
        )
        .groupby(["subject_id", "true_class"], as_index=False, sort=False)
        .first()
    )
    return selected
