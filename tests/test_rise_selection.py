import pandas as pd

from hcbf.rise_selection import select_participant_class_median_samples


def test_median_selection_is_deterministic():
    rows = []
    for sample, mean in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        for model in range(6):
            rows.append(
                {
                    "model": f"m{model}",
                    "xai_sample_id": sample,
                    "xai_sample_number": ord(sample),
                    "subject_id": "s1",
                    "true_class": "closed",
                    "relative_path": f"{sample}.png",
                    "normalized_deletion_auc": mean,
                }
            )
    selected = select_participant_class_median_samples(pd.DataFrame(rows))
    assert selected.iloc[0]["xai_sample_id"] == "b"
