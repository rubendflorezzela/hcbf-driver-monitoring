from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_level_frozen_anchors():
    tables = ROOT / "results/frozen/supplementary_tables"
    figures = ROOT / "results/frozen/figure_data"
    clean = pd.read_csv(tables / "table_S2A_clean_metrics.csv")
    zero = pd.read_csv(tables / "table_S2B_zero_shot_metrics.csv")
    target_domain = pd.read_csv(tables / "table_S6_matched_target_domain_effects.csv")
    deployment = pd.read_csv(figures / "deployment_summary.csv")
    xai = pd.read_csv(figures / "xai_primary_faithfulness.csv")
    insertion = pd.read_csv(figures / "xai_insertion_coverage.csv")

    assert abs(clean["macro_f1"].min() - 0.9566) < 5e-4
    assert abs(clean["macro_f1"].max() - 0.9794) < 5e-4
    assert abs(zero["macro_f1"].min() - 0.2066) < 5e-4
    assert abs(zero["macro_f1"].max() - 0.7771) < 5e-4
    assert abs(target_domain["delta_macro_f1_B2_minus_B1"].min() - (-0.0406)) < 5e-4
    assert abs(target_domain["delta_macro_f1_B2_minus_B1"].max() - 0.4807) < 5e-4
    assert deployment["primary_gate_pass"].astype(bool).sum() == 0
    assert abs(xai["subject_macro_mean"].min() - 0.5826) < 5e-4
    assert abs(xai["subject_macro_mean"].max() - 0.9113) < 5e-4
    assert abs(insertion["insertion_valid_percent"].min() - 17.6) < 0.05
    assert abs(insertion["insertion_valid_percent"].max() - 88.6) < 0.05
