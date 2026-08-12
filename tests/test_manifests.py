from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_subject_split_manifest_matches_protocol():
    split = pd.read_csv(ROOT / "manifests/mrl_subject_split_seed42.csv")
    counts = split["split"].value_counts().to_dict()
    assert counts == {"train": 26, "val": 6, "test": 5}


def test_six_model_checkpoint_manifest():
    checkpoints = pd.read_csv(ROOT / "manifests/checkpoint_inventory_six_models.csv")
    assert checkpoints["model"].nunique() == 6
    assert (checkpoints["trainable_parameters"] > 0).all()
