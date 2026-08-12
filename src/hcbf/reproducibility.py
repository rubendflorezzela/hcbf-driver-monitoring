from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_results(root: Path) -> list[str]:
    problems: list[str] = []
    tables = root / "results/frozen/supplementary_tables"
    figures = root / "results/frozen/figure_data"

    clean = pd.read_csv(tables / "table_S2A_clean_metrics.csv")
    zero = pd.read_csv(tables / "table_S2B_zero_shot_metrics.csv")
    target_domain = pd.read_csv(tables / "table_S6_matched_target_domain_effects.csv")
    deployment = pd.read_csv(figures / "deployment_summary.csv")
    xai = pd.read_csv(figures / "xai_primary_faithfulness.csv")
    insertion = pd.read_csv(figures / "xai_insertion_coverage.csv")

    checks = [
        (abs(clean["macro_f1"].min() - 0.9566) < 5e-4, "clean Macro-F1 minimum"),
        (abs(clean["macro_f1"].max() - 0.9794) < 5e-4, "clean Macro-F1 maximum"),
        (abs(zero["macro_f1"].min() - 0.2066) < 5e-4, "zero-shot Macro-F1 minimum"),
        (abs(zero["macro_f1"].max() - 0.7771) < 5e-4, "zero-shot Macro-F1 maximum"),
        (
            abs(target_domain["delta_macro_f1_B2_minus_B1"].min() - (-0.0406)) < 5e-4,
            "target-domain effect minimum",
        ),
        (
            abs(target_domain["delta_macro_f1_B2_minus_B1"].max() - 0.4807) < 5e-4,
            "target-domain effect maximum",
        ),
        (deployment["primary_gate_pass"].astype(bool).sum() == 0, "empty eligible set"),
        (abs(xai["subject_macro_mean"].min() - 0.5826) < 5e-4, "RISE deletion minimum"),
        (abs(xai["subject_macro_mean"].max() - 0.9113) < 5e-4, "RISE deletion maximum"),
        (abs(insertion["insertion_valid_percent"].min() - 17.6) < 0.05, "insertion coverage minimum"),
        (abs(insertion["insertion_valid_percent"].max() - 88.6) < 0.05, "insertion coverage maximum"),
    ]
    problems.extend(label for passed, label in checks if not passed)

    split = pd.read_csv(root / "manifests/mrl_subject_split_seed42.csv")
    if split["split"].value_counts().to_dict() != {"train": 26, "val": 6, "test": 5}:
        problems.append("MRL split counts")

    checkpoints = pd.read_csv(root / "manifests/checkpoint_inventory_six_models.csv")
    if checkpoints["model"].nunique() != 6:
        problems.append("six-model checkpoint inventory")

    return problems


def write_frozen_checksums(root: Path) -> Path:
    targets = []
    for folder in [
        root / "results/frozen",
        root / "manifests",
        root / "configs",
    ]:
        targets.extend(p for p in folder.rglob("*") if p.is_file())

    manifest = root / "manifests/frozen_files.sha256"
    lines = []
    for path in sorted(targets):
        if path == manifest:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def verify_frozen_checksums(root: Path) -> list[str]:
    manifest = root / "manifests/frozen_files.sha256"
    problems = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif sha256(path) != expected:
            problems.append(f"hash mismatch: {relative}")
    return problems
