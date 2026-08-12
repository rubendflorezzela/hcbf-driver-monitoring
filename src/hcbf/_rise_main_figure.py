#!/usr/bin/env python3
"""
Generate the main-manuscript qualitative RISE comparison.

The figure contains two common-correct MRL test images:
- one closed-eye image;
- one open-eye image;
- the images must belong to different participants.

Default selection rule
----------------------
For each image, the mean normalized deletion AUC across the six models is
computed. Among all closed/open pairs from different participants, the pair
that minimizes the sum of the absolute distances to the corresponding
class-specific medians is selected. Visual appearance is not used for
selection.

The script reads the already generated RISE saliency maps. It does not rerun
model inference or regenerate explanations.

Expected project structure
--------------------------
<ROOT>/
├── data/
│   └── mrlEyes_2018_01/
├── results/
│   ├── audit_45_xai_sample_manifest/
│   │   └── xai_primary_sample_manifest.csv
│   ├── audit_49_rise_full_saliency/
│   │   ├── rise_full_saliency_shard_registry.csv
│   │   └── shards/<model>/*.npz
│   └── audit_54_rise_full_faithfulness_curves/
│       └── rise_full_curve_metrics.csv
└── figures/

Usage
-----
Automatic reproducible selection:
    python generate_rise_qualitative_comparison.py --root "D:/Ruben/INVESTIGACIONES/HCBF-V2"

Manual override after inspecting the automatically selected samples:
    python generate_rise_qualitative_comparison.py ^
        --root "D:/Ruben/INVESTIGACIONES/HCBF-V2" ^
        --closed-id XAI_0058 ^
        --open-id XAI_0319

Outputs
-------
<output-dir>/fig_rise_qualitative_comparison.pdf
<output-dir>/fig_rise_qualitative_comparison.png
<output-dir>/fig_rise_qualitative_selection.csv
<output-dir>/fig_rise_qualitative_selection.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MODELS: Tuple[Tuple[str, str], ...] = (
    ("mobilenetv3", "MobileNetV3-Large"),
    ("shufflenetv2", "ShuffleNetV2 x1.0"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("deit_tiny", "DeiT-Tiny"),
    ("repvit_m1_0", "RepViT-M1.0"),
    ("efficientformer_l1", "EfficientFormer-L1"),
)

REQUIRED_METRIC_COLUMNS = {
    "model",
    "display_name",
    "shard_index",
    "local_index",
    "xai_sample_id",
    "xai_sample_number",
    "relative_path",
    "subject_id",
    "true_class",
    "normalized_deletion_auc",
}

REQUIRED_MANIFEST_COLUMNS = {
    "xai_sample_id",
    "relative_path",
    "subject_id",
    "true_class",
    "image_sha256",
}

REQUIRED_REGISTRY_COLUMNS = {
    "model",
    "shard_index",
    "npz_relative_path",
    "first_xai_sample_id",
    "last_xai_sample_id",
    "status",
}

REQUIRED_NPZ_ARRAYS = {
    "normalized_maps",
    "true_label",
    "xai_sample_number",
}


@dataclass(frozen=True)
class SelectedSample:
    true_class: str
    xai_sample_id: str
    xai_sample_number: int
    subject_id: str
    relative_path: str
    class_median_mean_deletion: float
    six_model_mean_deletion: float
    absolute_distance_to_class_median: float


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent if script_path.parent.name == "scripts" else Path.cwd()

    parser = argparse.ArgumentParser(
        description=(
            "Create a two-row qualitative RISE figure from the frozen "
            "six-model saliency corpus."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="HCBF project root.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Override rise_full_curve_metrics.csv.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Override xai_primary_sample_manifest.csv.",
    )
    parser.add_argument(
        "--saliency-dir",
        type=Path,
        default=None,
        help="Override results/audit_49_rise_full_saliency.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data/mrlEyes_2018_01.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/figures.",
    )
    parser.add_argument(
        "--closed-id",
        type=str,
        default=None,
        help="Optional manual closed-eye XAI sample ID.",
    )
    parser.add_argument(
        "--open-id",
        type=str,
        default=None,
        help="Optional manual open-eye XAI sample ID.",
    )
    parser.add_argument(
        "--allow-same-subject",
        action="store_true",
        help="Allow manual samples from the same participant.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Saliency overlay opacity in [0,1]. Default: 0.55.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600 dpi.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = args.root.expanduser().resolve()
    paths = {
        "root": root,
        "metrics": (
            args.metrics_csv.expanduser().resolve()
            if args.metrics_csv
            else root
            / "results"
            / "audit_54_rise_full_faithfulness_curves"
            / "rise_full_curve_metrics.csv"
        ),
        "manifest": (
            args.manifest_csv.expanduser().resolve()
            if args.manifest_csv
            else root
            / "results"
            / "audit_45_xai_sample_manifest"
            / "xai_primary_sample_manifest.csv"
        ),
        "saliency_dir": (
            args.saliency_dir.expanduser().resolve()
            if args.saliency_dir
            else root / "results" / "audit_49_rise_full_saliency"
        ),
        "data_dir": (
            args.data_dir.expanduser().resolve()
            if args.data_dir
            else root / "data" / "mrlEyes_2018_01"
        ),
        "output_dir": (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else root / "figures"
        ),
    }
    paths["registry"] = (
        paths["saliency_dir"] / "rise_full_saliency_shard_registry.csv"
    )

    for key in ("metrics", "manifest", "registry"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Required file not found: {paths[key]}")
    if not paths["data_dir"].exists():
        raise FileNotFoundError(f"MRL image directory not found: {paths['data_dir']}")

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def load_sources(paths: Mapping[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(paths["metrics"])
    manifest = pd.read_csv(paths["manifest"])
    registry = pd.read_csv(paths["registry"])

    require_columns(metrics, REQUIRED_METRIC_COLUMNS, paths["metrics"])
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, paths["manifest"])
    require_columns(registry, REQUIRED_REGISTRY_COLUMNS, paths["registry"])

    metrics["xai_sample_id"] = metrics["xai_sample_id"].astype(str)
    metrics["model"] = metrics["model"].astype(str)
    metrics["true_class"] = metrics["true_class"].astype(str).str.lower()
    metrics["relative_path"] = (
        metrics["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    metrics["normalized_deletion_auc"] = pd.to_numeric(
        metrics["normalized_deletion_auc"], errors="coerce"
    )

    manifest["xai_sample_id"] = manifest["xai_sample_id"].astype(str)
    manifest["true_class"] = manifest["true_class"].astype(str).str.lower()
    manifest["relative_path"] = (
        manifest["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )

    registry["model"] = registry["model"].astype(str)
    registry["npz_relative_path"] = (
        registry["npz_relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )

    expected_models = {model for model, _ in MODELS}
    observed_models = set(metrics["model"].unique())
    missing_models = sorted(expected_models - observed_models)
    if missing_models:
        raise ValueError(
            "The curve metrics do not contain all six models: "
            + ", ".join(missing_models)
        )

    return metrics, manifest, registry


def build_image_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    expected_models = {model for model, _ in MODELS}
    work = metrics[
        metrics["model"].isin(expected_models)
        & metrics["true_class"].isin({"closed", "open"})
        & np.isfinite(metrics["normalized_deletion_auc"])
    ].copy()

    duplicate = work.duplicated(["xai_sample_id", "model"], keep=False)
    if duplicate.any():
        rows = work.loc[duplicate, ["xai_sample_id", "model"]].drop_duplicates()
        raise ValueError(
            "Duplicate model/sample rows were found:\n"
            + rows.to_string(index=False)
        )

    summary = (
        work.groupby(
            ["xai_sample_id", "xai_sample_number", "subject_id", "true_class", "relative_path"],
            as_index=False,
        )
        .agg(
            n_models=("model", "nunique"),
            six_model_mean_deletion=("normalized_deletion_auc", "mean"),
        )
    )
    summary = summary[summary["n_models"] == len(MODELS)].copy()

    counts = summary.groupby("true_class")["xai_sample_id"].nunique().to_dict()
    if counts.get("closed", 0) == 0 or counts.get("open", 0) == 0:
        raise ValueError("No complete six-model samples were found for both classes.")

    summary["class_median_mean_deletion"] = summary.groupby("true_class")[
        "six_model_mean_deletion"
    ].transform("median")
    summary["distance_to_class_median"] = (
        summary["six_model_mean_deletion"]
        - summary["class_median_mean_deletion"]
    ).abs()
    return summary


def choose_automatic_pair(summary: pd.DataFrame) -> Tuple[SelectedSample, SelectedSample]:
    closed = summary[summary["true_class"] == "closed"].copy()
    open_eye = summary[summary["true_class"] == "open"].copy()

    candidate_pairs: List[Tuple[float, str, str, pd.Series, pd.Series]] = []
    for _, closed_row in closed.iterrows():
        for _, open_row in open_eye.iterrows():
            if str(closed_row["subject_id"]) == str(open_row["subject_id"]):
                continue
            total_distance = (
                float(closed_row["distance_to_class_median"])
                + float(open_row["distance_to_class_median"])
            )
            candidate_pairs.append(
                (
                    total_distance,
                    str(closed_row["xai_sample_id"]),
                    str(open_row["xai_sample_id"]),
                    closed_row,
                    open_row,
                )
            )

    if not candidate_pairs:
        raise ValueError(
            "No closed/open pair from different participants is available."
        )

    _, _, _, closed_row, open_row = min(candidate_pairs, key=lambda item: item[:3])
    return row_to_selected(closed_row), row_to_selected(open_row)


def choose_manual_pair(
    summary: pd.DataFrame,
    closed_id: str,
    open_id: str,
    allow_same_subject: bool,
) -> Tuple[SelectedSample, SelectedSample]:
    def get_one(sample_id: str, expected_class: str) -> pd.Series:
        rows = summary[summary["xai_sample_id"] == sample_id]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one complete six-model row for {sample_id}; found {len(rows)}."
            )
        row = rows.iloc[0]
        if row["true_class"] != expected_class:
            raise ValueError(
                f"{sample_id} is {row['true_class']}, not {expected_class}."
            )
        return row

    closed_row = get_one(closed_id, "closed")
    open_row = get_one(open_id, "open")
    if (
        not allow_same_subject
        and str(closed_row["subject_id"]) == str(open_row["subject_id"])
    ):
        raise ValueError(
            "Manual samples belong to the same participant. "
            "Use --allow-same-subject only if this is intentional."
        )
    return row_to_selected(closed_row), row_to_selected(open_row)


def row_to_selected(row: pd.Series) -> SelectedSample:
    return SelectedSample(
        true_class=str(row["true_class"]),
        xai_sample_id=str(row["xai_sample_id"]),
        xai_sample_number=int(row["xai_sample_number"]),
        subject_id=str(row["subject_id"]),
        relative_path=str(row["relative_path"]),
        class_median_mean_deletion=float(row["class_median_mean_deletion"]),
        six_model_mean_deletion=float(row["six_model_mean_deletion"]),
        absolute_distance_to_class_median=float(row["distance_to_class_median"]),
    )


def find_manifest_row(manifest: pd.DataFrame, sample: SelectedSample) -> pd.Series:
    rows = manifest[manifest["xai_sample_id"] == sample.xai_sample_id]
    if len(rows) != 1:
        raise ValueError(
            f"Manifest lookup for {sample.xai_sample_id} returned {len(rows)} rows."
        )
    row = rows.iloc[0]
    checks = {
        "subject_id": str(row["subject_id"]) == sample.subject_id,
        "true_class": str(row["true_class"]) == sample.true_class,
        "relative_path": str(row["relative_path"]) == sample.relative_path,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Manifest mismatch for {sample.xai_sample_id}: {', '.join(failed)}"
        )
    return row


def load_display_image(data_dir: Path, relative_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    image_path = data_dir / Path(relative_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    height, width = target_hw
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((width, height), resampling)
        return np.asarray(image)


def registry_row_for(
    registry: pd.DataFrame,
    model: str,
    shard_index: int,
) -> pd.Series:
    rows = registry[
        (registry["model"] == model)
        & (pd.to_numeric(registry["shard_index"], errors="coerce") == shard_index)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Registry lookup for model={model}, shard={shard_index} "
            f"returned {len(rows)} rows."
        )
    return rows.iloc[0]


def load_saliency_map(
    saliency_dir: Path,
    registry: pd.DataFrame,
    metric_row: pd.Series,
) -> np.ndarray:
    model = str(metric_row["model"])
    shard_index = int(metric_row["shard_index"])
    sample_number = int(metric_row["xai_sample_number"])

    reg = registry_row_for(registry, model, shard_index)
    npz_path = saliency_dir / Path(str(reg["npz_relative_path"]))
    if not npz_path.exists():
        raise FileNotFoundError(
            "The raw RISE saliency shard is missing:\n"
            f"  {npz_path}\n\n"
            "The compact audit_49 ZIP contains only the registry and lock files. "
            "This figure requires the full raw saliency corpus generated in "
            "results/audit_49_rise_full_saliency/shards/."
        )

    with np.load(npz_path, allow_pickle=False) as arrays:
        missing = REQUIRED_NPZ_ARRAYS - set(arrays.files)
        if missing:
            raise ValueError(
                f"{npz_path} is missing arrays: {', '.join(sorted(missing))}"
            )
        numbers = arrays["xai_sample_number"].astype(int)
        matches = np.flatnonzero(numbers == sample_number)
        if len(matches) != 1:
            raise ValueError(
                f"{npz_path}: sample number {sample_number} occurs {len(matches)} times."
            )
        saliency = np.asarray(arrays["normalized_maps"][matches[0]], dtype=np.float32)

    if saliency.ndim != 2 or not np.isfinite(saliency).all():
        raise ValueError(f"Invalid saliency map for {model}, sample {sample_number}.")
    if saliency.min() < -1e-6 or saliency.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Normalized saliency range is outside [0,1] for {model}: "
            f"[{saliency.min()}, {saliency.max()}]"
        )
    return np.clip(saliency, 0.0, 1.0)


def sample_metric_rows(metrics: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    rows = metrics[
        (metrics["xai_sample_id"] == sample_id)
        & metrics["model"].isin([model for model, _ in MODELS])
    ].copy()
    if rows["model"].nunique() != len(MODELS) or len(rows) != len(MODELS):
        raise ValueError(
            f"{sample_id} does not have exactly one metric row for each model."
        )
    return rows.set_index("model", drop=False)


def hide_image_axes(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def create_figure(
    selected: Sequence[SelectedSample],
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    registry: pd.DataFrame,
    paths: Mapping[str, Path],
    overlay_alpha: float,
    png_dpi: int,
) -> Tuple[Path, Path]:
    if not (0.0 <= overlay_alpha <= 1.0):
        raise ValueError("--overlay-alpha must be in [0,1].")

    nrows = len(selected)
    ncols = 1 + len(MODELS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.4, 5.1),
        squeeze=False,
        constrained_layout=True,
    )

    column_titles = [
        "Input",
        "MobileNetV3-\nLarge",
        "ShuffleNetV2\nx1.0",
        "EfficientNet-\nB0",
        "DeiT-\nTiny",
        "RepViT-\nM1.0",
        "EfficientFormer-\nL1",
    ]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10, pad=7)

    last_overlay = None
    for row_index, sample in enumerate(selected):
        find_manifest_row(manifest, sample)
        rows = sample_metric_rows(metrics, sample.xai_sample_id)

        first_model_row = rows.loc[MODELS[0][0]]
        first_map = load_saliency_map(
            paths["saliency_dir"], registry, first_model_row
        )
        image = load_display_image(
            paths["data_dir"],
            sample.relative_path,
            first_map.shape,
        )

        input_axis = axes[row_index, 0]
        input_axis.imshow(image)
        input_axis.set_xlabel(
            f"{sample.true_class.capitalize()} eye\n"
            f"Six-model mean D-nAUC = {sample.six_model_mean_deletion:.3f}",
            fontsize=8,
            labelpad=4,
        )
        input_axis.text(
            -0.08,
            0.5,
            f"({chr(ord('a') + row_index)})\n{sample.true_class.capitalize()}",
            transform=input_axis.transAxes,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        hide_image_axes(input_axis)

        for column_index, (model, _) in enumerate(MODELS, start=1):
            metric_row = rows.loc[model]
            saliency = (
                first_map
                if model == MODELS[0][0]
                else load_saliency_map(
                    paths["saliency_dir"], registry, metric_row
                )
            )
            axis = axes[row_index, column_index]
            axis.imshow(image)
            last_overlay = axis.imshow(
                saliency,
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=1.0,
            )
            deletion = float(metric_row["normalized_deletion_auc"])
            axis.set_xlabel(
                f"D-nAUC = {deletion:.3f}",
                fontsize=8,
                labelpad=4,
            )
            hide_image_axes(axis)

    if last_overlay is not None:
        colorbar = fig.colorbar(
            last_overlay,
            ax=axes[:, 1:].ravel().tolist(),
            fraction=0.015,
            pad=0.012,
            aspect=35,
        )
        colorbar.set_label("Normalized RISE importance", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    pdf_path = paths["output_dir"] / "fig_rise_qualitative_comparison.pdf"
    png_path = paths["output_dir"] / "fig_rise_qualitative_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_selection_audit(
    selected: Sequence[SelectedSample],
    metrics: pd.DataFrame,
    paths: Mapping[str, Path],
    mode: str,
) -> Tuple[Path, Path]:
    rows: List[dict] = []
    for sample in selected:
        model_rows = sample_metric_rows(metrics, sample.xai_sample_id)
        base = asdict(sample)
        base["selection_mode"] = mode
        for model, display_name in MODELS:
            row = dict(base)
            row["model"] = model
            row["display_name"] = display_name
            row["normalized_deletion_auc"] = float(
                model_rows.loc[model, "normalized_deletion_auc"]
            )
            rows.append(row)

    csv_path = paths["output_dir"] / "fig_rise_qualitative_selection.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = paths["output_dir"] / "fig_rise_qualitative_selection.json"
    payload = {
        "status": "PASS_QUALITATIVE_RISE_FIGURE",
        "selection_mode": mode,
        "selection_rule": (
            "Closed/open pair from different participants minimizing the "
            "combined absolute distance to the class-specific medians of the "
            "six-model mean normalized deletion AUC."
            if mode == "automatic_median_pair"
            else "Manual XAI sample IDs supplied by the user."
        ),
        "samples": [asdict(sample) for sample in selected],
        "outputs": {
            "pdf": str(paths["output_dir"] / "fig_rise_qualitative_comparison.pdf"),
            "png": str(paths["output_dir"] / "fig_rise_qualitative_comparison.png"),
            "selection_csv": str(csv_path),
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    metrics, manifest, registry = load_sources(paths)
    summary = build_image_summary(metrics)

    manual_requested = args.closed_id is not None or args.open_id is not None
    if manual_requested and not (args.closed_id and args.open_id):
        raise ValueError(
            "--closed-id and --open-id must be supplied together."
        )

    if manual_requested:
        selected = choose_manual_pair(
            summary,
            args.closed_id,
            args.open_id,
            args.allow_same_subject,
        )
        mode = "manual"
    else:
        selected = choose_automatic_pair(summary)
        mode = "automatic_median_pair"

    pdf_path, png_path = create_figure(
        selected=selected,
        metrics=metrics,
        manifest=manifest,
        registry=registry,
        paths=paths,
        overlay_alpha=args.overlay_alpha,
        png_dpi=args.png_dpi,
    )
    csv_path, json_path = write_selection_audit(
        selected, metrics, paths, mode
    )

    print("Qualitative RISE figure created successfully.")
    for sample in selected:
        print(
            f"  {sample.true_class}: {sample.xai_sample_id}, "
            f"subject={sample.subject_id}, "
            f"mean D-nAUC={sample.six_model_mean_deletion:.6f}, "
            f"distance to class median="
            f"{sample.absolute_distance_to_class_median:.6f}"
        )
    print(f"  PDF:  {pdf_path}")
    print(f"  PNG:  {png_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
