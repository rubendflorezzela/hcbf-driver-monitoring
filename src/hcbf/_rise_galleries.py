#!/usr/bin/env python3
"""
Generate Supplementary Figures S.7 and S.8:
participant-stratified qualitative RISE galleries.

Selection rule
--------------
For each available participant-class stratum:

1. retain images with complete normalized-deletion results for all six models;
2. compute the six-model mean normalized deletion AUC for each image;
3. compute the median of that mean within the participant-class stratum;
4. select the image with the smallest absolute distance to the median;
5. break exact ties deterministically by xai_sample_id.

No visual inspection is used for sample selection.

Important data limitation
-------------------------
Under the frozen 500-image XAI corpus used in the manuscript, participant
s0008 has closed-eye examples but no eligible open-eye example. The default
policy omits unavailable strata and records them explicitly in the audit.
Use --missing-policy placeholder to retain a blank row, or
--missing-policy error to stop execution.

The script reads frozen RISE maps; it does not rerun RISE or model inference.

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
└── supplementary/
    └── figures/

Usage
-----
python generate_rise_supplementary_galleries.py ^
    --root "D:/Ruben/INVESTIGACIONES/HCBF-V2"

Selection audit only, without loading raw saliency maps:
python generate_rise_supplementary_galleries.py ^
    --root "D:/Ruben/INVESTIGACIONES/HCBF-V2" ^
    --selection-only

Outputs
-------
figS7_rise_closed_gallery.pdf/.svg/.png
figS8_rise_open_gallery.pdf/.svg/.png
figS7_S8_selection_audit.csv
figS7_S8_selection_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MODELS: Tuple[Tuple[str, str, str], ...] = (
    ("mobilenetv3", "MobileNetV3-Large", "MobileNetV3-\nLarge"),
    ("shufflenetv2", "ShuffleNetV2 x1.0", "ShuffleNetV2\nx1.0"),
    ("efficientnet_b0", "EfficientNet-B0", "EfficientNet-\nB0"),
    ("deit_tiny", "DeiT-Tiny", "DeiT-\nTiny"),
    ("repvit_m1_0", "RepViT-M1.0", "RepViT-\nM1.0"),
    ("efficientformer_l1", "EfficientFormer-L1", "EfficientFormer-\nL1"),
)

MODEL_KEYS = tuple(item[0] for item in MODELS)

REQUIRED_METRIC_COLUMNS = {
    "model",
    "shard_index",
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
}

REQUIRED_NPZ_ARRAYS = {
    "normalized_maps",
    "xai_sample_number",
}


@dataclass(frozen=True)
class SelectedSample:
    subject_id: str
    true_class: str
    xai_sample_id: str
    xai_sample_number: int
    relative_path: str
    n_complete_models: int
    subject_class_sample_count: int
    participant_class_median_mean_deletion: float
    six_model_mean_deletion: float
    absolute_distance_to_median: float


def configure_matplotlib() -> None:
    """Set a restrained, journal-oriented style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = (
        script_path.parent.parent
        if script_path.parent.name == "scripts"
        else Path.cwd()
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate participant-stratified qualitative RISE galleries "
            "for Supplementary Figures S.7 and S.8."
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
        help=(
            "Output directory. Default: "
            "<root>/supplementary/figures."
        ),
    )
    parser.add_argument(
        "--expected-subjects",
        nargs="*",
        default=None,
        help=(
            "Optional ordered subject list. By default, use the union of "
            "subjects present in the frozen XAI manifest."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("omit", "placeholder", "error"),
        default="omit",
        help=(
            "How to handle a participant without an eligible image for one "
            "class. Default: omit."
        ),
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help=(
            "Matplotlib colormap for normalized RISE importance. "
            "Use the same colormap in the main qualitative figure."
        ),
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.62,
        help="Saliency overlay opacity in [0,1]. Default: 0.62.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600 dpi.",
    )
    parser.add_argument(
        "--show-sample-id",
        action="store_true",
        help="Include the XAI sample ID in each row label.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Create only the selection audit. Raw saliency maps and source "
            "images are not required."
        ),
    )
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    source: Path,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def natural_subject_key(subject_id: str) -> Tuple[str, int]:
    prefix = "".join(ch for ch in subject_id if not ch.isdigit())
    digits = "".join(ch for ch in subject_id if ch.isdigit())
    return prefix, int(digits) if digits else -1


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
            else root / "supplementary" / "figures"
        ),
    }
    paths["registry"] = (
        paths["saliency_dir"] / "rise_full_saliency_shard_registry.csv"
    )

    for key in ("metrics", "manifest"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Required file not found: {paths[key]}")

    if not args.selection_only:
        for key in ("registry", "data_dir"):
            if not paths[key].exists():
                raise FileNotFoundError(
                    f"Required path not found: {paths[key]}"
                )

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def normalize_frame_strings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "model",
        "xai_sample_id",
        "subject_id",
        "true_class",
        "relative_path",
    ):
        if column in result.columns:
            result[column] = result[column].astype(str)
    if "true_class" in result.columns:
        result["true_class"] = result["true_class"].str.lower()
    if "relative_path" in result.columns:
        result["relative_path"] = result["relative_path"].str.replace(
            "\\", "/", regex=False
        )
    return result


def load_sources(
    paths: Mapping[str, Path],
    selection_only: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    metrics = normalize_frame_strings(pd.read_csv(paths["metrics"]))
    manifest = normalize_frame_strings(pd.read_csv(paths["manifest"]))

    require_columns(metrics, REQUIRED_METRIC_COLUMNS, paths["metrics"])
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, paths["manifest"])

    metrics["normalized_deletion_auc"] = pd.to_numeric(
        metrics["normalized_deletion_auc"],
        errors="coerce",
    )
    metrics["xai_sample_number"] = pd.to_numeric(
        metrics["xai_sample_number"],
        errors="raise",
    ).astype(int)
    metrics["shard_index"] = pd.to_numeric(
        metrics["shard_index"],
        errors="raise",
    ).astype(int)

    observed_models = set(metrics["model"].unique())
    missing_models = sorted(set(MODEL_KEYS) - observed_models)
    if missing_models:
        raise ValueError(
            "The metrics file does not contain all six models: "
            + ", ".join(missing_models)
        )

    # Explicitly preserve the frozen common-correct requirement when the
    # manifest provides these fields.
    if "common_correct_all_six" in manifest.columns:
        manifest = manifest[
            manifest["common_correct_all_six"].astype(bool)
        ].copy()
    if "n_models_correct" in manifest.columns:
        manifest = manifest[
            pd.to_numeric(
                manifest["n_models_correct"],
                errors="coerce",
            )
            == len(MODEL_KEYS)
        ].copy()

    registry: Optional[pd.DataFrame] = None
    if not selection_only:
        registry = normalize_frame_strings(pd.read_csv(paths["registry"]))
        require_columns(
            registry,
            REQUIRED_REGISTRY_COLUMNS,
            paths["registry"],
        )
        registry["shard_index"] = pd.to_numeric(
            registry["shard_index"],
            errors="raise",
        ).astype(int)

    return metrics, manifest, registry


def build_complete_image_summary(
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    work = metrics[
        metrics["model"].isin(MODEL_KEYS)
        & metrics["true_class"].isin({"closed", "open"})
        & np.isfinite(metrics["normalized_deletion_auc"])
    ].copy()

    duplicate = work.duplicated(
        ["xai_sample_id", "model"],
        keep=False,
    )
    if duplicate.any():
        bad = (
            work.loc[duplicate, ["xai_sample_id", "model"]]
            .drop_duplicates()
            .sort_values(["xai_sample_id", "model"])
        )
        raise ValueError(
            "Duplicate model-sample metric rows were found:\n"
            + bad.to_string(index=False)
        )

    summary = (
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
            n_complete_models=("model", "nunique"),
            six_model_mean_deletion=(
                "normalized_deletion_auc",
                "mean",
            ),
        )
    )
    summary = summary[
        summary["n_complete_models"] == len(MODEL_KEYS)
    ].copy()

    eligible_ids = set(manifest["xai_sample_id"].astype(str))
    summary = summary[
        summary["xai_sample_id"].isin(eligible_ids)
    ].copy()

    if summary.empty:
        raise ValueError(
            "No complete six-model, common-correct images were found."
        )

    summary["subject_class_sample_count"] = (
        summary.groupby(["subject_id", "true_class"])[
            "xai_sample_id"
        ].transform("nunique")
    )
    summary["participant_class_median_mean_deletion"] = (
        summary.groupby(["subject_id", "true_class"])[
            "six_model_mean_deletion"
        ].transform("median")
    )
    summary["absolute_distance_to_median"] = (
        summary["six_model_mean_deletion"]
        - summary["participant_class_median_mean_deletion"]
    ).abs()

    return summary


def choose_subject_class_samples(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ordered = summary.sort_values(
        [
            "subject_id",
            "true_class",
            "absolute_distance_to_median",
            "xai_sample_id",
        ],
        kind="mergesort",
    )
    selected = (
        ordered.groupby(
            ["subject_id", "true_class"],
            as_index=False,
            sort=False,
        )
        .first()
        .copy()
    )
    return selected


def selected_row_to_dataclass(row: pd.Series) -> SelectedSample:
    return SelectedSample(
        subject_id=str(row["subject_id"]),
        true_class=str(row["true_class"]),
        xai_sample_id=str(row["xai_sample_id"]),
        xai_sample_number=int(row["xai_sample_number"]),
        relative_path=str(row["relative_path"]),
        n_complete_models=int(row["n_complete_models"]),
        subject_class_sample_count=int(
            row["subject_class_sample_count"]
        ),
        participant_class_median_mean_deletion=float(
            row["participant_class_median_mean_deletion"]
        ),
        six_model_mean_deletion=float(
            row["six_model_mean_deletion"]
        ),
        absolute_distance_to_median=float(
            row["absolute_distance_to_median"]
        ),
    )


def resolve_subject_order(
    manifest: pd.DataFrame,
    expected_subjects: Optional[Sequence[str]],
) -> List[str]:
    if expected_subjects:
        return list(dict.fromkeys(str(x) for x in expected_subjects))
    return sorted(
        manifest["subject_id"].astype(str).unique().tolist(),
        key=natural_subject_key,
    )


def build_gallery_rows(
    selected: pd.DataFrame,
    true_class: str,
    subjects: Sequence[str],
    missing_policy: str,
) -> Tuple[List[Tuple[str, Optional[SelectedSample]]], List[str]]:
    lookup = {
        str(row["subject_id"]): selected_row_to_dataclass(row)
        for _, row in selected[
            selected["true_class"] == true_class
        ].iterrows()
    }

    missing = [subject for subject in subjects if subject not in lookup]
    if missing and missing_policy == "error":
        raise ValueError(
            f"Missing {true_class} strata for: {', '.join(missing)}"
        )

    rows: List[Tuple[str, Optional[SelectedSample]]] = []
    for subject in subjects:
        if subject in lookup:
            rows.append((subject, lookup[subject]))
        elif missing_policy == "placeholder":
            rows.append((subject, None))

    if missing_policy == "omit":
        rows = [
            (subject, lookup[subject])
            for subject in subjects
            if subject in lookup
        ]

    if not rows:
        raise ValueError(
            f"No rows are available for the {true_class} gallery."
        )

    return rows, missing


def sample_metric_rows(
    metrics: pd.DataFrame,
    sample_id: str,
) -> pd.DataFrame:
    rows = metrics[
        (metrics["xai_sample_id"] == sample_id)
        & metrics["model"].isin(MODEL_KEYS)
    ].copy()

    if len(rows) != len(MODEL_KEYS):
        raise ValueError(
            f"{sample_id} does not have exactly six model rows."
        )
    if rows["model"].nunique() != len(MODEL_KEYS):
        raise ValueError(
            f"{sample_id} contains duplicate or missing model rows."
        )

    return rows.set_index("model", drop=False)


def registry_row_for(
    registry: pd.DataFrame,
    model: str,
    shard_index: int,
) -> pd.Series:
    rows = registry[
        (registry["model"] == model)
        & (registry["shard_index"] == shard_index)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Registry lookup for {model}, shard {shard_index} "
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

    registry_row = registry_row_for(
        registry,
        model,
        shard_index,
    )
    npz_path = saliency_dir / Path(
        str(registry_row["npz_relative_path"])
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            "Raw RISE saliency shard not found:\n"
            f"  {npz_path}\n"
            "The compact uploaded results do not include the full raw "
            "saliency corpus. Run this script in the local HCBF project."
        )

    with np.load(npz_path, allow_pickle=False) as arrays:
        missing = REQUIRED_NPZ_ARRAYS - set(arrays.files)
        if missing:
            raise ValueError(
                f"{npz_path} is missing arrays: "
                + ", ".join(sorted(missing))
            )

        numbers = np.asarray(
            arrays["xai_sample_number"],
            dtype=int,
        )
        matches = np.flatnonzero(numbers == sample_number)
        if len(matches) != 1:
            raise ValueError(
                f"{npz_path}: sample number {sample_number} "
                f"occurs {len(matches)} times."
            )

        saliency = np.asarray(
            arrays["normalized_maps"][matches[0]],
            dtype=np.float32,
        )

    if saliency.ndim != 2:
        raise ValueError(
            f"Expected a 2-D saliency map for {model}; "
            f"received shape {saliency.shape}."
        )
    if not np.isfinite(saliency).all():
        raise ValueError(
            f"Non-finite saliency values for {model}, "
            f"sample {sample_number}."
        )
    if saliency.min() < -1e-6 or saliency.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Normalized saliency outside [0,1] for {model}: "
            f"[{saliency.min()}, {saliency.max()}]"
        )

    return np.clip(saliency, 0.0, 1.0)


def load_display_image(
    data_dir: Path,
    relative_path: str,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    image_path = data_dir / Path(relative_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Source MRL image not found: {image_path}"
        )

    height, width = target_shape
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(
            (width, height),
            resampling,
        )
        return np.asarray(image)


def hide_image_axes(axis: mpl.axes.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_placeholder_row(
    axes: Sequence[mpl.axes.Axes],
    subject_id: str,
    true_class: str,
) -> None:
    for axis in axes:
        axis.set_facecolor("0.96")
        axis.text(
            0.5,
            0.5,
            "No eligible\nsample",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=8,
        )
        hide_image_axes(axis)

    axes[0].text(
        -0.08,
        0.5,
        f"{subject_id}\n{true_class}",
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
    )


def create_gallery(
    true_class: str,
    rows: Sequence[Tuple[str, Optional[SelectedSample]]],
    metrics: pd.DataFrame,
    registry: pd.DataFrame,
    paths: Mapping[str, Path],
    cmap_name: str,
    overlay_alpha: float,
    png_dpi: int,
    show_sample_id: bool,
) -> Dict[str, Path]:
    if not (0.0 <= overlay_alpha <= 1.0):
        raise ValueError("--overlay-alpha must be in [0,1].")

    try:
        cmap = mpl.colormaps[cmap_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Matplotlib colormap: {cmap_name}"
        ) from exc

    nrows = len(rows)
    ncols = 1 + len(MODELS)

    figure_height = max(4.6, 1.65 * nrows + 1.0)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.8, figure_height),
        squeeze=False,
        constrained_layout=True,
    )

    column_titles = ["Input"] + [item[2] for item in MODELS]
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(
            title,
            fontsize=9,
            pad=6,
            fontweight="semibold",
        )

    last_overlay = None

    for row_index, (subject_id, sample) in enumerate(rows):
        row_axes = axes[row_index, :]

        if sample is None:
            draw_placeholder_row(
                row_axes,
                subject_id,
                true_class,
            )
            continue

        metric_rows = sample_metric_rows(
            metrics,
            sample.xai_sample_id,
        )

        first_metric = metric_rows.loc[MODELS[0][0]]
        first_map = load_saliency_map(
            paths["saliency_dir"],
            registry,
            first_metric,
        )
        image = load_display_image(
            paths["data_dir"],
            sample.relative_path,
            first_map.shape,
        )

        input_axis = row_axes[0]
        input_axis.imshow(
            image,
            interpolation="nearest",
        )
        hide_image_axes(input_axis)

        row_label_lines = [
            subject_id,
            true_class.capitalize(),
            f"mean D-nAUC = {sample.six_model_mean_deletion:.3f}",
        ]
        if show_sample_id:
            row_label_lines.append(sample.xai_sample_id)

        input_axis.text(
            -0.08,
            0.5,
            "\n".join(row_label_lines),
            transform=input_axis.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
        )

        for column_index, (model, _, _) in enumerate(
            MODELS,
            start=1,
        ):
            metric_row = metric_rows.loc[model]
            saliency = (
                first_map
                if model == MODELS[0][0]
                else load_saliency_map(
                    paths["saliency_dir"],
                    registry,
                    metric_row,
                )
            )

            axis = row_axes[column_index]
            axis.imshow(
                image,
                interpolation="nearest",
            )
            last_overlay = axis.imshow(
                saliency,
                cmap=cmap,
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            deletion = float(
                metric_row["normalized_deletion_auc"]
            )
            axis.text(
                0.03,
                0.96,
                f"D-nAUC {deletion:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.17",
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.68,
                },
            )
            hide_image_axes(axis)

    if last_overlay is not None:
        colorbar = fig.colorbar(
            last_overlay,
            ax=axes[:, 1:].ravel().tolist(),
            fraction=0.012,
            pad=0.010,
            aspect=42,
        )
        colorbar.set_label(
            "Normalized RISE importance",
            fontsize=8.5,
        )
        colorbar.ax.tick_params(labelsize=7.5)

    figure_number = "S7" if true_class == "closed" else "S8"
    stem = (
        "figS7_rise_closed_gallery"
        if true_class == "closed"
        else "figS8_rise_open_gallery"
    )

    output_paths = {
        "pdf": paths["output_dir"] / f"{stem}.pdf",
        "svg": paths["output_dir"] / f"{stem}.svg",
        "png": paths["output_dir"] / f"{stem}.png",
    }

    fig.savefig(
        output_paths["pdf"],
        bbox_inches="tight",
    )
    fig.savefig(
        output_paths["svg"],
        bbox_inches="tight",
    )
    fig.savefig(
        output_paths["png"],
        dpi=png_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"Supplementary Figure {figure_number}: "
        f"{len(rows)} displayed subject rows."
    )
    return output_paths


def write_audit(
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    subjects: Sequence[str],
    missing_by_class: Mapping[str, Sequence[str]],
    output_dir: Path,
    missing_policy: str,
    figure_outputs: Optional[Mapping[str, Mapping[str, Path]]],
) -> Tuple[Path, Path]:
    records: List[dict] = []

    for _, selected_row in selected.iterrows():
        sample = selected_row_to_dataclass(selected_row)
        model_rows = sample_metric_rows(
            metrics,
            sample.xai_sample_id,
        )
        base_record = asdict(sample)

        for model, display_name, _ in MODELS:
            record = dict(base_record)
            record.update(
                {
                    "model": model,
                    "display_name": display_name,
                    "normalized_deletion_auc": float(
                        model_rows.loc[
                            model,
                            "normalized_deletion_auc",
                        ]
                    ),
                    "selection_rule": (
                        "nearest_to_participant_class_median_of_"
                        "six_model_mean_normalized_deletion_auc"
                    ),
                }
            )
            records.append(record)

    csv_path = output_dir / "figS7_S8_selection_audit.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    json_path = output_dir / "figS7_S8_selection_audit.json"
    payload = {
        "status": "PASS_SUPPLEMENTARY_RISE_GALLERY_SELECTION",
        "selection_rule": (
            "For every available participant-class stratum, select the "
            "common-correct image whose six-model mean normalized deletion "
            "AUC is closest to the median of that stratum. Resolve exact "
            "ties by xai_sample_id."
        ),
        "expected_subject_order": list(subjects),
        "missing_policy": missing_policy,
        "missing_subject_class_strata": {
            key: list(value)
            for key, value in missing_by_class.items()
        },
        "selected_samples": [
            asdict(selected_row_to_dataclass(row))
            for _, row in selected.iterrows()
        ],
        "figure_outputs": (
            {
                class_name: {
                    format_name: str(path)
                    for format_name, path in paths.items()
                }
                for class_name, paths in figure_outputs.items()
            }
            if figure_outputs
            else None
        ),
    }
    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    paths = resolve_paths(args)

    metrics, manifest, registry = load_sources(
        paths,
        args.selection_only,
    )
    summary = build_complete_image_summary(
        metrics,
        manifest,
    )
    selected = choose_subject_class_samples(summary)
    subjects = resolve_subject_order(
        manifest,
        args.expected_subjects,
    )

    gallery_rows: Dict[str, List[Tuple[str, Optional[SelectedSample]]]] = {}
    missing_by_class: Dict[str, List[str]] = {}

    for true_class in ("closed", "open"):
        rows, missing = build_gallery_rows(
            selected,
            true_class,
            subjects,
            args.missing_policy,
        )
        gallery_rows[true_class] = rows
        missing_by_class[true_class] = missing

        if missing:
            print(
                f"WARNING: no eligible {true_class} sample for: "
                + ", ".join(missing),
                file=sys.stderr,
            )

    figure_outputs: Optional[Dict[str, Dict[str, Path]]] = None

    if not args.selection_only:
        if registry is None:
            raise RuntimeError(
                "The saliency registry was not loaded."
            )

        figure_outputs = {}
        for true_class in ("closed", "open"):
            figure_outputs[true_class] = create_gallery(
                true_class=true_class,
                rows=gallery_rows[true_class],
                metrics=metrics,
                registry=registry,
                paths=paths,
                cmap_name=args.cmap,
                overlay_alpha=args.overlay_alpha,
                png_dpi=args.png_dpi,
                show_sample_id=args.show_sample_id,
            )

    csv_path, json_path = write_audit(
        selected=selected,
        metrics=metrics,
        subjects=subjects,
        missing_by_class=missing_by_class,
        output_dir=paths["output_dir"],
        missing_policy=args.missing_policy,
        figure_outputs=figure_outputs,
    )

    print("\nSelected participant-class samples:")
    print(
        selected[
            [
                "subject_id",
                "true_class",
                "xai_sample_id",
                "six_model_mean_deletion",
                "participant_class_median_mean_deletion",
                "absolute_distance_to_median",
            ]
        ].to_string(index=False)
    )
    print(f"\nSelection CSV:  {csv_path}")
    print(f"Selection JSON: {json_path}")

    if args.selection_only:
        print(
            "Selection-only mode completed. "
            "No figure files were generated."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
