from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import MODEL_LABELS, MODEL_ORDER

MARKERS = {
    "mobilenetv3": "o",
    "shufflenetv2": "s",
    "efficientnet_b0": "^",
    "deit_tiny": "D",
    "repvit_m1_0": "v",
    "efficientformer_l1": "P",
}

FULL_WIDTH = 7.15


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _cycle_colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _model_colors():
    colors = _cycle_colors()
    return {
        model: colors[index % len(colors)]
        for index, model in enumerate(MODEL_ORDER)
    }


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    result = frame.copy()
    result["_order"] = result["model"].map(order)
    return result.sort_values("_order").drop(columns="_order")


def figure_noise(data_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_dir / "mrl_corruption_family_results.csv")
    data = data[data["perturbation"].eq("noise")].copy()
    severities = ["mild", "moderate", "severe"]
    x = np.arange(3)
    colors = _model_colors()

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.45))
    for model in MODEL_ORDER:
        subset = (
            data[data["model"].eq(model)]
            .set_index("severity_label_original")
            .loc[severities]
        )
        y = subset["point_estimate_pooled"].to_numpy()
        low = subset["ci_low"].to_numpy()
        high = subset["ci_high"].to_numpy()
        ax.errorbar(
            x, y,
            yerr=np.vstack((y - low, high - y)),
            marker=MARKERS[model],
            color=colors[model],
            capsize=2.5,
            label=MODEL_LABELS[model],
        )
        collapse = (
            (subset["predicted_open_fraction"] <= 0.05)
            | (subset["predicted_open_fraction"] >= 0.95)
        )
        for xi, yi, flag in zip(x, y, collapse):
            if flag:
                ax.annotate(
                    "C", (xi, yi), xytext=(0, 7),
                    textcoords="offset points", ha="center",
                    fontsize=7, fontweight="bold",
                )
    ax.set_xticks(x, ["Mild", "Moderate", "Severe"])
    ax.set_xlabel("Gaussian-noise severity")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.27)
    _save(fig, output_dir / "fig03_noise_robustness")


def figure_target_domain(data_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_dir / "target_domain_effects.csv").sort_values(
        "delta_macro_f1_B2_minus_B1"
    )
    colors = _model_colors()
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.25))
    ax.axvline(0, linestyle="--", linewidth=0.9)

    for yi, row in zip(y, data.itertuples()):
        value = row.delta_macro_f1_B2_minus_B1
        low = row.delta_macro_f1_ci_low
        high = row.delta_macro_f1_ci_high
        filled = bool(row.macro_f1_ci_excludes_zero)
        ax.hlines(yi, low, high, color=colors[row.model], linewidth=1.5)
        ax.plot(
            value, yi,
            marker=MARKERS[row.model],
            markerfacecolor=colors[row.model] if filled else "none",
            markeredgecolor=colors[row.model],
            linestyle="none",
        )
        ax.text(high + 0.01, yi, f"{value:+.3f}", va="center", fontsize=7)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in data["model"]])
    ax.set_xlabel("Matched Macro-F1 change: target-domain OOF - zero-shot")
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.25)
    _save(fig, output_dir / "fig04_target_domain_effects")


def figure_operational(data_dir: Path, output_dir: Path) -> None:
    data = _ordered(pd.read_csv(data_dir / "deployment_summary.csv"))
    colors = _model_colors()
    deadline, recall_floor = 33.333, 0.50
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.8))
    ax.axvspan(0, deadline, ymin=recall_floor, ymax=1.0, alpha=0.08)
    ax.axvline(deadline, linestyle="--", linewidth=0.9)
    ax.axhline(recall_floor, linestyle=":", linewidth=0.9)

    for row in data.itertuples():
        collapsed = row.nonclean_collapse_count > 0
        ax.scatter(
            row.jetson_b2_p95_ms,
            row.worst_retained_class_recall,
            marker=MARKERS[row.model],
            facecolors="none" if collapsed else colors[row.model],
            edgecolors=colors[row.model],
            s=46,
        )
        suffix = f" C={int(row.nonclean_collapse_count)}" if collapsed else ""
        ax.annotate(
            MODEL_LABELS[row.model] + suffix,
            (row.jetson_b2_p95_ms, row.worst_retained_class_recall),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=6.7,
        )
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Jetson TensorRT FP32 P95 latency per binocular pair (ms)")
    ax.set_ylabel("Worst retained class recall across non-clean conditions")
    ax.grid(alpha=0.16)
    _save(fig, output_dir / "fig05_operational_constraint")


def figure_rise_deletion(data_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_dir / "xai_primary_faithfulness.csv").sort_values(
        "subject_macro_mean", ascending=False
    )
    colors = _model_colors()
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.2))
    for yi, row in zip(y, data.itertuples()):
        ax.hlines(yi, row.ci_lower, row.ci_upper, color=colors[row.model])
        ax.plot(
            row.subject_macro_mean, yi,
            marker=MARKERS[row.model],
            color=colors[row.model],
            linestyle="none",
        )
        ax.text(row.ci_upper + 0.008, yi, f"{row.subject_macro_mean:.3f}",
                va="center", fontsize=7)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in data["model"]])
    ax.set_xlabel("Normalized deletion AUC (lower is better)")
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.25)
    _save(fig, output_dir / "fig06_rise_deletion_forest")


def figure_efficiency_rank(data_dir: Path, output_dir: Path) -> None:
    data = _ordered(pd.read_csv(data_dir / "efficiency_rank_table.csv"))
    colors = _model_colors()
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.45))
    for row in data.itertuples():
        ranks = [
            row.parameters_rank,
            row.rtx_b1_rank,
            row.jetson_b1_rank,
        ]
        ax.plot(
            x, ranks,
            marker=MARKERS[row.model],
            color=colors[row.model],
            label=MODEL_LABELS[row.model],
        )
    ax.set_xticks(x, ["Parameter count", "RTX PyTorch\nbatch 1",
                      "Jetson TensorRT\nbatch 1"])
    ax.set_yticks(range(1, 7))
    ax.set_ylim(6.35, 0.65)
    ax.set_ylabel("Efficiency rank (1 = lowest cost / fastest)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.subplots_adjust(right=0.73)
    _save(fig, output_dir / "figS1_efficiency_rank_transfer")


def figure_insertion(data_dir: Path, output_dir: Path) -> None:
    data = _ordered(pd.read_csv(data_dir / "xai_insertion_coverage.csv"))
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.35))
    bars = ax.barh(y, data["insertion_valid_percent"], height=0.62)
    for bar, row in zip(bars, data.itertuples()):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{row.insertion_valid_percent:.1f}% ({row.insertion_valid_pairs}/500)",
            va="center", fontsize=7,
        )
    ax.axvline(9.8, linestyle="--", linewidth=0.9)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in data["model"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 102)
    ax.set_xlabel("Normalized insertion valid (%)")
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.25)
    _save(fig, output_dir / "figS2_insertion_coverage")


def _stability_plot(data, mean_col, p05_col, min_col, xlabel, stem, output_dir):
    ordered = data.sort_values(mean_col)
    y = np.arange(len(ordered))
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.3))
    for yi, row in zip(y, ordered.itertuples()):
        mean = getattr(row, mean_col)
        p05 = getattr(row, p05_col)
        minimum = getattr(row, min_col)
        ax.hlines(yi, minimum, mean, color=colors[row.model])
        ax.plot(minimum, yi, marker="D", markerfacecolor="none",
                markeredgecolor=colors[row.model], linestyle="none")
        ax.plot(p05, yi, marker="|", color=colors[row.model],
                markersize=10, linestyle="none")
        ax.plot(mean, yi, marker="o", color=colors[row.model],
                linestyle="none")
    ax.set_yticks(y, [MODEL_LABELS[m] for m in ordered["model"]])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.25)
    _save(fig, output_dir / stem)


def figure_stability(data_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_dir / "xai_stability.csv")
    _stability_plot(
        data, "mean_spearman", "p05_spearman", "min_spearman",
        "Pixelwise Spearman correlation across independent mask banks",
        "figS3_rise_spearman_stability", output_dir
    )
    _stability_plot(
        data, "mean_top10_iou", "p05_top10_iou", "min_top10_iou",
        "Top-10% saliency intersection over union",
        "figS4_rise_iou_stability", output_dir
    )


def figure_hcs_first(data_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_dir / "hcs_sensitivity.csv").copy()
    data["lattice"] = 100 * data["first_place_fractional_share"]
    data["dirichlet"] = 100 * data["dirichlet_first_share"]
    data = data.sort_values("lattice")
    y = np.arange(len(data))
    colors = _model_colors()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    for yi, row in zip(y, data.itertuples()):
        ax.hlines(yi, row.lattice, row.dirichlet, color=colors[row.model])
        ax.plot(row.lattice, yi, marker="o", color=colors[row.model],
                linestyle="none")
        ax.plot(row.dirichlet, yi, marker="s", markerfacecolor="none",
                markeredgecolor=colors[row.model], linestyle="none")
    ax.set_yticks(y, [MODEL_LABELS[m] for m in data["model"]])
    ax.set_xlabel("First-place share of preference space (%)")
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.25)
    _save(fig, output_dir / "figS5_hcs_first_place")


def figure_hcs_rank(data_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_dir / "hcs_sensitivity.csv").sort_values(
        "mean_competition_rank"
    )
    columns = [
        "rank_1_fraction", "rank_2_fraction",
        "rank_3_fraction", "rank_4_fraction",
    ]
    matrix = data[columns].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.7))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(4), ["Rank 1", "Rank 2", "Rank 3", "Rank 4"])
    ax.set_yticks(
        range(len(data)),
        [
            f"{MODEL_LABELS[m]} (mean {rank:.2f})"
            for m, rank in zip(data["model"], data["mean_competition_rank"])
        ],
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.31, right=0.93)
    _save(fig, output_dir / "figS6_hcs_rank_heatmap")


def make_all(data_dir: Path, output_dir: Path) -> None:
    publication_style()
    figure_noise(data_dir, output_dir)
    figure_target_domain(data_dir, output_dir)
    figure_operational(data_dir, output_dir)
    figure_rise_deletion(data_dir, output_dir)
    figure_efficiency_rank(data_dir, output_dir)
    figure_insertion(data_dir, output_dir)
    figure_stability(data_dir, output_dir)
    figure_hcs_first(data_dir, output_dir)
    figure_hcs_rank(data_dir, output_dir)
