"""
gradcam_figures.py
==================
Generates qualitative saliency heatmaps for the Results / Failure Case
Analysis (Section 5). For CNN models, uses Grad-CAM (Captum LayerGradCam) on
the last convolutional layer. For DeiT-Tiny, falls back to a model-agnostic
GradientShap map (Grad-CAM is not directly applicable to transformers).

Produces a grid: rows = a few example images, columns = models
(both correctly classified and misclassified, ideally under perturbation).

Usage:
    python gradcam_figures.py --n 6

Author: Ruben Dario Florez-Zela
"""

import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import config as C
import dataset as D
from models import build_model, gradcam_target_layer
from evaluate import load_trained, _gaussian_blur_baseline


def _denormalize(x):
    mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    img = x.squeeze(0).cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def saliency_map(model, model_name, x, target):
    """Return a (H,W) heatmap in [0,1]."""
    if C.MODEL_IS_CNN[model_name]:
        from captum.attr import LayerGradCam, LayerAttribution
        layer = gradcam_target_layer(model, model_name)
        lgc = LayerGradCam(model, layer)
        attr = lgc.attribute(x, target=target)
        attr = LayerAttribution.interpolate(attr, (C.IMG_SIZE, C.IMG_SIZE))
        hm = attr.squeeze().detach().cpu().numpy()
        hm = np.maximum(hm, 0)
    else:
        from captum.attr import GradientShap
        gs = GradientShap(model)
        baseline = torch.cat([torch.zeros_like(x),
                              _gaussian_blur_baseline(x)], dim=0)
        attr = gs.attribute(x, baselines=baseline, target=target,
                            n_samples=8, stdevs=0.1)
        hm = attr.squeeze(0).abs().sum(0).detach().cpu().numpy()
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    return hm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3,
                    help="number of example rows (3 keeps the figure wide and short)")
    ap.add_argument("--perturb", default="blur",
                    choices=["clean", "noise", "brightness", "blur"])
    ap.add_argument("--severity", type=int, default=2, help="0=mild..2=severe")
    args = ap.parse_args()

    random.seed(C.SEED)
    splits, _ = D.build_splits()
    sample = random.sample(splits["test"], args.n)
    tfm = D.eval_transform()
    perturb_fn = D.make_perturbation(args.perturb, args.severity)

    # Transposed layout (compact, wider than tall):
    #   rows    = examples (GT)        -> keep args.n small, e.g. 3
    #   columns = Input + the 4 models -> 5 columns
    n_rows = args.n
    n_cols = len(C.MODELS) + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.7 * n_cols, 1.7 * n_rows))
    # Ensure axes is always 2D even if n_rows == 1
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Pre-load models once (avoids reloading per example)
    models = {name: load_trained(name) for name in C.MODELS}

    # Column titles (only on the top row)
    col_titles = ["Input"] + [C.MODEL_DISPLAY[n] for n in C.MODELS]

    for i, (path, label) in enumerate(sample):
        img = Image.open(path).convert("RGB")
        arr = perturb_fn(np.array(img))
        x = tfm(Image.fromarray(arr)).unsqueeze(0).to(C.DEVICE)

        # Column 0: perturbed input
        axes[i, 0].imshow(_denormalize(x))
        axes[i, 0].set_ylabel(f"GT={C.CLASS_NAMES[label]}", fontsize=8)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontsize=9)

        # Columns 1..K: each model's prediction + saliency
        for j, name in enumerate(C.MODELS, start=1):
            model = models[name]
            with torch.no_grad():
                pred = int(model(x).argmax(1).item())
            hm = saliency_map(model, name, x, pred)
            axes[i, j].imshow(_denormalize(x))
            axes[i, j].imshow(hm, cmap="jet", alpha=0.45)
            ok = "OK" if pred == label else "X"
            axes[i, j].set_title(
                col_titles[j] if i == 0 else "", fontsize=9)
            axes[i, j].set_xlabel(f"{C.CLASS_NAMES[pred]} [{ok}]", fontsize=8)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.tight_layout()
    out = os.path.join(C.FIGURE_DIR, f"gradcam_{args.perturb}_s{args.severity}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}  ({n_rows} rows x {n_cols} cols)")


if __name__ == "__main__":
    main()