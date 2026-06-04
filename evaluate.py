"""
evaluate.py
===========
Computes the four HCBF dimensions for each trained model, following the
formal definitions in Section 3:

  Accuracy       (Eq. 2): mean of Top-1, macro F1, AUC-ROC
  Explainability (Eq. 3): 0.5 * ((1 - Deletion_AUC) + Insertion_AUC)
  Efficiency     (Eq. 4): min-max normalized (params, FLOPs, latency), inverted
  Robustness     (Eq. 5): mean F1 retention across 9 perturbation conditions

Deletion / Insertion follow Petsiuk et al. (RISE). Attribution maps are
computed with Captum (model-agnostic GradientShap) so the same method applies
to both CNN and transformer models. Grad-CAM heatmaps are produced separately
for qualitative figures (CNNs only; transformers use the GradientShap map).

Outputs a JSON with raw and normalized scores per model.

Usage:
    python evaluate.py --model all

Author: Ruben Dario Florez-Zela
"""

import os
import json
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

import config as C
import dataset as D
from models import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_trained(model_name, device=C.DEVICE):
    model = build_model(model_name, pretrained=False)
    ckpt = os.path.join(C.CHECKPOINT_DIR, f"{model_name}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model


def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dimension 1: Accuracy (Eq. 2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_accuracy(model, test_items, device=C.DEVICE):
    loader = D.make_loader(test_items, train=False)
    preds, probs, gts = [], [], []
    for x, y in tqdm(loader, desc="  accuracy", leave=False):
        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(p[:, 1].cpu().numpy())     # P(open) for AUC
        gts.append(y.numpy())
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    gts = np.concatenate(gts)

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    try:
        auc = roc_auc_score(gts, probs)
    except ValueError:
        auc = float("nan")
    alpha = np.nanmean([acc, f1, auc])
    return {"top1": float(acc), "f1": float(f1), "auc": float(auc),
            "alpha": float(alpha)}


# ---------------------------------------------------------------------------
# Dimension 2: Explainability (Eq. 3) - Deletion / Insertion AUC
# ---------------------------------------------------------------------------
def _gaussian_blur_baseline(x, ksize=C.DELINS_BLUR_KSIZE):
    """Return a heavily blurred version of x (1,C,H,W) as the insertion baseline."""
    import cv2
    img = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)   # HxWxC (normalized)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return torch.tensor(blurred.transpose(2, 0, 1)).unsqueeze(0).to(x.device)


def _delins_curve(model, x, attribution, target, mode, steps=C.DELINS_STEPS):
    """
    Compute a deletion or insertion probability curve for one image.
    x: (1,C,H,W) normalized tensor
    attribution: (H,W) importance (already aggregated over channels)
    mode: 'deletion' or 'insertion'
    Returns AUC in [0,1].
    """
    device = x.device
    C_, H, W = x.shape[1], x.shape[2], x.shape[3]
    n_pixels = H * W
    order = np.argsort(-attribution.reshape(-1))     # most important first

    if mode == "deletion":
        start = x.clone()
        end = torch.zeros_like(x)                     # remove -> zero
    else:  # insertion
        start = _gaussian_blur_baseline(x)
        end = x.clone()

    # Build a batch of `steps+1` progressively modified images
    per_step = max(1, n_pixels // steps)
    batch = [start.clone()]
    cur = start.clone()
    cur_flat = cur.view(1, C_, -1)
    end_flat = end.view(1, C_, -1)
    for s in range(steps):
        idx = order[s * per_step:(s + 1) * per_step]
        if len(idx) == 0:
            break
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        cur_flat[:, :, idx_t] = end_flat[:, :, idx_t]
        batch.append(cur_flat.view(1, C_, H, W).clone())
    batch_t = torch.cat(batch, dim=0)

    with torch.no_grad():
        probs = F.softmax(model(batch_t), dim=1)[:, target].cpu().numpy()
    # Normalized AUC (trapezoidal) over the curve
    auc = float(np.trapz(probs, dx=1.0 / (len(probs) - 1)))
    return auc


def eval_explainability(model, test_items, model_name,
                        n_samples=C.N_EXPLAIN_SAMPLES, device=C.DEVICE):
    from captum.attr import GradientShap
    set_seed()
    sample = random.sample(test_items, min(n_samples, len(test_items)))
    tfm = D.eval_transform()

    gs = GradientShap(model)
    del_aucs, ins_aucs = [], []

    from PIL import Image
    for path, _ in tqdm(sample, desc="  explainability", leave=False):
        img = Image.open(path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            target = int(model(x).argmax(1).item())

        # Model-agnostic attribution (works for CNN and transformer)
        baseline = torch.cat([torch.zeros_like(x),
                              _gaussian_blur_baseline(x)], dim=0)
        attr = gs.attribute(x, baselines=baseline, target=target,
                            n_samples=8, stdevs=0.1)
        attr_map = attr.squeeze(0).abs().sum(0).cpu().numpy()   # (H,W)

        del_aucs.append(_delins_curve(model, x, attr_map, target, "deletion"))
        ins_aucs.append(_delins_curve(model, x, attr_map, target, "insertion"))

    del_auc = float(np.mean(del_aucs))
    ins_auc = float(np.mean(ins_aucs))
    eps = 0.5 * ((1.0 - del_auc) + ins_auc)
    return {"deletion_auc": del_auc, "insertion_auc": ins_auc, "eps": float(eps)}


# ---------------------------------------------------------------------------
# Dimension 3: Efficiency (Eq. 4) - raw measurements (normalized later)
# ---------------------------------------------------------------------------
def eval_efficiency(model_name):
    from thop import profile
    model = build_model(model_name, pretrained=False).to(C.LATENCY_DEVICE).eval()
    dummy = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE, device=C.LATENCY_DEVICE)

    flops, params = profile(model, inputs=(dummy,), verbose=False)

    # Latency on CPU (worst-case embedded scenario)
    import time
    with torch.no_grad():
        for _ in range(C.LATENCY_WARMUP):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(C.LATENCY_RUNS):
            model(dummy)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / C.LATENCY_RUNS * 1000.0

    return {"params_M": params / 1e6,
            "flops_G": flops / 1e9,
            "latency_ms": float(latency_ms)}


# ---------------------------------------------------------------------------
# Dimension 4: Robustness (Eq. 5)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _f1_under(model, test_items, perturb_fn, device=C.DEVICE):
    loader = D.make_loader(test_items, train=False, perturb_fn=perturb_fn)
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).argmax(1).cpu().numpy())
        gts.append(y.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return f1_score(gts, preds, average="macro")


def eval_robustness(model, test_items, device=C.DEVICE):
    clean_f1 = _f1_under(model, test_items, D.make_perturbation("clean", 0), device)
    conditions = {}
    retentions = []
    for ptype in ("noise", "brightness", "blur"):
        for s_idx, s_label in enumerate(C.SEVERITY_LABELS):
            fn = D.make_perturbation(ptype, s_idx)
            f1 = _f1_under(model, test_items, fn, device)
            ret = f1 / clean_f1 if clean_f1 > 0 else 0.0
            conditions[f"{ptype}_{s_label}"] = {"f1": float(f1),
                                                "retention": float(ret)}
            retentions.append(ret)
    rho = float(np.mean(retentions))
    return {"clean_f1": float(clean_f1), "conditions": conditions, "rho": float(rho)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--skip_explain", action="store_true",
                    help="skip the (slow) explainability dimension")
    args = ap.parse_args()

    splits, _ = D.build_splits()
    test_items = splits["test"]
    names = C.MODELS if args.model == "all" else [args.model]

    results = {}
    for name in names:
        print(f"\n=== Evaluating {C.MODEL_DISPLAY.get(name, name)} ===")
        model = load_trained(name)
        res = {}
        res["accuracy"] = eval_accuracy(model, test_items)
        print(f"  accuracy: {res['accuracy']}")
        res["efficiency"] = eval_efficiency(name)
        print(f"  efficiency: {res['efficiency']}")
        res["robustness"] = eval_robustness(model, test_items)
        print(f"  robustness rho={res['robustness']['rho']:.4f}")
        if not args.skip_explain:
            res["explainability"] = eval_explainability(model, test_items, name)
            print(f"  explainability: {res['explainability']}")
        results[name] = res

    out = os.path.join(C.OUTPUT_DIR, "raw_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results -> {out}")
    print("Run analysis.py to compute normalized scores, Pareto and HCS.")


if __name__ == "__main__":
    main()
