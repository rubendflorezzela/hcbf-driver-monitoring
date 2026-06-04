"""
train.py
========
Two-stage fine-tuning of a single model, following the protocol in Section 4.3:
  - Warmup: first WARMUP_EPOCHS epochs train only the classification head.
  - Joint:  remaining epochs train all layers with cosine-annealed LR.
Early stopping is based on validation macro F1 (patience EARLY_STOP_PATIENCE).

Usage:
    python train.py --model mobilenetv3
    python train.py --model all
"""

import os
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm

import config as C
import dataset as D
from models import build_model, get_backbone_parameters, set_backbone_trainable


def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_f1(model, loader, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = logits.argmax(dim=1).cpu().numpy()
        preds.append(p)
        gts.append(y.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return f1_score(gts, preds, average="macro")


def train_one(model_name, splits, device=C.DEVICE, verbose=True):
    set_seed()
    model = build_model(model_name, pretrained=True).to(device)

    train_loader = D.make_loader(splits["train"], train=True)
    val_loader = D.make_loader(splits["val"], train=False)

    backbone_params, head_params = get_backbone_parameters(model, model_name)
    criterion = nn.CrossEntropyLoss()

    # Stage 1: head-only
    set_backbone_trainable(model, model_name, trainable=False)
    optimizer = AdamW(head_params, lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    best_f1, best_state, patience = -1.0, None, 0
    history = {"train_loss": [], "val_f1": [], "stop_epoch": None}

    scheduler = None
    for epoch in range(C.MAX_EPOCHS):
        # Switch to joint training after warmup
        if epoch == C.WARMUP_EPOCHS:
            set_backbone_trainable(model, model_name, trainable=True)
            optimizer = AdamW(model.parameters(), lr=C.LR,
                              weight_decay=C.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=C.MAX_EPOCHS - C.WARMUP_EPOCHS, eta_min=C.LR_MIN
            )

        model.train()
        running = 0.0
        pbar = tqdm(train_loader, disable=not verbose,
                    desc=f"[{model_name}] epoch {epoch + 1}/{C.MAX_EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        if scheduler is not None:
            scheduler.step()

        train_loss = running / len(train_loader.dataset)
        val_f1 = evaluate_f1(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        if verbose:
            print(f"  train_loss={train_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                history["stop_epoch"] = epoch + 1
                if verbose:
                    print(f"  early stopping at epoch {epoch + 1}")
                break

    if history["stop_epoch"] is None:
        history["stop_epoch"] = C.MAX_EPOCHS

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = os.path.join(C.CHECKPOINT_DIR, f"{model_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(C.CHECKPOINT_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if verbose:
        print(f"  saved checkpoint -> {ckpt_path} (best val_f1={best_f1:.4f})")
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    help="model name or 'all'")
    args = ap.parse_args()

    splits, meta = D.build_splits()
    print(f"Split type: {meta['split_type']}")
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(splits[k])} images")
    with open(os.path.join(C.OUTPUT_DIR, "split_meta.json"), "w") as f:
        json.dump({"meta": meta,
                   "counts": {k: len(splits[k]) for k in splits}}, f, indent=2)

    names = C.MODELS if args.model == "all" else [args.model]
    for name in names:
        print(f"\n=== Training {C.MODEL_DISPLAY.get(name, name)} ===")
        train_one(name, splits, device=C.DEVICE)


if __name__ == "__main__":
    main()
