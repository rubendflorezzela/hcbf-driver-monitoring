"""
dataset.py
==========
MRL Eye Dataset loading with subject-level train/val/test split, training
augmentations, and perturbation wrappers for the robustness dimension.

Supports two on-disk layouts:
  (A) Original MRL flat folder with encoded filenames:
      sXXXX_YYYYY_<gender>_<glasses>_<eyestate>_<reflection>_<lighting>_<sensor>.png
      Field index 0 -> subject id ; field index 4 -> eye state (0 closed, 1 open)
  (B) Folder-based: DATA_DIR/Open_Eyes/*  and  DATA_DIR/Close_Eyes/*
      In this layout subject ids are unknown, so the split falls back to a
      stratified IMAGE-level split and a warning is printed.

Author: Ruben Dario Florez-Zela
"""

import os
import glob
import random
import warnings
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config as C


# ---------------------------------------------------------------------------
# Filename parsing for the original MRL layout
# ---------------------------------------------------------------------------
def _parse_mrl_filename(fname):
    """Return (subject_id, eye_state) or None if the filename is not MRL-encoded."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 5:
        return None
    subject = parts[0]              # e.g. 's0001'
    try:
        eye_state = int(parts[4])   # 0 closed, 1 open
    except ValueError:
        return None
    if eye_state not in (0, 1):
        return None
    return subject, eye_state


def _scan_flat_layout(data_dir):
    """Scan original MRL flat layout. Returns list of (path, label, subject)."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    items = []
    for p in paths:
        parsed = _parse_mrl_filename(p)
        if parsed is not None:
            subject, label = parsed
            items.append((p, label, subject))
    return items


def _scan_folder_layout(data_dir):
    """Scan folder-based layout. Returns list of (path, label, subject=None)."""
    mapping = {
        "open": 1, "open_eyes": 1, "openeyes": 1, "1": 1,
        "closed": 0, "close": 0, "close_eyes": 0, "closed_eyes": 0,
        "closeeyes": 0, "0": 0,
    }
    items = []
    for sub in os.listdir(data_dir):
        full = os.path.join(data_dir, sub)
        if not os.path.isdir(full):
            continue
        key = sub.strip().lower().replace(" ", "_")
        if key not in mapping:
            continue
        label = mapping[key]
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in glob.glob(os.path.join(full, ext)):
                items.append((p, label, None))
    return items


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------
def build_splits(data_dir=C.DATA_DIR, seed=C.SEED):
    """
    Build train/val/test splits.

    Returns a dict: {'train': [(path,label),...], 'val': [...], 'test': [...]}
    Uses subject-level split when subject ids are available (original MRL
    layout), otherwise falls back to a stratified image-level split.
    """
    rng = random.Random(seed)

    items = _scan_flat_layout(data_dir)
    subject_level = len(items) > 0

    if not subject_level:
        items = _scan_folder_layout(data_dir)
        if len(items) == 0:
            raise FileNotFoundError(
                f"No images found under '{data_dir}'. Set MRL_DATA_DIR or "
                f"edit config.DATA_DIR. Expected either the original MRL flat "
                f"layout with encoded filenames, or Open_Eyes/Close_Eyes folders."
            )
        warnings.warn(
            "Subject ids could not be parsed from filenames. Falling back to a "
            "stratified IMAGE-level split. NOTE: this may overestimate "
            "performance relative to the subject-level split described in the "
            "paper. For a faithful reproduction, use the original MRL layout "
            "with encoded filenames."
        )

    if subject_level:
        # Group by subject, then split subjects (not images)
        by_subject = defaultdict(list)
        for p, label, subj in items:
            by_subject[subj].append((p, label))
        subjects = sorted(by_subject.keys())
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(round(C.TRAIN_FRAC * n))
        n_val = int(round(C.VAL_FRAC * n))
        train_subj = subjects[:n_train]
        val_subj = subjects[n_train:n_train + n_val]
        test_subj = subjects[n_train + n_val:]

        def collect(subjs):
            out = []
            for s in subjs:
                out.extend(by_subject[s])
            return out

        splits = {
            "train": collect(train_subj),
            "val": collect(val_subj),
            "test": collect(test_subj),
        }
        meta = {
            "split_type": "subject-level",
            "n_subjects": {"train": len(train_subj),
                           "val": len(val_subj),
                           "test": len(test_subj)},
        }
    else:
        # Stratified image-level split
        pairs = [(p, label) for (p, label, _) in items]
        by_label = defaultdict(list)
        for p, label in pairs:
            by_label[label].append((p, label))
        splits = {"train": [], "val": [], "test": []}
        for label, lst in by_label.items():
            rng.shuffle(lst)
            n = len(lst)
            n_train = int(round(C.TRAIN_FRAC * n))
            n_val = int(round(C.VAL_FRAC * n))
            splits["train"].extend(lst[:n_train])
            splits["val"].extend(lst[n_train:n_train + n_val])
            splits["test"].extend(lst[n_train + n_val:])
        meta = {"split_type": "image-level (fallback)", "n_subjects": None}

    for k in splits:
        rng.shuffle(splits[k])
    return splits, meta


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def train_transform():
    return T.Compose([
        T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),  # factor in [0.8, 1.2]
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Perturbations for robustness (operate on a uint8 HxWxC RGB array)
# ---------------------------------------------------------------------------
def apply_gaussian_noise(img_u8, sigma):
    noise = np.random.normal(0.0, sigma, img_u8.shape)
    out = img_u8.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness(img_u8, factor):
    out = img_u8.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_motion_blur(img_u8, ksize):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize          # horizontal motion
    return cv2.filter2D(img_u8, -1, kernel)


def make_perturbation(ptype, severity_idx):
    """Return a function uint8->uint8 for the given perturbation/severity."""
    if ptype == "clean":
        return lambda x: x
    if ptype == "noise":
        sigma = C.GAUSSIAN_SIGMAS[severity_idx]
        return lambda x: apply_gaussian_noise(x, sigma)
    if ptype == "brightness":
        factor = C.BRIGHTNESS_FACTORS[severity_idx]
        return lambda x: apply_brightness(x, factor)
    if ptype == "blur":
        ksize = C.MOTION_BLUR_KSIZES[severity_idx]
        return lambda x: apply_motion_blur(x, ksize)
    raise ValueError(f"Unknown perturbation type: {ptype}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MRLEyeDataset(Dataset):
    """
    items: list of (path, label)
    perturb_fn: optional uint8->uint8 function applied BEFORE normalization
                (used for the robustness dimension).
    """

    def __init__(self, items, transform, perturb_fn=None):
        self.items = items
        self.transform = transform
        self.perturb_fn = perturb_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.perturb_fn is not None:
            arr = np.array(img)                       # HxWxC uint8 RGB
            arr = self.perturb_fn(arr)
            img = Image.fromarray(arr)
        tensor = self.transform(img)
        return tensor, label


def make_loader(items, train=False, perturb_fn=None, batch_size=C.BATCH_SIZE,
                shuffle=None):
    tfm = train_transform() if train else eval_transform()
    ds = MRLEyeDataset(items, tfm, perturb_fn=perturb_fn)
    if shuffle is None:
        shuffle = train
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=C.NUM_WORKERS, pin_memory=True)
