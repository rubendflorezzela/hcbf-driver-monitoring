"""
config.py
=========
Central configuration for the Human-Centered Benchmarking Framework (HCBF).
All values match the experimental protocol described in Sections 3 and 4
of the paper.

Author: Ruben Dario Florez-Zela
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Point DATA_DIR to the folder that contains the MRL Eye Dataset images.
# The loader supports two layouts (see dataset.py):
#   (A) Original MRL flat folder with encoded filenames
#       e.g. s0001_00123_0_0_1_0_1_01.png
#   (B) Folder-based: DATA_DIR/Open_Eyes/*.png and DATA_DIR/Close_Eyes/*.png
#       (subject-level split is NOT possible in this layout; see warning)
DATA_DIR = os.environ.get("MRL_DATA_DIR", "./data/mrlEyes_2018_01")

OUTPUT_DIR = "./results"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Latency is always measured on CPU to reflect worst-case embedded deployment
# (see Section 4.4). Training uses DEVICE.
LATENCY_DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Image / data
# ---------------------------------------------------------------------------
IMG_SIZE = 224
NUM_CLASSES = 2                      # 0 = closed, 1 = open
CLASS_NAMES = ["closed", "open"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Subject-level split fractions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15                     # remainder

# ---------------------------------------------------------------------------
# Training protocol (Section 4.3)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 30
WARMUP_EPOCHS = 5                    # head-only training before unfreezing
LR = 1e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Models to benchmark (Section 4.2)
# ---------------------------------------------------------------------------
MODELS = ["mobilenetv3", "shufflenetv2", "efficientnet_b0", "deit_tiny"]

MODEL_DISPLAY = {
    "mobilenetv3": "MobileNetV3-Large",
    "shufflenetv2": "ShuffleNetV2 x1.0",
    "efficientnet_b0": "EfficientNet-B0",
    "deit_tiny": "DeiT-Tiny",
}

# Whether the model is a CNN (Grad-CAM applicable) or transformer
MODEL_IS_CNN = {
    "mobilenetv3": True,
    "shufflenetv2": True,
    "efficientnet_b0": True,
    "deit_tiny": False,
}

# ---------------------------------------------------------------------------
# Explainability (Section 3.3, 4.5)
# ---------------------------------------------------------------------------
N_EXPLAIN_SAMPLES = 500              # random test images for Del/Ins AUC
DELINS_STEPS = 50                    # steps in the deletion/insertion curve
DELINS_BLUR_KSIZE = 31              # kernel for the blurred baseline (insertion)

# ---------------------------------------------------------------------------
# Efficiency (Section 3.4)
# ---------------------------------------------------------------------------
LATENCY_RUNS = 1000
LATENCY_WARMUP = 50

# ---------------------------------------------------------------------------
# Robustness (Section 3.5)
# ---------------------------------------------------------------------------
# Each perturbation has three severity levels: mild, moderate, severe
GAUSSIAN_SIGMAS = [10, 25, 40]                 # in [0,255] intensity space
BRIGHTNESS_FACTORS = [0.5, 0.7, 1.5]
MOTION_BLUR_KSIZES = [5, 11, 17]
SEVERITY_LABELS = ["mild", "moderate", "severe"]

# ---------------------------------------------------------------------------
# Human-Centered Score weighting scenarios (Section 3.6, Table 1)
# Order of weights: (accuracy, explainability, efficiency, robustness)
# ---------------------------------------------------------------------------
HCS_SCENARIOS = {
    "HCS-A (Safety)":     {"alpha": 0.30, "eps": 0.20, "eta": 0.20, "rho": 0.30},
    "HCS-B (Deployment)": {"alpha": 0.25, "eps": 0.20, "eta": 0.35, "rho": 0.20},
    "HCS-C (Balanced)":   {"alpha": 0.25, "eps": 0.25, "eta": 0.25, "rho": 0.25},
}
