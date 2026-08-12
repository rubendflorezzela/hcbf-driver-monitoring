from __future__ import annotations

MODEL_ORDER = (
    "mobilenetv3",
    "shufflenetv2",
    "efficientnet_b0",
    "deit_tiny",
    "repvit_m1_0",
    "efficientformer_l1",
)

MODEL_LABELS = {
    "mobilenetv3": "MobileNetV3-Large",
    "shufflenetv2": "ShuffleNetV2 x1.0",
    "efficientnet_b0": "EfficientNet-B0",
    "deit_tiny": "DeiT-Tiny",
    "repvit_m1_0": "RepViT-M1.0",
    "efficientformer_l1": "EfficientFormer-L1",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_THRESHOLD = 0.5
MRL_SUBJECTS = tuple(f"s{i:04d}" for i in range(1, 38))
