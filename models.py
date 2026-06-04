"""
models.py
=========
Builds the four benchmarked architectures with ImageNet-pretrained weights
and a replaced 2-class head:
  - MobileNetV3-Large   (torchvision)
  - ShuffleNetV2 x1.0   (torchvision)
  - EfficientNet-B0     (torchvision)
  - DeiT-Tiny           (timm)

Also exposes the layer used as Grad-CAM target for CNN models.

Author: Ruben Dario Florez-Zela
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

import config as C


def build_model(name, num_classes=C.NUM_CLASSES, pretrained=True):
    """Return an nn.Module with a 2-neuron classification head."""
    name = name.lower()

    if name == "mobilenetv3":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, num_classes)

    elif name == "shufflenetv2":
        weights = tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.shufflenet_v2_x1_0(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)

    elif name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)

    elif name == "deit_tiny":
        import timm
        model = timm.create_model(
            "deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


def get_backbone_parameters(model, name):
    """
    Return (backbone_params, head_params) for two-stage fine-tuning.
    During warmup only head_params are trained.
    """
    name = name.lower()
    head_params, backbone_params = [], []

    if name == "mobilenetv3":
        head_ids = set(id(p) for p in model.classifier[3].parameters())
    elif name == "shufflenetv2":
        head_ids = set(id(p) for p in model.fc.parameters())
    elif name == "efficientnet_b0":
        head_ids = set(id(p) for p in model.classifier[1].parameters())
    elif name == "deit_tiny":
        head_ids = set(id(p) for p in model.get_classifier().parameters())
    else:
        raise ValueError(f"Unknown model: {name}")

    for p in model.parameters():
        if id(p) in head_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return backbone_params, head_params


def set_backbone_trainable(model, name, trainable):
    backbone_params, _ = get_backbone_parameters(model, name)
    for p in backbone_params:
        p.requires_grad = trainable


def gradcam_target_layer(model, name):
    """
    Return the convolutional layer used as Grad-CAM target for CNNs.
    Returns None for transformers (Grad-CAM not directly applicable).
    """
    name = name.lower()
    if name == "mobilenetv3":
        return model.features[-1]
    if name == "shufflenetv2":
        return model.conv5
    if name == "efficientnet_b0":
        return model.features[-1]
    return None  # deit_tiny: use attention-based or model-agnostic attribution
