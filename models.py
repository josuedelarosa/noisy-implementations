# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm


# -------------------------
# Utilities
# -------------------------

def _replace_first_conv(module: nn.Module, in_channels: int) -> None:
    """
    Replace the first conv layer of a torchvision ResNet-like model
    so it can accept in_channels != 3 (e.g., 1 for CT patches).
    Keeps kernel/stride/padding identical.
    """
    if not hasattr(module, "conv1"):
        raise ValueError("Model has no conv1 attribute to replace.")

    old: nn.Conv2d = module.conv1
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    # If the old conv had 3 input channels and new has 1, initialize by averaging weights.
    # If new has >3, repeat/scale.
    with torch.no_grad():
        if old.weight.shape[1] == 3 and in_channels == 1:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        elif old.weight.shape[1] == 3 and in_channels != 3:
            # Repeat as needed, then trim.
            rep = int((in_channels + 2) // 3)
            w = old.weight.repeat(1, rep, 1, 1)[:, :in_channels, :, :]
            # Scale to keep variance roughly stable
            w = w * (3.0 / float(in_channels))
            new.weight.copy_(w)
        else:
            # Fallback: Kaiming init
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
            if new.bias is not None:
                nn.init.zeros_(new.bias)

    module.conv1 = new


class Identity(nn.Module):
    def forward(self, x):
        return x


# -------------------------
# Small CNN baseline (recommended to start for LIDC patches)
# -------------------------

class SmallCNN(nn.Module):
    """
    Lightweight CNN for 2D patches (e.g., 64/128/256).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        width: int = 32,
        dropout: float = 0.2,
        return_features: bool = False,
    ):
        super().__init__()
        self.return_features = return_features

        # Conv blocks: (Conv -> BN -> ReLU) x2 + MaxPool
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            block(in_channels, width),            # /2
            block(width, width * 2),              # /4
            block(width * 2, width * 4),          # /8
            block(width * 4, width * 4),          # /16
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = width * 4

        head = []
        if dropout and dropout > 0:
            head.append(nn.Dropout(p=dropout))
        head.append(nn.Linear(feat_dim, num_classes))
        self.classifier = nn.Sequential(*head)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)  # (N, C)
        logits = self.classifier(feat)
        if self.return_features:
            return logits, feat
        return logits


# -------------------------
# ResNet wrappers
# -------------------------

class ResNetClassifier(nn.Module):
    """
    Wrap torchvision ResNet (18/34/50...) for arbitrary input channels and classes.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        in_channels: int = 1,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.0,
        return_features: bool = False,
    ):
        super().__init__()
        self.return_features = return_features

        if not hasattr(tvm, arch):
            raise ValueError(f"Unknown torchvision model: {arch}")

        # Newer torchvision uses weights=..., but to keep compatibility we use this:
        # If you want pretrained, you can adapt to your torchvision version.
        base = getattr(tvm, arch)(weights="DEFAULT" if pretrained else None)

        # Replace first conv for grayscale CT patches
        if in_channels != 3:
            _replace_first_conv(base, in_channels=in_channels)

        # Replace classifier head
        feat_dim = base.fc.in_features
        base.fc = Identity()
        self.backbone = base

        head = []
        if dropout and dropout > 0:
            head.append(nn.Dropout(p=dropout))
        head.append(nn.Linear(feat_dim, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat = self.backbone(x)          # (N, D)
        logits = self.classifier(feat)   # (N, C)
        if self.return_features:
            return logits, feat
        return logits


# -------------------------
# Factory
# -------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str = "smallcnn"     # "smallcnn", "resnet18", "resnet34"
    in_channels: int = 1
    num_classes: int = 2
    pretrained: bool = False   # only used for resnet*
    dropout: float = 0.2
    width: int = 32            # only used for smallcnn
    return_features: bool = False


def build_model(cfg: ModelConfig) -> nn.Module:
    name = cfg.name.lower()

    if name in ("smallcnn", "cnn", "lidc_cnn"):
        return SmallCNN(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            width=cfg.width,
            dropout=cfg.dropout,
            return_features=cfg.return_features,
        )

    if name in ("resnet18", "resnet34", "resnet50", "resnet101"):
        return ResNetClassifier(
            arch=name,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
            return_features=cfg.return_features,
        )

    raise ValueError(
        f"Unknown model name: {cfg.name}. "
        "Supported: smallcnn, resnet18, resnet34, resnet50, resnet101"
    )


# -------------------------
# Optional: loss helpers for different label types
# -------------------------

class OrdinalRegressionHead(nn.Module):
    """
    Optional head for ordinal targets (e.g., malignancy 1..5) using K-1 logits (CORN-style / cumulative).

    For K classes, outputs K-1 logits. Training typically uses BCE on thresholds.
    """
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Ordinal regression needs at least 2 classes.")
        layers = []
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, num_classes - 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ordinal_targets_from_class(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert class labels y in {0..K-1} to K-1 binary thresholds.
    Example K=5:
      y=0 -> [0,0,0,0]
      y=1 -> [1,0,0,0]
      y=2 -> [1,1,0,0]
      y=3 -> [1,1,1,0]
      y=4 -> [1,1,1,1]
    """
    # y: (N,)
    thresholds = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)  # (1, K-1)
    return (y.unsqueeze(1) > thresholds).float()
