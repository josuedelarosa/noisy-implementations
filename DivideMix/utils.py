# utils.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism note: may slow down.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Logging helpers
# -------------------------

@dataclass
class AverageMeter:
    name: str
    fmt: str = ":.4f"
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} (avg {self.avg:{self.fmt}})"


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy for integer targets."""
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


# -------------------------
# MixMatch / DivideMix ops
# -------------------------

@torch.no_grad()
def sharpen(p: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    p_pow = p ** (1.0 / T)
    return p_pow / p_pow.sum(dim=1, keepdim=True)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with soft targets (N,C). Targets should be normalized (sum=1).
    """
    logp = F.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()


def linear_rampup(current: int, rampup_length: int) -> float:
    if rampup_length <= 0:
        return 1.0
    current = np.clip(current, 0, rampup_length)
    return float(current) / float(rampup_length)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)

    idx = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[idx], y[idx]
    x_mix = lam * x + (1.0 - lam) * x2
    y_mix = lam * y + (1.0 - lam) * y2
    return x_mix, y_mix


# -------------------------
# Checkpointing
# -------------------------

def save_checkpoint(
    path: str,
    epoch: int,
    model_a: nn.Module,
    model_b: nn.Module,
    opt_a: torch.optim.Optimizer,
    opt_b: torch.optim.Optimizer,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_a": model_a.state_dict(),
        "model_b": model_b.state_dict(),
        "opt_a": opt_a.state_dict(),
        "opt_b": opt_b.state_dict(),
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model_a: nn.Module,
    model_b: nn.Module,
    opt_a: Optional[torch.optim.Optimizer] = None,
    opt_b: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> int:
    ckpt = torch.load(path, map_location=map_location)
    model_a.load_state_dict(ckpt["model_a"])
    model_b.load_state_dict(ckpt["model_b"])
    if opt_a is not None:
        opt_a.load_state_dict(ckpt["opt_a"])
    if opt_b is not None:
        opt_b.load_state_dict(ckpt["opt_b"])
    return int(ckpt.get("epoch", 0))
