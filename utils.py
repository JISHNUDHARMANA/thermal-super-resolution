"""
Utility functions: losses, metrics, logging, and helpers.
"""

import math
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# LOSSES
# ============================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 approximation).
    Better than L1 for image restoration — differentiable everywhere,
    avoids gradient instability near zero.

    L = sqrt((x - y)^2 + eps^2)
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class PSNRLoss(nn.Module):
    """
    PSNR-based loss: -10 * log10(MSE).
    Directly optimizes for PSNR metric.
    """

    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        # Add small eps to avoid log(0)
        psnr = -10.0 * torch.log10(mse + 1e-8)
        return -psnr  # Negate because we minimize


def get_loss_fn(cfg) -> nn.Module:
    """Build loss function from config."""
    name = cfg.train.loss_fn.lower()
    if name == 'l1':
        return nn.L1Loss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'charbonnier':
        return CharbonnierLoss(eps=cfg.train.charbonnier_eps)
    elif name == 'psnr':
        return PSNRLoss()
    else:
        raise ValueError(f"Unknown loss: {name}. Choose from: l1, mse, charbonnier, psnr")


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def calc_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR between two image tensors."""
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(max_val ** 2 / mse)


@torch.no_grad()
def calc_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Calculate SSIM between two image tensors.
    Simplified but correct implementation.
    """
    pred = pred.float()
    target = target.float()
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)  # 2D Gaussian
    window = window.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    channels = pred.shape[1]
    window = window.expand(channels, -1, -1, -1)

    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ============================================================
# TRAINING HELPERS
# ============================================================

class AverageMeter:
    """Tracks running average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingLogger:
    """Simple JSON-lines logger for training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train_log.jsonl"
        self.history = []

    def log(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_history(self) -> list:
        return self.history


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_psnr: float,
    save_path: str,
):
    """Save a training checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_psnr': best_psnr,
    }, save_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict:
    """Load a training checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    return ckpt


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
