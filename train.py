"""
Training script for NAFNet-SR thermal 4× super-resolution.

LR input (32×32 patches) → model → HR output (128×128) → loss vs HR ground truth.

Usage:
    python train.py
"""

import os
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import get_config
from nafnet import NAFNetSR, count_params
from dataset import build_dataloaders
from utils import (
    get_loss_fn, calc_psnr, calc_ssim,
    AverageMeter, TrainingLogger,
    save_checkpoint, load_checkpoint, set_seed,
)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(cfg.train.min_lr / cfg.train.lr,
                       0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg, epoch):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()

    for step, batch in enumerate(loader):
        lr_input = batch['input'].to(device, non_blocking=True)    # (B, 3, 32, 32)
        hr_target = batch['target'].to(device, non_blocking=True)  # (B, 3, 128, 128)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.train.use_amp):
            hr_pred = model(lr_input)             # (B, 3, 128, 128)
            loss = criterion(hr_pred, hr_target)

        scaler.scale(loss).backward()

        if cfg.train.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_meter.update(loss.item(), lr_input.size(0))

        if (step + 1) % cfg.train.log_every == 0 or step == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [Epoch {epoch:03d}] Step {step+1:04d}/{len(loader)} | "
                  f"Loss: {loss_meter.avg:.6f} | LR: {lr:.2e}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    """Validate and compute PSNR/SSIM on HR outputs."""
    model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in loader:
        lr_input = batch['input'].to(device, non_blocking=True)
        hr_target = batch['target'].to(device, non_blocking=True)

        with autocast(enabled=cfg.train.use_amp):
            hr_pred = model(lr_input)
            loss = criterion(hr_pred, hr_target)

        hr_pred = hr_pred.float().clamp(0.0, 1.0)

        loss_meter.update(loss.item(), lr_input.size(0))
        psnr_meter.update(calc_psnr(hr_pred, hr_target), lr_input.size(0))
        ssim_meter.update(calc_ssim(hr_pred, hr_target), lr_input.size(0))

    return {
        'val_loss': loss_meter.avg,
        'val_psnr': psnr_meter.avg,
        'val_ssim': ssim_meter.avg,
    }


def main():
    cfg = get_config()
    set_seed(cfg.train.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data ---
    print("\n--- Loading data ---")
    train_loader, val_loader = build_dataloaders(cfg)

    # --- Model ---
    print("\n--- Building NAFNet-SR ---")
    model = NAFNetSR(
        img_channels=cfg.model.img_channels,
        width=cfg.model.width,
        num_blocks=cfg.model.num_blocks,
        upscale=cfg.data.upscale_factor,
    ).to(device)
    print(f"Params: {count_params(model)}")

    # Quick shape check
    with torch.no_grad():
        dummy = torch.randn(1, cfg.model.img_channels, 15, 22).to(device)
        out = model(dummy)
        print(f"Shape check: {dummy.shape} → {out.shape} "
              f"({cfg.data.upscale_factor}× upscale)")

    # --- Loss, Optimizer, Scheduler ---
    criterion = get_loss_fn(cfg).to(device)
    print(f"Loss: {cfg.train.loss_fn}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.9),
        eps=1e-8,
    )

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.train.use_amp)

    # --- Resume ---
    start_epoch = 1
    best_psnr = 0.0

    if cfg.train.resume_from and os.path.exists(cfg.train.resume_from):
        print(f"\nResuming from: {cfg.train.resume_from}")
        ckpt = load_checkpoint(cfg.train.resume_from, model, optimizer, scheduler, scaler)
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"  Epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    logger = TrainingLogger(cfg.train.log_dir)

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"THERMAL 4× SUPER-RESOLUTION TRAINING")
    print(f"Epochs: {cfg.train.epochs} | Batch: {cfg.train.batch_size} | "
          f"LR patch: {cfg.data.patch_size // cfg.data.upscale_factor}×"
          f"{cfg.data.patch_size // cfg.data.upscale_factor} → "
          f"HR patch: {cfg.data.patch_size}×{cfg.data.patch_size}")
    print(f"LR: {cfg.train.lr} → {cfg.train.min_lr} "
          f"(cosine, {cfg.train.warmup_epochs} warmup epochs)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, cfg, epoch
        )

        val_metrics = {}
        if epoch % cfg.train.val_every == 0:
            val_metrics = validate(model, val_loader, criterion, device, cfg)

        epoch_time = time.time() - epoch_start

        metrics = {
            'train_loss': train_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_s': round(epoch_time, 1),
            **val_metrics,
        }
        logger.log(epoch, metrics)

        val_str = ""
        if val_metrics:
            val_str = (f" | Val: {val_metrics['val_loss']:.6f}"
                       f" | PSNR: {val_metrics['val_psnr']:.2f}"
                       f" | SSIM: {val_metrics['val_ssim']:.4f}")

        print(f"Epoch {epoch:03d}/{cfg.train.epochs} | "
              f"Train: {train_loss:.6f}{val_str} | {epoch_time:.1f}s")

        if val_metrics and val_metrics['val_psnr'] > best_psnr:
            best_psnr = val_metrics['val_psnr']
            save_path = os.path.join(cfg.train.save_dir, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr, save_path)
            print(f"  ★ New best PSNR: {best_psnr:.2f} dB → {save_path}")

        if epoch % cfg.train.save_every == 0:
            save_path = os.path.join(cfg.train.save_dir, f"epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr, save_path)

    print(f"\n{'='*60}")
    print(f"Done! Best PSNR: {best_psnr:.2f} dB")
    print(f"Model: {cfg.train.save_dir}/best_model.pth")
    print(f"Log: {cfg.train.log_dir}/train_log.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
