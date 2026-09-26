"""
Inference for NAFNet-SR thermal 4× super-resolution.

Takes LR thermal images, outputs HR images at 4× resolution.

Usage:
    python infer.py
    python infer.py --checkpoint ./checkpoints/best_model.pth --input ./test_lr --output ./results
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image

from config import get_config
from nafnet import NAFNetSR
from dataset import load_image, IMG_EXTENSIONS
from utils import calc_psnr, calc_ssim


def pad_to_multiple(img: torch.Tensor, multiple: int = 4) -> tuple:
    """Pad image so H, W are multiples of `multiple`."""
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, (h, w)


@torch.no_grad()
def infer_single(model, img_path: str, device, cfg) -> np.ndarray:
    """Run SR inference on a single LR image."""
    # Load LR image
    img = load_image(Path(img_path), channels=cfg.model.img_channels)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    # HWC → BCHW
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    # Pad to multiple of 4 (for PixelShuffle)
    img_t, orig_size = pad_to_multiple(img_t.to(device), 4)

    with autocast(enabled=cfg.infer.use_amp):
        output = model(img_t)

    # Crop to expected HR size (remove padding)
    hr_h = orig_size[0] * cfg.data.upscale_factor
    hr_w = orig_size[1] * cfg.data.upscale_factor
    output = output[:, :, :hr_h, :hr_w].cpu()

    # Tensor → uint8 numpy
    output = output.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    output = (output * 255.0).round().astype(np.uint8)

    if output.shape[2] == 1:
        output = output.squeeze(2)

    return output


def main():
    parser = argparse.ArgumentParser(description="NAFNet-SR Inference")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="LR image or directory")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--target", type=str, default=None,
                        help="Optional HR target directory for metrics")
    args = parser.parse_args()

    cfg = get_config()
    if args.checkpoint:
        cfg.infer.checkpoint_path = args.checkpoint
    if args.input:
        cfg.infer.input_dir = args.input
    if args.output:
        cfg.infer.output_dir = args.output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Model ---
    model = NAFNetSR(
        img_channels=cfg.model.img_channels,
        width=cfg.model.width,
        num_blocks=cfg.model.num_blocks,
        upscale=cfg.data.upscale_factor,
    ).to(device)

    ckpt = torch.load(cfg.infer.checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {cfg.infer.checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # --- Input images ---
    input_path = Path(cfg.infer.input_dir)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

    print(f"\nProcessing {len(image_paths)} LR images → {cfg.data.upscale_factor}× SR\n")

    output_dir = Path(cfg.infer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = Path(args.target) if args.target else None
    psnr_vals, ssim_vals = [], []

    total_time = 0.0
    for i, img_path in enumerate(image_paths):
        start = time.time()
        result = infer_single(model, str(img_path), device, cfg)
        elapsed = time.time() - start
        total_time += elapsed

        save_path = output_dir / img_path.name
        Image.fromarray(result).save(save_path)

        status = f"[{i+1}/{len(image_paths)}] {img_path.name} → {save_path.name} ({elapsed:.2f}s)"

        if target_dir:
            target_path = target_dir / img_path.name
            if target_path.exists():
                target_img = load_image(target_path, cfg.model.img_channels)
                if target_img.ndim == 2:
                    target_img = target_img[:, :, np.newaxis]
                result_f = result.astype(np.float32) / 255.0
                if result_f.ndim == 2:
                    result_f = result_f[:, :, np.newaxis]

                r_t = torch.from_numpy(result_f.transpose(2, 0, 1)).unsqueeze(0)
                t_t = torch.from_numpy(target_img.transpose(2, 0, 1)).unsqueeze(0)
                psnr = calc_psnr(r_t, t_t)
                ssim = calc_ssim(r_t, t_t)
                psnr_vals.append(psnr)
                ssim_vals.append(ssim)
                status += f" | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"

        print(status)

    print(f"\nDone! {len(image_paths)} images, {total_time:.1f}s total")
    print(f"Results: {output_dir}")

    if psnr_vals:
        print(f"\nAvg PSNR: {np.mean(psnr_vals):.2f} dB | Avg SSIM: {np.mean(ssim_vals):.4f}")


if __name__ == "__main__":
    main()
