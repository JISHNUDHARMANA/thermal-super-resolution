"""
Plotting utilities for training analysis.
Run this on Colab after training to visualize curves.

Usage:
    python plot_training.py
    # Or in a Colab cell:
    # %run plot_training.py
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_log(log_path: str = "./logs/train_log.jsonl") -> list:
    """Load training log from JSONL file."""
    entries = []
    with open(log_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def plot_training_curves(log_path: str = "./logs/train_log.jsonl", save_dir: str = "./logs"):
    """Generate and save training plots."""
    entries = load_log(log_path)
    save_dir = Path(save_dir)

    epochs = [e['epoch'] for e in entries]
    train_loss = [e['train_loss'] for e in entries]
    lr = [e['lr'] for e in entries]

    # Filter entries that have validation metrics
    val_entries = [e for e in entries if 'val_psnr' in e]
    val_epochs = [e['epoch'] for e in val_entries]
    val_loss = [e['val_loss'] for e in val_entries]
    val_psnr = [e['val_psnr'] for e in val_entries]
    val_ssim = [e['val_ssim'] for e in val_entries]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NAFNet Thermal Restoration — Training Progress', fontsize=14, fontweight='bold')

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label='Train Loss', color='#2196F3', linewidth=1.5)
    if val_loss:
        ax.plot(val_epochs, val_loss, label='Val Loss', color='#F44336', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # --- PSNR ---
    ax = axes[0, 1]
    if val_psnr:
        ax.plot(val_epochs, val_psnr, color='#4CAF50', linewidth=1.5, marker='o', markersize=2)
        best_idx = val_psnr.index(max(val_psnr))
        ax.axhline(y=max(val_psnr), color='#4CAF50', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {max(val_psnr):.2f} dB (ep {val_epochs[best_idx]})',
                    xy=(val_epochs[best_idx], max(val_psnr)),
                    fontsize=9, color='#2E7D32', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR')
    ax.grid(True, alpha=0.3)

    # --- SSIM ---
    ax = axes[1, 0]
    if val_ssim:
        ax.plot(val_epochs, val_ssim, color='#FF9800', linewidth=1.5, marker='o', markersize=2)
        ax.axhline(y=max(val_ssim), color='#FF9800', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM')
    ax.grid(True, alpha=0.3)

    # --- Learning Rate ---
    ax = axes[1, 1]
    ax.plot(epochs, lr, color='#9C27B0', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_curves()
