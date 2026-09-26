"""
Centralized configuration for thermal image restoration.
All hyperparameters and paths in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # --- Dataset paths (set these before training) ---
    train_thermal_dir: str = "./data/train/thermal"
    train_target_dir: str = "./data/train/visible"
    val_thermal_dir: str = "./data/val/thermal"
    val_target_dir: str = "./data/val/visible"

    # --- Preprocessing ---
    patch_size: int = 128            # Random crop size during training (on HR)
    img_channels: int = 3           # 3 for RGB thermal, 1 for grayscale
    upscale_factor: int = 4          # 4× super-resolution (LR 90×60 → HR 360×240)
    augment: bool = True            # Enable random augmentations

    # --- Loader ---
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    img_channels: int = 3
    width: int = 48                  # Feature channel width
    num_blocks: int = 16             # Number of NAFBlocks (reduced for small dataset)


@dataclass
class TrainConfig:
    # --- Core ---
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0      # 0 = disabled

    # --- Scheduler ---
    scheduler: str = "cosine"        # "cosine" or "step"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # --- Loss ---
    loss_fn: str = "charbonnier"     # "l1", "mse", "charbonnier"
    charbonnier_eps: float = 1e-3

    # --- Checkpointing & Logging ---
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 10             # Save checkpoint every N epochs
    val_every: int = 1               # Validate every N epochs
    log_every: int = 50              # Log every N steps

    # --- Mixed precision ---
    use_amp: bool = True

    # --- Reproducibility ---
    seed: int = 42

    # --- Resume ---
    resume_from: Optional[str] = None


@dataclass
class InferConfig:
    checkpoint_path: str = r"C:\Users\JISHNU\.gemini\antigravity-ide\scratch\thermal_restoration\best_model.pth"
    input_dir: str = "./data/test/thermal"
    output_dir: str = "./results"
    tile_size: int = 256             # For tiled inference on large images
    tile_overlap: int = 32
    use_amp: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def get_config() -> Config:
    """Returns the default config. Modify in code or extend with argparse."""
    return Config()
