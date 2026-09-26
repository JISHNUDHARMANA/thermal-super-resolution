"""
Dataset for thermal image 4× super-resolution.

Pipeline:
  1. Load LR (90×60) and HR (360×240) as separate images
  2. Random crop at CORRESPONDING positions:
     - LR crop: 32×32 (patch_size / upscale_factor)
     - HR crop: 128×128 (patch_size)
     - Same spatial region, different resolutions
  3. Augment both identically (flip, rotate)
  4. Model input: LR patch (32×32)
     Model target: HR patch (128×128)
     Loss: between model output (128×128) and HR target (128×128)
"""

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def get_image_paths(directory: str) -> list:
    """Get sorted list of image paths from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = sorted([
        p for p in directory.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {directory}")

    return paths


def load_image(path: Path, channels: int = 3) -> np.ndarray:
    """Load image as float32 numpy array in [0, 1]."""
    img = Image.open(path)

    if channels == 1:
        img = img.convert('L')
    elif channels == 3:
        img = img.convert('RGB')

    img = np.array(img, dtype=np.float32)

    # Normalize based on bit depth
    if img.max() > 255.0:
        img = img / 65535.0
    elif img.max() > 1.0:
        img = img / 255.0

    return img


class PairedSRDataset(Dataset):
    """
    Paired LR → HR super-resolution dataset.

    Folder structure:
        lr_dir/  (90×60 low-resolution thermal images)
            img_001.png
            ...
        hr_dir/  (360×240 high-resolution thermal images)
            img_001.png
            ...
    """

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        patch_size: int = 128,
        upscale_factor: int = 4,
        channels: int = 3,
        augment: bool = True,
        is_train: bool = True,
    ):
        super().__init__()
        lr_all = get_image_paths(lr_dir)
        hr_all = get_image_paths(hr_dir)

        def clean_stem(s):
            for suffix in ['_lr', '_hr', '_LR', '_HR', '-lr', '-hr', '-LR', '-HR']:
                if s.endswith(suffix):
                    return s[:-len(suffix)]
            return s

        lr_dict = {clean_stem(p.stem): p for p in lr_all}
        hr_dict = {clean_stem(p.stem): p for p in hr_all}
        common = sorted(list(set(lr_dict.keys()) & set(hr_dict.keys())))

        if len(common) > 0:
            self.lr_paths = [lr_dict[k] for k in common]
            self.hr_paths = [hr_dict[k] for k in common]
        else:
            n = min(len(lr_all), len(hr_all))
            self.lr_paths = lr_all[:n]
            self.hr_paths = hr_all[:n]

        self.patch_size = patch_size          # HR patch size (e.g., 128)
        self.lr_patch_size = patch_size // upscale_factor  # LR patch size (e.g., 32)
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.augment = augment and is_train
        self.is_train = is_train

        print(f"{'Train' if is_train else 'Val'} dataset: {len(self)} pairs | "
              f"LR patch: {self.lr_patch_size}×{self.lr_patch_size} → "
              f"HR patch: {self.patch_size}×{self.patch_size} | "
              f"aug={self.augment}")

    def __len__(self) -> int:
        return len(self.lr_paths)

    def _random_crop_pair(
        self, lr: np.ndarray, hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random crop at CORRESPONDING spatial positions.
        LR crop is (lr_patch × lr_patch), HR crop is (patch × patch).
        """
        lr_h, lr_w = lr.shape[:2]
        lr_ps = self.lr_patch_size

        # Ensure LR image is big enough
        if lr_h < lr_ps or lr_w < lr_ps:
            pad_h = max(0, lr_ps - lr_h)
            pad_w = max(0, lr_ps - lr_w)
            if lr.ndim == 3:
                lr = np.pad(lr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                hr = np.pad(hr, ((0, pad_h * self.upscale_factor),
                                 (0, pad_w * self.upscale_factor), (0, 0)), mode='reflect')
            else:
                lr = np.pad(lr, ((0, pad_h), (0, pad_w)), mode='reflect')
                hr = np.pad(hr, ((0, pad_h * self.upscale_factor),
                                 (0, pad_w * self.upscale_factor)), mode='reflect')
            lr_h, lr_w = lr.shape[:2]

        # Random position on LR grid
        top_lr = random.randint(0, lr_h - lr_ps)
        left_lr = random.randint(0, lr_w - lr_ps)

        # Corresponding position on HR grid
        top_hr = top_lr * self.upscale_factor
        left_hr = left_lr * self.upscale_factor
        hr_ps = self.patch_size

        lr_crop = lr[top_lr:top_lr+lr_ps, left_lr:left_lr+lr_ps]
        hr_crop = hr[top_hr:top_hr+hr_ps, left_hr:left_hr+hr_ps]

        return lr_crop, hr_crop

    def _augment_pair(
        self, lr: np.ndarray, hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical augmentations to both LR and HR."""
        # Horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()

        # Vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()

        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()

        return lr, hr

    def __getitem__(self, idx: int) -> dict:
        # 1. Load LR and HR as float32 [0, 1]
        lr_img = load_image(self.lr_paths[idx], self.channels)
        hr_img = load_image(self.hr_paths[idx], self.channels)

        # Ensure 3D
        if lr_img.ndim == 2:
            lr_img = lr_img[:, :, np.newaxis]
        if hr_img.ndim == 2:
            hr_img = hr_img[:, :, np.newaxis]

        # 2. Random crop at corresponding positions (train only)
        if self.is_train:
            lr_img, hr_img = self._random_crop_pair(lr_img, hr_img)

        # 3. Augment identically (train only)
        if self.augment:
            lr_img, hr_img = self._augment_pair(lr_img, hr_img)

        # 4. HWC → CHW tensor
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).float().clamp(0, 1)
        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1)).float().clamp(0, 1)

        return {
            'input': lr_tensor,     # LR patch (e.g., 3×32×32)
            'target': hr_tensor,    # HR patch (e.g., 3×128×128)
            'filename': self.lr_paths[idx].name,
        }


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders."""
    train_dataset = PairedSRDataset(
        lr_dir=cfg.data.train_thermal_dir,
        hr_dir=cfg.data.train_target_dir,
        patch_size=cfg.data.patch_size,
        upscale_factor=cfg.data.upscale_factor,
        channels=cfg.data.img_channels,
        augment=cfg.data.augment,
        is_train=True,
    )

    val_dataset = PairedSRDataset(
        lr_dir=cfg.data.val_thermal_dir,
        hr_dir=cfg.data.val_target_dir,
        patch_size=cfg.data.patch_size,
        upscale_factor=cfg.data.upscale_factor,
        channels=cfg.data.img_channels,
        augment=False,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    return train_loader, val_loader
