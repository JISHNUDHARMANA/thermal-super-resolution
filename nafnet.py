"""
NAFNet-SR — NAFNet adapted for 4× Super-Resolution.

Key difference from restoration NAFNet:
  - NO encoder-decoder U-Net (don't want to lose spatial info by downsampling)
  - Processes entirely at LR resolution (fast!)
  - PixelShuffle 4× upsampling at the end (learned, not bicubic)
  - No global residual (input/output are different sizes)

Architecture:
  LR Input (B,3,H,W)
    → Shallow feature extraction (3×3 conv)
    → N × NAFBlocks at LR resolution (deep feature extraction)
    → PixelShuffle 2× → PixelShuffle 2× (total 4× upsampling)
    → Output conv (3×3)
  HR Output (B,3,4H,4W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


class SimpleGate(nn.Module):
    """Split channels in half, multiply element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global avg pool → 1×1 conv → sigmoid gating."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_channels, num_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """Core NAFNet block with SimpleGate, SCA, and learnable skip scaling."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_c = c * dw_expand
        ffn_c = c * ffn_expand

        # Spatial mixing
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, 1, 1, groups=dw_c)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_c // 2)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1, 1, 0)

        # Channel mixing (FFN)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1, 1, 0)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1, 1, 0)

        # Learnable skip scaling (initialized to zero for stable training)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial path
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        # Channel path (FFN)
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.gate2(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x


class PixelShuffleUpsampler(nn.Module):
    """
    Learned 4× upsampling using two stages of PixelShuffle (2× each).
    Much better than bicubic — the model learns the upsampling filters.
    """

    def __init__(self, channels: int, upscale: int = 4):
        super().__init__()
        layers = []
        # 4× = 2× + 2×
        num_stages = {2: 1, 4: 2, 8: 3}[upscale]
        for _ in range(num_stages):
            layers.extend([
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.GELU(),  # Light activation between upsample stages
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NAFNetSR(nn.Module):
    """
    NAFNet for Super-Resolution.

    Unlike restoration NAFNet (U-Net), this is a straight-through backbone:
    all processing happens at LR resolution for efficiency, then PixelShuffle
    upsamples to HR at the end.

    Args:
        img_channels: Input/output image channels (3 for RGB)
        width: Feature channel width
        num_blocks: Number of NAFBlocks in the body
        upscale: Upscaling factor (4)
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 48,
        num_blocks: int = 16,
        upscale: int = 4,
    ):
        super().__init__()
        self.upscale = upscale

        # --- Shallow feature extraction ---
        self.intro = nn.Conv2d(img_channels, width, 3, 1, 1)

        # --- Deep feature extraction (all at LR resolution) ---
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(width, width, 3, 1, 1)  # Before residual add

        # --- Upsampling (learned, not bicubic) ---
        self.upsampler = PixelShuffleUpsampler(width, upscale)

        # --- Output reconstruction ---
        self.ending = nn.Conv2d(width, img_channels, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        """Careful initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) low-resolution input
        Returns:
            (B, C, H*upscale, W*upscale) high-resolution output
        """
        # Shallow features
        shallow = self.intro(x)

        # Deep features with body-level residual
        deep = self.body(shallow)
        deep = self.body_tail(deep) + shallow  # Residual at LR level

        # Upsample to HR
        up = self.upsampler(deep)

        # Final reconstruction
        out = self.ending(up)

        return out


def count_params(model: nn.Module) -> str:
    """Return human-readable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1e6:
        return f"{total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable"
    else:
        return f"{total/1e3:.1f}K total, {trainable/1e3:.1f}K trainable"


# --- Quick sanity check ---
if __name__ == "__main__":
    model = NAFNetSR(img_channels=3, width=48, num_blocks=16, upscale=4)
    print(f"NAFNet-SR params: {count_params(model)}")

    # Test: LR 90×60 → HR 360×240
    x = torch.randn(1, 3, 60, 90)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")       # (1, 3, 60, 90)
    print(f"Output: {y.shape}")       # (1, 3, 240, 360)
    assert y.shape == (1, 3, 240, 360), f"Expected (1,3,240,360), got {y.shape}"
    print("✓ Forward pass OK — 4× upscaling confirmed")
