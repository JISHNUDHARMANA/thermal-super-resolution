"""
Interactive Dual-Mode Web Demo for Thermal Super-Resolution & Enhancement.
Supports:
1. 4× Super-Resolution (Raw low-res e.g. 90×60 -> 360×240)
2. Same-Size Enhancement (Pre-stretched blurry image -> Sharp image at same size)
3. Auto-Detect based on input dimensions.
"""

import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr


# ============================================================
# NAFNet-SR ARCHITECTURE
# ============================================================
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return y * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=[0, 2, 3]), grad_output.sum(dim=[0, 2, 3]), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_c = c * DW_Expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, 1, 1, groups=dw_c)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedCA(dw_c // 2)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1, 1, 0)

        ffn_c = c * FFN_Expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1, 1, 0)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1, 1, 0)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.gate2(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, channels, upscale=4):
        super().__init__()
        layers = []
        for _ in range(2):  # 2x * 2x = 4x
            layers.extend([
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NAFNetSR(nn.Module):
    def __init__(self, img_channels=3, width=48, num_blocks=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(img_channels, width, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(width, width, 3, 1, 1)
        self.upsampler = PixelShuffleUpsampler(width, upscale)
        self.ending = nn.Conv2d(width, img_channels, 3, 1, 1)

    def forward(self, x):
        shallow = self.intro(x)
        deep = self.body(shallow)
        deep = self.body_tail(deep) + shallow
        up = self.upsampler(deep)
        return self.ending(up)


# ============================================================
# LOAD MODEL
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NAFNetSR(img_channels=3, width=48, num_blocks=16, upscale=4).to(DEVICE)

def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ['model', 'model_state_dict', 'state_dict', 'net', 'params']:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt

def find_checkpoint():
    explicit_path = Path(r"C:\Users\JISHNU\.gemini\antigravity-ide\scratch\thermal_restoration\best_model.pth")
    if explicit_path.exists():
        return str(explicit_path)
    env_path = os.environ.get('MODEL_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "best_model.pth",
        base_dir / "checkpoints" / "best_model.pth",
        Path.cwd() / "best_model.pth",
        Path.cwd() / "checkpoints" / "best_model.pth",
        Path("/content/drive/MyDrive/thermal_sr/checkpoints/best_model.pth"),
        Path("/content/best_model.pth"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p)
    return None

CKPT_PATH = find_checkpoint()
MODEL_LOADED = False
if CKPT_PATH:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state)
    MODEL_LOADED = True
    print(f"Loaded checkpoint from {CKPT_PATH}")
else:
    print("Notice: No checkpoint file found in auto-discovery search paths.")

model.eval()


# ============================================================
# CORE PROCESSING ENGINE
# ============================================================
def run_model_forward(img_pil):
    """Feed PIL image through NAFNet 4x upsampler."""
    img_np = np.array(img_pil.convert('RGB'), dtype=np.float32)

    # Normalize
    if img_np.max() > 255.0:
        img_np /= 65535.0
    elif img_np.max() > 1.0:
        img_np /= 255.0

    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    # Dynamic padding to multiple of 4
    _, _, h, w = tensor.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = model(tensor)

    # Remove padding & clamp
    out = out[:, :, :h * 4, :w * 4]
    out = out.clamp(0.0, 1.0).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_uint8 = (out * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_uint8)


def process_image(input_image, mode):
    if input_image is None:
        return None, "⚠️ Please upload an image."

    t0 = time.time()
    img = Image.fromarray(input_image).convert('RGB') if isinstance(input_image, np.ndarray) else input_image.convert('RGB')
    orig_w, orig_h = img.size

    # Auto-detection logic
    selected_mode = mode
    if mode == "Auto-Detect":
        if orig_w <= 160 and orig_h <= 120:
            selected_mode = "4× Super-Resolution (Small / Raw Sensor)"
        else:
            selected_mode = "Restore at Same Size (Big but Blurry)"

    # Execution based on mode
    if "4× Super-Resolution" in selected_mode:
        # Standard 4x direct super-resolution
        result_img = run_model_forward(img)
        final_w, final_h = result_img.size
        action_note = f"Direct 4× super-resolution: `{orig_w}×{orig_h}` ➔ `{final_w}×{final_h}`"
    else:
        # Same-Size Restoration: downsample to native sensor grid, then 4x restore
        target_sensor_w = max(4, orig_w // 4)
        target_sensor_h = max(4, orig_h // 4)
        sensor_lr = img.resize((target_sensor_w, target_sensor_h), Image.Resampling.BICUBIC)
        
        # Restore through NAFNet
        restored_4x = run_model_forward(sensor_lr)
        # Match original canvas size exactly
        result_img = restored_4x.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        final_w, final_h = result_img.size
        action_note = f"Restored high-frequency thermal edges at original canvas size `{orig_w}×{orig_h}`"

    elapsed_ms = (time.time() - t0) * 1000

    ckpt_status = "✅ `best_model.pth` loaded" if MODEL_LOADED else "⚠️ `best_model.pth` not found (running untrained weights)"
    info_text = (
        f"### ⚡ Processed in {elapsed_ms:.1f} ms\n"
        f"- **Checkpoint:** {ckpt_status}\n"
        f"- **Applied Mode:** `{selected_mode}`\n"
        f"- **Input Dimensions:** `{orig_w} × {orig_h}` px\n"
        f"- **Output Dimensions:** `{final_w} × {final_h}` px\n"
        f"- **Device:** `{DEVICE.type.upper()}`\n"
        f"- **Summary:** {action_note}"
    )

    return result_img, info_text


# ============================================================
# GRADIO INTERFACE
# ============================================================
with gr.Blocks(title="Thermal Image Super-Resolution & Restoration", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔥 Thermal Vision Super-Resolution & Restoration
        ### Powered by Lightweight NAFNet (0.47M Parameters)
        Choose between **4× Super-Resolution** for raw small sensor images or **Same-Size Restoration** for stretched blurry thermal photos.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_ui = gr.Image(type="numpy", label="Upload Thermal Image")
            
            mode_selector = gr.Radio(
                choices=[
                    "Auto-Detect",
                    "4× Super-Resolution (Small / Raw Sensor)",
                    "Restore at Same Size (Big but Blurry)",
                ],
                value="Auto-Detect",
                label="Select Enhancement Method",
                info="Auto-Detect automatically chooses 4× for small images (≤160×120) and Same-Size Restoration for larger blurry images.",
            )

            btn_run = gr.Button("⚡ Enhance Thermal Image", variant="primary", size="lg")
            stats_box = gr.Markdown("Ready to process...")

        with gr.Column(scale=1):
            output_ui = gr.Image(type="pil", label="Enhanced Thermal Output")

    btn_run.click(
        fn=process_image,
        inputs=[input_ui, mode_selector],
        outputs=[output_ui, stats_box],
    )

if __name__ == '__main__':
    # Launch for Hugging Face Spaces (or local browser)
    demo.launch()
