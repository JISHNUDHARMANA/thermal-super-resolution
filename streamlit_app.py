"""
🔥 Thermal Vision Super-Resolution & Edge Restoration
Streamlit Web Application for NAFNet-SR (0.47M parameters)
Deployable on Streamlit Community Cloud (100% Free)
"""

import os
import time
import io
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Thermal Vision Super-Resolution",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def extract_state_dict(ckpt):
    """Robustly extract model state_dict from various checkpoint formats."""
    if not isinstance(ckpt, dict):
        return ckpt
    
    # Check all common keys where model weights may be stored
    for key in ['model', 'model_state_dict', 'state_dict', 'net', 'params']:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        state = ckpt
        
    # Strip 'module.' prefix if trained with DataParallel / DistributedDataParallel
    clean_state = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith('module.'):
            clean_state[k[7:]] = v
        else:
            clean_state[k] = v
    return clean_state


# ============================================================
# CACHED MODEL LOADER (INTERNAL & INVISIBLE TO USERS)
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NAFNetSR(img_channels=3, width=48, num_blocks=16, upscale=4).to(device)

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "best_model.pth",
        Path.cwd() / "best_model.pth",
        Path(r"C:\Users\JISHNU\.gemini\antigravity-ide\scratch\thermal_restoration\best_model.pth"),
        Path("/content/drive/MyDrive/thermal_sr/checkpoints/best_model.pth"),
    ]
    model_path = next((p for p in candidates if p.exists() and p.is_file()), None)

    loaded = False
    if model_path:
        try:
            ckpt = torch.load(model_path, map_location=device)
            state = extract_state_dict(ckpt)
            model.load_state_dict(state)
            loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")

    model.eval()
    return model, device, loaded


# ============================================================
# INFERENCE PIPELINE
# ============================================================
def run_model_forward(model, device, img_pil):
    img_np = np.array(img_pil.convert('RGB'), dtype=np.float32)

    # Normalize dynamic range
    if img_np.max() > 255.0:
        img_np /= 65535.0
    elif img_np.max() > 1.0:
        img_np /= 255.0

    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Reflection pad to multiple of 4
    _, _, h, w = tensor.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = model(tensor)

    # Crop reflection padding & clamp to valid unit range
    out = out[:, :, :h * 4, :w * 4]
    out = out.clamp(0.0, 1.0).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_uint8 = (out * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_uint8)


def restore_thermal_image(model, device, input_image, mode):
    t0 = time.time()
    orig_w, orig_h = input_image.size

    # Auto-detection rule
    selected_mode = mode
    if mode == "Auto-Detect":
        if orig_w <= 160 and orig_h <= 120:
            selected_mode = "4× Super-Resolution (Small / Sensor Native)"
        else:
            selected_mode = "Restore at Same Size (Pre-stretched Blur)"

    if "4× Super-Resolution" in selected_mode:
        enhanced = run_model_forward(model, device, input_image)
        final_w, final_h = enhanced.size
        summary = f"Direct 4× Upscale: `{orig_w}×{orig_h}` ➔ `{final_w}×{final_h}`"
    else:
        # Pre-stretched thermal image restoration
        target_sensor_w = max(4, orig_w // 4)
        target_sensor_h = max(4, orig_h // 4)
        sensor_lr = input_image.resize((target_sensor_w, target_sensor_h), Image.Resampling.BICUBIC)
        restored_4x = run_model_forward(model, device, sensor_lr)
        enhanced = restored_4x.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        final_w, final_h = enhanced.size
        summary = f"Sub-pixel Edge Sharpening at original canvas: `{orig_w}×{orig_h}`"

    elapsed_ms = (time.time() - t0) * 1000
    return enhanced, elapsed_ms, selected_mode, summary


# ============================================================
# SYNTHETIC TEST SAMPLE GENERATOR
# ============================================================
def create_sample_thermal():
    """Generate a realistic test thermal pattern if user has no image."""
    h, w = 60, 90
    y, x = np.ogrid[:h, :w]
    # Heat spot 1 (person/engine)
    spot1 = np.exp(-((x - 30)**2 + (y - 30)**2) / 120.0)
    # Heat spot 2
    spot2 = 0.8 * np.exp(-((x - 65)**2 + (y - 25)**2) / 80.0)
    # Background gradient & subtle noise
    bg = 0.2 + 0.1 * np.sin(x / 10.0) + np.random.normal(0, 0.03, (h, w))
    raw = np.clip(spot1 + spot2 + bg, 0, 1)

    # Colorize with iron-like thermal gradient
    r = np.clip(raw * 2.5, 0, 1)
    g = np.clip(raw * 1.8 - 0.2, 0, 1)
    b = np.clip(raw * 0.8 - 0.4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


# ============================================================
# USER INTERFACE
# ============================================================
def main():
    st.markdown('<div class="main-header">🔥 NAFNet Thermal Super-Resolution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ultra-lightweight Deep Learning (0.47M parameters) for Infrared Thermal Vision Restoration</div>',
        unsafe_allow_html=True,
    )

    # Load model silently in background
    model, device, loaded = load_model()

    # Sidebar: Clean Status & About (No weights or paths exposed)
    st.sidebar.header("⚡ System Status")
    if loaded:
        st.sidebar.success("● Model Engine: Online & Ready")
    else:
        st.sidebar.error("● Model Engine: Offline")

    st.sidebar.markdown(f"**Acceleration:** `{device.type.upper()}`")
    st.sidebar.markdown("**Network:** `NAFNet-SR (0.47M params)`")
    st.sidebar.markdown("**Scaling Factor:** `4× Spatial`")

    st.sidebar.divider()
    st.sidebar.markdown("### 🖼️ Display Size")
    display_width = st.sidebar.slider(
        "Image Display Width (px):",
        min_value=260,
        max_value=720,
        value=480,
        step=20,
        help="Adjust the display size of the thermal output image on screen.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 📖 How to Use")
    st.sidebar.markdown(
        "1. Upload any low-resolution thermal image (or click **🧪 Use Sample**).\n"
        "2. Choose your enhancement mode.\n"
        "3. Inspect the restored image and download the high-res result."
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "NAFNet (Nonlinear Activation Free Network) avoids heavy non-linearities, "
        "enabling state-of-the-art thermal edge restoration with negligible inference latency."
    )

    # Main Area: Mode & Input
    col_input_ctrl, col_sample = st.columns([3, 1])
    with col_input_ctrl:
        mode = st.radio(
            "Select Enhancement Mode:",
            [
                "4× Super-Resolution (Small / Sensor Native)",
                "Restore at Same Size (Pre-stretched Blur)",
                "Auto-Detect",
            ],
            index=0,
            horizontal=True,
            help="4× Super-Resolution directly upscales raw low-resolution thermal captures 4× spatially.",
        )

    with col_sample:
        use_sample = st.button("🧪 Use Sample Thermal Image", use_container_width=True)

    uploaded_file = st.file_uploader(
        "Upload a low-resolution or blurry thermal image (PNG, JPG, TIFF, BMP):",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )

    # Determine input image
    input_img = None
    input_id = None
    if uploaded_file is not None:
        input_img = Image.open(uploaded_file).convert('RGB')
        input_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
    elif use_sample or 'sample_img' in st.session_state:
        if use_sample:
            st.session_state['sample_img'] = create_sample_thermal()
            st.session_state['sample_id'] = time.time()
        input_img = st.session_state.get('sample_img')
        input_id = f"sample_{st.session_state.get('sample_id', 0)}"

    if input_img is None:
        st.info("👆 Upload a thermal image above, or click **🧪 Use Sample Thermal Image** to test the model immediately.")
        return

    # Process image (with session caching to allow instant resize without re-running model)
    cache_key = f"{input_id}_{mode}"
    if st.session_state.get('last_cache_key') != cache_key:
        with st.spinner("⚡ Running NAFNet thermal enhancement..."):
            result_img, latency_ms, applied_mode, summary = restore_thermal_image(model, device, input_img, mode)
            st.session_state['last_result'] = (result_img, latency_ms, applied_mode, summary)
            st.session_state['last_cache_key'] = cache_key
    else:
        result_img, latency_ms, applied_mode, summary = st.session_state['last_result']

    orig_w, orig_h = input_img.size
    new_w, new_h = result_img.size

    # Performance Metrics Banner
    st.markdown("### 📊 Inference Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input Size", f"{orig_w} × {orig_h} px")
    m2.metric("Output Size", f"{new_w} × {new_h} px", delta=f"{new_w//orig_w}×" if orig_w else "4×")
    m3.metric("Latency", f"{latency_ms:.1f} ms", delta="Fast" if latency_ms < 100 else None)
    m4.metric("Applied Mode", applied_mode.split('(')[0].strip())

    st.caption(f"**Action Note:** {summary}")

    # Dynamic styling for the image and download button size
    st.markdown(
        f"""
        <style>
        div[data-testid="stImage"] {{
            max-width: {display_width}px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: flex !important;
            justify-content: center !important;
        }}
        div[data-testid="stImage"] img {{
            max-width: 100% !important;
            height: auto !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        }}
        div[data-testid="stDownloadButton"] {{
            max-width: {display_width}px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        div[data-testid="stDownloadButton"] button {{
            width: 100% !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced Thermal Output (Centered, Crisp Display)
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            f"<h3 style='text-align: center; margin-top: 1rem;'>✨ Enhanced Thermal Output</h3>"
            f"<p style='text-align: center; color: #888; font-size: 0.95rem; margin-top: -0.5rem;'>Resolution: <b>{new_w} × {new_h} px</b></p>",
            unsafe_allow_html=True,
        )
        st.image(result_img, use_container_width=True)

        # Download Button
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        btn_download = st.download_button(
            label="📥 Download Enhanced Thermal Image (PNG)",
            data=buf.getvalue(),
            file_name="thermal_super_resolved.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )


if __name__ == '__main__':
    main()
