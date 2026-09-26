# 🔥 Thermal Image Super-Resolution & Restoration (NAFNet-SR)

An ultra-lightweight Deep Learning system (0.47M parameters) for **4× Super-Resolution** and **Edge Restoration** in thermal infrared imaging.

Designed to enhance low-cost, low-resolution thermal sensor captures (e.g., 90×60) to high-definition (360×240) with sharp thermal boundaries and reduced noise.

---

## ⚡ Highlights
- **Ultra-Lightweight Architecture**: Only **0.47M parameters (~6 MB checkpoint)**, making it 35× smaller than ESRGAN and 25× smaller than SwinIR.
- **Fast Real-Time Inference**: Sub-20ms latency on GPU and ~50ms on CPU.
- **Dual Restoration Modes**:
  - **4× Direct Super-Resolution**: Upscales raw sensor resolution 4× spatially.
  - **Same-Size Restoration**: Restores pre-stretched blurry thermal captures back to sharp native clarity.
  - **Auto-Detect**: Automatically selects optimal restoration strategy based on input dimensions.
- **Interactive Streamlit Web App**: Complete with side-by-side visual comparison, latency metrics, and 1-click PNG download.

---

## 📁 Repository Structure

```text
├── streamlit_app.py               # Production Streamlit Web Application
├── app.py                         # Standalone Gradio Web Application
├── best_model.pth                 # Trained NAFNet-SR weights checkpoint (~6 MB)
├── nafnet.py                      # NAFNet-SR architecture definition
├── dataset.py                     # Thermal data loading & augmentations
├── train.py                       # Training pipeline with AMP & Cosine Annealing
├── infer.py                       # Batch test inference script
├── config.py                      # Centralized hyperparameters & paths
├── utils.py                       # PSNR/SSIM metrics & logging helpers
├── plot_training.py               # Visualizer for training loss/PSNR curves
├── requirements.txt               # Dependencies
└── NAFNet_SR_Thermal_Colab.ipynb  # Free-tier Colab training notebook
```

---

## 🚀 Quickstart — Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/thermal-super-resolution.git
cd thermal-super-resolution
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web App
```bash
python -m streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔬 Architecture Details

NAFNet-SR replaces traditional heavy nonlinear activations (ReLU, GELU, Sigmoid) with **SimpleGate** (element-wise multiplication of channel halves) and **Simplified Channel Attention (SCA)**:
- **Input**: Low-Resolution Thermal Image $(B, 3, H, W)$
- **Feature Extraction**: $3 \times 3$ Conv $\rightarrow 16 \times \text{NAFBlocks}$ at native resolution
- **Upsampling**: Dual $2\times$ PixelShuffle blocks (total $4\times$ spatial upscale)
- **Output**: Super-Resolved Thermal Image $(B, 3, 4H, 4W)$

---

## 📊 Training
The model can be trained locally with:
```bash
python train.py
```
Or directly on Google Colab using `NAFNet_SR_Thermal_Colab.ipynb` with GPU acceleration.

---

## 📜 License
MIT License
