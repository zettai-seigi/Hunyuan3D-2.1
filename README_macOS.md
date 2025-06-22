# Hunyuan3D-2.1 for macOS

<p align="center">
  <img src="assets/images/teaser.jpg">
</p>

A macOS-compatible version of Tencent's Hunyuan3D-2.1, the first production-ready 3D asset generation model. This repository provides full macOS support with Apple Silicon MPS acceleration and graceful fallbacks for CUDA-dependent components.

## 🍎 macOS Features

- ✅ **Apple Silicon Support**: Native MPS (Metal Performance Shaders) acceleration
- ✅ **Intel Mac Compatible**: CPU fallbacks when MPS unavailable  
- ✅ **No CUDA Required**: Removes all CUDA dependencies
- ✅ **Automated Setup**: One-command installation script
- ✅ **Web Interface**: Gradio app optimized for macOS
- ✅ **Platform Detection**: Automatic device selection (MPS → CPU)

## 🚀 Quick Start

### Prerequisites
- **macOS 10.15+** (Catalina or later)
- **Python 3.11-3.12** (Python 3.13 not supported by PyTorch)
- **8GB+ RAM** (16GB+ recommended)
- **10GB+ free disk space**

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
```

2. **Create Python environment:**
```bash
# Using conda (recommended)
conda create -n hunyuan3d python=3.11 -y
conda activate hunyuan3d

# Or using venv
python3.11 -m venv hunyuan3d_env
source hunyuan3d_env/bin/activate
```

3. **Run automated macOS installer:**
```bash
bash install-macos.sh
```

**Or manual installation:**
```bash
# Install PyTorch for macOS
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install macOS-compatible dependencies
pip install -r requirements-macos.txt

# Install additional web dependencies
pip install gradio fastapi uvicorn pymeshlab
```

## 🎯 Usage

### Web Interface (Recommended)
```bash
# Simple launch with auto-detection
./run_gradio_macos.sh

# Or manually
python gradio_app.py --device mps --low_vram_mode --disable_tex
```

Open your browser to: **http://127.0.0.1:8081**

### Python API
```python
import sys
sys.path.insert(0, './hy3dshape')

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# Load model (auto-detects MPS/CPU)
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

# Generate 3D mesh from image
mesh = pipeline(image='path/to/image.png')[0]

# Save as file
mesh.export('output.obj')
```

### Command Line Demo
```bash
# macOS-optimized demo
python demo_macos.py

# Original demo (may have issues on macOS)
python demo.py
```

## 📊 Performance on macOS

| Device | Performance | Memory Usage | Notes |
|--------|-------------|--------------|-------|
| **Apple Silicon (M1/M2/M3)** | ~5-10x slower than CUDA | 8-16GB RAM | MPS acceleration |
| **Intel Mac** | ~10-15x slower than CUDA | 8-16GB RAM | CPU only |
| **Shape Generation** | 2-5 minutes | ~4GB RAM | Works well |
| **Texture Generation** | Limited/Disabled | ~8GB RAM | Requires xatlas |

## 🔧 Web Interface Guide

### 1. Upload Image
- Click the image area or drag & drop
- Supports: PNG, JPG, JPEG
- Best results: Clear objects, white/simple background

### 2. Generate 3D Shape
- Click **"Gen Shape"** button
- Wait 2-5 minutes for processing
- Monitor progress in browser console

### 3. Export & Download
- **Target Face Number**: Controls mesh complexity
  - `1,000` = Low detail, small file
  - `10,000` = Medium detail (recommended)
  - `100,000+` = High detail, large file
- **File Type**: GLB, OBJ, PLY, STL
- Click **"Transform"** to process
- Click **"Download"** to save file

### 4. Advanced Options
- **Seed**: For reproducible results
- **Inference Steps**: Quality vs speed (5-50)
- **Guidance Scale**: How closely to follow input
- **Remove Background**: Auto-removes image background

## 🛠 macOS-Specific Files

- **`requirements-macos.txt`**: CUDA-free dependencies
- **`install-macos.sh`**: Automated installation script
- **`run_gradio_macos.sh`**: Optimized web app launcher  
- **`demo_macos.py`**: macOS-compatible demo
- **`platform_utils.py`**: Cross-platform compatibility
- **`gradio_macos.py`**: Alternative launcher

## ⚠️ Limitations on macOS

- **No CUDA Support**: Custom rasterizer uses CPU fallback
- **Slower Performance**: 5-15x slower than CUDA systems
- **Limited Texture Generation**: Some features disabled by default
- **Memory Usage**: Limited by system RAM vs GPU VRAM
- **Training**: DeepSpeed unavailable, uses DDP fallback

## 🔍 Troubleshooting

### Common Issues

**1. Python Version Error**
```bash
# Check version
python --version
# Should be 3.11 or 3.12, not 3.13

# Fix: Install correct Python version
conda install python=3.11
```

**2. MPS Fallback Warnings**
```bash
# Normal on macOS - some operations fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**3. Memory Issues**
```bash
# Reduce resolution and batch size
python gradio_app.py --low_vram_mode
```

**4. Package Installation Failures**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Use conda for better ARM64 support
conda install -c conda-forge pymeshlab
```

**5. Gradio Connection Issues**
- Check firewall settings
- Try different port: `--port 8082`
- Use localhost only: `--host 127.0.0.1`

### Performance Optimization

1. **Close other applications** to free RAM
2. **Use lower Target Face Number** (1000-5000)
3. **Reduce Inference Steps** (5-15)
4. **Enable low VRAM mode** (`--low_vram_mode`)
5. **Use smaller input images** (512x512 or less)

## 📁 Directory Structure

```
Hunyuan3D-2.1/
├── hy3dshape/              # Shape generation models
├── hy3dpaint/              # Texture generation (limited on macOS)
├── assets/                 # Example images and templates
├── requirements-macos.txt  # macOS dependencies
├── install-macos.sh        # Installation script
├── run_gradio_macos.sh     # Web app launcher
├── demo_macos.py           # Command line demo
├── gradio_app.py           # Web interface
├── platform_utils.py      # Platform compatibility
└── README_macOS.md         # This file
```

## 🎨 Examples

| Input Image | Generated 3D Model | Time (Apple Silicon) |
|-------------|-------------------|----------------------|
| Portrait | Detailed head mesh | ~3 minutes |
| Object | Full 3D reconstruction | ~2-4 minutes |
| Animal | Organic shape | ~2-5 minutes |

## 🔗 Links

- **Original Repository**: [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- **Models**: [HuggingFace Hub](https://huggingface.co/tencent/Hunyuan3D-2.1)
- **Demo**: [Online Demo](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1)
- **Technical Report**: [arXiv:2506.15442](https://arxiv.org/abs/2506.15442)

## 📄 License

This project follows the original Hunyuan3D-2.1 license terms. See `LICENSE` for details.

## 🙏 Acknowledgments

- **Tencent Hunyuan3D Team** for the original model and codebase
- **Apple** for Metal Performance Shaders (MPS) framework
- **PyTorch Team** for macOS MPS support
- **Community contributors** for testing and feedback

---

**🍎 Made with ❤️ for macOS users**

For issues specific to this macOS port, please check the troubleshooting section or create an issue with the `[macOS]` tag.