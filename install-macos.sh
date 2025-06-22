#!/bin/bash
# macOS Installation Script for Hunyuan3D-2.1
# This script provides macOS-compatible installation with fallbacks

set -e

echo "🍎 Installing Hunyuan3D-2.1 for macOS..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "📍 Python version: $python_version"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ required"
    exit 1
fi

# Detect Apple Silicon vs Intel
arch=$(uname -m)
if [[ "$arch" == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon (M1/M2/M3)"
    MPS_AVAILABLE=true
else
    echo "💻 Detected Intel Mac"
    MPS_AVAILABLE=false
fi

# Check for virtual environment or suggest creating one
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  No virtual environment detected."
    echo "📦 Creating virtual environment for safer installation..."
    python3 -m venv hunyuan3d_env
    source hunyuan3d_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install PyTorch for macOS (CPU + MPS support)
echo "🔥 Installing PyTorch for macOS..."
if [[ "$MPS_AVAILABLE" == true ]]; then
    echo "📱 Installing with MPS (Metal Performance Shaders) support"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
else
    echo "💻 Installing CPU-only version"
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install macOS-compatible requirements
echo "📦 Installing macOS-compatible dependencies..."
pip install -r requirements-macos.txt

# Try to install problematic packages with fallbacks
echo "🔧 Installing packages that may need special handling..."

# PyMeshLab (may fail on ARM64)
if ! pip install pymeshlab==2022.2.post3; then
    echo "⚠️  pymeshlab installation failed - you may need to install it manually"
    echo "   Try: conda install -c conda-forge pymeshlab"
fi

# Handle custom rasterizer compilation
echo "🔨 Setting up custom rasterizer..."
cd hy3dpaint/custom_rasterizer

# Create a CPU fallback setup.py for macOS
cat > setup_macos.py << 'EOF'
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import warnings

# Custom rasterizer with CPU fallback for macOS
print("🍎 Setting up custom rasterizer for macOS (CPU fallback)")

# Only use CPU extensions on macOS
custom_rasterizer_module = CppExtension(
    "custom_rasterizer_kernel_cpu",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        # Note: Excluding CUDA files for macOS compatibility
    ],
    extra_compile_args=['-std=c++14']
)

setup(
    packages=find_packages(),
    version="0.1-macos",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)
EOF

# Try CPU-only compilation
echo "🔨 Attempting CPU-only rasterizer compilation..."
if python3 setup_macos.py build_ext --inplace; then
    echo "✅ Custom rasterizer compiled successfully (CPU fallback)"
else
    echo "⚠️  Custom rasterizer compilation failed - texture generation will use fallback"
fi

cd ../..

# Handle mesh painter compilation
echo "🔨 Compiling mesh painter for macOS..."
cd hy3dpaint/DifferentiableRenderer

# Check if we have the required C++ compiler
if command -v clang++ &> /dev/null; then
    echo "✅ clang++ found"
    # Modify compile script for macOS
    clang++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) mesh_inpaint_processor.cpp -o mesh_inpaint_processor$(python3-config --extension-suffix) || echo "⚠️  Mesh painter compilation failed"
else
    echo "❌ clang++ not found. Please install Xcode Command Line Tools:"
    echo "   xcode-select --install"
fi

cd ../..

# Download required model checkpoints
echo "📥 Downloading required checkpoints..."
mkdir -p hy3dpaint/ckpt
if ! [ -f "hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
    curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -o "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    echo "✅ RealESRGAN checkpoint downloaded"
else
    echo "✅ RealESRGAN checkpoint already exists"
fi

echo ""
echo "🎉 macOS installation complete!"
echo ""
echo "📝 Important Notes:"
echo "   • CUDA acceleration is not available on macOS"
echo "   • Performance will be slower than CUDA systems"
echo "   • MPS acceleration available on Apple Silicon"
echo "   • Some texture generation features may be limited"
echo ""
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "💡 Remember to activate the virtual environment:"
    echo "   source hunyuan3d_env/bin/activate"
    echo ""
fi
echo "🚀 To test the installation, run:"
echo "   python demo_macos.py"