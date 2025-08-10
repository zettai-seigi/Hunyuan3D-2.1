#!/bin/bash
# Smart launcher script for Gradio app with automatic hardware detection

echo "🍎 Starting Hunyuan3D-2.1 Gradio App for macOS"
echo "🚀 Using conda environment: proj-Huanyuan3D-2.1"

# Set MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use the correct Python from conda environment
PYTHON_PATH="/opt/homebrew/Caskroom/miniconda/base/envs/proj-Huanyuan3D-2.1/bin/python"

echo "📍 Python: $PYTHON_PATH"
echo "🔍 Auto-detecting hardware for optimal settings..."
echo ""

# Run the gradio app with smart defaults
# The script will auto-detect your hardware and configure accordingly
# Pass any additional arguments through
$PYTHON_PATH gradio_macos.py "$@"