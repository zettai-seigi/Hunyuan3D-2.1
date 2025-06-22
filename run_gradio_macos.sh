#!/bin/bash
# Simple script to run Gradio app with the correct conda environment

echo "🍎 Starting Hunyuan3D-2.1 Gradio App for macOS"
echo "🚀 Using conda environment: proj-Huanyuan3D-2.1"

# Set MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use the correct Python from conda environment
PYTHON_PATH="/opt/homebrew/Caskroom/miniconda/base/envs/proj-Huanyuan3D-2.1/bin/python"

echo "📍 Python: $PYTHON_PATH"
echo "🔧 Settings: MPS device, Low VRAM mode, Texture generation disabled"
echo "🌐 Server: http://127.0.0.1:8081"
echo ""

# Run the gradio app
$PYTHON_PATH gradio_app.py \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --texgen_model_path tencent/Hunyuan3D-2.1 \
  --device mps \
  --port 8081 \
  --host 127.0.0.1 \
  --low_vram_mode \
  --disable_tex