#!/usr/bin/env python3
"""
macOS-specific launcher for Hunyuan3D-2.1 Gradio app
This script handles macOS-specific setup and compatibility.
"""

import os
import sys
import subprocess

def main():
    print("🍎 Starting Hunyuan3D-2.1 Gradio App for macOS")
    
    # Set MPS fallback for unsupported operations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Import platform detection
    try:
        from platform_utils import get_platform, configure_torch_device
        platform_info = get_platform()
        device_info = configure_torch_device()
        device = str(device_info.device)  # Ensure device is a string
        print(f"🍎 Platform: {platform_info}")
        print(f"🍎 Auto-detected device: {device}")
    except Exception as e:
        print(f"⚠️  Platform utils error: {e}")
        # Try to detect device without platform_utils
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            print(f"🍎 Fallback device detection: {device}")
        except ImportError:
            device = 'cpu'
            print(f"🍎 Using fallback device: {device}")
    
    # Ensure device is a string (safety check)
    device = str(device)
    
    # Debug: print the args to see what's being passed
    print(f"Debug: device type = {type(device)}, device value = {device}")
    
    # Use the correct Python executable (conda environment)
    # Instead of sys.executable which points to system Python, use 'python'
    python_cmd = 'python'
    
    # Prepare command line arguments
    args = [
        python_cmd, 'gradio_app.py',
        '--model_path', 'tencent/Hunyuan3D-2.1',
        '--subfolder', 'hunyuan3d-dit-v2-1',
        '--texgen_model_path', 'tencent/Hunyuan3D-2.1',
        '--device', device,
        '--port', '8080',
        '--host', '127.0.0.1',  # Localhost only for security
        '--low_vram_mode',  # Enable by default on macOS
        '--disable_tex',  # Disable texture generation by default on macOS
    ]
    
    # Add any additional arguments passed to this script
    args.extend(sys.argv[1:])
    
    print("🚀 Launching Gradio app with the following settings:")
    print(f"   Device: {device}")
    print(f"   Low VRAM mode: Enabled")
    print(f"   Texture generation: Disabled (use --enable_tex to override)")
    print(f"   Host: 127.0.0.1:8080")
    print("")
    print("📝 To enable texture generation (if xatlas is available):")
    print("   python gradio_macos.py --enable_tex")
    print("")
    
    try:
        subprocess.run(args, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Gradio app: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())