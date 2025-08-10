#!/usr/bin/env python3
"""
macOS-specific launcher for Hunyuan3D-2.1 Gradio app
This script handles macOS-specific setup and compatibility.
Now with smart hardware detection for optimal performance!
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def detect_hardware():
    """Detect hardware and get optimal settings"""
    try:
        # Try to import and use hardware detector
        from hardware_detector import HardwareDetector
        detector = HardwareDetector()
        settings = detector.get_recommended_settings()
        print("🔍 Hardware detection successful!")
        return settings
    except Exception as e:
        print(f"⚠️  Could not detect hardware: {e}")
        # Return conservative defaults
        return {
            'low_vram_mode': True,
            'enable_tex': False,
            'compile': False
        }

def main():
    print("🍎 Starting Hunyuan3D-2.1 Gradio App for macOS")
    
    # Detect hardware and get optimal settings
    hw_settings = detect_hardware()
    
    # Print hardware summary if available
    if 'chip' in hw_settings:
        print(f"🖥️  Detected: {hw_settings['chip']} with {hw_settings['memory_gb']}GB RAM")
        if hw_settings.get('recommendations'):
            print(f"⚡ {hw_settings['recommendations'][0]}")
    
    # Set MPS fallback for unsupported operations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Import platform detection
    try:
        from platform_utils import get_platform, configure_torch_device
        platform_info = get_platform()
        device_info = configure_torch_device()
        device = str(device_info.device)  # Ensure device is a string
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
        except ImportError:
            device = 'cpu'
    
    # Ensure device is a string (safety check)
    device = str(device)
    
    # Use the correct Python executable (conda environment)
    python_cmd = 'python'
    
    # Check for command line overrides
    force_low_vram = '--low_vram' in sys.argv
    force_high_perf = '--high_perf' in sys.argv
    enable_tex_override = '--enable_tex' in sys.argv
    disable_tex_override = '--disable_tex' in sys.argv
    
    # Determine settings with overrides
    if force_high_perf:
        low_vram_mode = False
        print("⚡ High performance mode forced via --high_perf")
    elif force_low_vram:
        low_vram_mode = True
        print("🔋 Low VRAM mode forced via --low_vram")
    else:
        low_vram_mode = hw_settings.get('low_vram_mode', True)
    
    if disable_tex_override:
        enable_tex = False
    elif enable_tex_override:
        enable_tex = True
    else:
        enable_tex = hw_settings.get('enable_tex', False)
    
    # Prepare command line arguments
    args = [
        python_cmd, 'gradio_app.py',
        '--model_path', 'tencent/Hunyuan3D-2.1',
        '--subfolder', 'hunyuan3d-dit-v2-1',
        '--texgen_model_path', 'tencent/Hunyuan3D-2.1',
        '--device', device,
        '--port', '8080',
        '--host', '127.0.0.1',  # Localhost only for security
    ]
    
    # Add hardware-optimized settings
    if low_vram_mode:
        args.append('--low_vram_mode')
    
    if not enable_tex:
        args.append('--disable_tex')
    
    if hw_settings.get('compile', False):
        args.append('--compile')
    
    if hw_settings.get('enable_flashvdm', False):
        args.append('--enable_flashvdm')
    
    # Filter out our custom flags from remaining arguments
    custom_flags = ['--enable_tex', '--disable_tex', '--low_vram', '--high_perf']
    additional_args = [arg for arg in sys.argv[1:] if arg not in custom_flags]
    args.extend(additional_args)
    
    print("🚀 Launching Gradio app with the following settings:")
    print(f"   Device: {device}")
    print(f"   Low VRAM mode: {'Enabled' if low_vram_mode else 'Disabled (Full Performance)'}")
    print(f"   Texture generation: {'Enabled (with xatlas)' if enable_tex else 'Disabled'}")
    if hw_settings.get('resolution'):
        print(f"   Resolution: {hw_settings['resolution']}px")
    if hw_settings.get('max_num_views'):
        print(f"   Max views: {hw_settings['max_num_views']}")
    print(f"   Host: 127.0.0.1:8080")
    print("")
    print("📝 Command line options:")
    print("   --high_perf   : Force high performance mode (disable low VRAM)")
    print("   --low_vram    : Force low VRAM mode")
    print("   --enable_tex  : Force enable texture generation")
    print("   --disable_tex : Force disable texture generation")
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