#!/usr/bin/env python3
"""
Direct launch for Hunyuan3D-2.1 Gradio app on macOS
This script directly imports and runs the gradio app without subprocess
"""

import os
import sys

def main():
    print("🍎 Starting Hunyuan3D-2.1 Gradio App for macOS")
    
    # Set MPS fallback for unsupported operations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Set default arguments for macOS
    sys.argv = [
        'gradio_app.py',
        '--model_path', 'tencent/Hunyuan3D-2.1',
        '--subfolder', 'hunyuan3d-dit-v2-1',
        '--texgen_model_path', 'tencent/Hunyuan3D-2.1',
        '--port', '8080',
        '--host', '127.0.0.1',
        '--low_vram_mode',
        '--disable_tex',
    ]
    
    print("🚀 Launching Gradio app with macOS optimizations...")
    print("   • MPS fallback enabled for unsupported operations")
    print("   • Low VRAM mode enabled")
    print("   • Texture generation disabled (reduce dependencies)")
    print("   • Host: 127.0.0.1:8080")
    print("")
    
    # Import and run gradio_app directly
    try:
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Import the gradio app main code
        import gradio_app
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error running Gradio app: {e}")
        print("\n📝 Troubleshooting:")
        print("   • Make sure you're in the correct conda/virtual environment")
        print("   • Check that all dependencies are installed")
        print("   • Try: python gradio_app.py --help")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())