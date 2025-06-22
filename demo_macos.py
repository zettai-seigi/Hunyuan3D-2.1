#!/usr/bin/env python3
"""
macOS-compatible demo script for Hunyuan3D-2.1
This script includes fallbacks and proper error handling for macOS
"""

import sys
import os
import warnings
from pathlib import Path

# Add paths
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Import platform utilities first
from platform_utils import get_platform, configure_torch_device, check_cuda_dependency

def main():
    print("🍎 Hunyuan3D-2.1 macOS Demo")
    print("=" * 50)
    
    # Check platform
    platform = get_platform()
    print(f"Platform: {platform.system} {platform.machine}")
    print(f"Device: {platform.get_device_info()}")
    print(f"CUDA Available: {platform.cuda_available}")
    print(f"MPS Available: {platform.mps_available}")
    print(f"Custom Rasterizer Supported: {platform.supports_custom_rasterizer()}")
    print()
    
    # Configure device
    device = configure_torch_device()
    platform.warn_if_suboptimal()
    print()
    
    # Apply torchvision fix
    try:
        from torchvision_fix import apply_fix
        apply_fix()
        print("✅ Torchvision compatibility fix applied")
    except ImportError:
        print("⚠️  torchvision_fix not found - proceeding without compatibility fix")
    except Exception as e:
        print(f"⚠️  Failed to apply torchvision fix: {e}")
    
    print()
    
    # Test imports with fallback handling
    print("🔍 Testing imports...")
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return
    
    try:
        from hy3dshape.rembg import BackgroundRemover
        print("✅ Background remover imported")
    except ImportError as e:
        print(f"⚠️  Background remover failed: {e}")
    
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("✅ Shape pipeline imported")
    except ImportError as e:
        print(f"❌ Shape pipeline import failed: {e}")
        return
    
    try:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        print("✅ Texture pipeline imported")
    except ImportError as e:
        print(f"⚠️  Texture pipeline import failed: {e}")
        print("   Texture generation will not be available")
        texture_available = False
    else:
        texture_available = True
    
    print()
    
    # Test model loading (if available)
    print("🔄 Testing model loading...")
    
    try:
        print("Loading shape generation model...")
        model_path = 'tencent/Hunyuan3D-2.1'
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=device.type
        )
        print("✅ Shape model loaded successfully")
        
        # Test with demo image
        image_path = 'assets/demo.png'
        if os.path.exists(image_path):
            print(f"Processing demo image: {image_path}")
            
            image = Image.open(image_path).convert("RGBA")
            if image.mode == 'RGB':
                try:
                    rembg = BackgroundRemover()
                    image = rembg(image)
                    print("✅ Background removed")
                except:
                    print("⚠️  Background removal failed - using original image")
            
            print("Generating 3D mesh...")
            mesh = pipeline_shapegen(image=image)[0]
            
            output_path = 'demo_macos.glb'
            mesh.export(output_path)
            print(f"✅ Mesh saved to {output_path}")
            
            # Test texture generation if available
            if texture_available and platform.supports_custom_rasterizer():
                print("Testing texture generation...")
                try:
                    max_num_view = 4  # Reduced for macOS
                    resolution = 512
                    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
                    
                    # Set macOS-specific paths
                    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
                    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
                    
                    paint_pipeline = Hunyuan3DPaintPipeline(conf)
                    
                    textured_output = 'demo_macos_textured.glb'
                    paint_pipeline(
                        mesh_path=output_path,
                        image_path=image_path,
                        output_mesh_path=textured_output
                    )
                    print(f"✅ Textured mesh saved to {textured_output}")
                    
                except Exception as e:
                    print(f"⚠️  Texture generation failed: {e}")
                    print("   This is expected on macOS due to CUDA limitations")
            
            elif texture_available:
                print("⚠️  Texture generation skipped - custom rasterizer not supported on this platform")
            
        else:
            print(f"⚠️  Demo image not found at {image_path}")
            print("   Place a demo image at this path to test the full pipeline")
            
    except Exception as e:
        print(f"❌ Model loading/processing failed: {e}")
        print("   This might be due to:")
        print("   - Missing model files")
        print("   - Insufficient memory")
        print("   - Platform compatibility issues")
    
    print()
    print("🎉 Demo completed!")
    print()
    print("📝 Notes for macOS users:")
    print("   • Performance will be slower than CUDA systems")
    print("   • Texture generation has limited functionality")
    print("   • Use --low_vram_mode for memory-constrained systems")
    print("   • Consider using smaller resolution settings")


if __name__ == "__main__":
    main()