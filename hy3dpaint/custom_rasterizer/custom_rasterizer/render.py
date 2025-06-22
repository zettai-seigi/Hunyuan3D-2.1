# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
import sys
import os

# Add platform utilities for CUDA detection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from platform_utils import safe_cuda_import, check_cuda_dependency

# Try to import CUDA rasterizer, fallback to CPU if not available
custom_rasterizer_kernel = safe_cuda_import(
    'custom_rasterizer_kernel',
    "Custom CUDA rasterizer not available - using CPU fallback"
)

# Import CPU fallback
from .render_fallback import rasterize_cpu_fallback, interpolate_cpu_fallback


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    """
    Rasterize triangles to screen space with automatic CUDA/CPU fallback
    """
    assert pos.device == tri.device
    
    if custom_rasterizer_kernel is not None and check_cuda_dependency("Custom rasterizer"):
        # Use CUDA implementation
        try:
            findices, barycentric = custom_rasterizer_kernel.rasterize_image(
                pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
            )
            return findices, barycentric
        except Exception as e:
            print(f"⚠️  CUDA rasterizer failed: {e}")
            print("   Falling back to CPU implementation")
    
    # Use CPU fallback
    return rasterize_cpu_fallback(pos, tri, resolution, clamp_depth, use_depth_prior)


def interpolate(col, findices, barycentric, tri):
    """
    Interpolate vertex attributes with automatic CUDA/CPU fallback
    """
    if custom_rasterizer_kernel is not None and check_cuda_dependency("Custom interpolation"):
        # Use original CUDA implementation
        try:
            f = findices - 1 + (findices == 0)
            vcol = col[0, tri.long()[f.long()]]
            result = barycentric.view(*barycentric.shape, 1) * vcol
            result = torch.sum(result, axis=-2)
            return result.view(1, *result.shape)
        except Exception as e:
            print(f"⚠️  CUDA interpolation failed: {e}")
            print("   Falling back to CPU implementation")
    
    # Use CPU fallback
    return interpolate_cpu_fallback(col, findices, barycentric, tri)
