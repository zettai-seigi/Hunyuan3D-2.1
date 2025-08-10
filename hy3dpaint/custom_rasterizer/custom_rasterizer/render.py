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
import warnings

# Try to import CUDA rasterizer
try:
    import custom_rasterizer_kernel
    HAS_CUDA_RASTERIZER = True
except ImportError:
    custom_rasterizer_kernel = None
    HAS_CUDA_RASTERIZER = False
    warnings.warn(
        "Custom CUDA rasterizer not available. Using CPU fallback (slower performance).",
        UserWarning
    )

# Import CPU fallback
from .render_fallback import rasterize_cpu_fallback, interpolate_cpu_fallback


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    """
    Rasterize triangles to screen space with automatic CUDA/CPU fallback
    """
    assert pos.device == tri.device
    
    if HAS_CUDA_RASTERIZER and pos.device.type == 'cuda':
        # Use CUDA implementation
        try:
            findices, barycentric = custom_rasterizer_kernel.rasterize_image(
                pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
            )
            return findices, barycentric
        except Exception as e:
            warnings.warn(f"CUDA rasterizer failed: {e}. Falling back to CPU implementation.")
    
    # Use CPU fallback for MPS or when CUDA fails
    return rasterize_cpu_fallback(pos, tri, resolution, clamp_depth, use_depth_prior)


def interpolate(col, findices, barycentric, tri):
    """
    Interpolate vertex attributes with automatic CUDA/CPU fallback
    """
    if HAS_CUDA_RASTERIZER and col.device.type == 'cuda':
        # Use original CUDA implementation
        try:
            f = findices - 1 + (findices == 0)
            vcol = col[0, tri.long()[f.long()]]
            result = barycentric.view(*barycentric.shape, 1) * vcol
            result = torch.sum(result, axis=-2)
            return result.view(1, *result.shape)
        except Exception as e:
            warnings.warn(f"CUDA interpolation failed: {e}. Falling back to CPU implementation.")
    
    # Use CPU fallback for MPS or when CUDA fails
    return interpolate_cpu_fallback(col, findices, barycentric, tri)