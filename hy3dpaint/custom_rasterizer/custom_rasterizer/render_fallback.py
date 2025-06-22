# CPU fallback implementation for custom rasterizer
# This provides a software-based rasterization when CUDA is not available

import torch
import numpy as np
import warnings
from typing import Tuple


def rasterize_cpu_fallback(pos: torch.Tensor, tri: torch.Tensor, resolution: Tuple[int, int], 
                          clamp_depth: torch.Tensor = None, use_depth_prior: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for mesh rasterization
    
    This is a simplified software rasterizer that provides basic functionality
    when the CUDA rasterizer is not available (e.g., on macOS)
    
    Args:
        pos: Vertex positions [batch, N, 3]
        tri: Triangle indices [M, 3]
        resolution: (width, height) of output image
        clamp_depth: Depth clamping values (ignored in fallback)
        use_depth_prior: Use depth prior (ignored in fallback)
    
    Returns:
        findices: Face indices for each pixel
        barycentric: Barycentric coordinates for each pixel
    """
    warnings.warn(
        "Using CPU fallback rasterizer. Performance will be significantly slower than CUDA version.",
        UserWarning
    )
    
    batch_size = pos.shape[0] if len(pos.shape) == 3 else 1
    if len(pos.shape) == 2:
        pos = pos.unsqueeze(0)
    
    width, height = resolution
    device = pos.device
    
    # Initialize output tensors
    findices = torch.zeros((batch_size, height, width), dtype=torch.long, device=device)
    barycentric = torch.zeros((batch_size, height, width, 3), dtype=torch.float32, device=device)
    depth_buffer = torch.full((batch_size, height, width), float('inf'), dtype=torch.float32, device=device)
    
    # Convert to numpy for CPU processing
    pos_np = pos.detach().cpu().numpy()
    tri_np = tri.detach().cpu().numpy()
    
    for batch_idx in range(batch_size):
        vertices = pos_np[batch_idx]
        
        # Simple rasterization loop
        for face_idx, face in enumerate(tri_np):
            if face_idx >= len(vertices) - 2:  # Safety check
                break
                
            # Get triangle vertices
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Convert to screen coordinates (simple orthographic projection)
            # Assuming vertices are already in NDC space [-1, 1]
            x0, y0, z0 = int((v0[0] + 1) * width * 0.5), int((v0[1] + 1) * height * 0.5), v0[2]
            x1, y1, z1 = int((v1[0] + 1) * width * 0.5), int((v1[1] + 1) * height * 0.5), v1[2]
            x2, y2, z2 = int((v2[0] + 1) * width * 0.5), int((v2[1] + 1) * height * 0.5), v2[2]
            
            # Bounding box
            min_x = max(0, min(x0, x1, x2))
            max_x = min(width - 1, max(x0, x1, x2))
            min_y = max(0, min(y0, y1, y2))
            max_y = min(height - 1, max(y0, y1, y2))
            
            # Rasterize triangle
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Compute barycentric coordinates
                    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                    if abs(denom) < 1e-10:
                        continue
                        
                    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
                    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
                    c = 1 - a - b
                    
                    # Check if point is inside triangle
                    if a >= 0 and b >= 0 and c >= 0:
                        # Interpolate depth
                        z = a * z0 + b * z1 + c * z2
                        
                        # Depth test
                        if z < depth_buffer[batch_idx, y, x]:
                            depth_buffer[batch_idx, y, x] = z
                            findices[batch_idx, y, x] = face_idx + 1  # 1-indexed
                            barycentric[batch_idx, y, x, 0] = a
                            barycentric[batch_idx, y, x, 1] = b
                            barycentric[batch_idx, y, x, 2] = c
    
    return findices, barycentric


def interpolate_cpu_fallback(col: torch.Tensor, findices: torch.Tensor, 
                           barycentric: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """
    CPU fallback for attribute interpolation
    
    Args:
        col: Vertex colors/attributes [batch, N, channels]
        findices: Face indices from rasterization
        barycentric: Barycentric coordinates from rasterization
        tri: Triangle indices
    
    Returns:
        Interpolated attributes for each pixel
    """
    # Handle face indexing (convert from 1-indexed to 0-indexed)
    f = findices - 1 + (findices == 0)
    f = torch.clamp(f, 0, len(tri) - 1)
    
    # Get vertex indices for each pixel
    vertex_indices = tri[f.long()]  # [batch, height, width, 3]
    
    # Gather vertex colors
    if len(col.shape) == 2:
        col = col.unsqueeze(0)
    
    batch_size, height, width = findices.shape
    channels = col.shape[-1]
    
    # Expand dimensions for gathering
    vertex_indices = vertex_indices.unsqueeze(-1).expand(-1, -1, -1, -1, channels)
    col_expanded = col.unsqueeze(1).unsqueeze(1).expand(-1, height, width, -1, -1)
    
    # Gather colors for triangle vertices
    vcol = torch.gather(col_expanded, 3, vertex_indices)  # [batch, height, width, 3, channels]
    
    # Interpolate using barycentric coordinates
    barycentric_expanded = barycentric.unsqueeze(-1)  # [batch, height, width, 3, 1]
    result = (barycentric_expanded * vcol).sum(dim=3)  # [batch, height, width, channels]
    
    # Mask out background pixels
    mask = (findices > 0).unsqueeze(-1).float()
    result = result * mask
    
    return result


class CPUFallbackRasterizer:
    """
    CPU fallback rasterizer class that mimics the CUDA rasterizer interface
    """
    
    @staticmethod
    def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
        return rasterize_cpu_fallback(pos, tri, resolution, clamp_depth, use_depth_prior)
    
    @staticmethod
    def interpolate(col, findices, barycentric, tri):
        return interpolate_cpu_fallback(col, findices, barycentric, tri)


# Provide compatibility interface
def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    """Compatibility function that matches the original CUDA rasterizer interface"""
    return rasterize_cpu_fallback(pos, tri, resolution, clamp_depth, use_depth_prior)


def interpolate(col, findices, barycentric, tri):
    """Compatibility function that matches the original CUDA rasterizer interface"""
    return interpolate_cpu_fallback(col, findices, barycentric, tri)