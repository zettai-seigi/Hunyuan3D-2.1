# Platform utility functions for cross-platform compatibility
# Handles CUDA detection and fallback logic for macOS

import os
import platform
import warnings
import torch
from typing import Tuple, Optional


class PlatformInfo:
    """Platform detection and capability information"""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.is_macos = self.system == "Darwin"
        self.is_apple_silicon = self.is_macos and self.machine == "arm64"
        self.is_linux = self.system == "Linux"
        self.is_windows = self.system == "Windows"
        
        # CUDA availability
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_count = torch.cuda.device_count() if self.cuda_available else 0
        
        # MPS (Metal Performance Shaders) for Apple Silicon
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Determine best device
        self.device = self._get_best_device()
        
    def _get_best_device(self) -> str:
        """Determine the best available device for computation"""
        if self.cuda_available:
            return "cuda"
        elif self.mps_available:
            return "mps"
        else:
            return "cpu"
    
    def get_device_info(self) -> str:
        """Get human-readable device information"""
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0) if self.cuda_available else "Unknown"
            return f"CUDA GPU: {gpu_name}"
        elif self.device == "mps":
            return "Apple Metal Performance Shaders (MPS)"
        else:
            return "CPU"
    
    def warn_if_suboptimal(self):
        """Warn user about performance limitations"""
        if not self.cuda_available:
            if self.is_macos:
                if self.mps_available:
                    print("⚠️  Running on macOS with MPS acceleration")
                    print("   Performance will be slower than CUDA systems")
                else:
                    print("⚠️  Running on macOS with CPU-only acceleration")
                    print("   Performance will be significantly slower than CUDA systems")
            else:
                print("⚠️  CUDA not available - using CPU")
                print("   Performance will be significantly slower")
    
    def supports_custom_rasterizer(self) -> bool:
        """Check if custom CUDA rasterizer is supported"""
        return self.cuda_available and not self.is_macos


def get_platform() -> PlatformInfo:
    """Get platform information singleton"""
    if not hasattr(get_platform, '_instance'):
        get_platform._instance = PlatformInfo()
    return get_platform._instance


def check_cuda_dependency(package_name: str = "CUDA operation") -> bool:
    """
    Check if CUDA is available for a specific operation
    Returns True if CUDA available, False otherwise with appropriate warning
    """
    platform_info = get_platform()
    
    if not platform_info.cuda_available:
        if platform_info.is_macos:
            print(f"🍎 {package_name} not available on macOS - using CPU fallback")
        else:
            print(f"⚠️  {package_name} requires CUDA - using CPU fallback")
        return False
    
    return True


def safe_cuda_import(module_name: str, fallback_message: Optional[str] = None):
    """
    Safely import CUDA-dependent modules with fallback handling
    
    Args:
        module_name: Name of the module to import
        fallback_message: Custom message to show if import fails
    
    Returns:
        Imported module or None if import fails
    """
    try:
        # Try to import the module
        if '.' in module_name:
            parts = module_name.split('.')
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_name)
        
        return module
    
    except ImportError as e:
        platform_info = get_platform()
        
        if fallback_message:
            print(f"⚠️  {fallback_message}")
        else:
            if platform_info.is_macos:
                print(f"🍎 {module_name} not available on macOS")
            else:
                print(f"⚠️  Failed to import {module_name}: {e}")
        
        return None
    
    except Exception as e:
        print(f"❌ Unexpected error importing {module_name}: {e}")
        return None


def configure_torch_device(prefer_cuda: bool = True) -> torch.device:
    """
    Configure PyTorch device based on platform capabilities
    
    Args:
        prefer_cuda: Whether to prefer CUDA over other accelerators
    
    Returns:
        torch.device configured for the platform
    """
    platform_info = get_platform()
    
    if prefer_cuda and platform_info.cuda_available:
        device = torch.device("cuda")
        print(f"🚀 Using CUDA: {platform_info.get_device_info()}")
    elif platform_info.mps_available and platform_info.is_apple_silicon:
        device = torch.device("mps")
        print(f"🍎 Using MPS: {platform_info.get_device_info()}")
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU: {platform_info.get_device_info()}")
        platform_info.warn_if_suboptimal()
    
    return device


def get_memory_info() -> Tuple[Optional[float], Optional[float]]:
    """
    Get memory information for the current device
    
    Returns:
        Tuple of (used_memory_gb, total_memory_gb) or (None, None) if not available
    """
    platform_info = get_platform()
    
    if platform_info.cuda_available:
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return used, total
    elif platform_info.mps_available:
        # MPS doesn't provide direct memory info
        return None, None
    else:
        # CPU memory info would require psutil
        return None, None


if __name__ == "__main__":
    # Test the platform detection
    platform_info = get_platform()
    
    print("🔍 Platform Detection Results:")
    print(f"   System: {platform_info.system}")
    print(f"   Architecture: {platform_info.machine}")
    print(f"   Device: {platform_info.device}")
    print(f"   Device Info: {platform_info.get_device_info()}")
    print(f"   CUDA Available: {platform_info.cuda_available}")
    print(f"   MPS Available: {platform_info.mps_available}")
    print(f"   Custom Rasterizer Supported: {platform_info.supports_custom_rasterizer()}")
    
    # Test device configuration
    device = configure_torch_device()
    
    # Test memory info
    used, total = get_memory_info()
    if used is not None and total is not None:
        print(f"   GPU Memory: {used:.1f}GB / {total:.1f}GB")