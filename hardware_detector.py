#!/usr/bin/env python3
"""
Hardware detection and auto-configuration for macOS
Automatically determines optimal settings based on system capabilities
"""

import subprocess
import re
import json
from typing import Dict, Optional, Tuple

class HardwareDetector:
    """Detect Mac hardware and recommend optimal settings"""
    
    # Memory thresholds for configuration
    MEMORY_CONFIGS = {
        'ultra': 96,   # 96GB+ - No compromises, maximum quality
        'high': 64,    # 64GB+ - Full features, high quality
        'medium': 32,  # 32GB+ - Balanced performance
        'standard': 16, # 16GB+ - Standard settings
        'low': 8       # 8GB+ - Conservative settings
    }
    
    # Apple Silicon chip performance tiers
    CHIP_TIERS = {
        'ultra': ['M3 Max', 'M2 Max', 'M2 Ultra', 'M1 Ultra', 'M3 Ultra'],
        'pro': ['M3 Pro', 'M2 Pro', 'M1 Pro'],
        'standard': ['M3', 'M2', 'M1'],
        'intel': []  # Intel Macs (legacy)
    }
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.memory_gb = self._get_memory_gb()
        self.chip_name = self._get_chip_name()
        self.chip_tier = self._get_chip_tier()
        self.gpu_cores = self._get_gpu_cores()
        
    def _run_command(self, cmd: str) -> str:
        """Run system command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout
        except:
            return ""
    
    def _get_system_info(self) -> str:
        """Get full system hardware info"""
        return self._run_command("system_profiler SPHardwareDataType")
    
    def _get_memory_gb(self) -> int:
        """Extract memory in GB"""
        match = re.search(r'Memory:\s+(\d+)\s+GB', self.system_info)
        if match:
            return int(match.group(1))
        return 8  # Default fallback
    
    def _get_chip_name(self) -> str:
        """Extract Apple Silicon chip name"""
        match = re.search(r'Chip:\s+Apple\s+([^\n]+)', self.system_info)
        if match:
            return match.group(1).strip()
        
        # Check for Intel
        if 'Intel' in self.system_info:
            return 'Intel'
        
        return 'Unknown'
    
    def _get_gpu_cores(self) -> Optional[int]:
        """Extract GPU core count"""
        # Try to get GPU cores from system profiler
        gpu_info = self._run_command("system_profiler SPDisplaysDataType")
        
        # M3 Max can have 30 or 40 GPU cores
        if 'M3 Max' in self.chip_name:
            if '40-core' in gpu_info or '40-Core' in gpu_info:
                return 40
            elif '30-core' in gpu_info or '30-Core' in gpu_info:
                return 30
        
        # M2 Max has 30 or 38 GPU cores
        if 'M2 Max' in self.chip_name:
            if '38-core' in gpu_info or '38-Core' in gpu_info:
                return 38
            elif '30-core' in gpu_info or '30-Core' in gpu_info:
                return 30
                
        # Default GPU cores by chip
        default_cores = {
            'M3 Max': 30,
            'M3 Pro': 18,
            'M3': 10,
            'M2 Max': 30,
            'M2 Pro': 16,
            'M2': 10,
            'M1 Max': 24,
            'M1 Pro': 14,
            'M1': 8
        }
        
        return default_cores.get(self.chip_name, None)
    
    def _get_chip_tier(self) -> str:
        """Determine chip performance tier"""
        for tier, chips in self.CHIP_TIERS.items():
            if any(chip in self.chip_name for chip in chips):
                return tier
        return 'standard'
    
    def get_memory_config(self) -> str:
        """Get memory configuration tier"""
        for tier, min_gb in self.MEMORY_CONFIGS.items():
            if self.memory_gb >= min_gb:
                return tier
        return 'low'
    
    def get_recommended_settings(self) -> Dict:
        """Get recommended settings based on hardware"""
        memory_tier = self.get_memory_config()
        
        settings = {
            'chip': self.chip_name,
            'memory_gb': self.memory_gb,
            'gpu_cores': self.gpu_cores,
            'chip_tier': self.chip_tier,
            'memory_tier': memory_tier,
        }
        
        # Determine optimal settings
        if memory_tier == 'ultra' and self.chip_tier == 'ultra':
            # Top tier hardware - no compromises
            settings.update({
                'low_vram_mode': False,
                'max_batch_size': 4,
                'resolution': 1024,
                'max_num_views': 12,
                'enable_tex': True,  # Will use CPU fallback rasterizer on MPS (slower)
                'enable_flashvdm': True,
                'compile': False,  # torch.compile doesn't support MPS yet
                'device': 'mps',
                'recommendations': [
                    '🚀 Ultra Performance Mode',
                    'Full quality, no VRAM limitations',
                    'Maximum batch processing enabled',
                    'Shape generation optimized for M3 Max',
                    '⚠️  Texture generation uses CPU fallback (slower than CUDA)'
                ]
            })
        elif memory_tier in ['high', 'ultra'] and self.chip_tier in ['ultra', 'pro']:
            # High-end hardware
            settings.update({
                'low_vram_mode': False,
                'max_batch_size': 2,
                'resolution': 768,
                'max_num_views': 8,
                'enable_tex': True,  # Will use CPU fallback rasterizer on MPS (slower)
                'enable_flashvdm': True,
                'compile': False,
                'device': 'mps',
                'recommendations': [
                    '⚡ High Performance Mode',
                    'Full features enabled',
                    'Optimal quality/speed balance'
                ]
            })
        elif memory_tier == 'medium':
            # Mid-range hardware
            settings.update({
                'low_vram_mode': True,
                'max_batch_size': 1,
                'resolution': 512,
                'max_num_views': 6,
                'enable_tex': True,  # Will use CPU fallback rasterizer on MPS (slower)
                'enable_flashvdm': False,
                'compile': False,
                'device': 'mps',
                'recommendations': [
                    '⚖️ Balanced Mode',
                    'Memory-optimized settings',
                    'Good quality with stable performance'
                ]
            })
        else:
            # Conservative settings for lower-end hardware
            settings.update({
                'low_vram_mode': True,
                'max_batch_size': 1,
                'resolution': 512,
                'max_num_views': 4,
                'enable_tex': False,
                'enable_flashvdm': False,
                'compile': False,
                'device': 'mps',
                'recommendations': [
                    '🔋 Conservative Mode',
                    'Memory-saving enabled',
                    'Optimized for stability'
                ]
            })
        
        return settings
    
    def print_summary(self):
        """Print hardware summary and recommendations"""
        settings = self.get_recommended_settings()
        
        print("=" * 60)
        print("🖥️  HARDWARE DETECTION REPORT")
        print("=" * 60)
        print(f"📱 Chip: {settings['chip']}")
        print(f"💾 Memory: {settings['memory_gb']} GB")
        if settings['gpu_cores']:
            print(f"🎮 GPU Cores: {settings['gpu_cores']}")
        print(f"📊 Performance Tier: {settings['chip_tier'].upper()}")
        print(f"🗄️ Memory Tier: {settings['memory_tier'].upper()}")
        print()
        print("🎯 RECOMMENDED SETTINGS:")
        print("-" * 40)
        for rec in settings['recommendations']:
            print(f"  {rec}")
        print()
        print("⚙️ CONFIGURATION:")
        print(f"  • Low VRAM Mode: {'❌ Disabled' if not settings['low_vram_mode'] else '✅ Enabled'}")
        print(f"  • Max Batch Size: {settings['max_batch_size']}")
        print(f"  • Resolution: {settings['resolution']}px")
        print(f"  • Max Views: {settings['max_num_views']}")
        print(f"  • Texture Generation: {'✅ Enabled' if settings['enable_tex'] else '❌ Disabled'}")
        print(f"  • Flash VDM: {'✅ Enabled' if settings['enable_flashvdm'] else '❌ Disabled'}")
        
        # Special note for torch.compile on MPS
        if settings['chip_tier'] in ['ultra', 'pro', 'standard'] and not settings['compile']:
            print(f"  • Torch Compile: ❌ Disabled (MPS not yet supported)")
        else:
            print(f"  • Torch Compile: {'✅ Enabled' if settings['compile'] else '❌ Disabled'}")
        print("=" * 60)
        
        return settings

if __name__ == "__main__":
    detector = HardwareDetector()
    settings = detector.print_summary()
    
    # Save settings to JSON for other scripts to use
    with open('hardware_config.json', 'w') as f:
        json.dump(settings, f, indent=2)
    print("\n💾 Settings saved to hardware_config.json")