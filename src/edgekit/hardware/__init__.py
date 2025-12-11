"""
Hardware detection and inference engine recommendations.

Provides system hardware information and optimal inference engine selection.
"""

from .system_info import system_info
from .inference_engine import recommended_engine
from .hardware_schema import (
    HardwareProfile,
    OptimalInferenceEngine,
    CPU,
    GPU,
    RAM,
    Storage,
    # Platform-specific memory details
    MacOSMemoryDetails,
    WindowsMemoryDetails,
    LinuxMemoryDetails,
    # GPU schemas with new quirk fields
    NVIDIAGPU,
    AMDGPU,
    IntelAccelerator,
    AppleGPU,
)

# Platform-specific memory detection (for advanced usage)
# These imports are safe on all platforms - the functions handle platform checks internally
from .macos_memory import get_macos_memory_stats
from .linux_memory import get_linux_memory_stats
from .windows_memory import get_windows_memory_stats

# Backwards compatibility
systeminfo = system_info
optimal_inference_engine = recommended_engine

__all__ = [
    # New API
    "system_info",
    "recommended_engine",
    
    # Schemas
    "HardwareProfile",
    "OptimalInferenceEngine",
    "CPU",
    "GPU",
    "RAM",
    "Storage",
    
    # Platform-specific memory detail schemas
    "MacOSMemoryDetails",
    "WindowsMemoryDetails",
    "LinuxMemoryDetails",
    
    # GPU schemas (with quirk detection fields)
    "NVIDIAGPU",
    "AMDGPU",
    "IntelAccelerator",
    "AppleGPU",
    
    # Platform-specific memory detection functions
    "get_macos_memory_stats",
    "get_linux_memory_stats",
    "get_windows_memory_stats",
    
    # Backwards compatibility
    "systeminfo",
    "optimal_inference_engine",
]

