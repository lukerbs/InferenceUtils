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
)

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
    
    # Backwards compatibility
    "systeminfo",
    "optimal_inference_engine",
]

