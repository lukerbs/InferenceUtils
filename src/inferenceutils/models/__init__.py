"""
Model preflight checks and memory validation.

Validates if a model will fit in memory before loading.
"""

from .memory_validator import (
    model_preflight,
    can_load,
    PreflightResult,
    PreflightStatus,
    Engine,
)

from .model_inspector import ModelMetadata
from .memory_estimator import MemoryEstimate

__all__ = [
    # Primary API
    "model_preflight",
    "can_load", 
    "PreflightResult",
    "PreflightStatus",
    "Engine",
    
    # Advanced types
    "ModelMetadata",
    "MemoryEstimate",
]
