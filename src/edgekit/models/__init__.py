"""
Model preflight checks and memory validation.

Validates if a model will fit in memory before loading.
Includes lightweight remote inspection for bandwidth-efficient preflight checks.
"""

from .memory_validator import (
    model_preflight,
    can_load,
    PreflightResult,
    PreflightReason,
    Engine,
)
from .exceptions import PreflightValidationError

from .model_inspector import ModelMetadata
from .model_inspector_remote import (
    RemoteInspectError,
    inspect_model_remote,
)
from .memory_estimator import MemoryEstimate

__all__ = [
    # Primary API
    "model_preflight",
    "can_load", 
    "PreflightResult",
    "PreflightReason",
    "Engine",
    
    # Advanced types
    "ModelMetadata",
    "MemoryEstimate",
    
    # Remote inspection (for advanced usage)
    "RemoteInspectError",
    "inspect_model_remote",
    
    # Exceptions
    "PreflightValidationError",
]
