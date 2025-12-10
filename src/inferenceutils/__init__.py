"""
InferenceUtils - Hardware detection and model validation for LLM inference.

Submodules:
    - inferenceutils.hardware: System hardware detection
    - inferenceutils.models: Model preflight checks
    - inferenceutils.build: Build configuration utilities
"""

# Import submodules for namespace access (iu.hardware.system_info())
from . import hardware
from . import models
from . import build

# Top-level convenience exports (most common operations)
from .hardware import system_info, recommended_engine
from .models import model_preflight, can_load, PreflightResult, Engine
from .build import llama_cpp_args, install_command

__version__ = "0.2.0"

__all__ = [
    # Submodules
    "hardware",
    "models", 
    "build",
    
    # Primary API
    "system_info",
    "recommended_engine",
    "model_preflight",
    "can_load",
    "PreflightResult",
    "Engine",
    "llama_cpp_args",
    "install_command",
]
