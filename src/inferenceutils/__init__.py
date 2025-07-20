"""
InferenceUtils - Hardware Inspection with Pydantic Schemas

A comprehensive hardware inspection library with type-safe Pydantic schemas
for LLM inference engine recommendations.
"""

from .hardware_schema import HardwareProfile, OptimalInferenceEngine
from .system_info import systeminfo
from .llama_cpp_env import llama_cpp_build_args
from .inference_engine import optimal_inference_engine

__version__ = "0.1.0"
__all__ = [
    "HardwareProfile",
    "OptimalInferenceEngine",
    "systeminfo",
    "optimal_inference_engine",
    "llama_cpp_build_args",
]
