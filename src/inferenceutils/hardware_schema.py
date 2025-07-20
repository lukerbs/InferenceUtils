#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Schema Definitions

Pydantic BaseModel schemas that define the output structure of the 
HardwareInspector.inspect_all() method from engine_selector.py.

This module provides type-safe schemas for hardware inspection results,
enabling validation, serialization, and IDE support for hardware data.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class OperatingSystem(BaseModel):
    """Operating system information."""
    platform: Optional[str] = Field(None, description="Operating system platform (e.g., 'Linux', 'Windows', 'Darwin')")
    version: Optional[str] = Field(None, description="OS version/release")
    architecture: Optional[str] = Field(None, description="Machine architecture (e.g., 'x86_64', 'arm64')")


class CPU(BaseModel):
    """CPU information and capabilities."""
    brand_raw: Optional[str] = Field(None, description="Raw CPU brand string from hardware")
    arch: Optional[str] = Field(None, description="CPU architecture string")
    physical_cores: Optional[int] = Field(None, description="Number of physical CPU cores")
    logical_cores: Optional[int] = Field(None, description="Number of logical CPU cores")
    instruction_sets: Optional[List[str]] = Field(None, description="List of supported instruction sets (e.g., ['avx2', 'avx512f', 'neon'])")


class RAM(BaseModel):
    """Random Access Memory information."""
    total_gb: Optional[float] = Field(None, description="Total RAM in gigabytes")
    available_gb: Optional[float] = Field(None, description="Available RAM in gigabytes")


class Storage(BaseModel):
    """Storage device information."""
    primary_type: Optional[str] = Field(None, description="Primary storage type ('SSD/NVMe', 'HDD', 'Unknown')")


class NVIDIAGPU(BaseModel):
    """NVIDIA GPU information."""
    model: str = Field(..., description="GPU model name")
    vram_gb: float = Field(..., description="VRAM size in gigabytes")
    driver_version: str = Field(..., description="NVIDIA driver version")
    cuda_version: Optional[str] = Field(None, description="CUDA version supported by driver")
    compute_capability: Optional[float] = Field(None, description="Compute capability (e.g., 8.9 for RTX 4090)")
    cuda_cores: Optional[int] = Field(None, description="Number of CUDA cores")
    tensor_cores: Optional[int] = Field(None, description="Number of Tensor cores")


class AMDGPU(BaseModel):
    """AMD GPU information."""
    model: Optional[str] = Field(None, description="GPU model name")
    vram_gb: Optional[float] = Field(None, description="VRAM size in gigabytes")
    driver_version: Optional[str] = Field(None, description="AMD driver version")
    rocm_compatible: bool = Field(..., description="Whether GPU is ROCm compatible")
    compute_units: Optional[int] = Field(None, description="Number of compute units")


class IntelAccelerator(BaseModel):
    """Intel GPU or NPU information."""
    model: str = Field(..., description="Device model name")
    type: str = Field(..., description="Device type ('dGPU', 'iGPU', 'NPU')")
    execution_units: Optional[int] = Field(None, description="Number of execution units (GPU only)")
    vram_gb: Optional[float] = Field(None, description="VRAM size in gigabytes (GPU only)")
    driver_version: Optional[str] = Field(None, description="Driver version (GPU only)")


class AppleGPU(BaseModel):
    """Apple Silicon GPU information."""
    model: Optional[str] = Field(None, description="GPU model/architecture")
    vram_gb: Optional[float] = Field(None, description="Unified memory size in gigabytes")
    gpu_cores: Optional[int] = Field(None, description="Number of GPU cores")
    metal_supported: bool = Field(..., description="Whether Metal framework is supported")


class GPU(BaseModel):
    """Comprehensive GPU information across all vendors."""
    detected_vendor: Optional[str] = Field(None, description="Detected GPU vendor ('NVIDIA', 'AMD', 'Intel', 'Apple')")
    vulkan_api_version: Optional[str] = Field(None, description="Vulkan API version if available")
    nvidia: Optional[List[NVIDIAGPU]] = Field(None, description="List of NVIDIA GPUs")
    amd: Optional[List[AMDGPU]] = Field(None, description="List of AMD GPUs")
    intel: Optional[List[IntelAccelerator]] = Field(None, description="List of Intel accelerators (GPUs/NPUs)")
    apple: Optional[AppleGPU] = Field(None, description="Apple Silicon GPU information")


class NPU(BaseModel):
    """Neural Processing Unit information."""
    vendor: str = Field(..., description="NPU vendor ('Intel', 'Apple', 'AMD')")
    model_name: str = Field(..., description="NPU model name")
    npu_cores: Optional[int] = Field(None, description="Number of NPU cores")


class EngineRecommendation(BaseModel):
    """LLM inference engine recommendation."""
    name: str = Field(..., description="Recommended engine name")
    reason: str = Field(..., description="Reason for the recommendation")


class OptimalInferenceEngine(BaseModel):
    """Optimal inference engine recommendation with dependencies."""
    name: str = Field(..., description="Name of the recommended inference engine")
    dependencies: List[str] = Field(..., description="List of Python libraries needed to use this inference engine")
    reason: str = Field(..., description="Reason why this inference engine was selected")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        use_enum_values = True
        validate_assignment = True


class HardwareProfile(BaseModel):
    """
    Complete hardware profile from HardwareInspector.inspect_all().
    
    This schema represents the full output structure of the hardware inspection,
    including all detected components and the final engine recommendation.
    """
    os: OperatingSystem = Field(..., description="Operating system information")
    python_version: Optional[str] = Field(None, description="Python version")
    cpu: CPU = Field(..., description="CPU information and capabilities")
    ram: RAM = Field(..., description="Random access memory information")
    storage: Storage = Field(..., description="Storage device information")
    gpu: GPU = Field(..., description="Comprehensive GPU information")
    npus: List[NPU] = Field(default_factory=list, description="List of detected NPUs")
    recommended_engine: Optional[EngineRecommendation] = Field(None, description="Recommended LLM inference engine")

    class Config:
        """Pydantic configuration."""
        # Allow extra fields for future compatibility
        extra = "allow"
        # Use enum values for validation
        use_enum_values = True
        # Validate assignments
        validate_assignment = True


# Convenience types for partial data
class PartialHardwareProfile(BaseModel):
    """Partial hardware profile for cases where not all data is available."""
    os: Optional[OperatingSystem] = None
    python_version: Optional[str] = None
    cpu: Optional[CPU] = None
    ram: Optional[RAM] = None
    storage: Optional[Storage] = None
    gpu: Optional[GPU] = None
    npus: List[NPU] = Field(default_factory=list)
    recommended_engine: Optional[EngineRecommendation] = None

    class Config:
        extra = "allow"
        use_enum_values = True
        validate_assignment = True


# Type aliases for common use cases
HardwareInfo = HardwareProfile
PartialHardwareInfo = PartialHardwareProfile


def validate_hardware_data(data: Dict[str, Any]) -> HardwareProfile:
    """
    Validate hardware inspection data against the schema.
    
    Args:
        data: Dictionary containing hardware inspection results
        
    Returns:
        Validated HardwareProfile instance
        
    Raises:
        ValidationError: If data doesn't match the schema
    """
    return HardwareProfile(**data)


def create_hardware_profile(**kwargs) -> HardwareProfile:
    """
    Create a HardwareProfile instance with the given data.
    
    Args:
        **kwargs: Hardware profile data
        
    Returns:
        HardwareProfile instance
    """
    return HardwareProfile(**kwargs) 