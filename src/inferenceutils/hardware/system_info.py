#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Information Module

Provides a simple interface to get system hardware information
as a validated Pydantic BaseModel.
"""

from typing import Dict, Any
from .engine_selector import HardwareInspector
from .hardware_schema import HardwareProfile


def system_info() -> HardwareProfile:
    """
    Get comprehensive system hardware information as a validated Pydantic BaseModel.
    
    This function uses HardwareInspector.inspect_all() to gather hardware data
    and returns it as a validated HardwareProfile instance.
    
    Returns:
        HardwareProfile: A validated Pydantic BaseModel containing all hardware information
        
    Example:
        >>> from inferenceutils.hardware import system_info
        >>> hw = system_info()
        >>> print(f"OS: {hw.os.platform}")
        >>> print(f"CPU: {hw.cpu.brand_raw}")
        >>> print(f"RAM: {hw.ram.total_gb} GB")
    """
    # Create hardware inspector and gather data
    inspector = HardwareInspector()
    hardware_data: Dict[str, Any] = inspector.inspect_all()
    
    # Validate and return as Pydantic BaseModel
    return HardwareProfile(**hardware_data)
