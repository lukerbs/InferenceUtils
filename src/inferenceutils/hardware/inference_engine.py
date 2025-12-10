#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Engine Recommendation Module

Provides a simple interface to get optimal inference engine recommendations
with dependencies as a validated Pydantic BaseModel.
"""

from typing import List
from .system_info import system_info
from .engine_selector import Recommender
from .hardware_schema import OptimalInferenceEngine


def recommended_engine() -> OptimalInferenceEngine:
    """
    Get optimal inference engine recommendation based on detected hardware.
    
    Returns:
        OptimalInferenceEngine: Engine name, pip dependencies, and reasoning
        
    Example:
        >>> from inferenceutils.hardware import recommended_engine
        >>> engine = recommended_engine()
        >>> print(f"Use: {engine.name}")
        >>> print(f"Install: pip install {' '.join(engine.dependencies)}")
    """
    # Get hardware information
    hw_info = system_info()
    
    # Get engine recommendation
    recommender = Recommender()
    recommendation = recommender.recommend(hw_info.model_dump())
    
    # Map engine names to their dependencies
    engine_dependencies = {
        "TensorRT-LLM": ["tensorrt-llm"],
        "vLLM": ["vllm"],
        "MLX": ["mlx-lm"],
        "OpenVINO": ["openvino"],
        "llama.cpp": ["llama-cpp-python"]
    }
    
    # Get dependencies for the recommended engine
    engine_name = recommendation["name"]
    dependencies = engine_dependencies.get(engine_name, [engine_name.lower()])
    
    # Create and return the structured object
    return OptimalInferenceEngine(
        name=engine_name,
        dependencies=dependencies,
        reason=recommendation["reason"]
    ) 