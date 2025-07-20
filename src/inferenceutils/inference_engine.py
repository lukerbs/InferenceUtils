#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Engine Recommendation Module

Provides a simple interface to get optimal inference engine recommendations
with dependencies as a validated Pydantic BaseModel.
"""

from typing import List
from .system_info import systeminfo
from .engine_selector import Recommender
from .hardware_schema import OptimalInferenceEngine


def optimal_inference_engine() -> OptimalInferenceEngine:
    """
    Get optimal inference engine recommendation with dependencies as a validated Pydantic BaseModel.
    
    This function uses systeminfo() to detect hardware capabilities and the Recommender
    to determine the best inference engine, then returns a structured object containing
    the engine name, required dependencies, and reasoning.
    
    Returns:
        OptimalInferenceEngine: A validated Pydantic BaseModel containing engine recommendation
        
    Example:
        >>> engine = optimal_inference_engine()
        >>> print(f"Engine: {engine.name}")
        >>> print(f"Dependencies: {engine.dependencies}")
        >>> print(f"Reason: {engine.reason}")
    """
    # Get hardware information
    hw_info = systeminfo()
    
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