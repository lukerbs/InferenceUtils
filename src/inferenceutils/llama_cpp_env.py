#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA.cpp Build Arguments Generator

This module provides intelligent hardware-aware build argument generation for
llama-cpp-python, the Python bindings for llama.cpp. The primary goal is to
automatically determine the optimal CMAKE build configuration that maximizes
hardware acceleration and performance for LLM inference on the user's system.

Key Purpose:
-----------
The llama_cpp_build_args() function solves a common problem: users often install
llama-cpp-python with default settings, missing out on significant performance
gains from hardware acceleration. This utility automatically detects available
hardware capabilities and generates the appropriate CMAKE arguments to enable
the most optimal backend for their specific system.

Hardware Detection & Optimization:
---------------------------------
- **Apple Silicon Macs**: Enables Metal GPU acceleration and Accelerate framework BLAS
- **NVIDIA GPUs**: Enables CUDA with automatic compute capability detection
- **AMD GPUs**: Enables ROCm/HIP acceleration for AMD graphics cards
- **Intel GPUs/NPUs**: Enables SYCL with Intel oneAPI for Intel accelerators
- **Vulkan**: Provides cross-platform GPU acceleration as fallback
- **CPU Optimizations**: Intel oneMKL (AVX-512), OpenBLAS (AVX2), OpenMP (multi-core)
- **ARM CPUs**: KleidiAI acceleration for ARM processors with AI features

Use Cases:
----------
1. **Development Setup**: Automatically configure optimal build for development machines
2. **CI/CD Pipelines**: Generate hardware-specific build configurations for different runners
3. **User Documentation**: Provide users with exact install commands for their hardware
4. **Performance Optimization**: Ensure users get maximum inference speed from their hardware
5. **Cross-Platform Deployment**: Handle different OS-specific build requirements

Example Usage:
--------------
>>> from inferenceutils import llama_cpp_build_args, get_llama_cpp_install_command
>>>
>>> # Get optimal build arguments for current system
>>> args = llama_cpp_build_args()
>>> print(args)
['-DGGML_METAL=ON', '-DGGML_BLAS=ON', '-DGGML_BLAS_VENDOR=Accelerate']
>>>
>>> # Get complete install command
>>> cmd = get_llama_cpp_install_command()
>>> print(cmd)
CMAKE_ARGS="-DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate" pip install llama-cpp-python

Performance Impact:
------------------
Using optimized build arguments can provide:
- 2-10x faster inference on GPU-accelerated systems
- 20-50% faster CPU inference with optimized BLAS libraries
- Better memory utilization with unified memory support
- Reduced latency for real-time applications

This utility ensures users don't miss out on these performance gains by providing
the exact build configuration needed for their specific hardware setup.
"""

from typing import List, Dict, Any
from .system_info import systeminfo


def llama_cpp_build_args() -> List[str]:
    """
    Generate optimal CMAKE build arguments for llama-cpp-python based on detected hardware.

    This function uses systeminfo() to detect hardware capabilities and returns
    the most optimal CMAKE arguments for building llama-cpp-python with maximum
    hardware acceleration.

    Returns:
        List[str]: List of CMAKE arguments for optimal hardware acceleration

    Example:
        >>> args = llama_cpp_build_args()
        >>> print(" ".join(args))
        -DGGML_METAL=ON -DGGML_SVE=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate
    """
    hw_info = systeminfo()
    build_args = []

    # --- Step 1: Prioritize the Primary GPU Backend ---
    gpu_backend_selected = False

    # Apple Silicon (Highest Priority on macOS)
    if hw_info.os.platform == "Darwin" and hw_info.os.architecture == "arm64":
        print("✅ Apple Silicon detected. Enabling Metal backend.")
        build_args.append("-DGGML_METAL=ON")
        # CRITICAL: Prevents build from hanging on Apple Silicon
        build_args.append("-DGGML_SVE=OFF")
        gpu_backend_selected = True

    # NVIDIA CUDA
    elif hw_info.gpu.detected_vendor == "NVIDIA":
        print("✅ NVIDIA GPU detected. Enabling CUDA backend.")
        build_args.append("-DGGML_CUDA=ON")
        if hw_info.gpu.nvidia:
            # Join all unique compute capabilities with a semicolon for multi-GPU systems
            all_archs = {str(int(gpu.compute_capability * 10)) for gpu in hw_info.gpu.nvidia if gpu.compute_capability}
            if all_archs:
                build_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(sorted(list(all_archs)))}")
        # Unified Memory is a CUDA-specific feature for Linux
        if hw_info.os.platform == "Linux" and hw_info.ram.total_gb and hw_info.ram.total_gb >= 32:
            build_args.append("-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON")
        gpu_backend_selected = True

    # AMD ROCm / HIP
    elif hw_info.gpu.detected_vendor == "AMD":
        print("✅ AMD GPU detected. Enabling HIP/ROCm backend.")
        build_args.append("-DGGML_HIPBLAS=ON")
        gpu_backend_selected = True

    # Intel SYCL
    elif hw_info.gpu.detected_vendor == "Intel":
        print("✅ Intel Accelerator detected. Enabling SYCL backend.")
        build_args.append("-DGGML_SYCL=ON")
        # oneAPI compilers are required for SYCL
        build_args.extend(["-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"])
        gpu_backend_selected = True

    # Vulkan as a fallback
    elif hw_info.gpu.vulkan_api_version:
        print("✅ Vulkan SDK detected. Enabling Vulkan backend as fallback.")
        build_args.append("-DGGML_VULKAN=ON")
        gpu_backend_selected = True

    # --- Step 2: Add CPU BLAS Optimizations ---
    # This complements any GPU backend by accelerating prompt processing.
    if hw_info.os.platform == "Darwin":
        # macOS has its own optimized BLAS library: Accelerate
        print("✅ macOS detected. Enabling Accelerate framework for BLAS.")
        build_args.extend(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=Accelerate"])
    else:
        # For Linux/Windows, prioritize based on CPU features
        if hw_info.cpu.instruction_sets:
            if any(inst in hw_info.cpu.instruction_sets for inst in ["avx512f", "avx512bw"]):
                print("✅ AVX-512 CPU detected. Enabling oneMKL for BLAS.")
                build_args.extend(
                    [
                        "-DGGML_BLAS=ON",
                        "-DGGML_BLAS_VENDOR=Intel10_64lp",
                        "-DCMAKE_C_COMPILER=icx",
                        "-DCMAKE_CXX_COMPILER=icpx",
                        "-DGGML_NATIVE=ON",
                    ]
                )
            elif "avx2" in hw_info.cpu.instruction_sets:
                print("✅ AVX2 CPU detected. Enabling OpenBLAS for BLAS.")
                build_args.extend(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"])

    # --- Step 3: Add General CPU Optimizations ---
    # KleidiAI for ARM CPUs (e.g., Windows on ARM, Linux servers)
    if hw_info.os.architecture == "arm64" and "dotprod" in (hw_info.cpu.instruction_sets or []):
        print("✅ ARM CPU with dot-product support detected. Enabling KleidiAI.")
        build_args.append("-DGGML_CPU_KLEIDIAI=ON")

    # OpenMP for multi-core parallelism, good for all systems
    if hw_info.cpu.logical_cores and hw_info.cpu.logical_cores >= 4:
        build_args.append("-DGGML_OPENMP=ON")

    # Return the unique set of arguments
    return sorted(list(set(build_args)))


def get_llama_cpp_install_command() -> str:
    """
    Generate the complete pip install command for llama-cpp-python with optimal build args.

    Returns:
        str: Complete pip install command with CMAKE_ARGS

    Example:
        >>> cmd = get_llama_cpp_install_command()
        >>> print(cmd)
        CMAKE_ARGS="-DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate" pip install llama-cpp-python
    """
    build_args = llama_cpp_build_args()

    if not build_args:
        return "pip install llama-cpp-python"

    # Join build args into a single string
    cmake_args = " ".join(build_args)

    # Return the complete install command
    return f'CMAKE_ARGS="{cmake_args}" pip install llama-cpp-python'


def get_llama_cpp_install_command_windows() -> str:
    """
    Generate the complete pip install command for llama-cpp-python on Windows.

    Returns:
        str: Complete pip install command with CMAKE_ARGS for Windows

    Example:
        >>> cmd = get_llama_cpp_install_command_windows()
        >>> print(cmd)
        $env:CMAKE_ARGS = "-DGGML_CUDA=ON"; pip install llama-cpp-python
    """
    build_args = llama_cpp_build_args()

    if not build_args:
        return "pip install llama-cpp-python"

    # Join build args into a single string
    cmake_args = " ".join(build_args)

    # Return the Windows PowerShell command
    return f'$env:CMAKE_ARGS = "{cmake_args}"; pip install llama-cpp-python'
