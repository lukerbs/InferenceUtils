"""
LLaMA.cpp Build Arguments Generator

Generates optimal CMAKE build arguments for llama-cpp-python based on
detected hardware capabilities.
"""

import platform
from typing import List

from ..hardware import system_info


def llama_cpp_args() -> List[str]:
    """
    Generate optimal CMAKE build arguments for llama-cpp-python.
    
    Detects hardware capabilities and returns CMAKE arguments that enable
    the best available acceleration (Metal, CUDA, ROCm, SYCL, etc.).
    
    Returns:
        List[str]: CMAKE arguments for optimal hardware acceleration
        
    Example:
        >>> from inferenceutils.build import llama_cpp_args
        >>> args = llama_cpp_args()
        >>> print(" ".join(args))
        -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate -DGGML_METAL=ON
    """
    hw_info = system_info()
    build_args = []

    # --- GPU Backend Selection ---
    
    # Apple Silicon (Metal)
    if hw_info.os.platform == "Darwin" and hw_info.os.architecture == "arm64":
        build_args.append("-DGGML_METAL=ON")
        build_args.append("-DGGML_SVE=OFF")  # Prevents build hang

    # NVIDIA CUDA
    elif hw_info.gpu.detected_vendor == "NVIDIA":
        build_args.append("-DGGML_CUDA=ON")
        if hw_info.gpu.nvidia:
            all_archs = {str(int(gpu.compute_capability * 10)) 
                        for gpu in hw_info.gpu.nvidia if gpu.compute_capability}
            if all_archs:
                build_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(sorted(list(all_archs)))}")
        if hw_info.os.platform == "Linux" and hw_info.ram.total_gb and hw_info.ram.total_gb >= 32:
            build_args.append("-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON")

    # AMD ROCm / HIP
    elif hw_info.gpu.detected_vendor == "AMD":
        build_args.append("-DGGML_HIPBLAS=ON")

    # Intel SYCL
    elif hw_info.gpu.detected_vendor == "Intel":
        build_args.append("-DGGML_SYCL=ON")
        build_args.extend(["-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"])

    # Vulkan fallback
    elif hw_info.gpu.vulkan_api_version:
        build_args.append("-DGGML_VULKAN=ON")

    # --- CPU BLAS Optimizations ---
    
    if hw_info.os.platform == "Darwin":
        build_args.extend(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=Accelerate"])
    elif hw_info.cpu.instruction_sets:
        if any(inst in hw_info.cpu.instruction_sets for inst in ["avx512f", "avx512bw"]):
            build_args.extend([
                "-DGGML_BLAS=ON",
                "-DGGML_BLAS_VENDOR=Intel10_64lp",
                "-DCMAKE_C_COMPILER=icx",
                "-DCMAKE_CXX_COMPILER=icpx",
                "-DGGML_NATIVE=ON",
            ])
        elif "avx2" in hw_info.cpu.instruction_sets:
            build_args.extend(["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"])

    # --- General CPU Optimizations ---
    
    # KleidiAI for ARM
    if hw_info.os.architecture == "arm64" and "dotprod" in (hw_info.cpu.instruction_sets or []):
        build_args.append("-DGGML_CPU_KLEIDIAI=ON")

    # OpenMP for multi-core
    if hw_info.cpu.logical_cores and hw_info.cpu.logical_cores >= 4:
        build_args.append("-DGGML_OPENMP=ON")

    return sorted(list(set(build_args)))


def install_command(shell: str = "auto") -> str:
    """
    Generate the complete pip install command for llama-cpp-python.
    
    Args:
        shell: Shell type - "bash", "powershell", or "auto" (detect from OS)
        
    Returns:
        str: Complete pip install command with CMAKE_ARGS
        
    Example:
        >>> from inferenceutils.build import install_command
        >>> print(install_command())
        CMAKE_ARGS="-DGGML_METAL=ON ..." pip install llama-cpp-python
    """
    args = llama_cpp_args()
    
    if not args:
        return "pip install llama-cpp-python"
    
    cmake_args = " ".join(args)
    
    # Auto-detect shell from OS
    if shell == "auto":
        shell = "powershell" if platform.system() == "Windows" else "bash"
    
    if shell == "powershell":
        return f'$env:CMAKE_ARGS = "{cmake_args}"; pip install llama-cpp-python'
    else:
        return f'CMAKE_ARGS="{cmake_args}" pip install llama-cpp-python'
