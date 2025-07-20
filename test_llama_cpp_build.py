#!/usr/bin/env python3
"""
Test script for the llama_cpp_build_args() function
"""

from inferenceutils import (
    systeminfo,
    llama_cpp_build_args,
    get_llama_cpp_install_command,
    get_llama_cpp_install_command_windows,
)
import platform


def main():
    print("=== LLaMA.cpp Build Arguments Generator ===\n")

    # Get system info
    print("Detecting hardware...")
    hw_info = systeminfo()

    print(f"OS: {hw_info.os.platform} {hw_info.os.version} ({hw_info.os.architecture})")
    print(f"CPU: {hw_info.cpu.brand_raw}")
    print(f"  Cores: {hw_info.cpu.physical_cores} physical, {hw_info.cpu.logical_cores} logical")
    print(f"  Instruction Sets: {hw_info.cpu.instruction_sets}")
    print(f"RAM: {hw_info.ram.total_gb} GB")

    if hw_info.gpu.detected_vendor:
        print(f"GPU: {hw_info.gpu.detected_vendor}")
        if hw_info.gpu.nvidia:
            for gpu in hw_info.gpu.nvidia:
                print(f"  NVIDIA: {gpu.model} (Compute: {gpu.compute_capability})")
        elif hw_info.gpu.apple:
            print(f"  Apple: {hw_info.gpu.apple.model} (Metal: {hw_info.gpu.apple.metal_supported})")
        elif hw_info.gpu.amd:
            for gpu in hw_info.gpu.amd:
                print(f"  AMD: {gpu.model} (ROCm: {gpu.rocm_compatible})")
        elif hw_info.gpu.intel:
            for accel in hw_info.gpu.intel:
                print(f"  Intel: {accel.model} ({accel.type})")

    if hw_info.gpu.vulkan_api_version:
        print(f"Vulkan API: {hw_info.gpu.vulkan_api_version}")

    print("\n" + "=" * 50)

    # Get optimal build arguments
    print("\nGenerating optimal LLaMA.cpp build arguments...")
    build_args = llama_cpp_build_args()

    if build_args:
        print(f"\nOptimal CMAKE arguments:")
        for arg in build_args:
            print(f"  {arg}")

        print(f"\nComplete CMAKE_ARGS string:")
        print(f"  {' '.join(build_args)}")

        # Show install command
        print(f"\nInstall command for {platform.system()}:")
        if platform.system() == "Windows":
            install_cmd = get_llama_cpp_install_command_windows()
        else:
            install_cmd = get_llama_cpp_install_command()

        print(f"  {install_cmd}")

        # Explain what each argument does
        print(f"\nExplanation of build arguments:")
        for arg in build_args:
            if arg == "-DGGML_METAL=ON":
                print(f"  {arg}: Enable Apple Metal GPU acceleration")
            elif arg == "-DGGML_CUDA=ON":
                print(f"  {arg}: Enable NVIDIA CUDA GPU acceleration")
            elif arg == "-DGGML_HIPBLAS=ON":
                print(f"  {arg}: Enable AMD ROCm/HIP GPU acceleration")
            elif arg == "-DGGML_SYCL=ON":
                print(f"  {arg}: Enable Intel SYCL for GPU/NPU acceleration")
            elif arg == "-DGGML_VULKAN=ON":
                print(f"  {arg}: Enable Vulkan GPU acceleration")
            elif arg == "-DGGML_BLAS=ON":
                print(f"  {arg}: Enable BLAS acceleration for CPU")
            elif arg == "-DGGML_BLAS_VENDOR=Accelerate":
                print(f"  {arg}: Use Apple's Accelerate framework for BLAS")
            elif arg == "-DGGML_BLAS_VENDOR=OpenBLAS":
                print(f"  {arg}: Use OpenBLAS for CPU acceleration")
            elif arg == "-DGGML_BLAS_VENDOR=Intel10_64lp":
                print(f"  {arg}: Use Intel oneMKL for AVX-512 optimization")
            elif arg == "-DGGML_OPENMP=ON":
                print(f"  {arg}: Enable OpenMP for multi-threading")
            elif arg == "-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON":
                print(f"  {arg}: Enable CUDA unified memory for large models")
            elif arg == "-DGGML_CPU_KLEIDIAI=ON":
                print(f"  {arg}: Enable Arm KleidiAI for ARM CPU optimization")
            elif arg.startswith("-DCMAKE_CUDA_ARCHITECTURES="):
                arch = arg.split("=")[1]
                print(f"  {arg}: Target CUDA compute capability {arch}")
            elif arg in ["-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"]:
                print(f"  {arg}: Use Intel oneAPI compiler for optimization")
            elif arg == "-DGGML_NATIVE=ON":
                print(f"  {arg}: Enable native CPU optimizations")
            else:
                print(f"  {arg}: Custom build option")
    else:
        print("No specific optimizations detected. Using default build.")
        print("Install command: pip install llama-cpp-python")

    print(f"\n" + "=" * 50)
    print("Note: Make sure you have the required dependencies installed")
    print("before running the install command (e.g., CUDA toolkit for NVIDIA GPUs)")


if __name__ == "__main__":
    main()
