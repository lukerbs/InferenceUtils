#!/usr/bin/env python3
"""
Test script for macOS memory detection using Mach kernel API
"""

import platform
from inferenceutils import system_info


def main():
    print("=== macOS Memory Detection Test ===\n")
    
    if platform.system() != 'Darwin':
        print("⚠️  Skipping: Not running on macOS")
        print("This test is only applicable on macOS systems.")
        return
    
    print("Detecting hardware...")
    hw = system_info()
    
    print(f"\n=== Operating System ===")
    print(f"Platform: {hw.os.platform}")
    print(f"Version: {hw.os.version}")
    print(f"Architecture: {hw.os.architecture}")
    print(f"Python Version: {hw.python_version}")
    
    print(f"\n=== CPU Information ===")
    print(f"Brand: {hw.cpu.brand_raw}")
    print(f"Architecture: {hw.cpu.arch}")
    print(f"Physical Cores: {hw.cpu.physical_cores}")
    print(f"Logical Cores: {hw.cpu.logical_cores}")
    print(f"Instruction Sets: {', '.join(hw.cpu.instruction_sets) if hw.cpu.instruction_sets else 'None detected'}")
    
    print(f"\n=== RAM Information ===")
    print(f"Total RAM: {hw.ram.total_gb} GB")
    print(f"Available RAM: {hw.ram.available_gb} GB")
    print(f"Utilization: {((hw.ram.total_gb - hw.ram.available_gb) / hw.ram.total_gb * 100):.1f}%")
    
    print(f"\n=== Storage ===")
    print(f"Primary Type: {hw.storage.primary_type}")
    
    print(f"\n=== GPU Detection ===")
    print(f"Detected Vendor: {hw.gpu.detected_vendor or 'None'}")
    print(f"Vulkan API Version: {hw.gpu.vulkan_api_version or 'Not available'}")
    
    if hw.gpu.apple:
        print(f"\n=== Apple GPU ===")
        print(f"Model: {hw.gpu.apple.model}")
        print(f"GPU Cores: {hw.gpu.apple.gpu_cores}")
        print(f"Metal Supported: {hw.gpu.apple.metal_supported}")
        print(f"Unified Memory: {hw.gpu.apple.vram_gb} GB")
    
    if hw.gpu.nvidia:
        print(f"\n=== NVIDIA GPU(s) ===")
        for i, gpu in enumerate(hw.gpu.nvidia):
            print(f"GPU {i}: {gpu.model}")
            print(f"  VRAM: {gpu.vram_gb} GB")
            print(f"  Driver: {gpu.driver_version}")
            print(f"  CUDA Version: {gpu.cuda_version or 'N/A'}")
            print(f"  Compute Capability: {gpu.compute_capability or 'N/A'}")
            print(f"  CUDA Cores: {gpu.cuda_cores or 'N/A'}")
            print(f"  Tensor Cores: {gpu.tensor_cores or 'N/A'}")
    
    if hw.gpu.amd:
        print(f"\n=== AMD GPU(s) ===")
        for i, gpu in enumerate(hw.gpu.amd):
            print(f"GPU {i}: {gpu.model}")
            print(f"  VRAM: {gpu.vram_gb} GB")
            print(f"  Driver: {gpu.driver_version}")
            print(f"  ROCm Compatible: {gpu.rocm_compatible}")
            print(f"  Compute Units: {gpu.compute_units or 'N/A'}")
    
    if hw.gpu.intel:
        print(f"\n=== Intel Accelerator(s) ===")
        for i, accel in enumerate(hw.gpu.intel):
            print(f"Device {i}: {accel.model}")
            print(f"  Type: {accel.type}")
            print(f"  Execution Units: {accel.execution_units or 'N/A'}")
            print(f"  VRAM: {accel.vram_gb or 'N/A'} GB")
            print(f"  Driver: {accel.driver_version or 'N/A'}")
    
    if hw.npus:
        print(f"\n=== NPU(s) ===")
        for i, npu in enumerate(hw.npus):
            print(f"NPU {i}: {npu.vendor} - {npu.model_name}")
            print(f"  NPU Cores: {npu.npu_cores or 'N/A'}")
    
    if hw.recommended_engine:
        print(f"\n=== Recommended Inference Engine ===")
        print(f"Engine: {hw.recommended_engine.name}")
        print(f"Reason: {hw.recommended_engine.reason}")
    
    if hw.ram.details:
        print(f"\n=== macOS Memory Details (via Mach API) ===")
        print(f"Page Size: {hw.ram.details.page_size_bytes:,} bytes")
        print(f"App Memory: {hw.ram.details.app_memory_gb:.2f} GB")
        print(f"Wired Memory: {hw.ram.details.wired_gb:.2f} GB")
        print(f"Cached Files: {hw.ram.details.cached_files_gb:.2f} GB")
        print(f"Compressed: {hw.ram.details.compressed_gb:.2f} GB")
        print(f"Speculative: {hw.ram.details.speculative_gb:.2f} GB")
        
        print(f"\n=== Memory Breakdown ===")
        total = hw.ram.total_gb
        print(f"App Memory:    {hw.ram.details.app_memory_gb:6.2f} GB ({hw.ram.details.app_memory_gb/total*100:5.1f}%)")
        print(f"Wired Memory:  {hw.ram.details.wired_gb:6.2f} GB ({hw.ram.details.wired_gb/total*100:5.1f}%)")
        print(f"Compressed:    {hw.ram.details.compressed_gb:6.2f} GB ({hw.ram.details.compressed_gb/total*100:5.1f}%)")
        print(f"Cached Files:  {hw.ram.details.cached_files_gb:6.2f} GB ({hw.ram.details.cached_files_gb/total*100:5.1f}%)")
        print(f"Available:     {hw.ram.available_gb:6.2f} GB ({hw.ram.available_gb/total*100:5.1f}%)")
        
        # Validation checks
        print(f"\n=== Validation Checks ===")
        
        # Check 1: Page size based on architecture
        arch = hw.os.architecture.lower()
        expected_page = 16384 if 'arm' in arch else 4096
        actual_page = hw.ram.details.page_size_bytes
        
        if actual_page == expected_page:
            print(f"✅ Page size correct for {hw.os.architecture}: {actual_page:,} bytes")
        else:
            print(f"❌ Page size mismatch!")
            print(f"   Expected for {hw.os.architecture}: {expected_page:,} bytes")
            print(f"   Actual: {actual_page:,} bytes")
        
        # Check 2: Basic sanity checks
        checks_passed = True
        
        if hw.ram.total_gb <= 0:
            print("❌ Total RAM must be positive")
            checks_passed = False
        else:
            print(f"✅ Total RAM is positive: {hw.ram.total_gb} GB")
        
        if hw.ram.available_gb < 0:
            print("❌ Available RAM cannot be negative")
            checks_passed = False
        else:
            print(f"✅ Available RAM is non-negative: {hw.ram.available_gb} GB")
        
        if hw.ram.available_gb > hw.ram.total_gb:
            print("❌ Available RAM cannot exceed total RAM")
            checks_passed = False
        else:
            print(f"✅ Available RAM ≤ Total RAM")
        
        if hw.ram.details.wired_gb < 0 or hw.ram.details.app_memory_gb < 0:
            print("❌ Memory categories cannot be negative")
            checks_passed = False
        else:
            print(f"✅ All memory categories are non-negative")
        
        # Check 3: Available memory formula verification
        # Available should be roughly: free + speculative + external
        # We can't verify exact equality without access to free_count, but we can check reasonableness
        if hw.ram.available_gb < hw.ram.details.speculative_gb:
            print(f"⚠️  Warning: Available ({hw.ram.available_gb} GB) < Speculative ({hw.ram.details.speculative_gb} GB)")
            print("   This seems unusual but might be valid in edge cases")
        else:
            print(f"✅ Available memory calculation appears reasonable")
        
        print(f"\n=== Overall Result ===")
        if checks_passed and actual_page == expected_page:
            print("✅ All validation checks passed!")
            print("✅ macOS memory detection is working correctly!")
        else:
            print("⚠️  Some validation checks failed. Please review above.")
    else:
        print("\n❌ ERROR: macOS details missing!")
        print("Expected 'details' field in RAM data but got None.")
        print("This indicates the Mach API call may have failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

