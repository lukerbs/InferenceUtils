#!/usr/bin/env python3
"""
Test script for the system_info() function
"""

from inferenceutils import system_info


def main():
    print("Getting system information...")

    # Get system info as Pydantic BaseModel
    info = system_info()

    print(f"\n=== System Information ===")
    print(f"OS: {info.os.platform} {info.os.version} ({info.os.architecture})")
    print(f"Python: {info.python_version}")
    print(f"CPU: {info.cpu.brand_raw}")
    print(f"  Cores: {info.cpu.physical_cores} physical, {info.cpu.logical_cores} logical")
    print(f"  Instruction Sets: {info.cpu.instruction_sets}")
    print(f"RAM: {info.ram.total_gb} GB total, {info.ram.available_gb} GB available")
    print(f"Storage: {info.storage.primary_type}")

    # GPU information
    if info.gpu.detected_vendor:
        print(f"GPU Vendor: {info.gpu.detected_vendor}")

        if info.gpu.nvidia:
            for gpu in info.gpu.nvidia:
                print(f"  NVIDIA: {gpu.model} ({gpu.vram_gb} GB VRAM)")
        elif info.gpu.amd:
            for gpu in info.gpu.amd:
                print(f"  AMD: {gpu.model} ({gpu.vram_gb} GB VRAM)")
        elif info.gpu.intel:
            for accel in info.gpu.intel:
                print(f"  Intel: {accel.model} ({accel.type})")
        elif info.gpu.apple:
            apple_gpu = info.gpu.apple
            print(f"  Apple: {apple_gpu.model} ({apple_gpu.vram_gb} GB unified memory)")
    else:
        print("GPU: No dedicated GPU detected")

    # NPU information
    if info.npus:
        print("NPUs:")
        for npu in info.npus:
            print(f"  {npu.vendor}: {npu.model_name}")
            if npu.npu_cores:
                print(f"    Cores: {npu.npu_cores}")

    # Vulkan support
    if info.gpu.vulkan_api_version:
        print(f"Vulkan API: {info.gpu.vulkan_api_version}")

    # Engine recommendation
    if info.recommended_engine:
        print(f"\n=== Engine Recommendation ===")
        print(f"Recommended: {info.recommended_engine.name}")
        print(f"Reason: {info.recommended_engine.reason}")

    # Demonstrate Pydantic features
    print(f"\n=== Pydantic Features ===")
    print(f"Type: {type(info)}")
    print(f"Validated: {info.model_validate(info.model_dump())}")

    # JSON serialization
    json_data = info.model_dump_json(indent=2)
    print(f"\n=== JSON Output (first 500 chars) ===")
    print(json_data[:500] + "..." if len(json_data) > 500 else json_data)


if __name__ == "__main__":
    main()
