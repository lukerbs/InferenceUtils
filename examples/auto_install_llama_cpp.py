#!/usr/bin/env python3
"""
Auto-Install Script for llama-cpp-python

Detects your hardware and generates the optimal installation command.
No more guessing CMAKE flags or debugging slow CPU-only builds.
"""

from edgekit import system_info, install_command

def main():
    print("=" * 70)
    print("llama-cpp-python Auto-Installer")
    print("=" * 70)
    print()
    
    # Detect hardware
    print("🔍 Detecting hardware...")
    hw = system_info()
    
    print(f"\n✓ System: {hw.os.platform} {hw.os.architecture}")
    print(f"✓ CPU: {hw.cpu.brand_raw}")
    print(f"✓ RAM: {hw.ram.total_gb} GB")
    
    if hw.gpu.detected_vendor:
        print(f"✓ GPU: {hw.gpu.detected_vendor}")
        if hw.gpu.apple:
            print(f"  └─ {hw.gpu.apple.model} with {hw.gpu.apple.gpu_cores} cores (Metal)")
        if hw.gpu.nvidia:
            for gpu in hw.gpu.nvidia:
                print(f"  └─ {gpu.model} (CUDA {gpu.cuda_version}, Compute {gpu.compute_capability})")
        if hw.gpu.amd:
            for gpu in hw.gpu.amd:
                rocm_status = "ROCm compatible" if gpu.rocm_compatible else "ROCm not available"
                print(f"  └─ {gpu.model} ({rocm_status})")
    
    print("\n" + "=" * 70)
    print("📦 Optimized Installation Command")
    print("=" * 70)
    print()
    
    # Generate install command
    cmd = install_command(shell="bash")
    
    print("Copy and run this command:\n")
    print(f"  {cmd}")
    print()
    
    # Explain what it's doing
    print("=" * 70)
    print("🔧 What this does:")
    print("=" * 70)
    
    if hw.gpu.apple:
        print("  • Enables Metal acceleration for Apple Silicon")
        print("  • Uses Accelerate framework for optimized BLAS")
        print("  • Configures for 16KB page size (M-series chips)")
    elif hw.gpu.nvidia:
        compute_cap = hw.gpu.nvidia[0].compute_capability
        print(f"  • Enables CUDA acceleration (compute capability {compute_cap})")
        print(f"  • Optimizes for your GPU architecture")
        print("  • Links against cuBLAS for matrix operations")
    elif hw.gpu.amd and any(g.rocm_compatible for g in hw.gpu.amd):
        print("  • Enables ROCm acceleration for AMD GPUs")
        print("  • Configures HIP for GPU compute")
    else:
        print("  • Optimizes for CPU inference")
        if "AVX2" in hw.cpu.instruction_sets or "AVX-512" in hw.cpu.instruction_sets:
            print(f"  • Enables {', '.join(hw.cpu.instruction_sets)} SIMD instructions")
    
    print()
    print("=" * 70)
    print("📝 Note: Without these flags, pip would install a slow CPU-only build")
    print("=" * 70)

if __name__ == "__main__":
    main()
