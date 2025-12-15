#!/usr/bin/env python3
"""
Test script for model_preflight() function
Tests if a specific model can run on the current hardware
"""

from edgekit import model_preflight, system_info

# Model to test
# LLM_MODEL = "mlx-community/gemma-3n-E4B-it-lm-4bit"
LLM_MODEL = "mlx-community/Mistral-Small-24B-Instruct-2501-4bit"
# LLM_MODEL = "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit"

def main():
    print("=== Model Preflight Check ===\n")
    
    # Show current hardware
    print("Detecting hardware...")
    hw = system_info()
    print(f"System: {hw.os.platform} {hw.os.architecture}")
    print(f"CPU: {hw.cpu.brand_raw}")
    print(f"RAM: {hw.ram.total_gb} GB total, {hw.ram.available_gb} GB available")
    
    if hw.gpu.detected_vendor:
        print(f"GPU: {hw.gpu.detected_vendor}")
        if hw.gpu.apple:
            print(f"  Apple GPU: {hw.gpu.apple.model} ({hw.gpu.apple.vram_gb} GB unified memory)")
        if hw.gpu.nvidia:
            for gpu in hw.gpu.nvidia:
                print(f"  NVIDIA: {gpu.model} ({gpu.vram_gb} GB VRAM)")
        if hw.gpu.amd:
            for gpu in hw.gpu.amd:
                print(f"  AMD: {gpu.model} ({gpu.vram_gb} GB VRAM)")
    
    print(f"\n{'='*60}")
    print(f"Testing Model: {LLM_MODEL}")
    print(f"{'='*60}\n")
    
    # Run preflight check
    print("Running preflight check...")
    result = model_preflight(LLM_MODEL, engine="mlx_lm")
    
    print(f"\n=== Preflight Result ===")
    print(f"Status: {'✅ PASSED' if result.status else '❌ FAILED'}")
    print(f"\nMemory Analysis:")
    print(f"  Required: {result.required_gb:.2f} GB")
    print(f"  Available: {result.available_gb:.2f} GB")
    print(f"  Utilization: {result.utilization*100:.1f}%")
    
    print(f"\nContext Window:")
    print(f"  Model Limit: {result.context_limit:,} tokens")
    print(f"  Usable on Device: {result.usable_context:,} tokens")
    if result.usable_context < result.context_limit:
        reduction_pct = (1 - result.usable_context / result.context_limit) * 100
        print(f"  (Context reduced by {reduction_pct:.0f}%)")
    
    if result.estimate:
        print(f"\nDetailed Breakdown:")
        print(f"  Model Weights: {result.estimate.weights_gb:.2f} GB")
        print(f"  KV Cache: {result.estimate.kv_cache_gb:.2f} GB")
        print(f"  Overhead: {result.estimate.overhead_gb:.2f} GB")
        print(f"  Quantization: {result.estimate.quantization}")
        print(f"  Parameters: {result.estimate.params_billions:.1f}B")
    
    print(f"\n{'='*60}")
    print(f"Reason: {result.reason}")
    print(f"{'='*60}\n")
    
    # Final recommendation
    if result.status:
        print("✅ MODEL CAN RUN")
        if result.utilization > 0.70:
            print("💡 Memory utilization >70% - consider closing other apps for best performance")
    else:
        print("❌ MODEL CANNOT RUN - Won't fit on this device")
    
    return 0 if result.status else 1


if __name__ == "__main__":
    exit(main())
