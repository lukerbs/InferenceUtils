from edgekit import system_info, recommended_engine, model_preflight

# Get system hardware information
hw = system_info()
print(f"CPU: {hw.cpu.brand_raw}")
print(f"RAM: {hw.ram.total_gb} GB")
print(f"GPU: {hw.gpu.detected_vendor}")

# Get engine recommendation
engine = recommended_engine()
print(f"\nRecommended engine: {engine.name}")
print(f"Reason: {engine.reason}")
print(f"Install: pip install {' '.join(engine.dependencies)}")

# Example: Validate if a model will fit in memory
result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx")
if result.can_load:
    print(f"\n✓ Model will fit: {result.recommended_context:,} tokens")
else:
    print(f"\n✗ Model won't fit: {result.message}")
