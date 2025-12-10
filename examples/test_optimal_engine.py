#!/usr/bin/env python3
"""
Test script for the recommended_engine() function
"""

from edgekit import recommended_engine, system_info


def main():
    print("=== Recommended Inference Engine ===\n")

    # Get system info for context
    print("Detecting hardware...")
    hw_info = system_info()

    print(f"OS: {hw_info.os.platform} {hw_info.os.version} ({hw_info.os.architecture})")
    print(f"CPU: {hw_info.cpu.brand_raw}")
    if hw_info.gpu.detected_vendor:
        print(f"GPU: {hw_info.gpu.detected_vendor}")

    print("\n" + "=" * 50)

    # Get recommended inference engine
    print("\nGetting recommended inference engine...")
    engine = recommended_engine()

    print(f"\n=== Recommendation ===")
    print(f"Engine: {engine.name}")
    print(f"Dependencies: {engine.dependencies}")
    print(f"Reason: {engine.reason}")

    # Show installation command
    print(f"\n=== Installation ===")
    if len(engine.dependencies) == 1:
        print(f"Install with: pip install {engine.dependencies[0]}")
    else:
        print(f"Install with: pip install {' '.join(engine.dependencies)}")

    # Special handling for llama.cpp
    if engine.name == "llama.cpp":
        print("\nNote: For optimal llama.cpp performance, consider using the build arguments:")
        print("from edgekit import llama_cpp_args")
        print("args = llama_cpp_args()")
        print("print(' '.join(args))")

    # Demonstrate Pydantic features
    print(f"\n=== Pydantic Features ===")
    print(f"Type: {type(engine)}")
    print(f"Validated: {engine.model_validate(engine.model_dump())}")

    # JSON serialization
    json_data = engine.model_dump_json(indent=2)
    print(f"\n=== JSON Output ===")
    print(json_data)


if __name__ == "__main__":
    main()
