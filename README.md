# EdgeKit

> **Stop guessing. Start engineering.**  
> Hardware introspection and memory validation for cross-platform LLM inference

**What it does:**
- Profile your hardware: CPU, GPU, RAM, NPU specs + live memory/VRAM availability across macOS, Windows, and Linux
- Validate if an LLM will fit in memory *before* you load it (prevents OOM crashes)
- Recommend the optimal inference engine for your hardware (MLX, vLLM, llama.cpp, TensorRT-LLM, OpenVINO)
- Calculate safe context window sizes accounting for KV cache growth

One cross-platform API that answers: *What AI hardware do I have? Can I run this model? Should I run this model? Which inference engine is best for my device?*

## 🎯 Designed for Edge AI & Local Deployment

**EdgeKit** is a utility/helper library built with a **primary focus on edge AI applications and local AI models running on consumer-grade hardware**. Whether you're deploying on MacBooks, gaming PCs, Raspberry Pis, or edge devices, this library helps you make the most of what you have—though it's equally useful for profiling and optimizing enterprise-grade GPU hardware (NVIDIA A100/H100, AMD MI300, etc.). The emphasis is on **democratizing local AI**: making it accessible and reliable regardless of whether you're working with a laptop or a server rack.

> **⚠️ Important:** EdgeKit is **NOT** an inference engine. It doesn't run models or generate text. It's infrastructure—the layer that helps you profile your available hardware, choose the right engine (MLX, vLLM, llama.cpp, etc.), validate memory requirements, and optimize build configurations **before** you load any models.

**EdgeKit** solves the "Matrix from Hell" in local AI: the complex interplay between your hardware, operating system, model architecture, and inference engine. It replaces trial and error with engineering rigor.

## The Problem

Building cross-platform LLM applications involves four recurring engineering challenges:

1. **"Can I run this?" / "Should I run this?"** - Will this model fit in RAM? Will it crash during inference when the KV cache fills up? Will it thrash and become unusable even if it technically loads?

2. **The Build Configuration Maze** - Installing `llama-cpp-python` with the right CMAKE flags for your exact hardware (Metal vs CUDA vs ROCm vs CPU-only), turning a simple `pip install` into a 20-minute debugging session.

3. **Hardware Detection Fragility** - Querying hardware specs across macOS/Linux/Windows, Apple Silicon/Intel/NVIDIA/AMD, unified memory vs discrete VRAM—each requiring different APIs and shell commands that break with OS updates.

4. **Inference Engine Selection Guesswork** - Choosing between MLX, vLLM, llama.cpp, TensorRT-LLM, and OpenVINO based on your hardware capabilities and model requirements.

**EdgeKit solves all four** with a single, type-safe interface that validates memory before loading, auto-generates build configurations, detects hardware cross-platform, and recommends optimal engines.

---

## Quick Start

```python
from edgekit.hardware import system_info, recommended_engine
from edgekit.models import model_preflight
from edgekit.build import llama_cpp_args

# Get highly detailed hardware specs and activity monitoring
hw = system_info()
print(f"CPU: {hw.cpu.brand_raw}, RAM: {hw.ram.available_gb} GB")
# Output: CPU: Apple M4 Pro, RAM: 17.0 GB

# Get recommended AI inference engine that is most-optimized for your available hardware
engine = recommended_engine()
print(f"Use {engine.name}: {engine.reason}")
# Output: Use MLX: Apple Silicon detected with Metal support

# Check if a specific model can / should be ran on your available hardware
result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx")
if result.can_load:
    print(f"✓ Safe to load with {result.recommended_context:,} token context")
    # Output: ✓ Safe to load with 131,072 token context
else:
    print(f"✗ Won't fit in RAM: {result.message}")
    # Example: Model requires 21.0 GB but only 15.0 GB available

# Generate optimal llama.cpp build arguments (useful for creating optimized llama.cpp builds for your device)
args = llama_cpp_args()
print(f"CMAKE_ARGS: {' '.join(args)}")
# Output: CMAKE_ARGS: -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate -DGGML_METAL=ON
```

For detailed API documentation, see [`docs/USAGE.md`](docs/USAGE.md). For runnable examples, explore the [`examples/`](examples/) directory.

---

## Key Features

- **Memory Validation** - Graduated warnings (< 70% = safe, 70-85% = tight, > 85% = failed) + safe context limits accounting for KV cache growth. Prevents OOM crashes during inference.

- **Build Optimization** - Auto-generates CMAKE flags for `llama-cpp-python` based on your hardware (Metal, CUDA, ROCm, AVX-512). Turns 20-minute debugging into copy-paste commands.

- **Cross-Platform Hardware API** - Unified interface for CPU, GPU, RAM, NPU across macOS/Linux/Windows. Native APIs (Mach kernel, nvidia-ml-py, amdsmi) instead of fragile shell parsing.

- **Smart Engine Selection** - Recommends optimal backend (MLX, vLLM, llama.cpp, TensorRT-LLM, OpenVINO) based on detected hardware and model requirements.

- **Architecture-Aware Math** - Accounts for GQA (Grouped Query Attention), MoE (Mixture of Experts), precise quantization (Q4_K_M = 4.85 bits, not 4.0), and backend-specific overhead.

- **Type Safety** - Pydantic schemas with validation, IDE autocomplete, and `.model_dump_json()` serialization.

---

## Installation

**From PyPI** (once published):

```bash
pip install edgekit
```

**From GitHub**:

```bash
pip install git+https://github.com/yourusername/EdgeKit.git
```

**From source**:

```bash
git clone https://github.com/yourusername/EdgeKit.git
cd EdgeKit
pip install -e .
```

---

## Supported Platforms

| Platform | Hardware Detected | Special Features |
|----------|-------------------|------------------|
| **macOS** | Apple Silicon (M1/M2/M3/M4), Intel | 16KB page size support, Mach kernel memory API, Metal detection, Neural Engine |
| **Linux** | NVIDIA, AMD (ROCm), Intel | NVML bindings, AMDSMI integration, CUDA compute capability |
| **Windows** | NVIDIA, Intel | WMI storage detection, CUDA version matching, Intel oneAPI |

---

## The Decision Matrix

EdgeKit automatically solves "The Matrix from Hell" - the complex interplay between your hardware and optimal inference engine:

| Your Hardware | Recommended Engine | Why |
|--------------|-------------------|-----|
| **NVIDIA Ampere/Hopper (RTX 4000, A100, H100)** | **TensorRT-LLM** | Compute Capability ≥ 8.0 enables FP8 quantization and state-of-the-art optimizations |
| **NVIDIA Turing/Volta (RTX 2000/3000, V100)** | **vLLM** | Compute Capability ≥ 7.0 ideal for PagedAttention and continuous batching |
| **NVIDIA Older (GTX 1000 series)** | **llama.cpp** | CUDA backend provides broad compatibility for older architectures |
| **Apple Silicon (M1-M4)** | **MLX** | Native Metal acceleration, unified memory, optimized for Apple Neural Engine |
| **AMD ROCm Compatible** | **llama.cpp** | Mature HIP backend, excellent ROCm support |
| **Intel dGPU/iGPU** | **OpenVINO** | Optimized software stack for Intel Arc, Iris, and Xe graphics |
| **Intel NPU (AI Boost)** | **OpenVINO** | Only framework that leverages Intel's low-power AI accelerator |
| **Intel CPU (with AMX)** | **OpenVINO** | Specialized optimizations for Advanced Matrix Extensions |
| **CPU (x86 with AVX-512)** | **llama.cpp** | Highly optimized AVX-512 kernels for maximum CPU performance |
| **CPU (x86 with AVX2)** | **llama.cpp** | Efficient AVX2 SIMD instructions for modern CPUs |
| **CPU (ARM with NEON)** | **llama.cpp** | ARM-optimized NEON kernels for efficient inference |
| **Generic/Unknown** | **llama.cpp** | Default fallback - broad compatibility and reliable performance |

**Note:** `recommended_engine()` analyzes your specific hardware and returns the optimal choice. The decision is based on compute capability (NVIDIA), instruction sets (CPU), and vendor-specific accelerators (Apple/Intel NPUs).

---

## How It Works

EdgeKit acts as a **physics engine for LLM inference**—simulating memory and performance before execution to prevent runtime failures.

### 1. Native API Integration
Queries hardware through stable binary interfaces (Mach kernel, nvidia-ml-py, amdsmi) instead of fragile shell parsing. Production-grade reliability that doesn't break with driver updates or locale changes.

### 2. Architecture-Aware Memory Simulation

Most calculators just check file size. EdgeKit simulates the full runtime:

- **Architecture Inspection** - Reads `config.json`/GGUF headers for `num_kv_heads` (GQA awareness), `num_experts` (MoE), vocabulary size
- **Backend-Specific Overhead** - MLX graph compilation scratch space, vLLM PagedAttention pre-allocation, llama.cpp KV cache initialization
- **OS Memory Calculation** - On macOS, adds evictable memory (`cached_files` + `speculative`) to available RAM—often unlocks gigabytes
- **Quantization Precision** - Calculates true bits-per-weight (Q4_K_M = 4.85 bits, not 4.0). That 0.85-bit difference determines fit vs failure.

### 3. Hardware-Optimized Builds
Detects CUDA compute capability (sm_89 for RTX 4090), Metal support, ROCm compatibility, AVX-512, then generates optimal CMAKE flags. Handles edge cases like the Apple Silicon SVE hang bug.

---

## What `system_info()` Returns

The `system_info()` function returns a single `HardwareProfile` object with complete hardware information in one call:

| Category | Key Fields | Description |
|----------|-----------|-------------|
| **OS** | `platform`, `version`, `architecture` | Operating system details |
| **CPU** | `brand_raw`, `cores`, `instruction_sets` | CPU model, cores, AI extensions (AVX-512, NEON, AMX) |
| **RAM** | `total_gb`, `available_gb`, `details.*` | Memory stats + macOS Mach kernel breakdown |
| **GPU** | `nvidia`, `amd`, `apple`, `intel` | Per-vendor arrays with VRAM, drivers, compute capability |
| **NPU** | `vendor`, `model_name`, `cores` | Neural processor detection (Apple Neural Engine, Intel AI Boost) |
| **Storage** | `primary_type` | SSD/NVMe vs HDD detection |
| **Engine** | `recommended_engine.name` | Optimal inference backend for your hardware |

**Example fields:**
- `hw.gpu.nvidia[0].compute_capability` → `8.9` (RTX 4090)
- `hw.gpu.apple.gpu_cores` → `20` (M4 Pro)
- `hw.ram.details.cached_files_gb` → `4.2` (macOS evictable memory)
- `hw.cpu.instruction_sets` → `["AVX2", "AVX-512"]`

All fields are type-safe Pydantic models with `.model_dump()` and `.model_dump_json()` for serialization.

See [`docs/USAGE.md`](docs/USAGE.md) for the complete schema reference.

---

## Why This Matters

> **"You shouldn't need a PhD in Computer Architecture to `pip install` an LLM."**

EdgeKit is the **missing middleware layer** between end-user applications (Ollama, LM Studio) and raw inference libraries (llama.cpp, vLLM, MLX). It democratizes the hardware intelligence that's currently locked inside closed applications, bringing it to every Python developer.

**Without EdgeKit:** Your README has 3 pages of platform-specific installation instructions. 40% of users fail and open GitHub issues.

**With EdgeKit:** 
```python
from edgekit import install_command
print(f"Run: {install_command()}")
# Output: CMAKE_ARGS="-DGGML_METAL=on ..." pip install llama-cpp-python
```

It just works. See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the full vision.

---

## Documentation

- **[API Reference](docs/USAGE.md)** - Complete function signatures, schemas, and usage patterns
- **[Philosophy](docs/PHILOSOPHY.md)** - Design principles and long-term vision
- **[Examples](examples/)** - Runnable scripts for all features:
  - `auto_install_llama_cpp.py` - Generate optimal installation commands
  - `test_model_preflight.py` - Check if models fit before loading
  - `test_systeminfo.py` - Comprehensive hardware detection
  - `test_optimal_engine.py` - Backend selection recommendations

