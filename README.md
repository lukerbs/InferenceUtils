# InferenceUtils

> **Stop guessing. Start engineering.**  
> Hardware introspection and memory validation for cross-platform LLM inference

A single, consistent interface for local AI that handles the hardware differences between Mac, Windows, and Linux—giving you one reliable answer to: *Can I run this LLM model? Should I run this LLM model? And which inference engine works best for my device?*

**InferenceUtils** solves the "Matrix from Hell" in local AI: the complex interplay between your hardware, operating system, model architecture, and inference engine. It replaces trial and error with engineering rigor.

## The Problem

Building cross-platform LLM applications involves four recurring engineering challenges:

1. **"Can I run this?" / "Should I run this?"** - Will this model fit in RAM? Will it crash during inference when the KV cache fills up? Will it thrash and become unusable even if it technically loads?

2. **The Build Configuration Maze** - Installing `llama-cpp-python` with the right CMAKE flags for your exact hardware (Metal vs CUDA vs ROCm vs CPU-only), turning a simple `pip install` into a 20-minute debugging session.

3. **Hardware Detection Fragility** - Querying hardware specs across macOS/Linux/Windows, Apple Silicon/Intel/NVIDIA/AMD, unified memory vs discrete VRAM—each requiring different APIs and shell commands that break with OS updates.

4. **Engine Selection Guesswork** - Choosing between MLX, vLLM, llama.cpp, TensorRT-LLM, and OpenVINO based on your hardware capabilities and model requirements.

**InferenceUtils solves all four** with a single, type-safe interface that validates memory before loading, auto-generates build configurations, detects hardware cross-platform, and recommends optimal engines.

---

## Quick Start

```python
from inferenceutils import system_info, recommended_engine, model_preflight, llama_cpp_args

# Detect hardware capabilities
hw = system_info()
print(f"CPU: {hw.cpu.brand_raw}, RAM: {hw.ram.available_gb} GB")
# Output: CPU: Apple M4 Pro, RAM: 17.0 GB

# Get recommended inference engine
engine = recommended_engine()
print(f"Use {engine.name}: {engine.reason}")
# Output: Use MLX: Apple Silicon detected with Metal support

# Validate model will fit in RAM before loading
result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx")
if result.can_load:
    print(f"✓ Safe to load with {result.recommended_context:,} token context")
    # Output: ✓ Safe to load with 131,072 token context
else:
    print(f"✗ Won't fit in RAM: {result.message}")
    # Example: Model requires 21.0 GB but only 15.0 GB available

# Generate optimal llama.cpp build arguments
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
pip install inferenceutils
```

**From GitHub**:

```bash
pip install git+https://github.com/yourusername/InferenceUtils.git
```

**From source**:

```bash
git clone https://github.com/yourusername/InferenceUtils.git
cd InferenceUtils
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

## How It Works

InferenceUtils acts as a **physics engine for LLM inference**—simulating memory and performance before execution to prevent runtime failures.

### 1. Native API Integration
Queries hardware through stable binary interfaces (Mach kernel, nvidia-ml-py, amdsmi) instead of fragile shell parsing. Production-grade reliability that doesn't break with driver updates or locale changes.

### 2. Architecture-Aware Memory Simulation

Most calculators just check file size. InferenceUtils simulates the full runtime:

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

## Documentation

- **[API Reference](docs/USAGE.md)** - Complete function signatures, schemas, and usage patterns
- **[Examples](examples/)** - Runnable scripts for all features (`test_model_preflight.py`, `test_systeminfo.py`, etc.)

