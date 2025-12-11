# Why EdgeKit Exists

> A primer on local AI infrastructure challenges and how EdgeKit addresses them.

---

## The Core Problem

**"How do I get AI / LLMs to run on this specific device?"**

This question appears simple but fractures into a set of infrastructure challenges that most developers encounter only after significant time investment:

- What inference engine is compatible with my hardware?
- Will this model fit in my available memory?
- How do I compile the inference library with correct acceleration flags?
- Why did my application crash even though the system shows "available" memory?

These are not AI problems. They are infrastructure problems that sit between the application layer and the raw inference engines.

---

## The Problem Space

### Hardware Fragmentation

The local AI ecosystem spans fundamentally different hardware architectures:

| Platform | Accelerator | Memory Model | Primary Engine |
|----------|-------------|--------------|----------------|
| Apple Silicon | Metal GPU | Unified (shared CPU/GPU) | MLX, llama.cpp |
| NVIDIA GPU | CUDA cores | Dedicated VRAM | vLLM, llama.cpp |
| AMD GPU | ROCm/HIP | Dedicated VRAM | llama.cpp |
| Intel GPU/NPU | SYCL/OpenVINO | Shared or dedicated | OpenVINO, llama.cpp |
| CPU only | AVX2/AVX-512/NEON | System RAM | llama.cpp |

Each platform has its own:
- Detection mechanisms (some reliable, some brittle)
- Memory reporting semantics (what "available" means differs)
- Build requirements (different compiler flags, SDKs)
- Failure modes (driver crashes, silent fallbacks)

A developer building a cross-platform application must handle all of these cases, or their application works on their machine but fails on users' machines.

Without a unifying layer, this means writing and maintaining adapter code for each platform:
- NVIDIA detection via `pynvml`
- AMD detection via `amdsmi`
- Apple Silicon via Metal + IOKit + Mach kernel APIs
- Windows via WMI and DirectX queries
- Intel via OpenVINO device enumeration

Each new platform you support adds code to *your* application. Each vendor API update risks breaking *your* detection logic.

### Memory Complexity

A common assumption is that if a model's file size fits in available memory, it will load successfully. This is incorrect.

The actual memory requirement for LLM inference is:

**Total = Model Weights + KV Cache + Runtime Overhead**

The KV cache is particularly problematic:
- It scales linearly with context length
- It depends on the model's attention architecture (MHA, GQA, MLA)
- It can exceed the weight memory for long-context models

For example, a 70B parameter model might have:
- Weights: ~40GB (Q4 quantized)
- KV cache at 128K context: ~40GB (standard attention) or ~4GB (with GQA)

Without understanding the model's architecture, memory estimation is guesswork.

### Build Friction

Inference engines like llama.cpp require compilation with hardware-specific flags. The Python bindings (`llama-cpp-python`) attempt to automate this, but the process frequently fails due to:

- Missing or mismatched CMake versions
- Compiler conflicts (Apple Clang vs GCC on macOS)
- CUDA/ROCm libraries in non-standard paths
- Silent fallback to CPU-only builds

A developer may successfully install the package but unknowingly end up with a CPU-only binary, resulting in inference speeds 10-100x slower than expected.

---

## What EdgeKit Provides

EdgeKit is a Python library that provides three categories of utilities:

### 1. Hardware Detection

A unified API for querying hardware capabilities across all platforms.

**What it does:**
- Detects CPU, GPU (NVIDIA/AMD/Intel/Apple), and NPU presence
- Reports available memory using platform-appropriate methods
- Identifies instruction set support (AVX-512, Metal, CUDA compute capability)
- Normalizes vendor-specific APIs into a consistent data structure

**Why this matters:**
Vendor APIs (like AMD's `amdsmi` or Intel's OpenVINO device enumeration) have inconsistent behavior across hardware generations. EdgeKit wraps these in defensive error handling and provides fallback detection methods.

### 2. Memory Validation (Preflight Checks)

Pre-load validation that determines whether a model will fit before attempting to load it.

**What it does:**
- Extracts model architecture from GGUF headers or HuggingFace configs
- Calculates memory requirements accounting for:
  - Weight size (with quantization awareness)
  - KV cache (with GQA and MLA support)
  - Runtime overhead (backend-specific)
- Compares against available hardware memory
- Reports maximum safe context length

**Why this matters:**
An Out-of-Memory crash after a 10-minute model load is a poor developer experience. Preflight checks surface this failure in seconds, before any large download or load occurs.

**Remote inspection:** For HuggingFace models, EdgeKit can extract the necessary metadata using HTTP Range requests (~500KB) rather than downloading the full model (potentially 40GB+). If remote inspection fails, it falls back to the traditional download-then-inspect approach.

### 3. Build Configuration

Automated generation of correct build flags for llama-cpp-python installation.

**What it does:**
- Detects available accelerators (Metal, CUDA, ROCm, SYCL)
- Generates the appropriate `CMAKE_ARGS` environment variable
- Handles platform-specific edge cases (disabling SVE on Apple Silicon, setting CUDA architectures)

**Why this matters:**
The difference between a working GPU-accelerated build and a silent CPU fallback is often a single missing flag. EdgeKit encodes the "tribal knowledge" of correct build configuration.

---

## How These Components Relate

```
                    ┌─────────────────────────┐
                    │   Hardware Detection    │
                    │   "What do I have?"     │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 │                 ▼
┌─────────────────────┐         │    ┌─────────────────────┐
│  Preflight Checks   │         │    │  Build Configuration│
│  "Will it fit?"     │         │    │  "How do I install?"│
└─────────────────────┘         │    └─────────────────────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Application loads the  │
                    │  model successfully     │
                    └─────────────────────────┘
```

Hardware detection is foundational—both preflight checks and build configuration depend on knowing what hardware is present.

Preflight checks answer "can I run this model on this hardware?" Build configuration answers "how do I set up the inference engine for this hardware?"

Together, they address the infrastructure layer between the application and the raw inference engine.

---

## The Pattern: Multiple Devices, One Interface

If you've used [LiteLLM](https://github.com/BerriAI/litellm), you're familiar with this pattern: "100+ cloud providers, one interface." You write to a single API, and LiteLLM handles the differences between OpenAI, Anthropic, Cohere, and dozens of other providers. Your application code stays clean; LiteLLM absorbs the heterogeneity.

EdgeKit applies the same pattern to local hardware.

| Ecosystem | Problem | Solution |
|-----------|---------|----------|
| Cloud LLM APIs | N provider SDKs with different formats | LiteLLM: one interface |
| Local hardware | N platform APIs with different semantics | EdgeKit: one interface |

The value proposition is identical: **portability without adapter code**.

- With LiteLLM, your app works with any cloud provider without provider-specific code.
- With EdgeKit, your app works on any device without platform-specific code.

You call `system_info()` and get a consistent `HardwareProfile` whether you're on an M4 Mac, an RTX 4090 workstation, or an AMD laptop. The platform differences are EdgeKit's problem, not yours.

**This is the portability layer for local AI applications.** If you're building a desktop app, a CLI tool, or a library that runs LLMs on user hardware, EdgeKit lets you write it once and deploy it everywhere — without maintaining separate code paths for each platform your users might have.

---

## What This Enables

With EdgeKit as an infrastructure layer, you can build:

**Hardware-aware installers** — Your library's `setup.py` can detect the user's hardware at install time and automatically configure the correct inference backend. Users run `pip install your-library` and get optimal performance on their device without manual configuration.

**Dynamic dependency selection** — Conditionally install MLX on Apple Silicon, vLLM on NVIDIA, or llama.cpp on AMD/CPU — all from the same package, decided at install time based on what hardware is present.

**Portable build scripts** — Generate correct CMAKE flags for llama-cpp-python compilation without hardcoding platform-specific logic. The same build script works on M4 Macs, RTX workstations, and AMD laptops.

**Pre-flight validation in applications** — Before your app attempts to load a model, check if it will fit. Surface the failure in seconds rather than after a 10-minute download and load attempt.

**Cross-platform AI applications** — Desktop apps, CLI tools, and libraries that genuinely work on your users' hardware, not just your development machine.

The common thread: **your code stays clean, EdgeKit absorbs the platform complexity**.

---

## The Expertise You Don't Need

To build truly portable local AI applications without EdgeKit, you'd need working knowledge of:

- **Hardware architectures** — Apple Silicon unified memory semantics, NVIDIA CUDA compute capabilities, AMD ROCm/HIP, Intel Arc/NPU, ARM vs x86 instruction sets
- **OS-specific memory APIs** — macOS Mach kernel (`host_statistics64`), Windows WMI and `NtQuerySystemInformation`, Linux `/proc/meminfo` variations across kernel versions
- **Model architectures** — Grouped Query Attention (GQA), Multi-Head Latent Attention (MLA), Mixture of Experts (MoE), sliding window attention, RoPE scaling variants
- **Quantization formats** — GGUF K-quants vs I-quants, MLX bit packing, Safetensors dtype detection, bits-per-weight calculations that account for scale/min overhead
- **Inference engine internals** — llama.cpp CMAKE flags and their interactions, vLLM PagedAttention pre-allocation, MLX graph compilation overhead
- **Build system quirks** — Apple Silicon SVE hang bugs, CUDA architecture strings, ROCm library paths, compiler conflicts between Clang and GCC

This is a lot of domain expertise. It takes time to acquire, and it's not what you set out to build.

EdgeKit encodes this knowledge so you don't have to. You call `system_info()` and get accurate hardware data. You call `model_preflight()` and get architecture-aware memory math. You call `llama_cpp_args()` and get correct build flags.

**You just want to build your application. EdgeKit handles the infrastructure so you can.**

---

## Who This Is For

**Application developers** building local AI features who don't want to write platform-specific hardware detection code.

**Library authors** wrapping inference engines who need to validate hardware requirements before attempting operations.

**Tool builders** creating installers or setup utilities that need to configure inference engines correctly.

**Anyone** who has encountered an OOM crash, a silent CPU fallback, or a "works on my machine" failure in local AI deployment.

---

## What EdgeKit Is Not

EdgeKit is not an inference engine. It does not load models or run inference. It is infrastructure tooling that sits alongside engines like llama.cpp, vLLM, and MLX.

EdgeKit is not a model repository or downloader. It integrates with HuggingFace Hub for model inspection but does not replace it.

EdgeKit is not a configuration framework. It provides information and recommendations; the application decides how to act on them.

---

## Summary

Local AI deployment involves a matrix of hardware platforms, memory constraints, and build configurations. EdgeKit provides utilities to navigate this matrix:

| Challenge | EdgeKit Response |
|-----------|------------------|
| "What hardware do I have?" | `system_info()` — unified hardware profile |
| "Will this model fit?" | `model_preflight()` — architecture-aware memory validation |
| "How do I install llama.cpp correctly?" | `install_command()` — correct CMAKE_ARGS for your hardware |
| "What engine should I use?" | `recommended_engine()` — hardware-appropriate recommendation |

The goal is straightforward: **portable local AI infrastructure**. Build once, deploy on any device, with reliable behavior across the hardware your users actually have.
