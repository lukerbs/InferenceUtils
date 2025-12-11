# Model Preflight Internals

> **Internal Documentation** — For EdgeKit maintainers and contributors.  
> Last updated: December 2024

This document explains how EdgeKit inspects models, estimates memory requirements, and validates preflight checks before loading. It covers the **how** (implementation approach), the **why** (rationale for design decisions), and **format-specific considerations & nuances** that aren't immediately obvious.

---

## Table of Contents

1. [Overview](#overview)
   - [The "Light Check First, Full Download Fallback" Strategy](#the-light-check-first-full-download-fallback-strategy)
2. [Model Inspection](#model-inspection)
   - [GGUF Format (Local)](#gguf-format-local)
   - [GGUF Format (Remote)](#gguf-format-remote)
   - [Safetensors/Transformers](#safetensorstransformers)
   - [MLX Models](#mlx-models)
   - [Architecture Prefixes](#architecture-prefixes)
3. [Memory Estimation](#memory-estimation)
   - [Weight Memory](#weight-memory)
   - [KV Cache Memory](#kv-cache-memory)
   - [Activation Overhead](#activation-overhead)
   - [DeepSeek MLA (Compressed KV)](#deepseek-mla-compressed-kv)
4. [Quantization Maps](#quantization-maps)
   - [GGUF K-Quants](#gguf-k-quants)
   - [GGUF I-Quants](#gguf-i-quants)
   - [MLX Group Quantization](#mlx-group-quantization)
5. [Remote Inspection](#remote-inspection)
   - [HTTP Range Requests](#http-range-requests)
   - [GGUF Binary Parsing](#gguf-binary-parsing)
   - [Safetensors Header Fetching](#safetensors-header-fetching)
   - [Authentication](#authentication)
6. [Preflight Validation](#preflight-validation)
   - [Thresholds and Status](#thresholds-and-status)
   - [Backend-Specific Validation](#backend-specific-validation)
   - [Context Length Calculation](#context-length-calculation)
7. [Future Considerations](#future-considerations)

---

## Quick API Reference

The `edgekit.models` module exports:

- **`model_preflight(model_id, engine)`** — Full preflight check returning a `PreflightResult` with status, memory breakdown, and context recommendations
- **`can_load(model_id, engine)`** — Simple boolean check for quick pass/fail decisions
- **`inspect_model_remote(model_id, engine)`** — Low-level remote metadata extraction (raises `RemoteInspectError` on failure)
- **`PreflightResult`** — Dataclass with `.passed`, `.warning`, `.failed`, `.can_load` properties
- **`ModelMetadata`** — Dataclass containing extracted model architecture parameters
- **`MemoryEstimate`** — Dataclass with weights/KV cache/overhead breakdown

---

## Overview

The model preflight system answers a critical question: **"Will this model fit in my available memory?"**

This question is surprisingly complex because:
- Memory requirements depend on model architecture, quantization, and context length
- Different inference backends (llama.cpp, MLX, vLLM) have different memory characteristics
- Downloading a 40GB model just to discover it won't fit wastes significant bandwidth

EdgeKit solves this with a multi-layered approach:

1. **Remote Inspection**: Extract metadata via HTTP Range requests (~500KB instead of 40GB)
2. **Local Inspection**: Parse headers from cached/downloaded model files
3. **Memory Estimation**: Calculate weights + KV cache + overhead based on architecture
4. **Preflight Validation**: Compare requirements against available hardware memory

### The "Light Check First, Full Download Fallback" Strategy

The `model_preflight()` function implements an intelligent two-stage workflow:

1. **Local path check**: If the model_id is a local filesystem path, skip remote inspection entirely and use local inspection directly.

2. **Remote inspection attempt**: For HuggingFace repository IDs, attempt lightweight remote inspection first. This fetches only ~500KB of metadata via HTTP Range requests.

3. **Fallback on failure**: If remote inspection fails (raises `RemoteInspectError`), fall back to the traditional workflow—download the full model via `huggingface_hub`, then inspect the cached files locally.

**When does remote inspection fail?**
- Gated model without authentication (401/403)
- Private repository without access
- Non-standard repository structure (missing expected files)
- Network errors or timeouts
- Corrupted or malformed model files
- Unsupported model format

**The fallback is intentional**: Remote inspection is an optimization, not a requirement. If it fails for any reason, we gracefully degrade to the traditional download-then-inspect workflow. This ensures preflight checks always work, even for edge cases the remote inspector doesn't handle.

---

## Model Inspection

The goal of model inspection is to extract **architectural metadata** required for memory estimation. The key values are:

| Field | Purpose |
|-------|---------|
| `num_layers` | Multiplier for KV cache and weight calculations |
| `hidden_size` | Dimension of the model, affects all tensor sizes |
| `num_attention_heads` | Used to calculate `head_dim` |
| `num_kv_heads` | **Critical** for GQA-aware KV cache calculation |
| `head_dim` | Per-head dimension (`hidden_size / num_attention_heads`) |
| `model_max_context` | Maximum context length for KV cache sizing |
| `bits_per_weight` | Quantization precision (e.g., 4.85 for Q4_K_M) |
| `kv_lora_rank` | DeepSeek MLA compressed KV dimension (if applicable) |

### GGUF Format (Local)

#### How It Works

We use the `gguf` Python library's `GGUFReader` to parse GGUF files. The reader opens the file in read-only mode and parses the binary header without loading tensor data into memory.

The GGUF format stores all metadata in a key-value (KV) block at the beginning of the file. Keys are namespaced by architecture (e.g., `llama.block_count`, `qwen2.embedding_length`).

The inspection process:
1. Open file with `GGUFReader(path, mode='r')`
2. Extract `general.architecture` to determine the metadata prefix
3. Query architecture-specific keys using the prefix (e.g., `{arch}.block_count`)
4. Fall back to `llama.` prefix as a "super-architecture" (covers Mistral, Yi, etc.)
5. Iterate tensors to calculate exact model size

#### Why This Approach

The `GGUFReader` is the same library used by llama.cpp's Python bindings. It understands the binary format natively and handles version differences (GGUF v2 vs v3).

We calculate exact model size by iterating tensors rather than trusting metadata because:
- Mixed-precision quantization (different quant types per tensor) is common
- The file might have been modified or repacked
- Tensor-level calculation catches edge cases

#### Platform-Specific Considerations & Nuances

- **Architecture prefix is mandatory**: GGUF uses `general.architecture` as the "Source of Truth" for all parameter namespacing. Keys like `general.block_count` do **not** exist in the spec—only architecture-prefixed keys like `llama.block_count` are valid.

- **Llama as super-architecture**: Many model families (Mistral, Yi, TinyLlama) use `general.architecture = "llama"` because they share the same transformer topology. This simplifies parsing but means you can't distinguish Mistral from Llama by architecture alone.

- **GGUFReader memory usage**: Despite the name, `GGUFReader` does not load weights. It memory-maps the file and only reads the header. Safe for multi-GB files.

- **Tensor type variations**: Individual tensors can have different quantization types (e.g., attention layers at Q6_K, FFN at Q4_K_M). We iterate all tensors and calculate a weighted average.

---

### GGUF Format (Remote)

#### How It Works

Remote GGUF inspection uses HTTP Range requests to fetch only the first 512KB of the file. This is sufficient to cover the GGUF header and KV metadata block for virtually all models.

The parsing process:
1. Resolve the repository URL using `hf_hub_url(repo_id, filename)`
2. Fetch bytes 0-524287 with `Range: bytes=0-524287` header
3. Validate GGUF magic bytes (`GGUF` = `0x47 0x47 0x55 0x46`)
4. Parse global header (version, tensor count, KV count)
5. Iterate KV pairs, extracting strings and values by type
6. Build `ModelMetadata` from extracted values

#### Why This Approach

The 512KB speculative fetch is empirically validated—it covers the KV block for all known model architectures including deep MoE models with extensive metadata. This single request replaces what would otherwise be a 40GB+ download.

The bandwidth savings are dramatic:
- Traditional: Download 40GB GGUF → Parse header → Fail if OOM
- EdgeKit: Download 512KB → Parse remotely → Validate → Download only if viable

#### Platform-Specific Considerations & Nuances

- **Binary parsing is architecture-dependent**: The GGUF format uses Little-Endian byte ordering and variable-length strings. We use Python's `struct` module for unpacking.

- **KV type handling**: GGUF supports 13 value types (uint8 through float64, plus strings and arrays). We implement parsing for all types but only extract the subset needed for memory estimation.

- **Security limits**: We enforce sanity bounds to prevent DoS from malicious files:
  - Max KV count: 10,000
  - Max string length: 64KB
  - Max header size: 100MB (Safetensors)

- **Buffer management**: If the KV block exceeds 512KB (extremely rare), parsing stops gracefully and falls back to full download.

---

### Safetensors/Transformers

#### How It Works

Safetensors models require a dual-source approach because the format separates architectural metadata from weight storage:

1. **`config.json`**: Contains architecture parameters (layers, hidden size, attention heads)
2. **Safetensors header**: Contains tensor shapes and dtypes

For local inspection, we:
1. Load `config.json` from the model directory
2. Handle nested configs (common in multimodal models via `text_config`)
3. Extract architecture fields with fallback aliases (e.g., `num_hidden_layers` vs `n_layer`)
4. Detect quantization from `quantization_config` if present
5. Calculate parameter count from config or estimate from model name

#### Why This Approach

Unlike GGUF, Safetensors doesn't embed architectural metadata in the weight file. The JSON header only describes tensor names, shapes, and dtypes—not model topology. We must read `config.json` for the architecture.

This split is intentional in the Transformers ecosystem: `config.json` is the source of truth for model behavior, while Safetensors is purely a weight container.

#### Platform-Specific Considerations & Nuances

- **Field name variations**: Different model families use different config keys:
  - Layers: `num_hidden_layers`, `n_layer`, `num_layers`
  - Hidden size: `hidden_size`, `n_embd`, `d_model`
  - KV heads: `num_key_value_heads`, `n_kv_heads`, `kv_heads`

- **GQA detection is critical**: The presence of `num_key_value_heads` < `num_attention_heads` indicates Grouped Query Attention. Using the wrong head count for KV cache calculation can cause 8x overestimation (e.g., Llama 3 70B).

- **Multi-query attention**: Some older models use `multi_query: true` in config, indicating a single KV head. We handle this explicitly.

- **RoPE scaling**: Long-context models use `rope_scaling` to extend context beyond training length. We support multiple scaling types:
  - `linear`: Simple linear interpolation
  - `dynamic`: Dynamic NTK-aware scaling
  - `yarn`: YaRN (Yet another RoPE extensioN)
  - `su`/`longrope`: Microsoft's LongRoPE
  
  The effective max context is `base_context_length × scaling_factor`. Missing this causes underestimation of memory needs.

---

### MLX Models

#### How It Works

MLX models on HuggingFace (typically in `mlx-community/`) use Safetensors for weight storage but have MLX-specific quantization handling.

The critical difference: **`config.json` is authoritative for quantization, not the tensor dtype**.

MLX 4-bit models pack 8 weights into a single 32-bit integer. The Safetensors header reports `dtype: I32` or `U32` for these tensors, which would imply 32 bits per weight if interpreted naively. The actual precision is defined in `config.json` under the `quantization_config` key, which contains `bits` (e.g., 4), `group_size` (e.g., 64), and `quant_method` (e.g., "mlx").

We extract `bits` and `group_size` to calculate effective bits-per-weight using the MLX formula.

#### Why This Approach

MLX's quantization is "logical" rather than "physical"—the storage format (packed int32) doesn't reflect the semantic precision (4-bit). Trusting the Safetensors dtype would produce wildly incorrect memory estimates.

The `config.json`-first approach ensures we interpret weights correctly regardless of storage format.

#### Platform-Specific Considerations & Nuances

- **Group size affects overhead**: MLX uses affine quantization with per-group scales and biases. Smaller groups = better quality but higher overhead:
  - Group 64: 4.50 BPW (default)
  - Group 32: 5.00 BPW (high quality)

- **Unified memory budget**: MLX runs on Apple Silicon unified memory. The same RAM pool serves CPU, GPU, and Neural Engine. We account for this in hardware detection.

- **Wiring spike**: MLX graph compilation creates a temporary memory spike (~15% overhead). We add this to activation overhead estimates.

---

### Architecture Prefixes

GGUF uses architecture-specific prefixes for all structural parameters. The `general.architecture` key defines which prefix to use.

| `general.architecture` | Model Families |
|------------------------|----------------|
| `llama` | Llama 1/2/3, Mistral, Yi, TinyLlama, Vicuna |
| `qwen2` | Qwen 1.5, Qwen 2, Qwen 2.5 |
| `qwen2vl` | Qwen 2 VL (vision-language) |
| `phi3` | Phi-3, Phi-3.5 |
| `phi2` | Phi-2 (legacy) |
| `deepseek2` | DeepSeek V2, DeepSeek V3 |
| `gemma` | Gemma 1 |
| `gemma2` | Gemma 2 |
| `command-r` | Command-R, Command-R+ |
| `falcon` | Falcon |
| `stablelm` | StableLM |
| `mpt` | MPT (Mosaic) |

#### The "Super-Architecture" Pattern

The `llama` prefix functions as a "super-architecture" for models that share Llama's transformer topology (RMSNorm, SwiGLU, RoPE). This is a deliberate design choice in llama.cpp to reduce code duplication.

When parsing, we:
1. Read `general.architecture` (e.g., `"qwen2"`)
2. Try `{arch}.{key}` first (e.g., `qwen2.block_count`)
3. Fall back to `llama.{key}` if not found

**Important**: Keys like `general.block_count` do **not** exist in the GGUF spec. The `general.*` namespace is only for descriptive metadata (`general.name`, `general.size_label`, `general.quantization_version`), not structural parameters.

---

## Memory Estimation

Memory estimation calculates the total memory footprint as the sum of **Weights + KV Cache + Overhead**.

### Weight Memory

Weight memory is the static cost of loading model parameters.

#### Formula

**Weights (GB) = (params × bits_per_weight) / 8 / (1024³)**

For models with `exact_model_size_gb` from tensor iteration, we use that directly.

#### MoE Overhead

Mixture-of-Experts models keep router/gating weights at higher precision (typically Q6_K or Q8_0 even when experts are Q4_K_M). We apply a 5% overhead multiplier (1.05×).

---

### KV Cache Memory

The KV cache stores attention keys and values for all previous tokens. This is the dominant memory consumer for long-context inference.

#### Standard Formula

**KV Cache (GB) = 2 × layers × kv_heads × head_dim × context × dtype_bytes / (1024³)**

The factor of 2 accounts for both Keys (K) and Values (V) being stored.

**Critical**: Use `num_kv_heads`, not `num_attention_heads`! Modern models use Grouped Query Attention (GQA) where KV heads < attention heads. Using the wrong value causes massive overestimation.

| Model | Attention Heads | KV Heads | Ratio |
|-------|-----------------|----------|-------|
| Llama 2 7B | 32 | 32 | 1:1 (MHA) |
| Llama 3 8B | 32 | 8 | 4:1 (GQA) |
| Llama 3 70B | 64 | 8 | 8:1 (GQA) |

#### KV Cache Precision

The default is FP16 (2 bytes per element), but some configurations support quantized KV:

| Dtype | Bytes per element | Notes |
|-------|-------------------|-------|
| FP16 | 2.0 | Default |
| FP8 | 1.0 | Halves cache size |
| Q8 | 1.0 | 8-bit quantized |
| Q4 | 0.5 | 4-bit (experimental) |

---

### Activation Overhead

Activation overhead accounts for runtime memory beyond weights and KV cache:
- Scratch buffers for matrix multiplication
- Intermediate activation tensors
- Runtime overhead (Python, CUDA context)
- Graph compilation (MLX)

#### Tiered Overhead

We use fixed tiers based on model size (empirically validated):

| Model Size | Base Overhead |
|------------|---------------|
| < 10B params | 1.2 GB |
| 10-30B params | 2.5 GB |
| 30B+ params | 4.0 GB |
| MoE (any size) | 3.0 GB |

For MLX, we add 15% for graph compilation ("wiring") spike by multiplying the base overhead by 1.15.

---

### DeepSeek MLA (Compressed KV)

DeepSeek V2 and V3 use Multi-Head Latent Attention (MLA), which compresses the KV cache into a low-rank latent space. This reduces KV cache memory by **90%+** compared to standard attention.

#### Detection

The presence of `{arch}.attention.kv_lora_rank` in GGUF metadata indicates MLA. We extract this value during model inspection for both local and remote GGUF files.

#### MLA Formula

When `kv_lora_rank` is set, we use the compressed formula:

**KV Cache (GB) = 2 × layers × kv_lora_rank × context × dtype_bytes / (1024³)**

Note: The formula uses `kv_lora_rank` directly instead of `kv_heads × head_dim`.

#### Impact

For DeepSeek V3 at 128K context:
- Standard formula: ~400 GB KV cache
- MLA formula: ~20 GB KV cache

Ignoring MLA causes catastrophic overestimation—the model would appear to require server-grade RAM when it actually fits on a MacBook Pro.

---

## Quantization Maps

Quantization reduces model size by storing weights at lower precision. The **effective bits-per-weight (BPW)** accounts for quantization overhead (scales, zeros, block structures).

### Full Precision Types

| Type | BPW | Notes |
|------|-----|-------|
| `F32` | 32.0 | Full 32-bit float |
| `F16` | 16.0 | Half precision |
| `BF16` | 16.0 | Brain float (better range than F16) |

### Legacy Quantization

Older formats still seen in some models:

| Type | Effective BPW | Notes |
|------|---------------|-------|
| `Q4_0` | 4.50 | Original 4-bit |
| `Q4_1` | 4.50 | 4-bit with bias |
| `Q5_0` | 5.50 | Original 5-bit |
| `Q5_1` | 5.50 | 5-bit with bias |
| `Q8_0` | 8.50 | 8-bit quantization |
| `Q8_1` | 8.50 | 8-bit with bias |

### GGUF K-Quants

K-quants use k-means clustering for quantization. They're the most common format on HuggingFace.

| Type | Effective BPW | Notes |
|------|---------------|-------|
| `Q2_K` | 2.56 | Extreme compression, quality loss |
| `Q3_K_S` | 3.44 | Small 3-bit |
| `Q3_K_M` | 3.91 | Medium 3-bit |
| `Q3_K_L` | 4.27 | Large 3-bit |
| `Q4_K_S` | 4.58 | Small 4-bit |
| `Q4_K_M` | 4.85 | **Most commonly used** |
| `Q5_K_S` | 5.54 | Small 5-bit |
| `Q5_K_M` | 5.69 | Medium 5-bit |
| `Q6_K` | 6.59 | High quality |
| `Q8_K` | 8.50 | Near-lossless |

The "K" suffix indicates k-means quantization. S/M/L suffixes indicate small/medium/large block sizes (affecting quality vs compression tradeoff).

### GGUF I-Quants

I-quants (importance-weighted quantization) use learned importance scores to allocate more bits to critical weights.

| Type | Effective BPW | Notes |
|------|---------------|-------|
| `IQ1_S` | 1.56 | Extreme (experimental) |
| `IQ2_XXS` | 2.06 | Very aggressive |
| `IQ2_XS` | 2.31 | Aggressive |
| `IQ2_S` | 2.50 | 2-bit |
| `IQ3_XXS` | 3.06 | Very small 3-bit |
| `IQ3_S` | 3.44 | Small 3-bit |
| `IQ3_M` | 3.66 | Medium 3-bit |
| `IQ4_NL` | 4.50 | Non-linear 4-bit |
| `IQ4_XS` | 4.25 | Extra-small 4-bit |

### MLX Group Quantization

MLX uses affine quantization where each group of weights shares a scale (FP16) and bias (FP16).

#### Formula

**Effective BPW = (group_size × bits + 32) / group_size**

The 32-bit overhead comes from the FP16 scale + FP16 bias per group.

| Type | Effective BPW | Notes |
|------|---------------|-------|
| `2bit` | 2.25 | Aggressive compression |
| `3bit` | 3.75 | Good balance |
| `4bit` | 4.50 | **Most common** (group 64) |
| `4bit_g32` | 5.00 | High quality (group 32) |
| `8bit` | 8.25 | Near-lossless |
| `fp16`/`bf16` | 16.0 | Full precision |

**Formula** (for reference): `(group_size × bits + 32) / group_size`

Note: The table values are empirically validated and may differ slightly from the formula due to implementation details.

---

## Remote Inspection

Remote inspection enables preflight checks without downloading multi-GB model files. This is the core innovation that makes EdgeKit practical for bandwidth-constrained users.

### HTTP Range Requests

The HTTP/1.1 `Range` header allows fetching specific byte ranges. We send a GET request with a `Range: bytes=0-524287` header to fetch only the first 512KB. The server responds with HTTP status `206 Partial Content` and returns only the requested bytes, not the full file.

#### Bandwidth Comparison

| Format | Traditional | EdgeKit | Savings |
|--------|-------------|---------|---------|
| GGUF 70B | 40 GB | 512 KB | **80,000×** |
| Safetensors 70B | 140 GB | 50 KB | **2,800,000×** |
| MLX 7B 4-bit | 4 GB | 5 KB | **800,000×** |

### GGUF Binary Parsing

We implement a pure-Python GGUF parser for remote inspection (the `gguf` library requires local files).

#### Header Structure

The GGUF binary header is laid out as:
- **Bytes 0-3**: Magic bytes ("GGUF")
- **Bytes 4-7**: Version (uint32, Little-Endian)
- **Bytes 8-15**: Tensor count (uint64)
- **Bytes 16-23**: KV pair count (uint64)
- **Bytes 24+**: KV pairs (variable length)

#### KV Pair Parsing

Each KV pair is:
1. Key length (uint64) + Key string (UTF-8)
2. Value type (uint32)
3. Value data (type-dependent)

We parse all 13 GGUF types but only extract keys relevant to memory estimation.

### Safetensors Header Fetching

Safetensors has a simpler structure:
- **Bytes 0-7**: Header length N (uint64, Little-Endian)
- **Bytes 8 to 8+N-1**: JSON header containing tensor metadata (UTF-8)
- **Bytes 8+N onward**: Raw tensor data

Remote inspection:
1. Fetch bytes 0-7 → parse header length `N`
2. Fetch bytes 8 to 8+N-1 → parse JSON header
3. Extract dtype from a representative tensor

The Safetensors header only contains tensor metadata, not architecture. We separately fetch `config.json` for that.

### Authentication

Many models on HuggingFace are "gated"—access requires accepting a license and authentication.

We read the HuggingFace token from `~/.cache/huggingface/token` (managed by `huggingface-cli login`) and include it as an `Authorization: Bearer` header in HTTP requests.

If authentication fails (401/403), we raise `RemoteInspectError` to trigger fallback to full download.

---

## Preflight Validation

Preflight validation compares estimated memory requirements against available hardware memory.

### Thresholds and Status

| Utilization | Status | Meaning |
|-------------|--------|---------|
| < 70% | `PASSED` | Safe to load, comfortable headroom |
| 70-85% | `WARNING` | Can load but tight, may swap under load |
| > 85% | `FAILED` | Do not attempt, will likely OOM |

These thresholds are intentionally conservative to account for:
- Runtime memory spikes
- KV cache growth during generation
- System processes competing for RAM

### PreflightResult API

The `PreflightResult` dataclass provides both raw data and convenience helpers:

**Status properties:**
- `.passed` — True if utilization < 70%
- `.warning` — True if utilization is 70-85%
- `.failed` — True if utilization > 85%
- `.can_load` — True if passed OR warning (model will load)

**Memory details:**
- `.required_gb` — Total memory needed
- `.available_gb` — Available memory detected
- `.utilization` — Fraction (0.0 to 1.0+)

**Context recommendations:**
- `.max_context` — Maximum safe context length in tokens
- `.recommended_context` — Conservative recommendation (80% of max)

**Detailed breakdown:**
- `.estimate` — `MemoryEstimate` object with weights/kv/overhead split

**Error handling:**
- `.raise_if_failed()` — Raises `MemoryError` if preflight failed

### Backend-Specific Validation

Each inference backend has different memory characteristics.

#### MLX (Apple Silicon)

- Uses unified memory (shared CPU/GPU RAM)
- Tiered safety buffer:
  - ≤16GB Macs: 3GB reserve (tight but viable)
  - >16GB Macs: 20% reserve (smooth operation)
- Graph compilation adds 15% temporary spike

#### llama.cpp (CPU)

- Uses system RAM
- 1GB safety buffer (2GB on Windows for DWM)
- Pre-allocates entire KV cache at startup

#### vLLM (NVIDIA GPU)

- Uses dedicated VRAM
- Two-stage validation:
  1. Can weights + overhead fit?
  2. How much context fits in remaining space?
- PagedAttention fills remaining VRAM with KV blocks
- Default `gpu_memory_utilization=0.9` (90% of VRAM)
- CUDA context overhead: ~0.5GB for NCCL/PyTorch

### Context Length Calculation

When full context doesn't fit, we calculate the maximum safe context by first computing the **KV budget** (available memory minus weights and overhead), then dividing by **bytes per token**.

**Bytes per token calculation:**
- **Standard attention**: 2 × layers × kv_heads × head_dim × dtype_bytes
- **MLA (DeepSeek)**: 2 × layers × kv_lora_rank × dtype_bytes

We enforce a minimum viable context of 16K tokens—below this, the model isn't useful for most applications.

**Fallback behavior**:
- If `model_max_context` is unknown, we default to 32K tokens for validation
- If calculated max context < 16K, preflight fails (model not viable)
- Recommended context is set to 80% of max safe context for conservative operation

---

## Future Considerations

Areas identified for potential future enhancement:

- **Sliding window memory optimization**: Models with `sliding_window` attention (Mistral, Phi-3) have bounded attention memory. Currently we conservatively use full context for validation.

- **Vocabulary size from token array**: For Qwen models, `vocab_size` may not be explicit—it's implicit in the length of `tokenizer.ggml.tokens`. We could count array entries as fallback.

- **Multi-GPU sharding**: vLLM tensor parallelism splits models across GPUs. Memory validation could account for aggregate VRAM.

- **Quantized KV cache**: llama.cpp and vLLM support FP8/Q8 KV caches. We could expose this as a parameter for tighter memory budgets.

- **Expert offloading for MoE**: Large MoE models (Mixtral 8x22B) can offload inactive experts to CPU. Memory requirements could be adjusted for this mode.

- **Dynamic overhead profiling**: Replace fixed overhead tiers with runtime measurement after initialization for more accurate planning.

- **GGUF v4 support**: Future GGUF versions may change header layout. The parser should handle version negotiation gracefully.
