# EdgeKit API Reference

Complete API documentation for EdgeKit, organized by module.

---

## Table of Contents

- [Module: `edgekit.hardware`](#module-edgekithardware)
  - [Functions](#hardware-functions)
    - [`system_info()`](#system_info)
    - [`recommended_engine()`](#recommended_engine)
  - [Schemas](#hardware-schemas)
    - [`HardwareProfile`](#hardwareprofile)
    - [`OperatingSystem`](#operatingsystem)
    - [`CPU`](#cpu)
    - [`RAM`](#ram)
    - [`MacOSMemoryDetails`](#macosmemorydetails)
    - [`GPU`](#gpu)
    - [`NVIDIAGPU`](#nvidiagpu)
    - [`AMDGPU`](#amdgpu)
    - [`AppleGPU`](#applegpu)
    - [`IntelAccelerator`](#intelaccelerator)
    - [`NPU`](#npu)
    - [`Storage`](#storage)
    - [`EngineRecommendation`](#enginerecommendation)
    - [`OptimalInferenceEngine`](#optimalinferenceengine)
- [Module: `edgekit.models`](#module-edgekitmodels)
  - [Functions](#models-functions)
    - [`model_preflight()`](#model_preflight)
    - [`can_load()`](#can_load)
    - [`inspect_model_remote()`](#inspect_model_remote)
  - [Schemas](#models-schemas)
    - [`PreflightResult`](#preflightresult)
    - [`PreflightReason`](#preflightreason)
    - [`Engine`](#engine)
    - [`MemoryEstimate`](#memoryestimate)
    - [`ModelMetadata`](#modelmetadata)
    - [`RemoteInspectError`](#remoteinspecterror)
- [Module: `edgekit.build`](#module-edgekitbuild)
  - [Functions](#build-functions)
    - [`llama_cpp_args()`](#llama_cpp_args)
    - [`install_command()`](#install_command)
- [Platform-Specific Notes](#platform-specific-notes)
- [Error Handling](#error-handling)
- [Type Hints & IDE Support](#type-hints--ide-support)
- [JSON Serialization](#json-serialization)

---

## Module: `edgekit.hardware`

Hardware detection and inference engine recommendations.

### Hardware Functions

#### `system_info()`

Get comprehensive system hardware information as a validated Pydantic model.

**Signature:**
```python
def system_info() -> HardwareProfile
```

**Parameters:**
- None

**Returns:**
- [`HardwareProfile`](#hardwareprofile) - Complete hardware profile containing OS, CPU, RAM, GPU, NPU, storage, and engine recommendation

**Description:**

Detects all available hardware across all platforms in a single call. Uses direct API calls (Mach kernel for macOS, nvidia-ml-py, amdsmi, OpenVINO) instead of parsing shell commands for robustness.

**Example:**
```python
from edgekit import system_info

hw = system_info()
print(f"CPU: {hw.cpu.brand_raw}")
print(f"RAM: {hw.ram.available_gb} GB available")
print(f"GPU: {hw.gpu.detected_vendor}")
```

---

#### `recommended_engine()`

Get optimal inference engine recommendation based on detected hardware.

**Signature:**
```python
def recommended_engine() -> OptimalInferenceEngine
```

**Parameters:**
- None

**Returns:**
- [`OptimalInferenceEngine`](#optimalinferenceengine) - Engine name, pip dependencies, and reasoning

**Description:**

Analyzes detected hardware and recommends the best inference engine. Automatically selects:
- **MLX** for Apple Silicon (M1/M2/M3/M4)
- **vLLM** for NVIDIA GPUs (Turing+)
- **TensorRT-LLM** for NVIDIA GPUs (Ampere+)
- **OpenVINO** for Intel CPUs/GPUs/NPUs
- **llama.cpp** for universal CPU compatibility

**Example:**
```python
from edgekit import recommended_engine

engine = recommended_engine()
print(f"Use: {engine.name}")
print(f"Install: pip install {' '.join(engine.dependencies)}")
print(f"Reason: {engine.reason}")
```

---

### Hardware Schemas

#### `HardwareProfile`

Complete hardware profile from hardware detection.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `os` | [`OperatingSystem`](#operatingsystem) | Operating system information |
| `python_version` | `str \| None` | Python version string |
| `cpu` | [`CPU`](#cpu) | CPU information and capabilities |
| `ram` | [`RAM`](#ram) | Random access memory information |
| `storage` | [`Storage`](#storage) | Storage device information |
| `gpu` | [`GPU`](#gpu) | Comprehensive GPU information |
| `npus` | `List[`[`NPU`](#npu)`]` | List of detected NPUs |
| `recommended_engine` | [`EngineRecommendation`](#enginerecommendation) ` \| None` | Recommended inference engine |

**Methods:**
- `model_dump()` - Convert to Python dictionary
- `model_dump_json()` - Convert to JSON string

---

#### `OperatingSystem`

Operating system information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `platform` | `str \| None` | OS platform ("Linux", "Windows", "Darwin") |
| `version` | `str \| None` | OS version/release |
| `architecture` | `str \| None` | Machine architecture ("x86_64", "arm64") |

---

#### `CPU`

CPU information and capabilities.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `brand_raw` | `str \| None` | Raw CPU brand string from hardware |
| `arch` | `str \| None` | CPU architecture string |
| `physical_cores` | `int \| None` | Number of physical CPU cores |
| `logical_cores` | `int \| None` | Number of logical CPU cores |
| `instruction_sets` | `List[str] \| None` | Supported instruction sets (AVX-512, AVX2, NEON, AMX) |

---

#### `RAM`

Random Access Memory information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_gb` | `float \| None` | Total RAM in gigabytes |
| `available_gb` | `float \| None` | Available RAM in gigabytes |
| `details` | [`MacOSMemoryDetails`](#macosmemorydetails) ` \| None` | Platform-specific memory details (macOS only) |

---

#### `MacOSMemoryDetails`

macOS-specific memory breakdown using Mach kernel API.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `cached_files_gb` | `float` | File cache (instantly evictable) |
| `wired_gb` | `float` | Kernel wired memory (non-evictable) |
| `compressed_gb` | `float` | Memory held in compressed state |
| `app_memory_gb` | `float` | Active application memory |
| `speculative_gb` | `float` | Read-ahead cache (instantly evictable) |
| `page_size_bytes` | `int` | Hardware page size (16384 on Apple Silicon, 4096 on Intel) |

---

#### `GPU`

Comprehensive GPU information across all vendors.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `detected_vendor` | `str \| None` | Detected GPU vendor ("NVIDIA", "AMD", "Intel", "Apple") |
| `vulkan_api_version` | `str \| None` | Vulkan API version if available |
| `nvidia` | `List[`[`NVIDIAGPU`](#nvidiagpu)`] \| None` | List of NVIDIA GPUs |
| `amd` | `List[`[`AMDGPU`](#amdgpu)`] \| None` | List of AMD GPUs |
| `intel` | `List[`[`IntelAccelerator`](#intelaccelerator)`] \| None` | List of Intel accelerators (GPUs/NPUs) |
| `apple` | [`AppleGPU`](#applegpu) ` \| None` | Apple Silicon GPU information |

---

#### `NVIDIAGPU`

NVIDIA GPU information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | GPU model name ("RTX 4090", "A100") |
| `vram_gb` | `float` | VRAM size in gigabytes |
| `available_vram_gb` | `float \| None` | Currently available VRAM in gigabytes |
| `driver_version` | `str` | NVIDIA driver version |
| `cuda_version` | `str \| None` | CUDA version supported by driver |
| `compute_capability` | `float \| None` | Compute capability (e.g., 8.9 for RTX 4090) |
| `cuda_cores` | `int \| None` | Number of CUDA cores |
| `tensor_cores` | `int \| None` | Number of Tensor cores |
| `ecc_enabled` | `bool \| None` | Whether ECC memory is enabled (reduces usable VRAM by 6-12% on data center GPUs) |

---

#### `AMDGPU`

AMD GPU information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str \| None` | GPU model name |
| `vram_gb` | `float \| None` | VRAM size in gigabytes |
| `available_vram_gb` | `float \| None` | Currently available VRAM in gigabytes |
| `driver_version` | `str \| None` | AMD driver version |
| `rocm_compatible` | `bool` | Whether GPU is ROCm compatible |
| `compute_units` | `int \| None` | Number of compute units |
| `is_apu` | `bool \| None` | Whether this is an APU (integrated) vs discrete GPU |
| `gtt_total_gb` | `float \| None` | GTT (Graphics Translation Table) total size - system RAM accessible to GPU on APUs |
| `gtt_used_gb` | `float \| None` | GTT memory currently in use |

---

#### `AppleGPU`

Apple Silicon GPU information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str \| None` | GPU model/architecture ("M1", "M2", "M3", "M4") |
| `vram_gb` | `float \| None` | Unified memory size in gigabytes |
| `gpu_cores` | `int \| None` | Number of GPU cores |
| `metal_supported` | `bool` | Whether Metal framework is supported |

---

#### `IntelAccelerator`

Intel GPU or NPU information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Device model name |
| `type` | `str` | Device type ("dGPU", "iGPU", "NPU") |
| `execution_units` | `int \| None` | Number of execution units (GPU only) |
| `vram_gb` | `float \| None` | VRAM size in gigabytes (GPU only) |
| `driver_version` | `str \| None` | Driver version (GPU only) |

---

#### `NPU`

Neural Processing Unit information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `vendor` | `str` | NPU vendor ("Intel", "Apple", "AMD") |
| `model_name` | `str` | NPU model name |
| `npu_cores` | `int \| None` | Number of NPU cores |

---

#### `Storage`

Storage device information.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `primary_type` | `str \| None` | Primary storage type ("SSD/NVMe", "HDD", "Unknown") |

---

#### `EngineRecommendation`

LLM inference engine recommendation.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Recommended engine name |
| `reason` | `str` | Reason for the recommendation |

---

#### `OptimalInferenceEngine`

Optimal inference engine recommendation with dependencies.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the recommended inference engine |
| `dependencies` | `List[str]` | List of Python libraries needed for this engine |
| `reason` | `str` | Reason why this engine was selected |

**Methods:**
- `model_dump()` - Convert to Python dictionary
- `model_dump_json()` - Convert to JSON string

---

## Module: `edgekit.models`

Model preflight checks and memory validation.

### Models Functions

#### `model_preflight()`

Run preflight check to determine if a model will fit in memory.

**Signature:**
```python
def model_preflight(
    model_id: str,
    engine: Engine
) -> PreflightResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | HuggingFace repo ID (e.g., "mlx-community/Llama-3-8B-4bit") or local path (e.g., "/path/to/model.gguf") |
| `engine` | [`Engine`](#engine) | Inference engine - `"mlx_lm"`, `"llama_cpp"`, or `"vllm"` |

**Returns:**
- [`PreflightResult`](#preflightresult) - Validation result with status, memory details, and context recommendations

**Description:**

Validates whether a model will fit in memory before loading. For HuggingFace repository IDs, first attempts **lightweight remote inspection** using HTTP Range requests (~500KB bandwidth) to extract metadata without downloading the full model. If remote inspection fails, falls back to downloading the model.

The function inspects GGUF files or HuggingFace configs to extract model architecture, calculates memory requirements (weights + KV cache + overhead), and determines the maximum safe context window.

The function uses model-specific memory estimation for each backend:
- **MLX**: Checks macOS unified memory
- **llama_cpp**: Checks system RAM
- **vLLM**: Checks GPU VRAM (two-stage validation)

**Validation Thresholds:**
- **PASSED**: < 70% memory utilization
- **WARNING**: 70-85% memory utilization (can load but tight fit)
- **FAILED**: > 85% memory utilization (don't attempt to load)

**Example:**
```python
from edgekit import model_preflight

# Basic check with defaults
result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx_lm")

if result.status:
    print(f"✓ Model will run")
    print(f"Context: {result.usable_context:,} tokens")
    print(f"Memory usage: {result.utilization*100:.1f}%")
    print(f"Reason: {result.reason}")
    
    # Check if context was reduced
    if result.usable_context < result.context_limit:
        reduction = (1 - result.usable_context / result.context_limit) * 100
        print(f"⚠️ Context reduced by {reduction:.0f}% from designed {result.context_limit:,}")
else:
    print(f"✗ Won't fit: {result.reason}")

# Conservative: only use 70% of memory
result = model_preflight("model-id", engine="mlx_lm", max_utilization=0.70)

# Power user: push to 95%, accept lower context
result = model_preflight("model-id", engine="mlx_lm", max_utilization=0.95, min_context=2048)

# Check local GGUF file
result = model_preflight("/path/to/model.gguf", engine="llama_cpp")
```

**Raises:**
- `ValueError` - If engine is not one of "mlx_lm", "llama_cpp", or "vllm"
- `FileNotFoundError` - If local model path doesn't exist
- `Exception` - If model download or inspection fails

---

#### `can_load()`

Simple boolean check: can this model load on this hardware?

**Signature:**
```python
def can_load(
    model_id: str,
    engine: Engine
) -> bool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | HuggingFace repo ID or local path |
| `engine` | [`Engine`](#engine) | Inference engine - `"mlx_lm"`, `"llama_cpp"`, or `"vllm"` |

**Returns:**
- `bool` - `True` if model can load (passed or warning status), `False` otherwise

**Description:**

Convenience wrapper around [`model_preflight()`](#model_preflight) that returns a simple boolean. Useful for quick checks without needing detailed validation results.

**Example:**
```python
from edgekit import can_load

if can_load("mlx-community/Llama-3-8B-4bit", engine="mlx_lm"):
    # Proceed with loading
    from mlx_lm import load
    model, tokenizer = load("mlx-community/Llama-3-8B-4bit")
else:
    print("Model won't fit, choose a smaller one")
```

---

#### `inspect_model_remote()`

Extract model metadata from a remote HuggingFace repository without downloading the full model.

**Signature:**
```python
def inspect_model_remote(
    repo_id: str,
    engine: Engine
) -> ModelMetadata
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo_id` | `str` | HuggingFace repository ID (e.g., "meta-llama/Llama-3-8B") |
| `engine` | [`Engine`](#engine) | Inference engine - `"mlx_lm"`, `"llama_cpp"`, or `"vllm"` |

**Returns:**
- [`ModelMetadata`](#modelmetadata) - Extracted model metadata

**Raises:**
- [`RemoteInspectError`](#remoteinspecterror) - If remote inspection fails (authentication required, network error, unsupported format)

**Description:**

Performs lightweight remote inspection using HTTP Range requests. Fetches only the metadata needed for memory estimation:
- **GGUF models**: ~512KB (binary header + KV block)
- **Safetensors models**: ~50KB (config.json + tensor header)
- **MLX models**: ~5KB (config.json only)

This is an advanced function—most users should use `model_preflight()` which calls this internally and handles fallback to full download.

**Example:**
```python
from edgekit.models import inspect_model_remote, RemoteInspectError

try:
    metadata = inspect_model_remote("TheBloke/Llama-2-7B-GGUF", engine="llama_cpp")
    print(f"Layers: {metadata.num_layers}")
    print(f"KV heads: {metadata.num_kv_heads}")
    print(f"Quantization: {metadata.quantization_type}")
except RemoteInspectError as e:
    print(f"Remote inspection failed: {e}")
    # Fall back to downloading the model
```

---

### Models Schemas

#### `PreflightResult`

Result of model preflight check.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | `bool` | True if model can run, False if cannot |
| `reason` | [`PreflightReason`](#preflightreason) | Structured reason (enum with descriptive message) |
| `required_gb` | `float` | Memory required in GB (default: 0.0) |
| `available_gb` | `float` | Available memory in GB (default: 0.0) |
| `utilization` | `float` | Memory utilization ratio 0.0-1.0 (default: 0.0) |
| `context_limit` | `int` | Model's designed maximum context in tokens (default: 0) |
| `usable_context` | `int` | Context your device can support in tokens (default: 0) |
| `estimate` | [`MemoryEstimate`](#memoryestimate) ` \| None` | Detailed memory breakdown (optional) |

**Understanding Context Fields:**

The distinction between `context_limit` and `usable_context` is critical:

- **`context_limit`**: The model's inherent maximum context length (e.g., 32,768 tokens for Llama 3). This is determined by the model's architecture and training. This value never changes based on your hardware.

- **`usable_context`**: The actual context length your hardware can support while staying within memory limits (default 85% utilization). This may be:
  - **Equal to `context_limit`**: Model fits at full designed capacity
  - **Less than `context_limit`**: Context reduced to fit in memory, but still practical (≥4K tokens minimum)
  - **Zero**: Model cannot fit (status=False)

**Example:**
```python
result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx_lm")

if result.status:
    print(f"✓ {result.reason}")
    print(f"Memory: {result.utilization*100:.0f}%")
    
    # Check for context reduction
    if result.usable_context < result.context_limit:
        reduction = (1 - result.usable_context / result.context_limit) * 100
        print(f"⚠️ Context reduced by {reduction:.0f}%:")
        print(f"   Model designed for: {result.context_limit:,} tokens")
        print(f"   Device supports: {result.usable_context:,} tokens")
    else:
        print(f"✓ Full context: {result.context_limit:,} tokens")
else:
    print(f"✗ {result.reason}")
    print(f"   Context limit: {result.context_limit:,}")
    print(f"   Usable: {result.usable_context:,}")  # Will be 0
```

---


#### `Engine`

Type alias for inference engine parameter.

**Type:**
```python
Engine = Literal["mlx_lm", "llama_cpp", "vllm"]
```

**Valid Values:**
- `"mlx_lm"` - MLX framework (Apple Silicon)
- `"llama_cpp"` - llama.cpp backend
- `"vllm"` - vLLM backend (NVIDIA GPUs)

---

#### `MemoryEstimate`

Detailed memory breakdown for a model.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `weights_gb` | `float` | Model weights size in GB |
| `kv_cache_gb` | `float` | KV cache size in GB |
| `overhead_gb` | `float` | Backend overhead (activation buffers, runtime) in GB |
| `total_required_gb` | `float` | Total memory required in GB |
| `available_gb` | `float` | Available memory in GB |
| `context_length` | `int` | Context length used for calculation |
| `max_safe_context` | `int \| None` | Maximum safe context length (optional) |
| `params_billions` | `float \| None` | Model parameter count in billions (optional) |
| `quantization` | `str \| None` | Quantization type (optional) |

**Description:**

Provides detailed breakdown of memory requirements. Includes weights (from GGUF/config), KV cache (calculated from architecture with GQA awareness), and backend-specific overhead (MLX graph compilation, vLLM PagedAttention, llama.cpp buffers).

---

#### `ModelMetadata`

Comprehensive model metadata for memory estimation.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Model identifier |
| `model_type` | `str` | Model type ("gguf", "mlx", "transformers") |
| `params_billions` | `float \| None` | Parameter count in billions |
| `bits_per_weight` | `float` | Bits per weight (default: 16.0) |
| `quantization_type` | `str \| None` | Quantization type string |
| `exact_model_size_gb` | `float \| None` | Exact model size from file/tensor calculation |
| `num_layers` | `int \| None` | Number of transformer layers |
| `hidden_size` | `int \| None` | Hidden dimension size |
| `num_attention_heads` | `int \| None` | Number of attention heads |
| `num_kv_heads` | `int \| None` | Number of KV heads (GQA-aware) |
| `head_dim` | `int \| None` | Dimension per attention head |
| `vocab_size` | `int \| None` | Vocabulary size |
| `base_context_length` | `int \| None` | Base context length from config |
| `rope_scaling_factor` | `float` | RoPE scaling multiplier (default: 1.0) |
| `model_max_context` | `int \| None` | Actual supported context length |
| `sliding_window` | `int \| None` | Sliding window size |
| `is_moe` | `bool` | Whether model is Mixture of Experts |
| `num_experts` | `int \| None` | Number of experts (MoE models) |
| `is_multimodal` | `bool` | Whether model is multimodal |
| `kv_lora_rank` | `int \| None` | DeepSeek MLA compressed KV dimension (90%+ KV cache reduction) |
| `raw_config` | `Dict[str, Any]` | Raw configuration dictionary |

**Description:**

Extracted from GGUF binary headers or HuggingFace config.json files. Used for accurate memory estimation. This is typically not accessed directly by users—it's an internal detail of `model_preflight()`.

Note: The `kv_lora_rank` field is specific to DeepSeek V2/V3 models which use Multi-Head Latent Attention (MLA) to drastically reduce KV cache memory requirements.

---

#### `RemoteInspectError`

Exception raised when remote inspection fails.

**Description:**

This exception signals that the lightweight HTTP Range-based inspection could not complete. The caller should fall back to downloading the full model for local inspection.

**Common causes:**
- Authentication required for gated models (401/403)
- Private repository without access
- Non-standard repository structure
- Network errors or timeouts
- Unsupported model format

**Example:**
```python
from edgekit.models import inspect_model_remote, RemoteInspectError

try:
    metadata = inspect_model_remote("meta-llama/Llama-3-70B", engine="vllm")
except RemoteInspectError as e:
    print(f"Light inspection failed: {e}")
    # Fall back to full download
```

---

## Module: `edgekit.build`

Build configuration utilities for LLM inference engines.

### Build Functions

#### `llama_cpp_args()`

Generate optimal CMAKE build arguments for llama-cpp-python.

**Signature:**
```python
def llama_cpp_args() -> List[str]
```

**Parameters:**
- None

**Returns:**
- `List[str]` - List of CMAKE arguments for optimal hardware acceleration

**Description:**

Detects hardware capabilities and returns CMAKE arguments that enable the best available acceleration. Automatically configures:

**GPU Acceleration:**
- **NVIDIA CUDA** - With correct compute architecture (e.g., `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 4090)
- **Apple Metal** - For M-series Macs
- **AMD ROCm/HIP** - For AMD GPUs
- **Intel SYCL** - For Intel GPUs/NPUs
- **Vulkan** - Fallback for cross-platform GPU acceleration

**CPU Optimization:**
- **BLAS acceleration** - Accelerate framework (macOS), Intel MKL (AVX-512), OpenBLAS (AVX2)
- **KleidiAI** - ARM CPU optimization (NEON/dotprod)
- **OpenMP** - Multi-threading for multi-core CPUs

**Platform-Specific:**
- Disables SVE on Apple Silicon to prevent build hangs
- Enables unified memory for NVIDIA on high-RAM Linux systems (≥32GB)

**Example:**
```python
from edgekit import llama_cpp_args

args = llama_cpp_args()
print(" ".join(args))
# Example output on Apple Silicon:
# -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate -DGGML_METAL=ON -DGGML_OPENMP=ON

# Example output on RTX 4090:
# -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA=ON -DGGML_OPENMP=ON
```

---

#### `install_command()`

Generate the complete pip install command for llama-cpp-python.

**Signature:**
```python
def install_command(shell: str = "auto") -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shell` | `str` | `"auto"` | Shell type - `"bash"`, `"powershell"`, or `"auto"` (auto-detect from OS) |

**Returns:**
- `str` - Complete pip install command with CMAKE_ARGS

**Description:**

Generates a ready-to-run pip install command with optimal CMAKE arguments for your platform. Automatically formats for bash (macOS/Linux) or PowerShell (Windows).

**Example:**
```python
from edgekit import install_command

# Auto-detect shell from OS
cmd = install_command()
print(cmd)
# macOS/Linux: CMAKE_ARGS="-DGGML_METAL=ON ..." pip install llama-cpp-python
# Windows: $env:CMAKE_ARGS = "-DGGML_CUDA=ON ..."; pip install llama-cpp-python

# Specify shell explicitly
bash_cmd = install_command(shell="bash")
ps_cmd = install_command(shell="powershell")
```

---

## Platform-Specific Notes

### macOS
- Memory detection uses Mach kernel API (`host_statistics64`) for accurate unified memory reporting
- Automatically detects 16KB page size on Apple Silicon vs 4KB on Intel
- Metal support requires macOS 12.3+ for optimal performance

### Linux
- NVIDIA detection requires `nvidia-ml-py` package (auto-installed)
- AMD detection requires `amdsmi` package (auto-installed)
- Storage type detection reads from `/sys/block/*/queue/rotational`

### Windows
- NVIDIA detection requires NVIDIA driver installed
- Intel detection requires OpenVINO runtime
- Storage type detection uses WMI (`MSFT_PhysicalDisk`)

---

## Error Handling

All functions follow these error handling principles:

1. **Graceful degradation** - If a hardware component cannot be detected, its value is `None` rather than raising an exception
2. **Field validation** - Pydantic validates all returned data at runtime
3. **Explicit errors** - `model_preflight()` raises clear exceptions for invalid engines or missing models
4. **Safe imports** - Missing optional dependencies are handled gracefully (detection returns `None` instead of import errors)

**Common Exceptions:**

| Exception | When Raised | How to Handle |
|-----------|-------------|---------------|
| `ValueError` | Invalid engine parameter in `model_preflight()` or missing model metadata | Use `"mlx_lm"`, `"llama_cpp"`, or `"vllm"`. Check `result.reason` for metadata issues. |
| `FileNotFoundError` | Local model path doesn't exist | Verify path or use HuggingFace repo ID |
| `OSError` | macOS Mach kernel API fails | Falls back to psutil if available |

---

## Type Hints & IDE Support

All public functions and classes include full type hints for:
- **IDE autocomplete** - IntelliSense/code completion
- **Static type checking** - mypy, pyright, pylance
- **Runtime validation** - Pydantic models validate at runtime

**Example with type hints:**
```python
from edgekit import system_info, model_preflight
from edgekit.hardware import HardwareProfile
from edgekit.models import PreflightResult, Engine

# Fully typed
hw: HardwareProfile = system_info()
result: PreflightResult = model_preflight("model-id", engine="mlx_lm")
engine_type: Engine = "llama_cpp"  # Type-safe literal
```

---

## JSON Serialization

All Pydantic models support JSON serialization for:
- Logging hardware profiles
- Storing validation results
- API responses
- Configuration files

**Example:**
```python
import json
from edgekit import system_info, model_preflight

# Serialize to dict
hw = system_info()
hw_dict = hw.model_dump()

# Serialize to JSON string
hw_json = hw.model_dump_json(indent=2)
print(hw_json)

# Deserialize from dict
from edgekit.hardware import HardwareProfile
hw_restored = HardwareProfile(**hw_dict)

# Validation results
result = model_preflight("model-id", engine="mlx_lm")
result_json = result.model_dump_json()
with open("preflight.json", "w") as f:
    f.write(result_json)
```
