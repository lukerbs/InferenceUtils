# InferenceUtils - Intelligent Hardware Detection for LLM Inference

A comprehensive Python library that automatically detects your hardware capabilities and provides optimal recommendations for LLM inference engines and build configurations.

## 🚀 Quick Start

```python
from inferenceutils import systeminfo, optimal_inference_engine, llama_cpp_build_args

# Get comprehensive hardware information
hw = systeminfo()
print(f"CPU: {hw.cpu.brand_raw}")
print(f"GPU: {hw.gpu.detected_vendor}")
print(f"RAM: {hw.ram.total_gb} GB")

# Get optimal inference engine recommendation
engine = optimal_inference_engine()
print(f"Recommended: {engine.name}")
print(f"Install: pip install {' '.join(engine.dependencies)}")

# Get optimal build arguments for llama-cpp-python
args = llama_cpp_build_args()
print(f"CMAKE_ARGS: {' '.join(args)}")
```

## ✨ Key Features

### **🔍 Intelligent Hardware Detection**
- **Cross-platform**: macOS, Linux, Windows
- **Comprehensive**: CPU, GPU, RAM, storage, instruction sets
- **Type-safe**: All data validated with Pydantic schemas
- **Pure Python**: No external command execution required

### **🎯 Optimal Engine Recommendations**
- **Hardware-aware**: Automatically selects best engine for your system
- **Dependencies included**: Provides exact pip install commands
- **Detailed reasoning**: Explains why each engine was chosen
- **Performance-focused**: Prioritizes fastest available hardware

### **⚡ Build Optimization**
- **llama-cpp-python**: Optimal CMAKE arguments for your hardware
- **GPU acceleration**: CUDA, Metal, ROCm, Vulkan, SYCL
- **CPU optimization**: AVX-512, AVX2, OpenMP, KleidiAI
- **Platform-specific**: Different optimizations per OS

## 📦 Installation

```bash
# Install from source
git clone <repository-url>
cd InferenceUtils
pip install -e .

# Or install dependencies manually
pip install py-cpuinfo psutil nvidia-ml-py amdsmi openvino mlx pyobjc vulkan pydantic
```

## 🛠️ API Reference

### Core Functions

#### `systeminfo() -> HardwareProfile`
Get comprehensive hardware information as a validated Pydantic BaseModel.

```python
from inferenceutils import systeminfo

hw = systeminfo()

# Access typed data
print(f"OS: {hw.os.platform}")
print(f"CPU: {hw.cpu.brand_raw}")
print(f"RAM: {hw.ram.total_gb} GB")

# GPU information
if hw.gpu.detected_vendor == "NVIDIA":
    for gpu in hw.gpu.nvidia:
        print(f"NVIDIA: {gpu.model} ({gpu.vram_gb} GB)")
elif hw.gpu.detected_vendor == "Apple":
    print(f"Apple: {hw.gpu.apple.model}")

# Convert to JSON
json_data = hw.model_dump_json(indent=2)
```

#### `optimal_inference_engine() -> OptimalInferenceEngine`
Get the optimal inference engine recommendation with dependencies.

```python
from inferenceutils import optimal_inference_engine

engine = optimal_inference_engine()

print(f"Engine: {engine.name}")
print(f"Dependencies: {engine.dependencies}")
print(f"Reason: {engine.reason}")

# Install the recommended engine
install_cmd = f"pip install {' '.join(engine.dependencies)}"
print(f"Run: {install_cmd}")
```

#### `llama_cpp_build_args() -> List[str]`
Get optimal CMAKE build arguments for llama-cpp-python.

```python
from inferenceutils import llama_cpp_build_args, get_llama_cpp_install_command

# Get build arguments
args = llama_cpp_build_args()
print(f"CMAKE arguments: {' '.join(args)}")

# Get complete install command
install_cmd = get_llama_cpp_install_command()
print(f"Install command: {install_cmd}")
```

### Pydantic Schemas

#### `HardwareProfile`
Complete hardware profile with all detected components.

```python
from inferenceutils import HardwareProfile

# Validate hardware data
try:
    profile = HardwareProfile(**hardware_data)
    print("✅ Data is valid")
except ValidationError as e:
    print(f"❌ Validation failed: {e}")
```

#### `OptimalInferenceEngine`
Inference engine recommendation with dependencies.

```python
from inferenceutils import OptimalInferenceEngine

# Create recommendation
recommendation = OptimalInferenceEngine(
    name="MLX",
    dependencies=["mlx-lm"],
    reason="Optimized for Apple Silicon"
)
```

## 🎯 Supported Inference Engines

| Engine | Best For | Dependencies |
|--------|----------|--------------|
| **TensorRT-LLM** | High-end NVIDIA GPUs (Ampere+) | `tensorrt-llm` |
| **vLLM** | NVIDIA GPUs (Turing/Volta+) | `vllm` |
| **MLX** | Apple Silicon | `mlx-lm` |
| **OpenVINO** | Intel GPUs/NPUs | `openvino` |
| **llama.cpp** | AMD GPUs, high-performance CPUs | `llama-cpp-python` |

## 🔧 Hardware Acceleration Support

### GPU Backends
- **NVIDIA CUDA**: Automatic compute capability detection
- **Apple Metal**: Native Apple Silicon optimization
- **AMD ROCm**: HIP acceleration for AMD GPUs
- **Intel SYCL**: oneAPI for Intel accelerators
- **Vulkan**: Cross-platform GPU acceleration

### CPU Optimizations
- **Intel oneMKL**: AVX-512 optimization
- **OpenBLAS**: AVX2 acceleration
- **OpenMP**: Multi-core parallelism
- **KleidiAI**: ARM CPU optimization

## 📋 Example Output

### Hardware Detection
```json
{
  "os": {
    "platform": "Darwin",
    "version": "23.0.0",
    "architecture": "arm64"
  },
  "cpu": {
    "brand_raw": "Apple M2 Pro",
    "physical_cores": 10,
    "logical_cores": 10,
    "instruction_sets": ["neon"]
  },
  "ram": {
    "total_gb": 32.0,
    "available_gb": 24.5
  },
  "gpu": {
    "detected_vendor": "Apple",
    "apple": {
      "model": "Apple Silicon GPU",
      "vram_gb": 32.0,
      "metal_supported": true
    }
  }
}
```

### Engine Recommendation
```json
{
  "name": "MLX",
  "dependencies": ["mlx-lm"],
  "reason": "Natively designed for Apple Silicon. The system's unified memory architecture is best exploited by Apple's own MLX framework, which leverages the CPU, GPU, and Neural Engine."
}
```

### Build Arguments
```bash
# Apple Silicon
-DGGML_METAL=ON -DGGML_SVE=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Accelerate

# NVIDIA GPU
-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
```

## 🚀 Use Cases

### Development Setup
```python
from inferenceutils import systeminfo, optimal_inference_engine

# Quick hardware overview
hw = systeminfo()
print(f"Setting up development environment for {hw.cpu.brand_raw}")

# Get recommended engine
engine = optimal_inference_engine()
print(f"Installing {engine.name}...")
```

### CI/CD Pipelines
```python
from inferenceutils import llama_cpp_build_args

# Generate build args for different runners
args = llama_cpp_build_args()
print(f"Building with: {' '.join(args)}")
```

### User Documentation
```python
from inferenceutils import optimal_inference_engine, get_llama_cpp_install_command

# Generate user-specific instructions
engine = optimal_inference_engine()
if engine.name == "llama.cpp":
    install_cmd = get_llama_cpp_install_command()
    print(f"Run: {install_cmd}")
else:
    print(f"Run: pip install {' '.join(engine.dependencies)}")
```

## 🔍 Hardware Detection Capabilities

### CPU Detection
- Model and architecture
- Core count (physical/logical)
- Instruction sets (AVX-512, AVX2, NEON, AMX)
- Performance characteristics

### GPU Detection
- **NVIDIA**: Model, VRAM, compute capability, driver version
- **AMD**: Model, VRAM, ROCm compatibility, compute units
- **Intel**: Model, type (dGPU/iGPU/NPU), execution units
- **Apple**: Model, unified memory, Metal support, GPU cores

### Memory & Storage
- Total and available RAM
- Primary storage type (SSD/HDD)
- Memory bandwidth considerations

### NPU Detection
- **Apple Neural Engine**: Core count, availability
- **Intel AI Boost**: NPU detection and capabilities
- **AMD Ryzen AI**: CPU-based detection

## 🛠️ Dependencies

### Core Dependencies
- **py-cpuinfo**: CPU information
- **psutil**: System and process utilities
- **nvidia-ml-py**: NVIDIA GPU monitoring
- **openvino**: Intel accelerator support
- **mlx**: Apple Silicon support
- **pyobjc**: macOS system integration
- **vulkan**: Vulkan API support
- **pydantic**: Data validation and serialization
- **amdsmi**: AMD GPU monitoring (install with `pip install inferenceutils[amd]`)

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.