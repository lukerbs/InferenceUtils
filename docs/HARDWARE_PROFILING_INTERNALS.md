# Hardware Profiling Internals

> **Internal Documentation** — For EdgeKit maintainers and contributors.  
> Last updated: December 2024

This document explains how EdgeKit detects hardware capabilities and calculates available memory across different operating systems and hardware platforms. It covers the **how** (implementation approach), the **why** (rationale for design decisions), and **platform-specific considerations & nuances** that aren't immediately obvious.

---

## Table of Contents

1. [CPU Detection](#cpu-detection)
2. [System RAM Detection](#system-ram-detection)
   - [macOS (Mach Kernel API)](#macos-mach-kernel-api)
   - [Windows (Win32 + NtQuerySystemInformation)](#windows-win32--ntquerysysteminformation)
   - [Linux (/proc/meminfo)](#linux-procmeminfo)
3. [GPU Memory Detection](#gpu-memory-detection)
   - [NVIDIA (NVML)](#nvidia-nvml)
   - [AMD (amdsmi + sysfs)](#amd-amdsmi--sysfs)
   - [Intel (OpenVINO + DXGI)](#intel-openvino--dxgi)
   - [Apple Silicon (MLX + IOKit)](#apple-silicon-mlx--iokit)
4. [NPU Detection](#npu-detection)
5. [Vulkan Detection](#vulkan-detection)
6. [Storage Detection](#storage-detection)
7. [Engine Recommendation](#engine-recommendation)
8. [Memory Validator Logic](#memory-validator-logic)
9. [Utility Functions](#utility-functions)
10. [Future Considerations](#future-considerations)

---

## CPU Detection

### How It Works

We use the `py-cpuinfo` library to query CPU information including brand name, architecture, and instruction set flags. For core counts, we use `psutil` to get both physical and logical core counts.

The key focus is detecting AI-relevant instruction sets that affect inference performance:
- **AVX2**: Foundational SIMD instructions for modern x86 inference
- **AVX-512**: High-performance SIMD (Xeon, some consumer CPUs)
- **AMX**: Intel's Advanced Matrix Extensions (Sapphire Rapids+) for matrix operations
- **NEON**: ARM's SIMD equivalent (all Apple Silicon, ARM servers)

For ARM64 CPUs, we guarantee NEON support since it's mandatory for the architecture, even if `py-cpuinfo` doesn't explicitly list it in flags.

### Why This Approach

`py-cpuinfo` abstracts away platform differences and provides consistent instruction set flags across Windows, Linux, and macOS. It reads from CPUID on x86 and appropriate system files on ARM.

The instruction set detection directly influences engine recommendations - for example, OpenVINO is preferred when AMX is detected due to its specialized optimizations for this instruction set.

### Platform-Specific Considerations & Nuances

- **py-cpuinfo overhead**: The library can take 100-200ms to query full CPU info as it runs various detection routines. This is acceptable for one-time profiling but not for hot paths.

- **Fallback to platform.processor()**: If `py-cpuinfo` fails, we fall back to Python's built-in `platform.processor()` for basic CPU name, though this loses instruction set detection.

- **ARM flag detection**: NEON may not appear in `py-cpuinfo` flags for ARM64 even though it's always present, which is why we explicitly add it based on architecture detection.

---

## System RAM Detection

The goal of RAM detection is not just to report total and "free" memory, but to understand how much memory is **actually available for AI workloads**. Every operating system has caches and buffers that appear as "used" memory but are instantly reclaimable. Accurately detecting this reclaimable memory is critical for preventing false negatives in model preflight checks.

### macOS (Mach Kernel API)

#### How It Works

We query the XNU kernel directly using the Mach `host_statistics64` API. This provides page-level counters for different memory categories: free, active, inactive, wired, speculative, compressed, and external (file-backed) pages.

The available memory calculation uses the formula: **Free + Speculative + External pages**. This represents memory that can be reclaimed instantly without swapping or decompression.

We also query `hw.pagesize` via `sysctlbyname` to get the correct page size, then multiply page counts to get byte values.

#### Why This Approach

Standard tools like `psutil` report "available" memory on macOS, but their calculation doesn't account for speculative pages (read-ahead cache) which are instantly evictable. This leads to underreporting available memory by several gigabytes on systems with active file I/O.

The Mach kernel API gives us the same data that Activity Monitor uses, ensuring consistency with what users see in Apple's own tools.

#### Platform-Specific Considerations & Nuances

- **Page size varies by architecture**: Apple Silicon uses 16KB pages, Intel Macs use 4KB pages. Hardcoding 4096 bytes would cause a 4x error on M1/M2/M3/M4 Macs.

- **Unified Memory Architecture**: On Apple Silicon, the GPU and CPU share the same physical RAM. There is no separate VRAM. The "available" memory we report is the same pool that both CPU and GPU workloads draw from.

- **Wired memory is untouchable**: Wired pages are locked by the kernel and drivers (including GPU drivers). This memory cannot be reclaimed under any circumstances.

- **Compressed memory**: macOS aggressively compresses inactive pages before swapping. Compressed pages are still in RAM but require CPU cycles to decompress. We don't count these as "available" because decompression adds latency.

- **File cache vs App memory**: The "Cached Files" category (external + purgeable pages) represents file-backed data that can be instantly dropped. This is analogous to Linux's page cache.

---

### Windows (Win32 + NtQuerySystemInformation)

#### How It Works

We use two Windows APIs in combination:

1. **GlobalMemoryStatusEx**: Provides basic metrics including total physical RAM, available physical RAM, and commit charge (virtual memory usage including page file).

2. **NtQuerySystemInformation** with `SystemMemoryListInformation` (class 80): Returns the detailed breakdown of the Standby List, which is Windows' equivalent of file cache. This includes page counts by priority level (0-7), modified pages pending disk write, and truly free (zeroed) pages.

The Standby List is memory that appears "used" in Task Manager but is instantly reclaimable. We sum all priority levels to get total standby memory.

#### Why This Approach

The `GlobalMemoryStatusEx` API reports "available" memory, but this is a high-level estimate. The Standby List breakdown reveals exactly how much of that available memory is cache vs truly free, which matters for understanding allocation latency.

More importantly, `NtQuerySystemInformation` is the only way to get Standby List details without parsing performance counters or relying on external tools like RAMMap.

#### Platform-Specific Considerations & Nuances

- **NtQuerySystemInformation is undocumented**: While stable and widely used, this API is not officially documented by Microsoft. The structure layouts and information class numbers come from reverse engineering and community documentation.

- **Standby priority levels**: Priorities 0-4 are easily reclaimable (low-priority cache). Priorities 5-7 are higher priority and the system is more reluctant to evict them. For AI workload estimation, we treat all standby memory as reclaimable.

- **DWM (Desktop Window Manager) overhead**: On systems with multiple monitors or HDR enabled, DWM can consume 1-2GB of GPU memory for desktop composition. This affects VRAM availability, not system RAM, but is worth noting for GPU memory calculations.

- **Page file confusion**: Windows can allocate memory backed by the page file even when physical RAM is available. The "commit charge" can exceed physical RAM. For AI workloads, we only consider physical RAM since page file access is orders of magnitude slower.

- **Fallback behavior**: If `NtQuerySystemInformation` fails (e.g., on very old Windows versions or in sandboxed environments), we fall back to the basic `GlobalMemoryStatusEx` values and set detailed fields to None.

---

### Linux (/proc/meminfo)

#### How It Works

We parse `/proc/meminfo`, the kernel's authoritative source for memory statistics. This virtual file exposes dozens of memory counters maintained by the kernel.

The key field is `MemAvailable`, introduced in kernel 3.14. This is the kernel's own estimate of how much memory is available for new applications without swapping. It accounts for page cache, reclaimable slab memory, and low watermarks.

For kernels older than 3.14 (rare today), we calculate available memory manually as: **MemFree + Buffers + Cached + SReclaimable**.

#### Why This Approach

While `psutil` also reads `/proc/meminfo`, we expose the detailed breakdown (MemFree, Buffers, Cached, SReclaimable, Swap) for transparency. This allows downstream code to make informed decisions about memory pressure.

The kernel's `MemAvailable` is already a "smart" estimate similar to what we calculate manually on macOS, so we trust it directly rather than reimplementing the calculation.

#### Platform-Specific Considerations & Nuances

- **SReclaimable (Slab Reclaimable)**: This is kernel memory used for object caches (dentries, inodes, etc.). It can be reclaimed under memory pressure but doing so is CPU-intensive. On systems processing millions of files (e.g., ML dataset loading), SReclaimable can grow to multiple gigabytes.

- **MemAvailable vs MemFree**: MemFree is memory with nothing in it. MemAvailable is much larger because it includes cache that can be dropped. Always use MemAvailable for capacity planning.

- **Swap considerations**: We report swap statistics but don't include swap in "available" calculations. Swap is orders of magnitude slower than RAM and would cause catastrophic inference performance.

- **Memory pressure API**: Linux 4.20+ with PSI (Pressure Stall Information) enabled exposes `/proc/pressure/memory` which shows real-time memory contention. We optionally read this for diagnostics but don't use it in availability calculations.

- **Transparent Huge Pages (THP)**: Memory might be "available" in total bytes but fragmented (no contiguous 2MB blocks for huge pages). This can cause allocation latency. We don't currently check `/proc/buddyinfo` for fragmentation, but it's a potential future enhancement.

---

## GPU Memory Detection

GPU memory detection is more complex than system RAM because each vendor has its own APIs, driver models, and architectural quirks. The goal is to report both total VRAM capacity and currently available VRAM, accounting for vendor-specific overhead.

### NVIDIA (NVML)

#### How It Works

We use the NVIDIA Management Library (NVML) via the `pynvml` Python bindings. After initializing NVML, we enumerate all GPUs and query:

- Device name and driver version
- Total and free VRAM via `nvmlDeviceGetMemoryInfo`
- CUDA version supported by the driver
- ECC mode status via `nvmlDeviceGetEccMode`

We also maintain an internal lookup table for CUDA core counts, Tensor core counts, and compute capability since these aren't directly queryable via NVML.

#### Why This Approach

NVML is NVIDIA's official API for GPU management and monitoring. It's the same API that `nvidia-smi` uses internally. Using NVML directly (rather than parsing `nvidia-smi` output) gives us structured data and avoids locale/format parsing issues.

The lookup table for core counts is a pragmatic compromise. While this data exists in the GPU hardware, NVML doesn't expose it. The table includes popular consumer and data center GPUs (RTX 40/30 series, A100, H100, L40S, L4) with their CUDA core counts, Tensor core counts, and compute capability. We update the table when new GPU generations launch.

This table-based approach means newly released GPUs will report `None` for core counts until we add them, but all other fields (VRAM, driver version, ECC status) are dynamically queried.

#### Platform-Specific Considerations & Nuances

- **ECC memory tax**: Data center GPUs (A100, H100, L40S) support ECC for data integrity. When ECC is enabled, 6-12% of physical VRAM is reserved for parity data. The "total" memory reported by NVML reflects this reduction. Consumer GPUs use on-die ECC that doesn't reduce capacity.

- **CUDA context overhead**: The CUDA runtime allocates 300-800MB of VRAM just to initialize a context. This happens before any tensors are allocated. Multi-process applications multiply this overhead unless using NVIDIA MPS (Multi-Process Service).

- **Available vs Free VRAM**: We report `available_vram_gb` from `mem_info.free`. This is the current free memory, which may be less than total due to other applications. For preflight checks, current availability is what matters.

- **Driver version dependencies**: Older drivers may not support all NVML queries. We wrap each query in try/except to gracefully handle missing features.

---

### AMD (amdsmi + sysfs)

#### How It Works

We use the AMD SMI library (`amdsmi`) for querying AMD GPU information. This provides device enumeration, VRAM info, driver version, and compute unit counts.

For APU detection and GTT (Graphics Translation Table) information, we read directly from the Linux sysfs filesystem at `/sys/class/drm/card*/device/mem_info_gtt_*`.

APU detection uses a heuristic: if dedicated VRAM is less than 5% of system RAM, the device is likely an APU rather than a discrete GPU.

#### Why This Approach

AMD SMI is the official AMD library for GPU management on ROCm systems. It's required for ROCm compatibility anyway, so it's a natural choice.

The sysfs approach for GTT is necessary because AMD SMI doesn't directly expose GTT information. GTT is critical for APUs since it represents the system RAM accessible to the GPU beyond the small dedicated VRAM aperture.

#### Platform-Specific Considerations & Nuances

- **The 512MB VRAM anomaly**: AMD APUs often report only 512MB-2GB of "VRAM" because that's the dedicated aperture size configured in BIOS. The GPU can actually access much more system RAM via GTT. Naive tools that only check VRAM would incorrectly reject these devices as incapable.

- **APU detection heuristic**: We detect APUs by checking if dedicated VRAM is less than 5% of system RAM. For example, on a 64GB system, if VRAM reports as 512MB (0.8%), it's almost certainly an APU. This heuristic works because discrete GPUs have VRAM that's typically 10%+ of system RAM.

- **GTT (Graphics Translation Table)**: On APUs, GTT is how the GPU accesses the bulk of system RAM. The effective GPU memory is VRAM + GTT. We sum these for total accessible memory on APUs.

- **GTT size limits**: The kernel may cap GTT at 50-75% of system RAM by default. The `amdgpu.gttsize` kernel parameter can override this, but recent kernels have deprecated it in favor of TTM (Translation Table Manager) parameters.

- **ROCm compatibility flag**: If `amdsmi` initializes successfully, we set `rocm_compatible: True`. This indicates the GPU can be used with ROCm-based inference engines.

- **Discrete vs APU handling**: Discrete AMD GPUs (RX 7900, MI300) have large dedicated VRAM and small/zero GTT. APUs have small VRAM and large GTT. Our code handles both cases appropriately.

---

### Intel (OpenVINO + DXGI)

#### How It Works

We use the OpenVINO runtime to detect Intel GPUs and NPUs. OpenVINO's `Core().available_devices` enumerates all Intel accelerators, and we query properties like device name, execution units, and driver version.

For the critical shared memory limit on Windows, we query DXGI (DirectX Graphics Infrastructure) directly via ctypes. DXGI's `IDXGIAdapter1::GetDesc1` returns `SharedSystemMemory`, which is the WDDM-enforced cap on how much system RAM the iGPU can access.

On Linux, we detect whether the i915 or xe kernel driver is loaded, as this affects memory limits and behavior.

#### Why This Approach

OpenVINO is the de facto framework for Intel accelerator inference. Using it for detection ensures compatibility with the actual inference workload.

DXGI is the only reliable way to query the actual shared memory limit on Windows. Standard APIs like WMI or psutil don't expose this information. The alternative (assuming 50% of system RAM) is correct for default configurations but misses user-configured overrides.

#### Platform-Specific Considerations & Nuances

- **The 50% WDDM cap**: Windows caps Intel iGPU shared memory at 50% of system RAM by default. On a 64GB system, the iGPU can only address 32GB even if 60GB is physically free. This is a hard limit enforced by the video memory manager.

- **Shared Memory Override**: Newer Intel drivers (32.0.101.6987+) on Core Ultra processors allow users to increase this limit to 80-90% via "Shared GPU Memory Override" in Intel Graphics Command Center. Our DXGI query detects the actual configured limit.

- **DXGI COM interface**: We access DXGI via COM VTable traversal using ctypes. The VTable indices are stable (ABI hasn't changed since Windows 7/DX11) but must be calculated correctly by summing methods from inherited interfaces.

- **i915 vs xe drivers on Linux**: Intel is transitioning from the legacy i915 driver to the newer xe driver. Memory limits and behaviors differ between them. We detect which driver is loaded by checking `/proc/modules`.

- **iGPU vs dGPU detection**: Intel Arc discrete GPUs have dedicated VRAM and aren't subject to the shared memory cap. We distinguish iGPU from dGPU based on device enumeration patterns.

---

### Apple Silicon (MLX + IOKit)

#### How It Works

Apple Silicon GPU detection uses multiple sources:

1. **MLX** (`mlx.core.metal`): If MLX is available and Metal is supported, we query device info for architecture and total unified memory.

2. **PyTorch MPS**: As a fallback, we check `torch.backends.mps.is_available()` to confirm Metal support.

3. **IOKit**: We use PyObjC to query IOKit for granular hardware details like GPU core count and Neural Engine presence. This requires loading IOKit functions either via direct import (modern pyobjc) or NSBundle.loadFunctions (older pyobjc).

#### Why This Approach

MLX is Apple's own ML framework optimized for Apple Silicon. Using it for detection ensures we're querying through the same path that inference will use.

IOKit provides hardware-level details that aren't exposed through higher-level frameworks. While fragile (IOKit's API is not officially public), it's the only way to get GPU core counts and Neural Engine confirmation.

#### Platform-Specific Considerations & Nuances

- **Unified Memory Architecture**: There is no separate VRAM on Apple Silicon. The GPU shares the same physical RAM as the CPU. "GPU memory" and "system RAM" are the same pool.

- **Metal is required**: All GPU compute on Apple Silicon goes through Metal. If Metal isn't available (very rare edge case), GPU inference is impossible.

- **Neural Engine**: All Apple Silicon Macs have a Neural Engine (ANE), but its capabilities vary by generation. We confirm ANE presence via IOKit but don't currently detect specific ANE capabilities.

- **IOKit function loading**: Different pyobjc versions expose IOKit functions differently. Modern versions allow direct import; older versions require loading via NSBundle. We try both approaches with fallback logic.

- **GPU core counts**: M1 has 7-8 GPU cores, M1 Pro/Max have 14-32, M2/M3/M4 variants scale similarly. This affects parallel compute capacity but not memory availability.

---

## NPU Detection

Neural Processing Units are detected through multiple pathways depending on the vendor.

### Intel NPU (AI Boost)

Detected via OpenVINO's device enumeration. When `Core().available_devices` includes "NPU", we know the Intel AI Boost is present. This is available on Core Ultra ("Meteor Lake") and newer processors.

### Apple Neural Engine (ANE)

On Apple Silicon, the ANE is always present - we add it unconditionally for any arm64 Darwin system. We attempt to get the core count via IOKit by querying `AppleARMIODevice` services and looking for `ane-core-count` in devices with `compatible == "ane,2"`.

### AMD Ryzen AI (XDNA)

AMD NPU detection uses a heuristic approach since there's no direct API. We pattern-match the CPU brand string against known Ryzen AI series identifiers: "7040", "8040", "9040", "hawk point", "strix point", and "ryzen ai". This catches most Ryzen AI-equipped processors.

### Platform-Specific Considerations & Nuances

- **No unified NPU API**: Unlike GPUs which have vendor SDKs (CUDA, ROCm, Metal), NPUs lack standardized APIs for capability detection. Each vendor has proprietary methods.

- **AMD XDNA core counts unavailable**: We cannot currently query AMD NPU core counts through pure Pythonic methods. The `npu_cores` field will be `None` for AMD.

- **Intel NPU requires OpenVINO**: If OpenVINO isn't installed, we cannot detect the Intel NPU even if it's present. This is a soft dependency.

---

## Vulkan Detection

We optionally query the Vulkan API version using the `vulkan` Python library. This provides the `vkEnumerateInstanceVersion` call to get the system's Vulkan support level (e.g., "1.3.280").

This is informational and not currently used for engine selection, but useful for understanding the system's graphics API capabilities. If the `vulkan` library is not installed, this field will be `None`.

---

## Storage Detection

### How It Works

We detect whether the primary storage device is SSD/NVMe or HDD. This is useful for understanding I/O capabilities when loading large models.

**Linux**: Read the `rotational` flag from `/sys/block/{device}/queue/rotational`. A value of "0" indicates non-rotational (SSD/NVMe), "1" indicates HDD.

**Windows**: Use WMI (`root/Microsoft/Windows/Storage` namespace) to query `MSFT_PhysicalDisk.MediaType`. Value 4 = SSD, Value 3 = HDD.

**macOS**: Query IOKit for `IOBlockStorageDevice` services and check the "Solid State" property.

### Why This Approach

Each platform has native APIs that provide reliable storage type information without parsing command-line tools like `lsblk` or `diskutil`.

### Platform-Specific Considerations & Nuances

- **Virtual machines**: VMs may report incorrect storage types depending on hypervisor configuration.

- **RAID arrays**: May not report correctly if the physical devices are abstracted.

- **First device heuristic**: We check the first suitable block device found, which may not always be the boot device on multi-disk systems.

---

## Engine Recommendation

The `Recommender` class implements a priority-based decision tree to suggest the optimal inference engine based on detected hardware.

### Decision Tree Logic

The recommendation follows a tiered approach, prioritizing specialized hardware:

1. **TensorRT-LLM**: For NVIDIA GPUs with Compute Capability ≥ 8.0 (Ampere and newer). These GPUs support FP8 and advanced optimizations.

2. **vLLM**: For NVIDIA GPUs with Compute Capability ≥ 7.0 (Turing/Volta). PagedAttention and optimized CUDA kernels work well on these.

3. **MLX**: For Apple Silicon with Metal support. MLX is specifically designed for Apple's unified memory architecture.

4. **OpenVINO**: For Intel GPUs (iGPU or dGPU), Intel NPUs, or CPUs with AMX instructions. OpenVINO has specialized optimizations for all Intel hardware.

5. **llama.cpp (HIP)**: For ROCm-compatible AMD GPUs. The HIP backend provides excellent AMD GPU support.

6. **llama.cpp (CPU)**: For high-performance CPUs with AVX-512 or AVX2 (x86) or NEON (ARM). This is the fallback when no suitable GPU is available.

7. **llama.cpp (Generic)**: Default recommendation for maximum compatibility when no specialized hardware is detected.

### Why This Priority Order

The order reflects both performance and reliability. TensorRT-LLM and vLLM offer the highest throughput on NVIDIA hardware but require CUDA. MLX is the only framework that fully exploits Apple Silicon's architecture. OpenVINO is essential for Intel hardware where other frameworks have poor support. llama.cpp provides the broadest compatibility as a fallback.

---

## Memory Validator Logic

The memory validator uses hardware detection results to determine if a model will fit in memory before loading. This section describes the logic, safety margins, and thresholds.

### Preflight Thresholds

The validator uses two thresholds to classify preflight results:

- **SAFE_THRESHOLD (70%)**: Below 70% utilization = PASSED. Model will load comfortably.
- **WARNING_THRESHOLD (85%)**: 70-85% utilization = WARNING. Model will load but it's tight. Above 85% = FAILED.

These thresholds are intentionally conservative to account for runtime overhead, KV cache growth, and system processes.

### Safety Buffers

Memory buffers are deducted from available memory to ensure system stability:

| Backend | Safety Buffer | Rationale |
|---------|---------------|-----------|
| MLX (≤16GB Mac) | 3.0 GB | Fixed reserve for tight memory systems |
| MLX (>16GB Mac) | 20% of RAM | Percentage reserve for smooth operation |
| llama.cpp | 1.0 GB | Minimum for OS and basic processes |
| llama.cpp (Windows) | +1.0 GB | Additional DWM overhead on multi-monitor/HDR |
| vLLM | 0.5 GB | System overhead (vLLM already limits to 90%) |

### Backend-Specific Available Memory Calculation

Each inference backend has different memory characteristics:

**MLX (Apple Silicon)**
- Uses unified memory (system RAM)
- Tiered safety buffer: 3GB for ≤16GB Macs, 20% for >16GB Macs
- Formula: Available = RAM available - safety buffer

**vLLM (NVIDIA)**
- Uses dedicated GPU VRAM
- Prefers actual available VRAM from NVML (accounts for current usage)
- Falls back to 90% of total VRAM (vLLM's default gpu_memory_utilization)
- ECC tax is already reflected in NVML's reported available memory

**llama.cpp (CPU)**
- Uses system RAM
- Deducts 1GB safety buffer (2GB on Windows due to DWM)
- Formula: Available = RAM available - safety buffer

### AMD APU Handling

For AMD APUs, we sum VRAM and GTT for total accessible GPU memory, then apply an 85% utilization factor:
- Formula: Available = (VRAM free + GTT free) * 0.85

The 85% factor is more conservative than the 90% used for discrete GPUs because unified memory systems have more contention between CPU and GPU.

### Intel iGPU Handling

For Intel iGPUs on Windows, available memory is capped by the WDDM shared memory limit:
- Formula: Available = min(physical RAM free, SharedSystemMemory from DXGI)

This cap is the critical constraint - even if system RAM is abundant, the iGPU cannot exceed its WDDM limit.

### Context Length Calculation

After determining available memory, we calculate the maximum safe context length:

1. Subtract model weights and overhead from available memory to get "KV budget"
2. If KV budget ≤ 0, the model cannot load at all (FAILED)
3. Calculate max tokens that fit in KV budget based on model dimensions
4. If max context < 16K tokens (MIN_VIABLE_CONTEXT), return FAILED
5. Recommend using 80% of max safe context for conservative operation

---

## Utility Functions

### safe_import

A utility for safely importing optional dependencies. Returns the module if available, None otherwise. Used throughout the codebase for platform-specific modules (pynvml, amdsmi, openvino, mlx) and OS-specific modules (ctypes.wintypes on Windows).

### load_iokit_functions

A macOS-specific utility that loads IOKit functions with fallback for different pyobjc versions. Modern pyobjc exposes IOKit functions directly; older versions require loading via NSBundle.loadFunctions. This utility tries both approaches and returns a tuple of function references or None if unavailable.

---

## Future Considerations

Areas identified for potential future enhancement:

- **Linux memory fragmentation**: Check `/proc/buddyinfo` to detect fragmentation that could cause allocation issues with Transparent Huge Pages. High fragmentation means memory is "available" but may cause allocation latency.

- **NVIDIA Jetson (NvMap)**: Jetson devices use NvMap instead of standard CUDA memory management. GPU allocations appear as "Cached" in standard Linux tools, requiring special handling via `/sys/kernel/debug/nvmap/iovmm/clients`.

- **Qualcomm Hexagon**: Windows-on-ARM devices with Qualcomm Snapdragon have NPU memory managed via separate `cdsp` heaps with firmware-defined limits (often ~3.75GB max). Detection would require parsing DMA-BUF heaps.

- **Multi-GPU systems**: Current code primarily handles single-GPU scenarios (uses first GPU found). Multi-GPU memory aggregation, load balancing, and selection logic could be enhanced.

- **Linux CMA (Contiguous Memory Allocator)**: Edge devices may have CMA reservations that affect available memory for AI workloads. Checking `/proc/meminfo` for `CmaFree` could improve accuracy.

- **NVIDIA MPS detection**: Multi-Process Service amortizes CUDA context overhead when multiple processes share a GPU. Detecting MPS would allow more accurate context overhead estimation.

- **Dynamic overhead estimation**: Currently using fixed 300-800MB estimate for CUDA context. Could query actual overhead after initialization for more precise planning.
