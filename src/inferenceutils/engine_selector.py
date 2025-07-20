#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultra_pure_hardware_detector.py

An expert-level, cross-platform hardware and software detector for LLM inference environments.

This script adheres to a strict "ultra-pure" constraint: it does NOT execute any
external terminal commands (e.g., nvidia-smi, lscpu, dmidecode, system_profiler)
and parse their string output for hardware details. All information is gathered by:
1. Direct API calls using official or well-regarded Python libraries.
2. Reading from stable, well-defined OS-level filesystems (/proc, /sys) or registries.

This approach maximizes robustness and eliminates the fragility of parsing text output
that can change between OS versions, locales, or driver updates.

If a piece of information cannot be retrieved via a pure Pythonic method, its value
in the output dictionary will be None.
"""

import os
import sys
import platform
import json
from typing import Dict, Any, List, Optional, Union

# --- Constants ---
# Define instruction sets critical for AI/ML workloads
AI_INSTRUCTION_SETS = {
    'avx2', 'avx512f', 'avx512bw', 'avx512dq', 'avx512vl', 'neon', 'amx'
} # Added 'amx' for Intel CPUs


class HardwareInspector:
    """
    A class to inspect and collect detailed hardware and software information
    from the host system using pure Pythonic methods.
    """

    def __init__(self):
        """Initializes the HardwareInspector."""
        self.hw_info: Dict[str, Any] = {
            "os": {},
            "python_version": None,
            "cpu": {},
            "ram": {},
            "storage": {},
            "gpu": {
                "detected_vendor": None,
                "vulkan_api_version": None,
                "nvidia": None,
                "amd": None,
                "intel": None,
                "apple": None,
            },
            "npus": []
        }

    def _safe_import(self, module_name: str, package_name: str = None):
        """Safely imports a module and returns it, or None if import fails."""
        try:
            return __import__(module_name, fromlist=[''])
        except ImportError:
            # print(f"Info: '{package_name or module_name}' not found. Related features will be None.", file=sys.stderr)
            return None
        except Exception as e:
            # print(f"Warning: Failed to import '{module_name}': {e}. Features will be None.", file=sys.stderr)
            return None

    def _get_os_details(self):
        """Gathers basic OS and Python environment information using standard libraries."""
        self.hw_info["os"]["platform"] = platform.system()
        self.hw_info["os"]["version"] = platform.release()
        self.hw_info["os"]["architecture"] = platform.machine()
        self.hw_info["python_version"] = platform.python_version()

    def _get_cpu_details(self):
        """Gathers granular CPU information using py-cpuinfo."""
        cpu_data = {
            "brand_raw": None,
            "arch": None,
            "physical_cores": None,
            "logical_cores": None,
            "instruction_sets": None
        }
        cpuinfo = self._safe_import("cpuinfo")
        if cpuinfo:
            try:
                info = cpuinfo.get_cpu_info()
                cpu_data["brand_raw"] = info.get("brand_raw")
                cpu_data["arch"] = info.get("arch_string_raw")

                flags = info.get("flags", [])
                detected_instructions = []
                for inst_set in AI_INSTRUCTION_SETS:
                    if inst_set in flags:
                        detected_instructions.append(inst_set)
                
                # Guarantee NEON for ARM64 CPUs
                if cpu_data["arch"] and "arm64" in cpu_data["arch"].lower() and "neon" not in detected_instructions:
                    detected_instructions.append("neon")
                
                cpu_data["instruction_sets"] = detected_instructions if detected_instructions else None
            except Exception as e:
                # print(f"Warning: Could not get CPU details from py-cpuinfo: {e}", file=sys.stderr)
                cpu_data["brand_raw"] = platform.processor() # Fallback for basic name

        psutil = self._safe_import("psutil")
        if psutil:
            try:
                cpu_data["logical_cores"] = psutil.cpu_count(logical=True)
                cpu_data["physical_cores"] = psutil.cpu_count(logical=False)
            except Exception:
                # print("Warning: psutil failed to get CPU core counts.", file=sys.stderr)
                pass # Keep them None if psutil fails
            
        self.hw_info["cpu"] = cpu_data

    def _get_ram_details(self):
        """Gathers RAM total size and available RAM."""
        ram_data = {
            "total_gb": None,
            "available_gb": None,
        }
        
        psutil = self._safe_import("psutil")
        if psutil:
            try:
                mem = psutil.virtual_memory()
                ram_data["total_gb"] = round(mem.total / (1024**3), 2)
                ram_data["available_gb"] = round(mem.available / (1024**3), 2)
            except Exception as e:
                # print(f"Warning: psutil failed to get RAM usage. Error: {e}", file=sys.stderr)
                pass

        # RAM Speed: Explicitly removed as per directive. Cannot be obtained reliably
        # without elevated privileges or subprocess parsing.
        
        self.hw_info["ram"] = ram_data

    def _get_storage_details(self):
        """
        Determines the type of the primary storage device (SSD or HDD).
        Uses platform-specific pure methods.
        """
        storage_data = {
            "primary_type": None
        }
        
        system = self.hw_info["os"]["platform"]
        try:
            if system == "Linux":
                # Read from /sys filesystem for rotational flag
                # This is a heuristic; a more robust solution would trace the mount point of '/'.
                for device in sorted(os.listdir("/sys/block")):
                    if device.startswith(("sd", "nvme", "vd", "hd")): # Common block device prefixes
                        rotational_path = f"/sys/block/{device}/queue/rotational"
                        if os.path.exists(rotational_path):
                            with open(rotational_path, "r") as f:
                                # '0' is non-rotational (SSD/NVMe), '1' is rotational (HDD)
                                if f.read().strip() == "0":
                                    storage_data["primary_type"] = "SSD/NVMe"
                                else:
                                    storage_data["primary_type"] = "HDD"
                                break # Found the first plausible primary disk
            elif system == "Windows":
                wmi = self._safe_import("wmi")
                if wmi:
                    c = wmi.WMI(namespace="root/Microsoft/Windows/Storage")
                    # MediaType: 3=HDD, 4=SSD, 5=SCM, 0=Unspecified
                    for disk in c.MSFT_PhysicalDisk():
                        if disk.MediaType == 4 or disk.MediaType == 5:
                            storage_data["primary_type"] = "SSD/NVMe"
                        elif disk.MediaType == 3:
                            storage_data["primary_type"] = "HDD"
                        else:
                            storage_data["primary_type"] = "Unknown"
                        break # Check first physical disk
            elif system == "Darwin": # macOS
                # Use PyObjC to query IOKit
                Foundation = self._safe_import("Foundation")
                if Foundation:
                    try:
                        IOKit = Foundation.NSBundle.bundleWithIdentifier_('com.apple.framework.IOKit')
                        # Load necessary functions
                        functions = [
                            b"IOServiceMatching",
                            b"IOServiceGetMatchingServices",
                            b"IOIteratorNext",
                            b"IOObjectRelease",
                            b"IORegistryEntryCreateCFProperties"
                        ]
                        # Use a dictionary to load functions if `loadFunctions` is used directly, or access via `objc.currentBundle()`
                        # Modern pyobjc often exposes these directly via 'from IOKit import ...'
                        try: # Try direct import from IOKit
                            from IOKit import IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties
                        except ImportError: # Fallback for older pyobjc or if not directly exposed
                            _ = IOKit.loadFunctions(functions, globals()) # globals() will load them into global namespace
                            from IOKit import IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties

                        matching_dict = IOServiceMatching(b"IOBlockStorageDevice")
                        iterator = IOServiceGetMatchingServices(0, matching_dict) # 0 for kIOMasterPortDefault
                        
                        if iterator:
                            while (drive := IOIteratorNext(iterator)):
                                # Get properties of the drive
                                err, properties = IORegistryEntryCreateCFProperties(drive, None, None, 0)
                                if properties:
                                    if properties.get("Solid State"): # kIOPropertySolidStateKey
                                        storage_data["primary_type"] = "SSD/NVMe"
                                        IOObjectRelease(drive)
                                        break
                                    elif properties.get("Drive Type") == "Rotational":
                                        storage_data["primary_type"] = "HDD"
                                        IOObjectRelease(drive)
                                        break
                                IOObjectRelease(drive)
                            IOObjectRelease(iterator)
                    except Exception as e:
                        # print(f"Warning: Failed to get macOS storage info via IOKit: {e}", file=sys.stderr)
                        pass
        except Exception as e:
            # print(f"Warning: Could not infer storage type: {e}", file=sys.stderr)
            pass
        
        self.hw_info["storage"] = storage_data

    def _get_vulkan_version(self):
        """Gets Vulkan API version using the official vulkan library."""
        vulkan_api_version = None
        vulkan = self._safe_import("vulkan")
        if vulkan:
            try:
                from ctypes import byref, c_uint32
                pApiVersion = c_uint32(0)
                # vkEnumerateInstanceVersion is the direct API call
                result = vulkan.vkEnumerateInstanceVersion(byref(pApiVersion))
                if result == vulkan.VK_SUCCESS:
                    version = pApiVersion.value
                    major = vulkan.VK_VERSION_MAJOR(version)
                    minor = vulkan.VK_VERSION_MINOR(version)
                    patch = vulkan.VK_VERSION_PATCH(version)
                    vulkan_api_version = f"{major}.{minor}.{patch}"
            except Exception: # Catch any error including AttributeError if function not in loaded lib
                pass
        
        self.hw_info["gpu"]["vulkan_api_version"] = vulkan_api_version

    def _get_nvidia_gpus(self):
        """Gathers NVIDIA GPU details using pynvml."""
        nvidia_gpus_list = []
        pynvml = self._safe_import("pynvml.nvml")
        if pynvml:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                
                # Internal lookup table for cores (pragmatic solution for unavailable API data)
                # This needs to be updated manually for new GPUs
                core_lookup = {
                    "RTX 4090": {"cuda": 16384, "tensor": 512, "compute_cap": 8.9},
                    "RTX 4080": {"cuda": 9728, "tensor": 304, "compute_cap": 8.9},
                    "RTX 4070": {"cuda": 5888, "tensor": 184, "compute_cap": 8.9},
                    "RTX 3090": {"cuda": 10496, "tensor": 328, "compute_cap": 8.6},
                    "RTX 3080": {"cuda": 8704, "tensor": 272, "compute_cap": 8.6},
                    "A100": {"cuda": 6912, "tensor": 432, "compute_cap": 8.0},
                    "H100": {"cuda": 16896, "tensor": 528, "compute_cap": 9.0},
                    "L40S": {"cuda": 18176, "tensor": 568, "compute_cap": 8.9},
                    "L4": {"cuda": 7680, "tensor": 240, "compute_cap": 8.9},
                }

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get CUDA version supported by driver
                    cuda_version = None
                    try:
                        # nvmlSystemGetCudaDriverVersion may not be in all pynvml versions
                        cuda_version_int = pynvml.nvmlSystemGetCudaDriverVersion()
                        cuda_version = f"{cuda_version_int // 1000}.{(cuda_version_int % 1000) // 10}"
                    except (AttributeError, pynvml.NVMLError):
                        pass

                    # Use lookup table for CUDA/Tensor cores and Compute Capability
                    gpu_name_for_lookup = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                    matched_cores_cc = None
                    for key in core_lookup:
                        if key in gpu_name_for_lookup:
                            matched_cores_cc = core_lookup[key]
                            break

                    gpu_info = {
                        "model": name,
                        "vram_gb": round(mem_info.total / (1024**3), 2),
                        "driver_version": driver_version,
                        "cuda_version": cuda_version,
                        "compute_capability": matched_cores_cc.get("compute_cap") if matched_cores_cc else None,
                        "cuda_cores": matched_cores_cc.get("cuda") if matched_cores_cc else None,
                        "tensor_cores": matched_cores_cc.get("tensor") if matched_cores_cc else None
                    }
                    nvidia_gpus_list.append(gpu_info)
                    
                pynvml.nvmlShutdown()
                
                if nvidia_gpus_list:
                    self.hw_info["gpu"]["detected_vendor"] = "NVIDIA"
                    self.hw_info["gpu"]["nvidia"] = nvidia_gpus_list

            except pynvml.NVMLError:
                # print("Info: NVIDIA driver/GPU not found or NVML error.", file=sys.stderr)
                pass # Fail silently if no NVIDIA GPU or driver issues
            except Exception as e:
                # print(f"Warning: Unexpected error during NVIDIA GPU detection: {e}", file=sys.stderr)
                pass

    def _get_amd_gpus(self):
        """Gathers AMD GPU details and checks for ROCm compatibility using amdsmi."""
        # Skip AMD detection on macOS as it doesn't have AMD GPUs
        if self.hw_info["os"]["platform"] == "Darwin":
            return
            
        amd_gpus_list = []
        amdsmi = self._safe_import("amdsmi")
        if amdsmi:
            try:
                amdsmi.amdsmi_init()
                devices = amdsmi.amdsmi_get_processor_handles()
                
                if not devices:
                    amdsmi.amdsmi_shut_down()
                    return

                driver_version_info = amdsmi.amdsmi_get_driver_info()
                driver_version = driver_version_info.get('driver_version') if driver_version_info else None

                for dev_handle in devices:
                    gpu_info = {}
                    try:
                        name = amdsmi.amdsmi_get_gpu_product_name(dev_handle)
                        vram_info = amdsmi.amdsmi_get_gpu_vram_info(dev_handle)
                        asic_info = amdsmi.amdsmi_get_gpu_asic_info(dev_handle)
                        
                        gpu_info["model"] = name.decode('utf-8') if name else None
                        gpu_info["vram_gb"] = round(vram_info.get('vram_total', 0) / (1024**3), 2) if vram_info else None
                        gpu_info["driver_version"] = driver_version
                        gpu_info["rocm_compatible"] = True # If amdsmi works, ROCm is compatible
                        gpu_info["compute_units"] = asic_info.get('num_of_compute_units') if asic_info else None

                        amd_gpus_list.append(gpu_info)
                    except amdsmi.AmdSmiException:
                        # print(f"Warning: AMD SMI exception for device handle: {e}", file=sys.stderr)
                        continue # Skip device if specific queries fail
                
                amdsmi.amdsmi_shut_down()
                
                if amd_gpus_list:
                    self.hw_info["gpu"]["detected_vendor"] = "AMD"
                    self.hw_info["gpu"]["amd"] = amd_gpus_list

            except (amdsmi.AmdSmiException, Exception):
                # print(f"Info: AMD amdsmi library failed. No ROCm GPU detected. {e}", file=sys.stderr)
                pass # Fail silently if amdsmi fails or ROCm not installed

    def _get_intel_accelerators(self):
        """Gathers Intel GPU and NPU details using OpenVINO."""
        intel_devices_list = []
        ov = self._safe_import("openvino")
        if ov:
            try:
                core = ov.Core()
                available_devices = core.available_devices
                
                for device in available_devices:
                    if "GPU" in device or "NPU" in device:
                        accel_info = {}
                        
                        name = core.get_property(device, "FULL_DEVICE_NAME")
                        accel_info["model"] = name

                        if "GPU" in device:
                            accel_info["type"] = "dGPU" if ".1" in device else "iGPU"
                            accel_info["execution_units"] = core.get_property(device, "GPU_EXECUTION_UNITS_COUNT")
                            # VRAM for Intel GPUs might be shared or dedicated
                            accel_info["vram_gb"] = round(core.get_property(device, "GPU_MEMORY_SIZE") / (1024**3), 2)
                            accel_info["driver_version"] = core.get_property(device, "GPU_DRIVER_VERSION") # Pure API
                        else: # NPU
                            accel_info["type"] = "NPU"
                            accel_info["execution_units"] = None # N/A for NPU
                            accel_info["vram_gb"] = None # NPU uses system RAM directly
                            accel_info["driver_version"] = None # Not easily exposed for NPU

                        intel_devices_list.append(accel_info)

                if intel_devices_list:
                    self.hw_info["gpu"]["detected_vendor"] = "Intel"
                    self.hw_info["gpu"]["intel"] = intel_devices_list
                    # Check for Intel NPU explicitly
                    if any(dev.get("type") == "NPU" for dev in intel_devices_list):
                        self.hw_info["npus"].append({
                            "vendor": "Intel",
                            "model_name": "Intel AI Boost" # Generic name for now
                        })

            except (RuntimeError, Exception): # RuntimeError if device properties are missing
                # print(f"Info: OpenVINO failed to query Intel accelerator details. {e}", file=sys.stderr)
                pass

    def _get_apple_gpus_and_npus(self):
        """Gathers Apple Silicon GPU and NPU details using MLX and PyObjC."""
        if self.hw_info["os"]["platform"] == "Darwin" and self.hw_info["os"]["architecture"] == "arm64":
            # Guarantee Apple Neural Engine presence for Apple Silicon
            self.hw_info["npus"].append({
                "vendor": "Apple",
                "model_name": "Apple Neural Engine",
                "npu_cores": None  # Will be updated if PyObjC succeeds
            })
            apple_gpu_info = {
                "model": None,
                "vram_gb": None, # Unified memory, total system RAM
                "gpu_cores": None,
                "metal_supported": False
            }
            
            # Check for Metal availability via MLX and PyTorch
            mlx_metal = self._safe_import("mlx.core.metal")
            if mlx_metal and mlx_metal.is_available():
                apple_gpu_info["metal_supported"] = True
                device_info = mlx_metal.device_info()
                apple_gpu_info["model"] = device_info.get('architecture', 'Apple Silicon GPU') # MLX gives 'architecture' like 'arm64e'
                apple_gpu_info["vram_gb"] = round(device_info.get('memory_size', 0) / (1024**3), 2) # Total unified memory

            if not apple_gpu_info["metal_supported"]: # Fallback to PyTorch MPS if MLX not found
                torch = self._safe_import("torch")
                if torch and torch.backends.mps.is_available():
                    apple_gpu_info["metal_supported"] = True
                    # MPS doesn't give specific model/memory like MLX, so rely on OS data later if needed

            # Use PyObjC to query IOKit for more granular (but fragile) details
            Foundation = self._safe_import("Foundation")
            if Foundation:
                try:
                    # IOKit bridge setup (similar to Storage, ensures functions are loaded)
                    IOKit_bundle = Foundation.NSBundle.bundleWithIdentifier_('com.apple.framework.IOKit')
                    try:
                        from IOKit import IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties
                    except ImportError:
                        # Fallback for older pyobjc or if not directly exposed
                        functions = [
                            b"IOServiceMatching", b"IOServiceGetMatchingServices", b"IOIteratorNext",
                            b"IOObjectRelease", b"IORegistryEntryCreateCFProperties"
                        ]
                        _ = IOKit_bundle.loadFunctions(functions, globals())


                    # --- GPU Cores ---
                    matching = IOServiceMatching(b"AGXAccelerator")
                    iterator = IOServiceGetMatchingServices(0, matching)
                    if iterator:
                        service = IOIteratorNext(iterator)
                        if service:
                            err, props = IORegistryEntryCreateCFProperties(service, None, None, 0)
                            if props and 'gpu-core-count' in props:
                                apple_gpu_info["gpu_cores"] = props['gpu-core-count']
                            IOObjectRelease(service)
                        IOObjectRelease(iterator)
                    
                    # --- NPU Cores / Presence ---
                    ane_cores = None
                    matching_ane = IOServiceMatching(b"AppleARMIODevice") # ANE often found here
                    iterator_ane = IOServiceGetMatchingServices(0, matching_ane)
                    if iterator_ane:
                        while (service_ane := IOIteratorNext(iterator_ane)):
                            err, props_ane = IORegistryEntryCreateCFProperties(service_ane, None, None, 0)
                            if props_ane and props_ane.get("compatible") == "ane,2": # Heuristic for ANE
                                ane_cores = props_ane.get("ane-core-count")
                                break # Found it
                            IOObjectRelease(service_ane)
                        IOObjectRelease(iterator_ane)
                    
                    if ane_cores is not None:
                        # Update the existing Apple Neural Engine entry with core count
                        for npu in self.hw_info["npus"]:
                            if npu.get("vendor") == "Apple" and npu.get("model_name") == "Apple Neural Engine":
                                npu["npu_cores"] = ane_cores
                                break
                except Exception as e:
                    # print(f"Warning: Failed to get macOS GPU/NPU details via PyObjC/IOKit: {e}", file=sys.stderr)
                    pass # Keep defaults None if PyObjC fails

            # Set Apple GPU details if any usable info was found
            if apple_gpu_info["metal_supported"] or apple_gpu_info["model"]:
                self.hw_info["gpu"]["detected_vendor"] = "Apple"
                self.hw_info["gpu"]["apple"] = apple_gpu_info
            else:
                self.hw_info["gpu"]["apple"] = None # Explicitly set to None if no Apple GPU detected

    def inspect_all(self) -> Dict[str, Any]:
        """Runs all inspection methods to build the complete hardware profile."""
        # Order is important as some detections depend on others
        self._get_os_details()
        self._get_cpu_details()
        self._get_ram_details() # RAM info needed for Apple Silicon VRAM
        self._get_storage_details()
        self._get_vulkan_version()
        
        # GPU detection is vendor-specific and mutually exclusive for "detected_vendor"
        self._get_nvidia_gpus()
        if not self.hw_info["gpu"]["detected_vendor"]: # Only try next if no vendor found yet
            self._get_amd_gpus()
        if not self.hw_info["gpu"]["detected_vendor"]:
            self._get_intel_accelerators()
        if not self.hw_info["gpu"]["detected_vendor"]:
            self._get_apple_gpus_and_npus() # Combines Apple GPU and NPU logic
            
        # AMD Ryzen AI NPU (Heuristic based on CPU name)
        cpu_name = self.hw_info["cpu"].get("brand_raw", "").lower()
        # List of known Ryzen AI CPU series identifiers (updated based on common patterns)
        ryzen_ai_series = ["7040", "8040", "9040", "hawk point", "strix point", "ryzen ai"]
        if any(series in cpu_name for series in ryzen_ai_series):
            # Ensure it's not already added from Intel/Apple NPU detection (unlikely but safe)
            if not any(npu.get("vendor") == "AMD" for npu in self.hw_info["npus"]):
                self.hw_info["npus"].append({
                    "vendor": "AMD",
                    "model_name": "AMD Ryzen AI",
                    "npu_cores": None # Core count not directly available via pure methods
                })
        
        return self.hw_info

class Recommender:
    """
    Recommends an LLM inference engine based on hardware specifications.
    The decision tree prioritizes the most specialized hardware available.
    """

    def recommend(self, hw_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Applies a decision tree to the hardware info to recommend an engine.

        Args:
            hw_info: A dictionary of hardware information from HardwareInspector.

        Returns:
            A dictionary containing the recommended engine name and reason.
        """
        # Tier 1: Top-Tier NVIDIA GPU for TensorRT-LLM
        nvidia_gpus = hw_info.get("gpu", {}).get("nvidia")
        if nvidia_gpus:
            # Check for Ampere (CC 8.0) or newer (Hopper CC 9.0)
            top_tier_gpus = [gpu for gpu in nvidia_gpus if gpu.get("compute_capability") is not None and gpu["compute_capability"] >= 8.0]
            if top_tier_gpus:
                highest_cc = max(gpu["compute_capability"] for gpu in top_tier_gpus)
                return {
                    "name": "TensorRT-LLM",
                    "reason": f"Optimal choice for high-end NVIDIA hardware. A GPU with Compute Capability {highest_cc} (Ampere or newer) was detected, enabling state-of-the-art optimizations including FP8 support."
                }
            
            # Tier 4: High-Performance NVIDIA GPU for vLLM
            # Turing (CC 7.5) and Volta (CC 7.0) are excellent for vLLM
            capable_vllm_gpus = [gpu for gpu in nvidia_gpus if gpu.get("compute_capability") is not None and gpu["compute_capability"] >= 7.0]
            if capable_vllm_gpus:
                highest_cc_vllm = max(gpu["compute_capability"] for gpu in capable_vllm_gpus)
                return {
                    "name": "vLLM",
                    "reason": f"High-throughput engine for NVIDIA GPUs. A GPU with Compute Capability {highest_cc_vllm} (Turing/Volta or newer) was detected, ideal for vLLM's PagedAttention and optimized CUDA kernels."
                }

        # Tier 2: Apple Silicon for MLX
        apple_gpu_info = hw_info.get("gpu", {}).get("apple")
        if apple_gpu_info and apple_gpu_info.get("metal_supported"):
            return {
                "name": "MLX",
                "reason": "Natively designed for Apple Silicon. The system's unified memory architecture is best exploited by Apple's own MLX framework, which leverages the CPU, GPU, and Neural Engine."
            }

        # Tier 3: Intel Accelerator for OpenVINO
        intel_gpus = hw_info.get("gpu", {}).get("intel")
        if intel_gpus: # Check for Intel GPUs first
            # Check for dGPU as priority over iGPU for performance
            has_dgpu = any(dev.get("type") == "dGPU" for dev in intel_gpus)
            if has_dgpu:
                return {
                    "name": "OpenVINO",
                    "reason": "Intel discrete GPU detected. OpenVINO provides the most optimized software stack for inference on Intel GPU hardware."
                }
            # If no dGPU, check for iGPU
            has_igpu = any(dev.get("type") == "iGPU" for dev in intel_gpus)
            if has_igpu:
                return {
                    "name": "OpenVINO",
                    "reason": "Intel integrated GPU detected. OpenVINO provides optimized inference capabilities on Intel iGPUs."
                }
        
        # Check NPUs specifically (after GPUs, but before generic CPU)
        npus = hw_info.get("npus", [])
        if any(npu.get("vendor") == "Intel" for npu in npus):
            return {
                "name": "OpenVINO",
                "reason": "Intel NPU detected. OpenVINO is the only framework specifically optimized to leverage this low-power, high-efficiency AI accelerator."
            }
        
        # Tier 5: ROCm-Compatible AMD GPU for llama.cpp
        amd_gpus = hw_info.get("gpu", {}).get("amd")
        if amd_gpus and any(gpu.get("rocm_compatible") for gpu in amd_gpus):
            return {
                "name": "llama.cpp", # vLLM is also strong for AMD, but llama.cpp's HIP is very mature
                "reason": "ROCm-compatible AMD GPU detected. llama.cpp offers a robust and highly performant HIP backend for AMD hardware, providing excellent open-source support."
            }
        
        # Tier 6: High-Performance x86/ARM CPU for llama.cpp
        cpu_arch = hw_info.get("cpu", {}).get("arch")
        instruction_sets = hw_info.get("cpu", {}).get("instruction_sets", [])

        if cpu_arch and "x86" in cpu_arch.lower():
            if "avx512f" in instruction_sets:
                return {
                    "name": "llama.cpp",
                    "reason": "High-performance CPU with AVX-512 support detected. In the absence of a supported GPU, llama.cpp's AVX-512 kernels provide the best possible CPU performance."
                }
            if "avx2" in instruction_sets:
                return {
                    "name": "llama.cpp",
                    "reason": "CPU with AVX2 support detected. In the absence of a supported GPU, llama.cpp's AVX2 kernels offer significantly accelerated CPU performance."
                }
            if "amx" in instruction_sets: # Intel AMX for Sapphire Rapids+
                return {
                    "name": "OpenVINO", # OpenVINO specifically optimizes for AMX
                    "reason": "Intel CPU with AMX instruction set detected. OpenVINO offers specialized optimizations for AMX, providing superior CPU inference performance."
                }
        
        if cpu_arch and "arm" in cpu_arch.lower():
            if "neon" in instruction_sets:
                 return {
                    "name": "llama.cpp",
                    "reason": "ARM CPU with NEON support detected. In the absence of a supported GPU, llama.cpp's NEON optimizations provide the best available CPU performance."
                }
            # Check for ARM-based NPUs (e.g., AMD Ryzen AI, often detected by CPU name heuristic)
            if any(npu.get("vendor") == "AMD" for npu in npus):
                return {
                    "name": "OpenVINO", # OpenVINO (or other frameworks) would leverage these for inference
                    "reason": "AMD Ryzen AI NPU detected. This specialized AI accelerator will provide optimal performance for compatible models."
                }


        # Tier 7: Generic System (Default Fallback)
        # This covers older CPUs, or systems where no specialized hardware could be identified.
        return {
            "name": "llama.cpp",
            "reason": "Default recommendation for broad compatibility. llama.cpp is the most versatile engine and provides reliable performance on generic CPU hardware."
        }

def main():
    """
    Main function to run the hardware inspector and recommender,
    and print the results as a JSON object.
    """
    inspector = HardwareInspector()
    hardware_info = inspector.inspect_all()

    recommender = Recommender()
    recommendation = recommender.recommend(hardware_info)

    hardware_info["recommended_engine"] = recommendation

    # Print the final result as a nicely formatted JSON
    print(json.dumps(hardware_info, indent=4))

if __name__ == "__main__":
    main()