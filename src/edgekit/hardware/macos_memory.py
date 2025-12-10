#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS-specific memory detection using Mach kernel API.

This module provides accurate memory statistics for macOS (both Apple Silicon 
and Intel Macs) by querying the Mach kernel directly via host_statistics64.

The implementation correctly handles:
- 16KB page size on Apple Silicon (vs 4KB on Intel)
- Evictable memory calculation (free + speculative + external pages)
- App memory vs Cached files distinction
- VM Compressor accounting

Based on research: "Programmatic Detection of macOS Memory Categories 
for Apple Silicon ML Workloads"
"""

import ctypes
from ctypes import Structure, c_uint32, c_uint64, byref, sizeof
from typing import Dict, Any

# Mach host_statistics64 flavor constant (from mach/host_info.h)
HOST_VM_INFO64 = 4


class vm_statistics64(Structure):
    """
    Mach VM statistics structure (from mach/vm_statistics.h).
    
    This structure contains page-level counters from the XNU kernel.
    All count fields are in units of pages, not bytes.
    """
    _fields_ = [
        ("free_count", c_uint32),
        ("active_count", c_uint32),
        ("inactive_count", c_uint32),
        ("wire_count", c_uint32),
        ("zero_fill_count", c_uint64),
        ("reactivations", c_uint64),
        ("pageins", c_uint64),
        ("pageouts", c_uint64),
        ("faults", c_uint64),
        ("cow_faults", c_uint64),
        ("lookups", c_uint64),
        ("hits", c_uint64),
        ("purges", c_uint64),
        ("purgeable_count", c_uint32),
        ("speculative_count", c_uint32),
        ("decompressions", c_uint64),
        ("compressions", c_uint64),
        ("swapins", c_uint64),
        ("swapouts", c_uint64),
        ("compressor_page_count", c_uint32),
        ("throttled_count", c_uint32),
        ("external_page_count", c_uint32),
        ("internal_page_count", c_uint32),
        ("total_uncompressed_pages_in_compressor", c_uint64)
    ]


def get_macos_memory_stats() -> Dict[str, Any]:
    """
    Get accurate memory statistics for macOS using Mach kernel API.
    
    This function queries the kernel directly using host_statistics64 to obtain
    precise memory accounting. It correctly handles the 16KB page size on
    Apple Silicon and 4KB on Intel Macs.
    
    Returns:
        Dict[str, Any]: RAM data dictionary with keys:
            - total_gb: Total RAM in gigabytes
            - available_gb: Available (evictable) RAM in gigabytes
            - details: Dict with macOS-specific memory breakdown
        
    Raises:
        OSError: If kernel API calls fail
        
    Notes:
        - Available memory formula: free + speculative + external (evictable)
        - App memory formula: internal - purgeable (anonymous pages)
        - Cached files formula: external + purgeable (file-backed + volatile)
    """
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    
    # Get page size (16KB on Apple Silicon, 4KB on Intel)
    # CRITICAL: Never hardcode 4096 - this causes 4x error on M1/M2/M3
    page_size = c_uint64()
    size = c_uint64(sizeof(page_size))
    ret = libc.sysctlbyname(b"hw.pagesize", byref(page_size), byref(size), None, 0)
    if ret != 0:
        raise OSError("Failed to get page size via sysctlbyname")
    page_size_val = page_size.value
    
    # Get total physical RAM
    total_mem = c_uint64()
    size = c_uint64(sizeof(total_mem))
    ret = libc.sysctlbyname(b"hw.memsize", byref(total_mem), byref(size), None, 0)
    if ret != 0:
        raise OSError("Failed to get total memory via sysctlbyname")
    
    # Get VM statistics from kernel via Mach API
    host_port = libc.mach_host_self()
    vm_stat = vm_statistics64()
    count = c_uint32(sizeof(vm_statistics64) // 4)
    
    ret = libc.host_statistics64(host_port, HOST_VM_INFO64, byref(vm_stat), byref(count))
    if ret != 0:
        raise OSError(f"host_statistics64 failed with error code: {ret}")
    
    # Convert page counts to GB
    to_gb = lambda pages: (pages * page_size_val) / (1024**3)
    
    # Calculate memory categories (formulas from research report)
    free_gb = to_gb(vm_stat.free_count)
    speculative_gb = to_gb(vm_stat.speculative_count)
    
    # Cached Files = external (file-backed) + purgeable (voluntary cache)
    cached_files_gb = to_gb(vm_stat.external_page_count + vm_stat.purgeable_count)
    
    # Wired Memory = locked in RAM (kernel, GPU, drivers)
    wired_gb = to_gb(vm_stat.wire_count)
    
    # Compressed Memory = pages held in compressed state
    compressed_gb = to_gb(vm_stat.compressor_page_count)
    
    # App Memory = internal (anonymous) - purgeable (not truly essential)
    # Use max(0, ...) to prevent integer underflow if purgeable > internal
    app_memory_gb = to_gb(max(0, vm_stat.internal_page_count - vm_stat.purgeable_count))
    
    # Calculate safe evictable memory (research formula: Section 5.2)
    # This is memory that can be reclaimed instantly without compression/swap
    # Free: Zero cost to reclaim
    # Speculative: Read-ahead cache, zero cost to drop
    # External: File-backed pages, low cost to evict
    available_gb = free_gb + speculative_gb + to_gb(vm_stat.external_page_count)
    
    total_gb = total_mem.value / (1024**3)
    
    # Return as dict for consistency with HardwareInspector pattern
    return {
        "total_gb": round(total_gb, 2),
        "available_gb": round(available_gb, 2),
        "details": {
            "cached_files_gb": round(cached_files_gb, 2),
            "wired_gb": round(wired_gb, 2),
            "compressed_gb": round(compressed_gb, 2),
            "app_memory_gb": round(app_memory_gb, 2),
            "speculative_gb": round(speculative_gb, 2),
            "page_size_bytes": page_size_val,
        }
    }

