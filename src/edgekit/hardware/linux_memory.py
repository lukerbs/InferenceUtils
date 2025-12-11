#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux-specific memory detection via /proc/meminfo.

This module provides detailed memory statistics for Linux systems by parsing
the /proc/meminfo virtual filesystem, which is the kernel's authoritative
source for memory information.

The implementation exposes:
- MemAvailable: The kernel's estimate of reclaimable memory (includes file cache + slab)
- SReclaimable: Slab memory that can be reclaimed (kernel caches)
- Cached: File-backed page cache
- Buffers: Kernel buffer cache

This is the Linux equivalent of the macOS Mach kernel API integration,
providing transparency into reclaimable memory that generic tools may miss.
"""

from typing import Dict, Any, Optional


def _parse_meminfo_value(value_str: str) -> float:
    """
    Parse a value from /proc/meminfo and convert to GB.
    
    Args:
        value_str: Value string like "16384 kB" or "16384"
        
    Returns:
        Value in gigabytes
    """
    # Remove 'kB' suffix if present and strip whitespace
    value_str = value_str.strip().replace('kB', '').replace('KB', '').strip()
    try:
        kb_value = int(value_str)
        return kb_value / (1024 * 1024)  # kB to GB
    except ValueError:
        return 0.0


def get_linux_memory_stats() -> Dict[str, Any]:
    """
    Get detailed memory statistics for Linux by parsing /proc/meminfo.
    
    This function reads the kernel's memory statistics directly from the
    procfs virtual filesystem. Unlike psutil (which also reads this file),
    we expose the detailed breakdown for transparency.
    
    Returns:
        Dict[str, Any]: RAM data dictionary with keys:
            - total_gb: Total RAM in gigabytes
            - available_gb: Available RAM in gigabytes (kernel's MemAvailable)
            - details: LinuxMemoryDetails-compatible dict with breakdown
            
    Raises:
        FileNotFoundError: If /proc/meminfo doesn't exist (non-Linux system)
        PermissionError: If /proc/meminfo is not readable
        
    Notes:
        - MemAvailable is the kernel's estimate of how much memory is available
          for starting new applications without swapping. It accounts for
          page cache, reclaimable slab, and other factors.
        - This is already a "smart" available memory metric, similar to what
          we calculate manually on macOS.
    """
    meminfo: Dict[str, float] = {}
    
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            meminfo[key] = _parse_meminfo_value(value)
    
    # Extract key fields (all in GB now)
    total_gb = meminfo.get('MemTotal', 0.0)
    
    # MemAvailable is the kernel's smart estimate (since kernel 3.14)
    # It includes: MemFree + Active(file) + Inactive(file) + SReclaimable (with adjustments)
    available_gb = meminfo.get('MemAvailable', 0.0)
    
    # If MemAvailable is not present (very old kernels), calculate manually
    if available_gb == 0.0:
        mem_free = meminfo.get('MemFree', 0.0)
        buffers = meminfo.get('Buffers', 0.0)
        cached = meminfo.get('Cached', 0.0)
        sreclaimable = meminfo.get('SReclaimable', 0.0)
        available_gb = mem_free + buffers + cached + sreclaimable
    
    # Build details dict matching LinuxMemoryDetails schema
    details = {
        "mem_free_gb": round(meminfo.get('MemFree', 0.0), 2),
        "buffers_gb": round(meminfo.get('Buffers', 0.0), 2),
        "cached_gb": round(meminfo.get('Cached', 0.0), 2),
        "sreclaimable_gb": round(meminfo.get('SReclaimable', 0.0), 2),
        "swap_total_gb": round(meminfo.get('SwapTotal', 0.0), 2) if 'SwapTotal' in meminfo else None,
        "swap_free_gb": round(meminfo.get('SwapFree', 0.0), 2) if 'SwapFree' in meminfo else None,
    }
    
    return {
        "total_gb": round(total_gb, 2),
        "available_gb": round(available_gb, 2),
        "details": details,
    }


def get_linux_memory_pressure() -> Optional[Dict[str, Any]]:
    """
    Get memory pressure statistics from /proc/pressure/memory (if available).
    
    This provides insight into memory contention on Linux 4.20+ kernels with
    PSI (Pressure Stall Information) enabled.
    
    Returns:
        Dict with 'some' and 'full' pressure metrics, or None if unavailable
    """
    try:
        with open('/proc/pressure/memory', 'r') as f:
            pressure = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    metric_type = parts[0]  # 'some' or 'full'
                    # Parse avg10, avg60, avg300 values
                    for part in parts[1:]:
                        if '=' in part:
                            key, value = part.split('=')
                            pressure[f"{metric_type}_{key}"] = float(value)
            return pressure if pressure else None
    except (FileNotFoundError, PermissionError):
        return None
