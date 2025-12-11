#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows-specific memory detection via Win32 API.

This module provides detailed memory statistics for Windows systems by querying
the Windows Memory Manager's internal state, specifically the Standby List.

The Standby List is Windows' equivalent of macOS cached files - it contains
pages that are not actively in use but retained in RAM for quick access.
These pages are instantly reclaimable when applications need memory.

The implementation uses:
- GlobalMemoryStatusEx: Basic memory info (total, available)
- NtQuerySystemInformation: Detailed memory list breakdown (Standby, Modified, Free)

This is the Windows equivalent of the macOS Mach kernel API integration,
providing transparency into reclaimable memory that Task Manager may obscure.
"""

import ctypes
from ctypes import Structure, c_ulong, c_ulonglong, sizeof, byref, POINTER
from typing import Dict, Any, Optional

from ..utils import safe_import


class MEMORYSTATUSEX(Structure):
    """Windows MEMORYSTATUSEX structure for GlobalMemoryStatusEx."""
    _fields_ = [
        ("dwLength", c_ulong),
        ("dwMemoryLoad", c_ulong),
        ("ullTotalPhys", c_ulonglong),
        ("ullAvailPhys", c_ulonglong),
        ("ullTotalPageFile", c_ulonglong),
        ("ullAvailPageFile", c_ulonglong),
        ("ullTotalVirtual", c_ulonglong),
        ("ullAvailVirtual", c_ulonglong),
        ("ullAvailExtendedVirtual", c_ulonglong),
    ]


class SYSTEM_MEMORY_LIST_INFORMATION(Structure):
    """
    Windows SYSTEM_MEMORY_LIST_INFORMATION structure.
    
    This structure contains the memory list breakdown including:
    - ZeroPageCount: Pages that are zeroed and ready for allocation
    - FreePageCount: Free pages (not zeroed)
    - ModifiedPageCount: Dirty pages waiting to be written to disk
    - ModifiedNoWritePageCount: Modified but marked no-write
    - BadPageCount: Pages with hardware errors
    - PageCountByPriority[8]: Standby pages by priority (0-7)
    
    Priorities 0-4 are easily reclaimable, 5-7 are higher priority caches.
    """
    _fields_ = [
        ("ZeroPageCount", c_ulonglong),
        ("FreePageCount", c_ulonglong),
        ("ModifiedPageCount", c_ulonglong),
        ("ModifiedNoWritePageCount", c_ulonglong),
        ("BadPageCount", c_ulonglong),
        ("PageCountByPriority", c_ulonglong * 8),  # Standby list by priority
        ("RepurposedPagesByPriority", c_ulonglong * 8),
        ("ModifiedPageCountPageFile", c_ulonglong),
    ]


def _get_page_size() -> int:
    """Get the system page size in bytes."""
    wintypes = safe_import("ctypes.wintypes")
    if not wintypes:
        return 4096  # Default page size (not on Windows)
        
    try:
        from ctypes import windll
        kernel32 = windll.kernel32
        
        class SYSTEM_INFO(Structure):
            _fields_ = [
                ("wProcessorArchitecture", ctypes.c_ushort),
                ("wReserved", ctypes.c_ushort),
                ("dwPageSize", c_ulong),
                ("lpMinimumApplicationAddress", ctypes.c_void_p),
                ("lpMaximumApplicationAddress", ctypes.c_void_p),
                ("dwActiveProcessorMask", ctypes.POINTER(c_ulong)),
                ("dwNumberOfProcessors", c_ulong),
                ("dwProcessorType", c_ulong),
                ("dwAllocationGranularity", c_ulong),
                ("wProcessorLevel", ctypes.c_ushort),
                ("wProcessorRevision", ctypes.c_ushort),
            ]
        
        si = SYSTEM_INFO()
        kernel32.GetNativeSystemInfo(byref(si))
        return si.dwPageSize
    except Exception:
        return 4096  # Default to 4KB


def _get_basic_memory_stats() -> Dict[str, float]:
    """
    Get basic memory statistics using GlobalMemoryStatusEx.
    
    This is the fallback method if NtQuerySystemInformation fails.
    """
    wintypes = safe_import("ctypes.wintypes")
    if not wintypes:
        raise OSError("Windows API not available on this platform")
        
    from ctypes import windll
    kernel32 = windll.kernel32
    
    mem_status = MEMORYSTATUSEX()
    mem_status.dwLength = sizeof(MEMORYSTATUSEX)
    
    if not kernel32.GlobalMemoryStatusEx(byref(mem_status)):
        raise OSError("GlobalMemoryStatusEx failed")
    
    return {
        "total_gb": mem_status.ullTotalPhys / (1024**3),
        "available_gb": mem_status.ullAvailPhys / (1024**3),
        "total_page_file_gb": mem_status.ullTotalPageFile / (1024**3),
        "available_page_file_gb": mem_status.ullAvailPageFile / (1024**3),
        "memory_load_percent": mem_status.dwMemoryLoad,
    }


def _get_memory_list_info() -> Optional[Dict[str, float]]:
    """
    Get detailed memory list information using NtQuerySystemInformation.
    
    This function queries the Windows kernel for the Standby List breakdown,
    which shows how much "cached" memory is actually instantly reclaimable.
    
    Returns:
        Dict with memory list breakdown in GB, or None if API call fails
    """
    wintypes = safe_import("ctypes.wintypes")
    if not wintypes:
        return None
        
    try:
        from ctypes import windll
        ntdll = windll.ntdll
        
        # Get the memory list information
        mem_list = SYSTEM_MEMORY_LIST_INFORMATION()
        return_length = c_ulong(0)
        
        # NtQuerySystemInformation(SystemMemoryListInformation, buffer, size, return_length)
        # Information class 80 = SystemMemoryListInformation
        status = ntdll.NtQuerySystemInformation(
            80,  # SystemMemoryListInformation
            byref(mem_list),
            sizeof(mem_list),
            byref(return_length)
        )
        
        # NTSTATUS success is 0
        if status != 0:
            return None
        
        page_size = _get_page_size()
        to_gb = lambda pages: (pages * page_size) / (1024**3)
        
        # Calculate total standby (sum of all priorities)
        standby_pages = sum(mem_list.PageCountByPriority[i] for i in range(8))
        
        # Low priority standby (priorities 0-4) - easily reclaimable
        low_priority_standby = sum(mem_list.PageCountByPriority[i] for i in range(5))
        
        # High priority standby (priorities 5-7) - less easily reclaimable
        high_priority_standby = sum(mem_list.PageCountByPriority[i] for i in range(5, 8))
        
        return {
            "zero_pages_gb": to_gb(mem_list.ZeroPageCount),
            "free_pages_gb": to_gb(mem_list.FreePageCount),
            "modified_pages_gb": to_gb(mem_list.ModifiedPageCount),
            "standby_total_gb": to_gb(standby_pages),
            "standby_low_priority_gb": to_gb(low_priority_standby),
            "standby_high_priority_gb": to_gb(high_priority_standby),
        }
    except Exception:
        return None


def get_windows_memory_stats() -> Dict[str, Any]:
    """
    Get detailed memory statistics for Windows including Standby List.
    
    This function combines GlobalMemoryStatusEx (for basic stats) with
    NtQuerySystemInformation (for Standby List breakdown) to provide
    a complete picture of Windows memory state.
    
    The Standby List is Windows' equivalent of macOS cached files - memory
    that appears "used" but is instantly reclaimable for new allocations.
    
    Returns:
        Dict[str, Any]: RAM data dictionary with keys:
            - total_gb: Total RAM in gigabytes
            - available_gb: Available RAM in gigabytes
            - details: WindowsMemoryDetails-compatible dict with breakdown
            
    Raises:
        OSError: If GlobalMemoryStatusEx fails
    """
    # Get basic stats (always works)
    basic = _get_basic_memory_stats()
    
    # Try to get detailed memory list info
    mem_list = _get_memory_list_info()
    
    if mem_list:
        # Calculate detailed breakdown
        free_gb = mem_list.get("zero_pages_gb", 0) + mem_list.get("free_pages_gb", 0)
        standby_gb = mem_list.get("standby_total_gb", 0)
        modified_gb = mem_list.get("modified_pages_gb", 0)
        cached_gb = standby_gb + modified_gb
        
        details = {
            "standby_gb": round(standby_gb, 2),
            "modified_gb": round(modified_gb, 2),
            "free_gb": round(free_gb, 2),
            "cached_gb": round(cached_gb, 2),
            "commit_total_gb": round(basic.get("total_page_file_gb", 0) - basic.get("available_page_file_gb", 0), 2),
            "commit_limit_gb": round(basic.get("total_page_file_gb", 0), 2),
        }
    else:
        # Fallback: estimate from basic stats
        # Windows "available" is approximately free + standby, so we estimate
        available_gb = basic.get("available_gb", 0)
        
        details = {
            "standby_gb": None,  # Unknown without NtQuerySystemInformation
            "modified_gb": None,
            "free_gb": round(available_gb, 2),  # Approximation
            "cached_gb": None,
            "commit_total_gb": round(basic.get("total_page_file_gb", 0) - basic.get("available_page_file_gb", 0), 2),
            "commit_limit_gb": round(basic.get("total_page_file_gb", 0), 2),
        }
    
    return {
        "total_gb": round(basic.get("total_gb", 0), 2),
        "available_gb": round(basic.get("available_gb", 0), 2),
        "details": details,
    }
