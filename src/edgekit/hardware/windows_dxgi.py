#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows DXGI (DirectX Graphics Infrastructure) queries via ctypes.

This module provides low-level access to DXGI adapter information without
external dependencies, using COM VTable traversal via ctypes.

The primary use case is querying the SharedSystemMemory limit for Intel iGPUs,
which represents the WDDM-enforced cap on how much system RAM the iGPU can access.
By default, Windows caps this at 50% of system RAM, but newer Intel drivers
(32.0.101.6987+) allow users to override this limit.

This is the only reliable way to detect the actual limit - standard APIs like
psutil or WMI don't expose this information.
"""

import sys
import ctypes
from ctypes import (
    Structure, POINTER, c_void_p, c_ulonglong, c_uint, c_long, 
    c_wchar, byref, cast
)
from typing import Optional, Tuple


# --- Constants & GUIDs ---

DXGI_ERROR_NOT_FOUND = 0x887A0002

# IID_IDXGIFactory1: {770aae78-f26f-4dba-a829-253c83d1b387}
IID_IDXGIFactory1 = (ctypes.c_ubyte * 16)(
    0x78, 0xae, 0x0a, 0x77, 0x6f, 0xf2, 0xba, 0x4d,
    0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87
)

# Intel Vendor ID
INTEL_VENDOR_ID = 0x8086


# --- C Structures for DXGI ---

class LUID(Structure):
    """Locally Unique Identifier - used for adapter identification."""
    _fields_ = [("LowPart", c_uint), ("HighPart", c_long)]


class DXGI_ADAPTER_DESC1(Structure):
    """
    DXGI adapter description structure.
    
    Key fields:
    - DedicatedVideoMemory: VRAM on discrete GPUs
    - DedicatedSystemMemory: Reserved RAM (BIOS/Stolen memory)
    - SharedSystemMemory: WDDM limit on system RAM accessible to GPU
    """
    _fields_ = [
        ("Description", c_wchar * 128),
        ("VendorId", c_uint),
        ("DeviceId", c_uint),
        ("SubSysId", c_uint),
        ("Revision", c_uint),
        ("DedicatedVideoMemory", c_ulonglong),
        ("DedicatedSystemMemory", c_ulonglong),
        ("SharedSystemMemory", c_ulonglong),  # The WDDM limit we want
        ("AdapterLuid", LUID),
        ("Flags", c_uint),
    ]


# --- COM VTable Helper ---

def _call_com_method(interface_ptr, vtable_index, argtypes, restype, *args):
    """
    Call a COM method by dereferencing the VTable.
    
    COM interfaces are pointers to pointers to VTables (arrays of function pointers).
    This helper dereferences the interface pointer, finds the method at the given
    VTable index, and calls it.
    
    Args:
        interface_ptr: The COM interface pointer (e.g., IDXGIFactory1*)
        vtable_index: Index of the method in the VTable
        argtypes: ctypes argument types for the function
        restype: ctypes return type
        *args: Arguments to pass to the method (excluding 'this' pointer)
        
    Returns:
        The return value of the COM method (usually HRESULT)
    """
    if not interface_ptr:
        return -1
    
    # interface_ptr is a pointer to a pointer to the VTable
    vtable_ptr = cast(interface_ptr, POINTER(c_void_p)).contents.value
    
    # Calculate address of the method pointer in the VTable
    method_ptr_addr = vtable_ptr + (vtable_index * ctypes.sizeof(c_void_p))
    method_addr = cast(method_ptr_addr, POINTER(c_void_p)).contents.value
    
    # Define the C function prototype
    func = ctypes.WINFUNCTYPE(restype, *argtypes)(method_addr)
    
    # Call it (passing 'self' interface_ptr as first arg)
    return func(interface_ptr, *args)


# --- VTable Indices (stable ABI since Windows 7/DX11) ---
# 
# COM VTables are arrays of function pointers. Index = sum of methods in all
# inherited interfaces:
#
# IDXGIFactory1 inherits: IDXGIFactory -> IDXGIObject -> IUnknown
#   IUnknown (3): QueryInterface, AddRef, Release
#   IDXGIObject (3): SetPrivateData, SetPrivateDataInterface, GetParent  
#   IDXGIFactory (5): EnumAdapters, MakeWindowAssociation, GetWindowAssociation,
#                     CreateSwapChain, CreateSoftwareAdapter
#   IDXGIFactory1 (1): EnumAdapters1 = index 11
#
# IDXGIAdapter1 inherits: IDXGIAdapter -> IDXGIObject -> IUnknown
#   IUnknown (3) + IDXGIObject (3) = 6
#   IDXGIAdapter (3): EnumOutputs, GetDesc, CheckInterfaceSupport
#   IDXGIAdapter1 (1): GetDesc1 = index 9

# IUnknown methods
VTABLE_RELEASE = 2

# IDXGIFactory1 methods
VTABLE_ENUM_ADAPTERS1 = 11  # NOT 12 - that would be IsCurrent()

# IDXGIAdapter1 methods  
VTABLE_GET_DESC1 = 9  # NOT 10


# --- Public API ---

def get_intel_shared_memory_limit_bytes() -> Optional[int]:
    """
    Query DXGI for the SharedSystemMemory limit of the first Intel GPU.
    
    This returns the WDDM-enforced cap on how much system RAM the Intel iGPU
    can access. By default this is 50% of system RAM, but users with newer
    drivers can override it via "Shared GPU Memory Override" in Intel Graphics
    Command Center.
    
    Returns:
        SharedSystemMemory limit in bytes, or None if:
        - Not on Windows
        - DXGI is unavailable
        - No Intel adapter found
    """
    if sys.platform != 'win32':
        return None
    
    try:
        dxgi = ctypes.windll.dxgi
    except (AttributeError, OSError):
        return None
    
    # CreateDXGIFactory1
    CreateDXGIFactory1 = dxgi.CreateDXGIFactory1
    CreateDXGIFactory1.argtypes = [POINTER(ctypes.c_ubyte * 16), POINTER(c_void_p)]
    CreateDXGIFactory1.restype = c_long
    
    factory_ptr = c_void_p()
    hr = CreateDXGIFactory1(byref(IID_IDXGIFactory1), byref(factory_ptr))
    
    if hr != 0 or not factory_ptr:
        return None
    
    intel_shared_limit = None
    adapter_index = 0
    
    try:
        while True:
            adapter_ptr = c_void_p()
            
            # EnumAdapters1(UINT Adapter, IDXGIAdapter1 **ppAdapter)
            hr = _call_com_method(
                factory_ptr,
                VTABLE_ENUM_ADAPTERS1,
                [c_void_p, c_uint, POINTER(c_void_p)],
                c_long,
                adapter_index, byref(adapter_ptr)
            )
            
            if hr == DXGI_ERROR_NOT_FOUND:
                break  # No more adapters
            
            if hr != 0 or not adapter_ptr:
                adapter_index += 1
                continue
            
            try:
                # GetDesc1(DXGI_ADAPTER_DESC1 *pDesc)
                desc = DXGI_ADAPTER_DESC1()
                hr_desc = _call_com_method(
                    adapter_ptr,
                    VTABLE_GET_DESC1,
                    [c_void_p, POINTER(DXGI_ADAPTER_DESC1)],
                    c_long,
                    byref(desc)
                )
                
                if hr_desc == 0 and desc.VendorId == INTEL_VENDOR_ID:
                    intel_shared_limit = desc.SharedSystemMemory
                    break  # Found Intel, stop searching
                    
            finally:
                # Release adapter to prevent memory leaks
                _call_com_method(adapter_ptr, VTABLE_RELEASE, [c_void_p], c_long)
            
            adapter_index += 1
            
    finally:
        # Release factory
        _call_com_method(factory_ptr, VTABLE_RELEASE, [c_void_p], c_long)
    
    return intel_shared_limit


def get_intel_shared_memory_limit_gb() -> Optional[float]:
    """
    Query DXGI for the SharedSystemMemory limit of the first Intel GPU.
    
    Returns:
        SharedSystemMemory limit in gigabytes, or None if unavailable
    """
    limit_bytes = get_intel_shared_memory_limit_bytes()
    if limit_bytes is not None:
        return round(limit_bytes / (1024 ** 3), 2)
    return None


def get_all_adapter_info() -> list:
    """
    Query DXGI for information about all graphics adapters.
    
    Useful for debugging or getting a complete picture of the system's GPUs.
    
    Returns:
        List of dicts with adapter info, or empty list if unavailable
    """
    if sys.platform != 'win32':
        return []
    
    try:
        dxgi = ctypes.windll.dxgi
    except (AttributeError, OSError):
        return []
    
    CreateDXGIFactory1 = dxgi.CreateDXGIFactory1
    CreateDXGIFactory1.argtypes = [POINTER(ctypes.c_ubyte * 16), POINTER(c_void_p)]
    CreateDXGIFactory1.restype = c_long
    
    factory_ptr = c_void_p()
    hr = CreateDXGIFactory1(byref(IID_IDXGIFactory1), byref(factory_ptr))
    
    if hr != 0 or not factory_ptr:
        return []
    
    adapters = []
    adapter_index = 0
    
    try:
        while True:
            adapter_ptr = c_void_p()
            
            hr = _call_com_method(
                factory_ptr,
                VTABLE_ENUM_ADAPTERS1,
                [c_void_p, c_uint, POINTER(c_void_p)],
                c_long,
                adapter_index, byref(adapter_ptr)
            )
            
            if hr == DXGI_ERROR_NOT_FOUND:
                break
            
            if hr != 0 or not adapter_ptr:
                adapter_index += 1
                continue
            
            try:
                desc = DXGI_ADAPTER_DESC1()
                hr_desc = _call_com_method(
                    adapter_ptr,
                    VTABLE_GET_DESC1,
                    [c_void_p, POINTER(DXGI_ADAPTER_DESC1)],
                    c_long,
                    byref(desc)
                )
                
                if hr_desc == 0:
                    adapters.append({
                        "description": desc.Description,
                        "vendor_id": hex(desc.VendorId),
                        "device_id": hex(desc.DeviceId),
                        "dedicated_video_memory_gb": round(desc.DedicatedVideoMemory / (1024**3), 2),
                        "dedicated_system_memory_gb": round(desc.DedicatedSystemMemory / (1024**3), 2),
                        "shared_system_memory_gb": round(desc.SharedSystemMemory / (1024**3), 2),
                        "is_intel": desc.VendorId == INTEL_VENDOR_ID,
                    })
                    
            finally:
                _call_com_method(adapter_ptr, VTABLE_RELEASE, [c_void_p], c_long)
            
            adapter_index += 1
            
    finally:
        _call_com_method(factory_ptr, VTABLE_RELEASE, [c_void_p], c_long)
    
    return adapters
