"""Safe import utilities for optional dependencies."""

from typing import Optional, Tuple, Callable


def safe_import(module_name: str, package_name: str = None):
    """
    Safely import a module, returning None if unavailable.
    
    Use this for optional dependencies or platform-specific modules that may
    not be installed or available on all systems.
    
    Args:
        module_name: The module to import (e.g., "pynvml.nvml", "ctypes.wintypes")
        package_name: Display name for logging (optional, currently unused)
        
    Returns:
        The imported module, or None if import fails
        
    Examples:
        >>> pynvml = safe_import("pynvml.nvml")
        >>> if pynvml:
        ...     pynvml.nvmlInit()
        
        >>> wintypes = safe_import("ctypes.wintypes")
        >>> if not wintypes:
        ...     return None  # Not on Windows
    """
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError:
        return None
    except Exception:
        return None


def load_iokit_functions() -> Optional[Tuple[Callable, Callable, Callable, Callable, Callable]]:
    """
    Load macOS IOKit functions with fallback for different pyobjc versions.
    
    Modern pyobjc exposes IOKit functions directly, but older versions require
    loading them via NSBundle.loadFunctions(). This helper handles both cases.
    
    Returns:
        Tuple of (IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext,
                  IOObjectRelease, IORegistryEntryCreateCFProperties) or None if unavailable
                  
    Examples:
        >>> iokit = load_iokit_functions()
        >>> if iokit:
        ...     IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties = iokit
        ...     matching = IOServiceMatching(b"AGXAccelerator")
    """
    Foundation = safe_import("Foundation")
    if not Foundation:
        return None
    
    # Try direct import first (modern pyobjc)
    try:
        from IOKit import (
            IOServiceMatching, IOServiceGetMatchingServices,
            IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties
        )
        return (IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext,
                IOObjectRelease, IORegistryEntryCreateCFProperties)
    except ImportError:
        pass
    
    # Fallback: load via NSBundle (older pyobjc)
    try:
        bundle = Foundation.NSBundle.bundleWithIdentifier_('com.apple.framework.IOKit')
        if not bundle:
            return None
            
        functions = [
            b"IOServiceMatching",
            b"IOServiceGetMatchingServices",
            b"IOIteratorNext",
            b"IOObjectRelease",
            b"IORegistryEntryCreateCFProperties"
        ]
        bundle.loadFunctions(functions, globals())
        
        from IOKit import (
            IOServiceMatching, IOServiceGetMatchingServices,
            IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties
        )
        return (IOServiceMatching, IOServiceGetMatchingServices, IOIteratorNext,
                IOObjectRelease, IORegistryEntryCreateCFProperties)
    except Exception:
        return None
