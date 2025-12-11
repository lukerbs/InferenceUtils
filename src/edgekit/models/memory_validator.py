"""
Model Preflight Check

Validates if a model will fit in memory before loading.
Prevents OOM errors by checking hardware capacity against model requirements.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Tuple

from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

from ..hardware import system_info

# Module logger
logger = logging.getLogger(__name__)

# Type alias for inference engine parameter
Engine = Literal["mlx", "llama_cpp", "vllm"]

from .model_inspector import (
    inspect_gguf_model,
    inspect_transformers_model,
    ModelMetadata,
)
from .model_inspector_remote import (
    inspect_model_remote,
    RemoteInspectError,
)
from .memory_estimator import (
    estimate_llamacpp_memory,
    estimate_mlx_memory,
    estimate_vllm_memory,
    calculate_max_safe_context,
    MemoryEstimate
)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

SAFE_THRESHOLD = 0.70       # < 70% utilization = passed
WARNING_THRESHOLD = 0.85    # 70-85% = warning, > 85% = failed

LLAMACPP_SAFETY_BUFFER_GB = 1.0
VLLM_SYSTEM_OVERHEAD_GB = 0.5

MIN_VIABLE_CONTEXT = 16384  # 16K tokens minimum
DEFAULT_CONTEXT = 32768     # 32K tokens default


def get_macos_safety_buffer_gb(total_ram_gb: float) -> float:
    """
    Get tiered safety buffer for macOS unified memory.
    
    macOS requires memory reserved for:
    - WindowServer and display buffer (~1-2GB on high-res displays)
    - Kernel and system services
    - Wired memory limits (cannot swap ML tensors)
    
    Research-validated tiers:
    - <=16GB Macs: Fixed 3GB reserve (tight but viable)
    - >16GB Macs: 20% reserve (smooth operation zone)
    
    Args:
        total_ram_gb: Total system RAM in GB
        
    Returns:
        Safety buffer in GB
    """
    if total_ram_gb <= 16:
        return 3.0
    return total_ram_gb * 0.20


# ============================================================================
# PREFLIGHT RESULT
# ============================================================================

class PreflightStatus(Enum):
    """Preflight check status."""
    PASSED = "passed"       # Safe to load
    WARNING = "warning"     # Can load but tight fit
    FAILED = "failed"       # Don't attempt to load
    UNKNOWN = "unknown"     # Could not determine


@dataclass
class PreflightResult:
    """Result of model preflight check."""
    
    status: PreflightStatus
    message: str
    
    # Memory details
    required_gb: float = 0.0
    available_gb: float = 0.0
    utilization: float = 0.0
    
    # Context information
    requested_context: int = 0
    recommended_context: int = 0
    max_context: int = 0  # Renamed from max_safe_context
    
    # Detailed breakdown (optional)
    estimate: Optional[MemoryEstimate] = None
    
    # -------------------------------------------------------------------------
    # Pythonic helper properties
    # -------------------------------------------------------------------------
    
    @property
    def passed(self) -> bool:
        """True if model can load safely."""
        return self.status == PreflightStatus.PASSED
    
    @property
    def warning(self) -> bool:
        """True if model can load but is a tight fit."""
        return self.status == PreflightStatus.WARNING
    
    @property
    def failed(self) -> bool:
        """True if model won't fit - don't attempt to load."""
        return self.status == PreflightStatus.FAILED
    
    @property
    def can_load(self) -> bool:
        """True if model can load (passed or warning)."""
        return self.status in (PreflightStatus.PASSED, PreflightStatus.WARNING)
    
    def raise_if_failed(self) -> None:
        """Raise MemoryError if preflight failed."""
        if self.failed:
            raise MemoryError(self.message)


# Backwards compatibility aliases
ValidationStatus = PreflightStatus
ValidationResult = PreflightResult


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def get_available_memory(backend: str) -> Tuple[float, str]:
    """
    Get available memory for the specified backend.
    
    Uses platform-specific memory detection to provide accurate estimates.
    Accounts for quirks like:
    - macOS: Cached files and speculative memory (reclaimable)
    - Windows: Standby list, DWM overhead (~1GB on multi-monitor/HDR setups)
    - Linux: SReclaimable slab memory
    - NVIDIA: ECC tax (6-12% on data center GPUs), available VRAM vs total
    - AMD APU: GTT (system RAM accessible to GPU) beyond small VRAM aperture
    - Intel iGPU: WDDM 50% shared memory cap on Windows
    
    Args:
        backend: "mlx_lm", "vllm", or "llama_cpp"
        
    Returns:
        Tuple of (available_gb, memory_type_description)
    """
    try:
        hw = system_info()
        platform = hw.os.platform
        
        if backend == "mlx_lm":
            # Apple Silicon unified memory (already uses Mach API for accurate available)
            total_ram = hw.ram.total_gb or 16  # Default to 16GB if unknown
            available = hw.ram.available_gb or 0
            safety_buffer = get_macos_safety_buffer_gb(total_ram)
            usable = available - safety_buffer
            return max(0, usable), "unified memory (macOS)"
        
        elif backend == "vllm":
            # NVIDIA GPU VRAM
            if hw.gpu.nvidia and len(hw.gpu.nvidia) > 0:
                gpu = hw.gpu.nvidia[0]
                
                # Prefer actual available VRAM if we have it (accounts for current usage)
                if gpu.available_vram_gb is not None:
                    available_vram = gpu.available_vram_gb
                else:
                    # Fallback to total * 0.9 (vLLM default utilization)
                    available_vram = gpu.vram_gb * 0.9
                
                # Note: ECC tax is already reflected in available_vram_gb if ECC is enabled
                # We don't need to deduct it again - the NVML memory info already accounts for it
                
                return available_vram, "GPU VRAM"
            
            # Check AMD GPU with GTT (APU can run some inference)
            elif hw.gpu.amd and len(hw.gpu.amd) > 0:
                gpu = hw.gpu.amd[0]
                if gpu.is_apu and gpu.gtt_total_gb:
                    # APU: can use VRAM + GTT for total accessible memory
                    gtt_free = (gpu.gtt_total_gb or 0) - (gpu.gtt_used_gb or 0)
                    vram_free = gpu.available_vram_gb or gpu.vram_gb or 0
                    total_available = vram_free + gtt_free
                    return total_available * 0.85, "APU unified memory (VRAM+GTT)"
                elif gpu.available_vram_gb:
                    return gpu.available_vram_gb * 0.9, "GPU VRAM"
            
            return 0.0, "no compatible GPU detected"
        
        elif backend == "llama_cpp":
            # System RAM - llama.cpp runs on CPU primarily
            available = hw.ram.available_gb or 0
            
            # Apply platform-specific deductions for system overhead
            if platform == "Windows":
                # Windows DWM overhead: ~1GB on multi-monitor/HDR, ~0.5GB otherwise
                # Use conservative estimate since we can't detect monitor config easily
                dwm_overhead = 1.0
                available = available - dwm_overhead
            elif platform == "Linux":
                # Linux MemAvailable is already smart, but add small buffer for safety
                pass  # SReclaimable is already in MemAvailable calculation
            
            usable = available - LLAMACPP_SAFETY_BUFFER_GB
            return max(0, usable), "system RAM"
        
        else:
            # Unknown backend
            return 16.0, "unknown"
            
    except Exception:
        return 16.0, "estimated"


def get_gpu_available_memory() -> Tuple[float, str]:
    """
    Get available GPU memory, accounting for platform-specific quirks.
    
    Returns:
        Tuple of (available_gb, description)
    """
    try:
        hw = system_info()
        
        # NVIDIA
        if hw.gpu.nvidia and len(hw.gpu.nvidia) > 0:
            gpu = hw.gpu.nvidia[0]
            available = gpu.available_vram_gb or (gpu.vram_gb * 0.9)
            note = " (ECC enabled - 6-12% capacity reduction)" if gpu.ecc_enabled else ""
            return available, f"NVIDIA {gpu.model}{note}"
        
        # AMD
        if hw.gpu.amd and len(hw.gpu.amd) > 0:
            gpu = hw.gpu.amd[0]
            if gpu.is_apu and gpu.gtt_total_gb:
                gtt_free = (gpu.gtt_total_gb or 0) - (gpu.gtt_used_gb or 0)
                vram_free = gpu.available_vram_gb or 0
                return vram_free + gtt_free, f"AMD APU {gpu.model} (VRAM + GTT)"
            return gpu.available_vram_gb or gpu.vram_gb or 0, f"AMD {gpu.model}"
        
        # Intel
        if hw.gpu.intel and len(hw.gpu.intel) > 0:
            gpu = hw.gpu.intel[0]
            if gpu.type == "iGPU" and gpu.shared_memory_limit_gb:
                # iGPU is capped by WDDM limit
                return gpu.shared_memory_limit_gb, f"Intel iGPU (WDDM limit: {gpu.shared_memory_limit_gb}GB)"
            return gpu.vram_gb or 0, f"Intel {gpu.model}"
        
        # Apple (unified memory)
        if hw.gpu.apple:
            return hw.ram.available_gb or 0, "Apple unified memory"
        
        return 0.0, "no GPU detected"
        
    except Exception:
        return 0.0, "GPU detection failed"


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def _validate_gguf_memory(
    gguf_path: str,
    model_id: str
) -> PreflightResult:
    """
    Internal: Validate memory for a GGUF model (llama.cpp backend).
    
    Args:
        gguf_path: Path to the local GGUF file
        model_id: Original model ID
        
    Returns:
        ValidationResult with status and recommendations
    """
    try:
        # Extract metadata from GGUF file
        metadata = inspect_gguf_model(gguf_path, model_id)
        
        # Determine context length (use model's full max)
        if metadata.model_max_context:
            context = metadata.model_max_context  # Use full model max!
        else:
            context = DEFAULT_CONTEXT  # Fallback only if unknown
        
        # Get available memory
        available_gb, mem_type = get_available_memory("llama_cpp")
        
        if available_gb <= 0:
            return PreflightResult(
                status=PreflightStatus.UNKNOWN,
                message="Could not determine available memory",
                recommended_context=context
            )
        
        # Calculate memory requirements
        estimate = estimate_llamacpp_memory(metadata, context)
        estimate.available_gb = available_gb
        
        # Calculate utilization
        utilization = estimate.total_required_gb / available_gb
        
        # Calculate max safe context (handle negative budget)
        kv_budget = available_gb - estimate.weights_gb - estimate.overhead_gb
        if kv_budget <= 0:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason="Model weights and overhead alone exceed available memory"
                ),
                required_gb=estimate.total_required_gb,
                available_gb=available_gb,
                utilization=utilization,
                estimate=estimate
            )
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        # Determine status
        return _make_decision(
            utilization=utilization,
            estimate=estimate,
            metadata=metadata,
            requested_context=context,
            max_safe_context=max_safe_context,
            available_gb=available_gb,
            mem_type=mem_type
        )
        
    except Exception as e:
        return PreflightResult(
            status=PreflightStatus.UNKNOWN,
            message=f"Memory validation failed: {e}",
            recommended_context=DEFAULT_CONTEXT
        )


def _validate_mlx_memory(
    model_cache_path: str,
    model_id: str
) -> PreflightResult:
    """
    Internal: Validate memory for an MLX model.
    
    Args:
        model_cache_path: Path to the cached model directory
        model_id: Original model ID
        
    Returns:
        ValidationResult with status and recommendations
    """
    try:
        # Extract metadata from config.json
        metadata = inspect_transformers_model(model_cache_path, model_id)
        
        # Determine context length (use model's full max)
        if metadata.model_max_context:
            context = metadata.model_max_context  # Use full model max!
        else:
            context = DEFAULT_CONTEXT  # Fallback only if unknown
        
        # Get available memory
        available_gb, mem_type = get_available_memory("mlx_lm")
        
        if available_gb <= 0:
            return PreflightResult(
                status=PreflightStatus.UNKNOWN,
                message="Could not determine available memory",
                recommended_context=context
            )
        
        # Calculate memory requirements
        estimate = estimate_mlx_memory(metadata, context)
        estimate.available_gb = available_gb
        
        # Calculate utilization
        utilization = estimate.total_required_gb / available_gb
        
        # Calculate max safe context (handle negative budget)
        kv_budget = available_gb - estimate.weights_gb - estimate.overhead_gb
        if kv_budget <= 0:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason="Model weights and overhead alone exceed available memory"
                ),
                required_gb=estimate.total_required_gb,
                available_gb=available_gb,
                utilization=utilization,
                estimate=estimate
            )
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        # Determine status
        return _make_decision(
            utilization=utilization,
            estimate=estimate,
            metadata=metadata,
            requested_context=context,
            max_safe_context=max_safe_context,
            available_gb=available_gb,
            mem_type=mem_type
        )
        
    except Exception as e:
        return PreflightResult(
            status=PreflightStatus.UNKNOWN,
            message=f"Memory validation failed: {e}",
            recommended_context=DEFAULT_CONTEXT
        )


def _validate_vllm_memory(
    model_cache_path: str,
    model_id: str
) -> PreflightResult:
    """
    Internal: Validate memory for a vLLM model (two-stage validation).
    
    Stage 1: Can the model load at all? (weights + overhead < VRAM)
    Stage 2: How much context can we get? (remaining space for PagedAttention)
    
    Args:
        model_cache_path: Path to the cached model directory
        model_id: Original model ID
        
    Returns:
        ValidationResult with status and recommendations
    """
    try:
        # Extract metadata from config.json
        metadata = inspect_transformers_model(model_cache_path, model_id)
        
        # Get available VRAM
        available_gb, mem_type = get_available_memory("vllm")
        
        if available_gb <= 0:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message="No NVIDIA GPU detected. vLLM requires a CUDA-capable GPU.",
                required_gb=0,
                available_gb=0
            )
        
        # Stage 1: Can it load?
        estimate = estimate_vllm_memory(metadata)
        estimate.available_gb = available_gb
        
        load_required = estimate.total_required_gb
        
        if load_required > available_gb:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason="Model weights and overhead exceed available GPU memory"
                ),
                required_gb=load_required,
                available_gb=available_gb,
                estimate=estimate
            )
        
        # Stage 2: How much context?
        kv_budget = available_gb - load_required
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        if max_safe_context < MIN_VIABLE_CONTEXT:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason=f"Maximum safe context ({max_safe_context:,}) is below minimum viable ({MIN_VIABLE_CONTEXT:,})"
                ),
                required_gb=load_required,
                available_gb=available_gb,
                max_context=max_safe_context,
                estimate=estimate
            )
        
        # Calculate utilization for the full recommended context
        utilization = load_required / available_gb
        recommended_context = int(max_safe_context * 0.8)  # Conservative
        
        if utilization > WARNING_THRESHOLD:
            return PreflightResult(
                status=PreflightStatus.WARNING,
                message=_format_warning_message(
                    estimate=estimate,
                    max_context=max_safe_context,
                    recommended_context=recommended_context,
                    mem_type=mem_type,
                    extra_note="vLLM reserves 90% of VRAM by default."
                ),
                required_gb=load_required,
                available_gb=available_gb,
                utilization=utilization,
                max_context=max_safe_context,
                recommended_context=recommended_context,
                estimate=estimate
            )
        
        return PreflightResult(
            status=PreflightStatus.PASSED,
            message=_format_safe_message(
                max_context=max_safe_context,
                extra_note="vLLM reserves 90% of VRAM by default."
            ),
            required_gb=load_required,
            available_gb=available_gb,
            utilization=utilization,
            max_context=max_safe_context,
            recommended_context=max_safe_context,
            estimate=estimate
        )
        
    except Exception as e:
        return PreflightResult(
            status=PreflightStatus.UNKNOWN,
            message=f"Memory validation failed: {e}"
        )


# ============================================================================
# DECISION LOGIC
# ============================================================================

def _log_preflight_result(result: "PreflightResult") -> None:
    """Log preflight result details using the module logger."""
    status_icons = {
        PreflightStatus.PASSED: "[✓]",
        PreflightStatus.WARNING: "[!]",
        PreflightStatus.FAILED: "[✗]",
        PreflightStatus.UNKNOWN: "[?]"
    }
    icon = status_icons.get(result.status, "[?]")
    
    # Use appropriate log level based on status
    log_fn = logger.info
    if result.status == PreflightStatus.WARNING:
        log_fn = logger.warning
    elif result.status == PreflightStatus.FAILED:
        log_fn = logger.error
    
    log_fn(
        f"{icon} Preflight: {result.status.value.upper()} | "
        f"{result.required_gb:.1f}GB required, {result.available_gb:.1f}GB available "
        f"({result.utilization*100:.0f}%) | "
        f"max_context={result.max_context:,}, recommended={result.recommended_context:,}"
    )


def _make_decision(
    utilization: float,
    estimate: MemoryEstimate,
    metadata: ModelMetadata,
    requested_context: int,
    max_safe_context: int,
    available_gb: float,
    mem_type: str
) -> PreflightResult:
    """
    Make validation decision based on utilization and context.
    
    Args:
        utilization: Memory utilization ratio (required / available)
        estimate: Memory estimate breakdown
        metadata: Model metadata
        requested_context: User-requested context length
        max_safe_context: Maximum safe context we calculated
        available_gb: Available memory in GB
        mem_type: Memory type description
        
    Returns:
        ValidationResult with appropriate status
    """
    result: PreflightResult
    
    # Check if model fits at all
    if utilization > 1.0:
        result = PreflightResult(
            status=PreflightStatus.FAILED,
            message=_format_critical_message(
                estimate=estimate,
                available_gb=available_gb,
                mem_type=mem_type,
                reason="Model requirements exceed available memory"
            ),
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            requested_context=requested_context,
            max_context=max_safe_context,
            estimate=estimate
        )
    
    # Check if max context is viable
    elif max_safe_context < MIN_VIABLE_CONTEXT:
        result = PreflightResult(
            status=PreflightStatus.FAILED,
            message=_format_critical_message(
                estimate=estimate,
                available_gb=available_gb,
                mem_type=mem_type,
                reason=f"Maximum safe context ({max_safe_context:,}) is below minimum viable ({MIN_VIABLE_CONTEXT:,})"
            ),
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            requested_context=requested_context,
            max_context=max_safe_context,
            estimate=estimate
        )
    
    # Check thresholds
    elif utilization > WARNING_THRESHOLD:
        # Need to constrain context
        recommended_context = int(max_safe_context * 0.8)  # Conservative
        result = PreflightResult(
            status=PreflightStatus.WARNING,
            message=_format_warning_message(
                estimate=estimate,
                max_context=max_safe_context,
                recommended_context=recommended_context,
                mem_type=mem_type
            ),
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            requested_context=requested_context,
            recommended_context=recommended_context,
            max_context=max_safe_context,
            estimate=estimate
        )
    
    elif utilization > SAFE_THRESHOLD:
        # Mild warning but can proceed
        result = PreflightResult(
            status=PreflightStatus.WARNING,
            message=f"Model will use {utilization*100:.0f}% of available {mem_type}. "
                   f"Consider closing other applications for best performance.",
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            requested_context=requested_context,
            recommended_context=requested_context,
            max_context=max_safe_context,
            estimate=estimate
        )
    
    else:
        # Safe to proceed
        result = PreflightResult(
            status=PreflightStatus.PASSED,
            message=f"Memory validation passed ({utilization*100:.0f}% utilization)",
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            requested_context=requested_context,
            recommended_context=requested_context,
            max_context=max_safe_context,
            estimate=estimate
        )
    
    # Log and return
    _log_preflight_result(result)
    return result


# ============================================================================
# MESSAGE FORMATTING
# ============================================================================

# =============================================================================
# MESSAGE TEMPLATES
# =============================================================================

CRITICAL_MESSAGE_TEMPLATE = """
======================================================================
MEMORY VALIDATION FAILED
======================================================================

Model: {quantization} quantization, {params}B parameters

Memory Requirements:
  Model weights:    {weights_gb:>6.1f} GB
  KV cache:         {kv_cache_gb:>6.1f} GB
  System overhead:  {overhead_gb:>6.1f} GB
  -------------------------
  Total required:   {total_required_gb:>6.1f} GB

Available {mem_type}: {available_gb:.1f} GB

Reason: {reason}

Suggestions:
  - Use a smaller model (7B or 14B parameters)
  - Try higher quantization (3-bit instead of 4-bit)
  - Reduce context window if possible
  - Close other applications to free memory

======================================================================"""

WARNING_MESSAGE_TEMPLATE = """
------------------------------------------------------------
MEMORY VALIDATION WARNING
------------------------------------------------------------

Model will load with constrained context window:

  Available {mem_type}: {available_gb:.1f} GB
  Model requirements:   {total_required_gb:.1f} GB

  Maximum safe context:   {max_safe_context} tokens
  Recommended context:    {recommended_context} tokens

The model will be loaded with the recommended context limit.
This is sufficient for most agent tasks but may require
summarization for very long documents.

Close other applications for best performance.
{extra_note}
------------------------------------------------------------"""

SAFE_MESSAGE_TEMPLATE = "Model can load safely with up to {max_safe_context} token context.{extra_note}"


def _format_critical_message(estimate: MemoryEstimate, available_gb: float, mem_type: str, reason: str) -> str:
    """Format a critical (blocking) error message."""
    return CRITICAL_MESSAGE_TEMPLATE.format(
        quantization=estimate.quantization or 'unknown',
        params=estimate.params_billions or 'unknown',
        weights_gb=estimate.weights_gb,
        kv_cache_gb=estimate.kv_cache_gb,
        overhead_gb=estimate.overhead_gb,
        total_required_gb=estimate.total_required_gb,
        mem_type=mem_type,
        available_gb=available_gb,
        reason=reason
    )


def _format_warning_message(estimate: MemoryEstimate, max_safe_context: int, recommended_context: int, mem_type: str, extra_note: str = None) -> str:
    """Format a warning message for constrained context."""
    return WARNING_MESSAGE_TEMPLATE.format(
        mem_type=mem_type,
        available_gb=estimate.available_gb,
        total_required_gb=estimate.total_required_gb,
        max_safe_context=f"{max_safe_context:,}",
        recommended_context=f"{recommended_context:,}",
        extra_note=f"\nNote: {extra_note}" if extra_note else ""
    )


def _format_safe_message(max_safe_context: int, extra_note: str = None) -> str:
    """Format a safe validation message."""
    return SAFE_MESSAGE_TEMPLATE.format(
        max_safe_context=f"{max_safe_context:,}",
        extra_note=f" Note: {extra_note}" if extra_note else ""
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _resolve_model_path(model_id: str, engine: Engine) -> Tuple[str, str]:
    """
    Resolve model identifier to a local path.
    
    If model_id is a local path, returns it as-is.
    If model_id is a HuggingFace repo ID, downloads/caches it and returns the path.
    
    Args:
        model_id: Either a local path or HuggingFace repo ID
        engine: Inference engine (determines download strategy)
        
    Returns:
        Tuple of (local_path, model_name_for_display)
            - local_path: Path to the model on disk
            - model_name_for_display: Identifier for logs/metadata (repo ID or filename)
    """
    # Check if it's a local path
    if os.path.exists(model_id):
        # Extract a display name from the path
        display_name = os.path.basename(model_id).replace('.gguf', '')
        return model_id, display_name
    
    # Assume it's a HuggingFace repo ID - download/cache it
    if engine == "llama_cpp":
        # For GGUF files, try to find and download specific .gguf file
        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            if gguf_files:
                # Download the first GGUF file found
                local_path = hf_hub_download(repo_id=model_id, filename=gguf_files[0])
                return local_path, model_id
        except Exception:
            pass
        
        # Fallback: download entire repo
        local_path = snapshot_download(repo_id=model_id)
        return local_path, model_id
    else:
        # For MLX/vLLM, download the entire model repo
        local_path = snapshot_download(repo_id=model_id)
        return local_path, model_id


# ============================================================================
# LOCAL INSPECTION HELPER
# ============================================================================

def _inspect_local(model_path: str, model_id: str, engine: Engine) -> ModelMetadata:
    """
    Inspect a local model file or directory.
    
    Args:
        model_path: Path to the local model file or directory
        model_id: Original model ID (for display)
        engine: Inference engine determining the format
        
    Returns:
        ModelMetadata with extracted parameters
    """
    if engine == "llama_cpp":
        return inspect_gguf_model(model_path, model_id)
    else:
        # MLX and vLLM use transformers format
        return inspect_transformers_model(model_path, model_id)


# ============================================================================
# METADATA-BASED VALIDATION
# ============================================================================

def _validate_gguf_with_metadata(metadata: ModelMetadata) -> PreflightResult:
    """Validate GGUF model using pre-extracted metadata."""
    try:
        # Determine context length
        context = metadata.model_max_context or DEFAULT_CONTEXT
        
        # Get available memory
        available_gb, mem_type = get_available_memory("llama_cpp")
        
        if available_gb <= 0:
            return PreflightResult(
                status=PreflightStatus.UNKNOWN,
                message="Could not determine available memory",
                recommended_context=context
            )
        
        # Calculate memory requirements
        estimate = estimate_llamacpp_memory(metadata, context)
        estimate.available_gb = available_gb
        
        # Calculate utilization
        utilization = estimate.total_required_gb / available_gb
        
        # Calculate max safe context
        kv_budget = available_gb - estimate.weights_gb - estimate.overhead_gb
        if kv_budget <= 0:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason="Model weights and overhead alone exceed available memory"
                ),
                required_gb=estimate.total_required_gb,
                available_gb=available_gb,
                utilization=utilization,
                estimate=estimate
            )
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        return _make_decision(
            utilization=utilization,
            estimate=estimate,
            metadata=metadata,
            requested_context=context,
            max_safe_context=max_safe_context,
            available_gb=available_gb,
            mem_type=mem_type
        )
        
    except Exception as e:
        return PreflightResult(
            status=PreflightStatus.UNKNOWN,
            message=f"Memory validation failed: {e}",
            recommended_context=DEFAULT_CONTEXT
        )


def _validate_mlx_with_metadata(metadata: ModelMetadata) -> PreflightResult:
    """Validate MLX model using pre-extracted metadata."""
    try:
        # Determine context length
        context = metadata.model_max_context or DEFAULT_CONTEXT
        
        # Get available memory
        available_gb, mem_type = get_available_memory("mlx_lm")
        
        if available_gb <= 0:
            return PreflightResult(
                status=PreflightStatus.UNKNOWN,
                message="Could not determine available memory",
                recommended_context=context
            )
        
        # Calculate memory requirements
        estimate = estimate_mlx_memory(metadata, context)
        estimate.available_gb = available_gb
        
        # Calculate utilization
        utilization = estimate.total_required_gb / available_gb
        
        # Calculate max safe context
        kv_budget = available_gb - estimate.weights_gb - estimate.overhead_gb
        if kv_budget <= 0:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason="Model weights and overhead exceed available unified memory"
                ),
                required_gb=estimate.total_required_gb,
                available_gb=available_gb,
                utilization=utilization,
                estimate=estimate
            )
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        return _make_decision(
            utilization=utilization,
            estimate=estimate,
            metadata=metadata,
            requested_context=context,
            max_safe_context=max_safe_context,
            available_gb=available_gb,
            mem_type=mem_type
        )
        
    except Exception as e:
        return PreflightResult(
            status=PreflightStatus.UNKNOWN,
            message=f"Memory validation failed: {e}",
            recommended_context=DEFAULT_CONTEXT
        )


def _validate_vllm_with_metadata(metadata: ModelMetadata) -> PreflightResult:
    """Validate vLLM model using pre-extracted metadata."""
    try:
        # Get available VRAM
        available_gb, mem_type = get_available_memory("vllm")
        
        if available_gb <= 0:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message="No NVIDIA GPU detected. vLLM requires a CUDA-capable GPU.",
                required_gb=0,
                available_gb=0
            )
        
        # Stage 1: Can it load?
        estimate = estimate_vllm_memory(metadata)
        estimate.available_gb = available_gb
        
        load_required = estimate.total_required_gb
        
        if load_required > available_gb:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason="Model weights and overhead exceed available GPU memory"
                ),
                required_gb=load_required,
                available_gb=available_gb,
                estimate=estimate
            )
        
        # Stage 2: How much context?
        kv_budget = available_gb - load_required
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        if max_safe_context < MIN_VIABLE_CONTEXT:
            return PreflightResult(
                status=PreflightStatus.FAILED,
                message=_format_critical_message(
                    estimate=estimate,
                    available_gb=available_gb,
                    mem_type=mem_type,
                    reason=f"Maximum safe context ({max_safe_context:,}) is below minimum viable ({MIN_VIABLE_CONTEXT:,})"
                ),
                required_gb=load_required,
                available_gb=available_gb,
                max_context=max_safe_context,
                estimate=estimate
            )
        
        # Calculate utilization for the full recommended context
        utilization = load_required / available_gb
        recommended_context = int(max_safe_context * 0.8)  # Conservative
        
        if utilization > WARNING_THRESHOLD:
            return PreflightResult(
                status=PreflightStatus.WARNING,
                message=_format_warning_message(
                    estimate=estimate,
                    max_context=max_safe_context,
                    recommended_context=recommended_context,
                    mem_type=mem_type,
                    extra_note="vLLM reserves 90% of VRAM by default."
                ),
                required_gb=load_required,
                available_gb=available_gb,
                utilization=utilization,
                max_context=max_safe_context,
                recommended_context=recommended_context,
                estimate=estimate
            )
        
        return PreflightResult(
            status=PreflightStatus.PASSED,
            message=_format_safe_message(
                max_context=max_safe_context,
                extra_note="vLLM reserves 90% of VRAM by default."
            ),
            required_gb=load_required,
            available_gb=available_gb,
            utilization=utilization,
            max_context=max_safe_context,
            recommended_context=recommended_context,
            estimate=estimate
        )
        
    except Exception as e:
        return PreflightResult(
            status=PreflightStatus.UNKNOWN,
            message=f"Memory validation failed: {e}",
            recommended_context=DEFAULT_CONTEXT
        )


def _validate_with_metadata(metadata: ModelMetadata, engine: Engine) -> PreflightResult:
    """
    Validate memory using pre-extracted metadata.
    
    Dispatches to the appropriate engine-specific validator.
    """
    if engine == "llama_cpp":
        return _validate_gguf_with_metadata(metadata)
    elif engine == "mlx":
        return _validate_mlx_with_metadata(metadata)
    elif engine == "vllm":
        return _validate_vllm_with_metadata(metadata)
    else:
        raise ValueError(f"Invalid engine: {engine}")


# ============================================================================
# PUBLIC API - Unified preflight function
# ============================================================================

def model_preflight(
    model_id: str,
    engine: Engine
) -> PreflightResult:
    """
    Run preflight check to determine if a model will fit in memory.
    
    Checks if the model can load with its maximum context. If not, calculates
    the optimal constrained context that will fit safely.
    
    Uses lightweight remote inspection when possible (HTTP Range requests),
    falling back to full model download only if necessary.
    
    Bandwidth usage:
    - Light check (default): ~500KB for GGUF, ~50KB for Safetensors/MLX
    - Fallback: Full model download (can be 10-100GB)
    
    Args:
        model_id: HuggingFace repo ID (e.g., "mlx-community/Llama-3-8B-4bit")
                  or local path (e.g., "/path/to/model.gguf" or "/path/to/model/")
        engine: Inference engine - "mlx", "llama_cpp", or "vllm"
        
    Returns:
        PreflightResult with status and recommendations
        
    Examples:
        >>> from edgekit.models import model_preflight
        >>> 
        >>> # Option 1: HuggingFace repo ID (uses lightweight remote check)
        >>> result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx")
        >>> 
        >>> # Option 2: Local path (direct file inspection)
        >>> result = model_preflight("/path/to/model.gguf", engine="llama_cpp")
        >>> 
        >>> if result.can_load:
        ...     print(f"Use context: {result.recommended_context}")
        >>> else:
        ...     print(f"Won't fit: {result.message}")
    """
    # Validate engine
    if engine not in ("mlx", "llama_cpp", "vllm"):
        raise ValueError(
            f"Invalid engine: '{engine}'. "
            f"Must be one of: 'mlx', 'llama_cpp', or 'vllm'"
        )
    
    # Check if it's a local path
    if os.path.exists(model_id):
        # Local path: use direct inspection
        model_path, display_name = _resolve_model_path(model_id, engine)
        metadata = _inspect_local(model_path, display_name, engine)
        return _validate_with_metadata(metadata, engine)
    
    # Remote HuggingFace repo: try lightweight inspection first
    try:
        logger.debug(f"Attempting lightweight remote inspection for {model_id}")
        metadata = inspect_model_remote(model_id, engine)
        logger.debug(f"Remote inspection succeeded: {metadata}")
        return _validate_with_metadata(metadata, engine)
        
    except RemoteInspectError as e:
        # Light check failed, fall back to full download
        logger.info(f"Light check unavailable ({e}), downloading for inspection")
        model_path, display_name = _resolve_model_path(model_id, engine)
        metadata = _inspect_local(model_path, display_name, engine)
        return _validate_with_metadata(metadata, engine)


def can_load(
    model_id: str,
    engine: Engine
) -> bool:
    """
    Simple check: can this model load on this hardware?
    
    Returns True if the model can load (passed or warning status).
    Returns False if the model won't fit.
    
    Args:
        model_id: HuggingFace repo ID or local path
        engine: Inference engine - "mlx", "llama_cpp", or "vllm"
        
    Returns:
        True if model can load, False otherwise
        
    Examples:
        >>> from edgekit.models import can_load
        >>> 
        >>> # Check HuggingFace model
        >>> if can_load("mlx-community/Llama-3-8B-4bit", engine="mlx"):
        ...     load_the_model()
        >>> 
        >>> # Check local model
        >>> if can_load("/path/to/model.gguf", engine="llama_cpp"):
        ...     load_the_model()
    """
    result = model_preflight(model_id, engine)
    return result.can_load

