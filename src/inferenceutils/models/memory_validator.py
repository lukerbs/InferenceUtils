"""
Model Preflight Check

Validates if a model will fit in memory before loading.
Prevents OOM errors by checking hardware capacity against model requirements.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Tuple

from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

from ..hardware import system_info

# Type alias for inference engine parameter
Engine = Literal["mlx", "llama_cpp", "vllm"]

from .model_inspector import inspect_gguf_model, inspect_transformers_model, ModelMetadata
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

MLX_SAFETY_BUFFER_GB = 2.0  # Reduced from 4.0 - accurate memory detection allows tighter margins
LLAMACPP_SAFETY_BUFFER_GB = 1.0
VLLM_SYSTEM_OVERHEAD_GB = 0.5

MIN_VIABLE_CONTEXT = 16384  # 16K tokens minimum
DEFAULT_CONTEXT = 32768     # 32K tokens default


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
    
    Args:
        backend: "mlx", "vllm", or "gguf"
        
    Returns:
        Tuple of (available_gb, memory_type_description)
    """
    try:
        hw = system_info()
        
        if backend == "mlx_lm":
            # Apple Silicon unified memory
            available = hw.ram.available_gb
            usable = available - MLX_SAFETY_BUFFER_GB
            return max(0, usable), "unified memory (macOS)"
        
        elif backend == "vllm":
            # NVIDIA GPU VRAM
            if hw.gpu.nvidia and len(hw.gpu.nvidia) > 0:
                total_vram = hw.gpu.nvidia[0].vram_gb
                # vLLM uses gpu_memory_utilization default of 0.9
                usable = total_vram * 0.9
                return usable, "GPU VRAM"
            else:
                # No NVIDIA GPU detected
                return 0.0, "no GPU detected"
        
        elif backend == "llama_cpp":
            # System RAM
            available = hw.ram.available_gb
            usable = available - LLAMACPP_SAFETY_BUFFER_GB
            return max(0, usable), "system RAM"
        
        else:
            # Unknown backend
            return 16.0, "unknown"
            
    except Exception:
        return 16.0, "estimated"


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
    """Log preflight result details."""
    status_icons = {
        PreflightStatus.PASSED: "[✓]",
        PreflightStatus.WARNING: "[!]",
        PreflightStatus.FAILED: "[✗]",
        PreflightStatus.UNKNOWN: "[?]"
    }
    icon = status_icons.get(result.status, "[?]")
    
    print(f"{icon} Preflight: {result.status.value.upper()}")
    print(f"    Memory: {result.required_gb:.1f} GB required, {result.available_gb:.1f} GB available ({result.utilization*100:.0f}%)")
    print(f"    Context: max={result.max_context:,} tokens, recommended={result.recommended_context:,}")


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

def _resolve_model_path(model_id: str, engine: Engine) -> tuple:
    """
    Resolve model identifier to a local path.
    
    If model_id is a local path, returns it as-is.
    If model_id is a HuggingFace repo ID, downloads/caches it and returns the path.
    
    Args:
        model_id: Either a local path or HuggingFace repo ID
        engine: Inference engine (determines download strategy)
        
    Returns:
        tuple: (local_path, model_name_for_display)
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
    
    Automatically resolves HuggingFace repo IDs by downloading/caching the model,
    or uses local paths directly if provided.
    
    Args:
        model_id: HuggingFace repo ID (e.g., "mlx-community/Llama-3-8B-4bit")
                  or local path (e.g., "/path/to/model.gguf" or "/path/to/model/")
        engine: Inference engine - "mlx", "llama_cpp", or "vllm"
        
    Returns:
        PreflightResult with status and recommendations
        
    Examples:
        >>> from edgekit.models import model_preflight
        >>> 
        >>> # Option 1: HuggingFace repo ID (auto-downloads/caches)
        >>> result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx")
        >>> 
        >>> # Option 2: Local path
        >>> result = model_preflight("/path/to/model.gguf", engine="llama_cpp")
        >>> 
        >>> if result.can_load:
        ...     print(f"Use context: {result.recommended_context}")
        >>> else:
        ...     print(f"Won't fit: {result.message}")
    """
    # Resolve model ID to local path
    model_path, display_name = _resolve_model_path(model_id, engine)
    
    # Dispatch to appropriate validator (they handle max context internally)
    if engine == "llama_cpp":
        return _validate_gguf_memory(model_path, display_name)
    elif engine == "mlx":
        return _validate_mlx_memory(model_path, display_name)
    elif engine == "vllm":
        return _validate_vllm_memory(model_path, display_name)
    else:
        raise ValueError(
            f"Invalid engine: '{engine}'. "
            f"Must be one of: 'mlx', 'llama_cpp', or 'vllm'"
        )


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

