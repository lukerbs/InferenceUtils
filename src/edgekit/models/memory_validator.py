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
Engine = Literal["mlx_lm", "llama_cpp", "vllm"]

from .model_inspector import (
    inspect_gguf_model,
    inspect_transformers_model,
    ModelMetadata,
)
from .model_inspector_remote import (
    inspect_model_remote,
    RemoteInspectError,
)
from .exceptions import PreflightValidationError
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

# Memory utilization thresholds
WARNING_THRESHOLD = 0.85    # 85% utilization = target threshold for context reduction

# Context limits
MIN_VIABLE_CONTEXT = 4096   # 4K tokens minimum (research-backed, configurable via API)

# GPU memory utilization defaults
VLLM_GPU_UTILIZATION_DEFAULT = 0.9   # vLLM reserves 90% of VRAM by default
AMD_APU_MEMORY_RESERVE = 0.85         # Conservative reserve for APU shared memory

# Fallback for unknown backend/detection failures
UNKNOWN_MEMORY_GB = 0.0              # Fail gracefully with 0 instead of guessing


# ============================================================================
# PREFLIGHT RESULT
# ============================================================================

class PreflightReason(Enum):
    """Reason for preflight result."""
    
    # Failures (status=False)
    VALIDATION_FAILED = "Cannot validate - model metadata incomplete"
    MEMORY_EXCEEDED = "Model too large - weights and overhead exceed memory limit"
    CONTEXT_INSUFFICIENT = "Model would require impractically low context (<4K tokens) to fit"
    
    # Success (status=True)
    FULL_CONTEXT = "Model fits at full designed context"
    REDUCED_CONTEXT = "Model fits with context reduction (still practical)"
    
    def __str__(self):
        return self.value


@dataclass
class PreflightResult:
    """Result of model preflight check."""
    
    status: bool                # True = can run, False = cannot run
    reason: PreflightReason     # Structured reason with descriptive message
    
    # Memory details
    required_gb: float = 0.0
    available_gb: float = 0.0
    utilization: float = 0.0
    
    # Context information
    context_limit: int = 0      # Model's designed maximum context
    usable_context: int = 0     # Context supported by device
    
    # Detailed breakdown (optional)
    estimate: Optional[MemoryEstimate] = None


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def get_available_memory(backend: str) -> Tuple[float, str]:
    """
    Get available memory for the specified backend.
    
    Uses platform-specific memory detection to provide accurate estimates.
    The OS-reported available memory already accounts for system overhead,
    so no additional safety buffers are applied. The utilization thresholds
    (70%/85%) provide adequate safety margins.
    
    Platform-specific memory sources:
    - macOS: Mach VM API (free + speculative + external pages)
    - Windows: Available memory from GlobalMemoryStatusEx
    - Linux: MemAvailable from /proc/meminfo
    - NVIDIA: Available VRAM from NVML (accounts for ECC if enabled)
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
            # Apple Silicon unified memory
            # Mach VM API provides accurate instantly-reclaimable memory
            available = hw.ram.available_gb or 0
            return max(0, available), "unified memory (macOS)"
        
        elif backend == "vllm":
            # NVIDIA GPU VRAM
            if hw.gpu.nvidia and len(hw.gpu.nvidia) > 0:
                gpu = hw.gpu.nvidia[0]
                
                # Prefer actual available VRAM if we have it (accounts for current usage)
                if gpu.available_vram_gb is not None:
                    available_vram = gpu.available_vram_gb
                else:
                    # Fallback to total * VLLM_GPU_UTILIZATION_DEFAULT
                    available_vram = gpu.vram_gb * VLLM_GPU_UTILIZATION_DEFAULT
                
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
                    return total_available * AMD_APU_MEMORY_RESERVE, "APU unified memory (VRAM+GTT)"
                elif gpu.available_vram_gb:
                    return gpu.available_vram_gb * VLLM_GPU_UTILIZATION_DEFAULT, "GPU VRAM"
            
            return 0.0, "no compatible GPU detected"
        
        elif backend == "llama_cpp":
            # System RAM - llama.cpp runs on CPU primarily
            # OS-reported available memory already accounts for running processes
            # including DWM on Windows, kernel overhead, etc.
            available = hw.ram.available_gb or 0
            return max(0, available), "system RAM"
        
        else:
            # Unknown backend - fail gracefully
            return UNKNOWN_MEMORY_GB, "unknown"
            
    except Exception:
        return UNKNOWN_MEMORY_GB, "estimated"


# ============================================================================
# DECISION LOGIC
# ============================================================================

def _log_preflight_result(result: "PreflightResult") -> None:
    """Log preflight result details using the module logger."""
    icon = "[✓]" if result.status else "[✗]"
    status_text = "PASSED" if result.status else "FAILED"
    
    # Use appropriate log level based on status
    log_fn = logger.info if result.status else logger.error
    
    log_fn(
        f"{icon} Preflight: {status_text} | "
        f"{result.required_gb:.1f}GB required, {result.available_gb:.1f}GB available "
        f"({result.utilization*100:.0f}%) | "
        f"context_limit={result.context_limit:,}, usable={result.usable_context:,}"
    )


def _make_decision(
    utilization: float,
    estimate: MemoryEstimate,
    metadata: ModelMetadata,
    model_context_limit: int,
    max_safe_context: int,
    available_gb: float,
    mem_type: str,
    max_utilization: float = WARNING_THRESHOLD,
    min_context: int = MIN_VIABLE_CONTEXT
) -> PreflightResult:
    """
    Make validation decision based on utilization and context.
    
    Algorithm:
    1. Try full context first (always respect model design)
    2. If doesn't fit AND model < min_context: reject immediately (no reduction)
    3. If doesn't fit AND model >= min_context: calculate reduced context targeting max_utilization
    4. If reduced context < min_context: reject
    5. Otherwise: PASS with context info (user decides if acceptable)
    
    Args:
        utilization: Memory utilization ratio at full context (required / available)
        estimate: Memory estimate breakdown at full context
        metadata: Model metadata
        model_context_limit: Model's designed maximum context
        max_safe_context: Maximum safe context we calculated
        available_gb: Available memory in GB
        mem_type: Memory type description
        max_utilization: Target utilization threshold (default 0.85)
        min_context: Minimum viable context threshold (default 4096)
        
    Returns:
        PreflightResult with appropriate status
    """
    result: PreflightResult
    
    # Calculate base memory (weights + overhead only, no KV cache)
    base_memory_gb = estimate.weights_gb + estimate.overhead_gb
    base_utilization = base_memory_gb / available_gb
    
    # SCENARIO 1: Model base is too large even with zero context
    if base_utilization > max_utilization:
        result = PreflightResult(
            status=False,
            reason=PreflightReason.MEMORY_EXCEEDED,
            required_gb=base_memory_gb,
                available_gb=available_gb,
            utilization=base_utilization,
            context_limit=model_context_limit,
            usable_context=0,
            estimate=estimate
        )
        _log_preflight_result(result)
        return result
    
    # Check if model fits at full designed context
    if utilization <= max_utilization:
        # Model fits at full context!
        result = PreflightResult(
            status=True,
            reason=PreflightReason.FULL_CONTEXT,
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            context_limit=model_context_limit,
            usable_context=model_context_limit,  # Full context!
            estimate=estimate
        )
        _log_preflight_result(result)
        return result
    
    # Model doesn't fit at full context - check if reduction is viable
    
    # Special case: Sub-min_context models cannot be reduced further
    if model_context_limit < min_context:
        result = PreflightResult(
            status=False,
            reason=PreflightReason.CONTEXT_INSUFFICIENT,
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            context_limit=model_context_limit,
            usable_context=0,
            estimate=estimate
        )
        _log_preflight_result(result)
        return result
    
    # Calculate reduced context targeting max_utilization
    target_memory_gb = available_gb * max_utilization
    kv_budget_at_target = target_memory_gb - base_memory_gb
    
    # Calculate what context achieves the target utilization
    context_at_target = calculate_max_safe_context(kv_budget_at_target, metadata, safety_factor=1.0)
    
    # Check if reduced context meets minimum viability
    if context_at_target < min_context:
        result = PreflightResult(
            status=False,
            reason=PreflightReason.CONTEXT_INSUFFICIENT,
            required_gb=estimate.total_required_gb,
            available_gb=available_gb,
            utilization=utilization,
            context_limit=model_context_limit,
            usable_context=0,
            estimate=estimate
        )
        _log_preflight_result(result)
        return result
    
    # Model can run with reduced context
        result = PreflightResult(
        status=True,
        reason=PreflightReason.REDUCED_CONTEXT,
        required_gb=target_memory_gb,
            available_gb=available_gb,
        utilization=max_utilization,
        context_limit=model_context_limit,
        usable_context=context_at_target,
            estimate=estimate
        )
    _log_preflight_result(result)
    return result


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

def _validate_gguf_with_metadata(
    metadata: ModelMetadata,
    max_utilization: float = WARNING_THRESHOLD,
    min_context: int = MIN_VIABLE_CONTEXT
) -> PreflightResult:
    """Validate GGUF model using pre-extracted metadata."""
    try:
        # Check required metadata
        if not metadata.model_max_context:
            return PreflightResult(
                status=False,
                reason=PreflightReason.VALIDATION_FAILED,
                context_limit=0,
                usable_context=0
            )
        
        context = metadata.model_max_context
        
        # Get available memory
        available_gb, mem_type = get_available_memory("llama_cpp")
        
        if available_gb <= 0:
            return PreflightResult(
                status=False,
                reason=PreflightReason.VALIDATION_FAILED,
                context_limit=context,
                usable_context=0
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
                status=False,
                reason=PreflightReason.MEMORY_EXCEEDED,
                required_gb=estimate.total_required_gb,
                available_gb=available_gb,
                utilization=utilization,
                context_limit=context,
                usable_context=0,
                estimate=estimate
            )
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        return _make_decision(
            utilization=utilization,
            estimate=estimate,
            metadata=metadata,
            model_context_limit=context,
            max_safe_context=max_safe_context,
            available_gb=available_gb,
            mem_type=mem_type,
            max_utilization=max_utilization,
            min_context=min_context
        )
        
    except Exception as e:
        return PreflightResult(
            status=False,
            reason=PreflightReason.VALIDATION_FAILED,
            context_limit=metadata.model_max_context or 0,
            usable_context=0
        )


def _validate_mlx_with_metadata(
    metadata: ModelMetadata,
    max_utilization: float = WARNING_THRESHOLD,
    min_context: int = MIN_VIABLE_CONTEXT
) -> PreflightResult:
    """Validate MLX model using pre-extracted metadata."""
    try:
        print(f"\n🔧 DEBUG: MLX Memory Validation")
        print(f"   Model Type: {metadata.model_type}")
        if metadata.params_billions:
            print(f"   Params: {metadata.params_billions:.1f}B")
        print(f"   Quantization: {metadata.quantization_type or 'unknown'}")
        print(f"   BPW: {metadata.bits_per_weight}")
        
        # Check required metadata
        if not metadata.model_max_context:
            print(f"❌ DEBUG: Missing model_max_context")
            return PreflightResult(
                status=False,
                reason=PreflightReason.VALIDATION_FAILED,
                context_limit=0,
                usable_context=0
            )
        
        context = metadata.model_max_context
        print(f"   Context: {context:,}")
        
        # Get available memory
        print(f"   Getting available memory...")
        available_gb, mem_type = get_available_memory("mlx_lm")
        print(f"   Available: {available_gb:.2f} GB ({mem_type})")
        
        if available_gb <= 0:
            print(f"❌ DEBUG: Available memory is {available_gb} GB - returning UNKNOWN")
            return PreflightResult(
                status=False,
                reason=PreflightReason.VALIDATION_FAILED,
                context_limit=context,
                usable_context=0
            )
        
        # Calculate memory requirements
        print(f"   Calculating memory estimate...")
        estimate = estimate_mlx_memory(metadata, context)
        estimate.available_gb = available_gb
        print(f"   Weights: {estimate.weights_gb:.2f} GB")
        print(f"   KV Cache: {estimate.kv_cache_gb:.2f} GB")
        print(f"   Overhead: {estimate.overhead_gb:.2f} GB")
        print(f"   Total Required: {estimate.total_required_gb:.2f} GB")
        
        # Calculate utilization
        utilization = estimate.total_required_gb / available_gb
        
        # Calculate max safe context
        kv_budget = available_gb - estimate.weights_gb - estimate.overhead_gb
        if kv_budget <= 0:
            return PreflightResult(
                status=False,
                reason=PreflightReason.MEMORY_EXCEEDED,
                required_gb=estimate.total_required_gb,
                available_gb=available_gb,
                utilization=utilization,
                context_limit=context,
                usable_context=0,
                estimate=estimate
            )
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        return _make_decision(
            utilization=utilization,
            estimate=estimate,
            metadata=metadata,
            model_context_limit=context,
            max_safe_context=max_safe_context,
            available_gb=available_gb,
            mem_type=mem_type,
            max_utilization=max_utilization,
            min_context=min_context
        )
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in MLX validation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return PreflightResult(
            status=False,
            reason=PreflightReason.VALIDATION_FAILED,
            context_limit=metadata.model_max_context or 0,
            usable_context=0
        )


def _validate_vllm_with_metadata(
    metadata: ModelMetadata,
    max_utilization: float = WARNING_THRESHOLD,
    min_context: int = MIN_VIABLE_CONTEXT
) -> PreflightResult:
    """Validate vLLM model using pre-extracted metadata."""
    try:
        # Check required metadata
        if not metadata.model_max_context:
            return PreflightResult(
                status=False,
                reason=PreflightReason.VALIDATION_FAILED,
                context_limit=0,
                usable_context=0
            )
        
        # Get available VRAM
        available_gb, mem_type = get_available_memory("vllm")
        
        if available_gb <= 0:
            return PreflightResult(
                status=False,
                reason=PreflightReason.VALIDATION_FAILED,
                required_gb=0,
                available_gb=0,
                context_limit=metadata.model_max_context,
                usable_context=0
            )
        
        # Stage 1: Can it load?
        estimate = estimate_vllm_memory(metadata)
        estimate.available_gb = available_gb
        
        load_required = estimate.total_required_gb
        
        if load_required > available_gb:
            return PreflightResult(
                status=False,
                reason=PreflightReason.MEMORY_EXCEEDED,
                required_gb=load_required,
                available_gb=available_gb,
                context_limit=metadata.model_max_context,
                usable_context=0,
                estimate=estimate
            )
        
        # Stage 2: Calculate context that fits in remaining space
        # vLLM uses PagedAttention to dynamically allocate context
        context = metadata.model_max_context
        kv_budget = available_gb - load_required
        max_safe_context = calculate_max_safe_context(kv_budget, metadata)
        estimate.max_safe_context = max_safe_context
        
        # For vLLM, we already know weights fit (Stage 1 passed)
        # Now calculate full memory with a reasonable context
        # Use the target context to get accurate utilization
        from .memory_estimator import calculate_kv_cache_gb
        kv_lora_rank = getattr(metadata, 'kv_lora_rank', None)
        
        if all([metadata.num_layers, metadata.num_kv_heads, metadata.head_dim]) or kv_lora_rank:
            if not metadata.num_layers:
                raise PreflightValidationError("Cannot calculate KV cache - missing num_layers")
            if not kv_lora_rank and (not metadata.num_kv_heads or not metadata.head_dim):
                raise PreflightValidationError("Cannot calculate KV cache - missing num_kv_heads or head_dim")
            
            from .memory_estimator import KV_DTYPE_FP16
            kv_cache_gb = calculate_kv_cache_gb(
                num_layers=metadata.num_layers,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                context_length=context,
                dtype_bytes=KV_DTYPE_FP16,
                kv_lora_rank=kv_lora_rank
            )
            estimate.kv_cache_gb = kv_cache_gb
            estimate.total_required_gb = load_required + kv_cache_gb
        
        utilization = estimate.total_required_gb / available_gb
        
        # Use standard decision logic like other backends
        return _make_decision(
            utilization=utilization,
                    estimate=estimate,
            metadata=metadata,
            model_context_limit=context,
            max_safe_context=max_safe_context,
                available_gb=available_gb,
            mem_type=mem_type,
            max_utilization=max_utilization,
            min_context=min_context
        )
        
    except Exception as e:
        return PreflightResult(
            status=False,
            reason=PreflightReason.VALIDATION_FAILED,
            context_limit=metadata.model_max_context or 0,
            usable_context=0
        )


def _validate_with_metadata(
    metadata: ModelMetadata, 
    engine: Engine,
    max_utilization: float = WARNING_THRESHOLD,
    min_context: int = MIN_VIABLE_CONTEXT
) -> PreflightResult:
    """
    Validate memory using pre-extracted metadata.
    
    Dispatches to the appropriate engine-specific validator.
    """
    if engine == "llama_cpp":
        return _validate_gguf_with_metadata(metadata, max_utilization, min_context)
    elif engine == "mlx_lm":
        return _validate_mlx_with_metadata(metadata, max_utilization, min_context)
    elif engine == "vllm":
        return _validate_vllm_with_metadata(metadata, max_utilization, min_context)
    else:
        raise PreflightValidationError(f"Invalid engine: {engine}")


# ============================================================================
# PUBLIC API - Unified preflight function
# ============================================================================

def model_preflight(
    model_id: str,
    engine: Engine,
    max_utilization: float = WARNING_THRESHOLD,
    min_context: int = MIN_VIABLE_CONTEXT
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
        engine: Inference engine - "mlx_lm", "llama_cpp", or "vllm"
        max_utilization: Target memory utilization threshold (0.0-1.0).
                        Models exceeding this will be rejected or context-reduced.
                        Default: 0.85 (85%)
        min_context: Minimum viable context length in tokens.
                    Models requiring less than this will be rejected.
                    Default: 4096 tokens
        
    Returns:
        PreflightResult with status and recommendations
        
    Examples:
        >>> from edgekit.models import model_preflight
        >>> 
        >>> # Basic usage with defaults
        >>> result = model_preflight("mlx-community/Llama-3-8B-4bit", engine="mlx_lm")
        >>> 
        >>> # Conservative: only use 70% of memory
        >>> result = model_preflight(model_id, engine="mlx_lm", max_utilization=0.70)
        >>> 
        >>> # Power user: accept lower context, push memory to 95%
        >>> result = model_preflight(model_id, engine="mlx_lm", 
        ...                          max_utilization=0.95, min_context=2048)
        >>> 
        >>> if result.status:
        ...     print(f"Model: {result.context_limit}, Device: {result.usable_context}")
        >>> else:
        ...     print(f"Won't fit: {result.reason}")
    """
    # Validate engine
    if engine not in ("mlx_lm", "llama_cpp", "vllm"):
        raise PreflightValidationError(
            f"Invalid engine: '{engine}'. "
            f"Must be one of: 'mlx_lm', 'llama_cpp', or 'vllm'"
        )
    
    # Check if it's a local path
    if os.path.exists(model_id):
        # Local path: use direct inspection
        model_path, display_name = _resolve_model_path(model_id, engine)
        metadata = _inspect_local(model_path, display_name, engine)
        return _validate_with_metadata(metadata, engine, max_utilization, min_context)
    
    # Remote HuggingFace repo: try lightweight inspection first
    try:
        print(f"🔍 DEBUG: Attempting lightweight remote inspection for {model_id}")
        logger.debug(f"Attempting lightweight remote inspection for {model_id}")
        metadata = inspect_model_remote(model_id, engine)
        print(f"✅ DEBUG: Remote inspection succeeded!")
        print(f"   Model Type: {metadata.model_type}")
        print(f"   Quantization: {metadata.quantization_type or 'unknown'}")
        if metadata.params_billions:
            print(f"   Params: {metadata.params_billions:.1f}B")
        print(f"   Layers: {metadata.num_layers or 'unknown'}")
        print(f"   Context: {metadata.model_max_context or 'unknown'}")
        logger.debug(f"Remote inspection succeeded: {metadata}")
        return _validate_with_metadata(metadata, engine, max_utilization, min_context)
        
    except RemoteInspectError as e:
        # Light check failed, fall back to full download
        print(f"⚠️  DEBUG: Light check failed: {e}")
        print(f"   Falling back to full download...")
        logger.info(f"Light check unavailable ({e}), downloading for inspection")
        model_path, display_name = _resolve_model_path(model_id, engine)
        metadata = _inspect_local(model_path, display_name, engine)
        return _validate_with_metadata(metadata, engine, max_utilization, min_context)


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
        engine: Inference engine - "mlx_lm", "llama_cpp", or "vllm"
        
    Returns:
        True if model can load, False otherwise
        
    Examples:
        >>> from edgekit.models import can_load
        >>> 
        >>> # Check HuggingFace model
        >>> if can_load("mlx-community/Llama-3-8B-4bit", engine="mlx_lm"):
        ...     load_the_model()
        >>> 
        >>> # Check local model
        >>> if can_load("/path/to/model.gguf", engine="llama_cpp"):
        ...     load_the_model()
    """
    result = model_preflight(model_id, engine)
    return result.status

