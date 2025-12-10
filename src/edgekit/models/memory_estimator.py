"""
Memory Estimator for LLM Inference

Provides accurate memory estimation for different inference backends.
Uses fixed overhead tiers and precise KV cache calculations.
"""

from dataclasses import dataclass
from typing import Optional

from .model_inspector import ModelMetadata


@dataclass
class MemoryEstimate:
    """Detailed memory breakdown for a model."""
    
    # Component sizes in GB
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    
    # Total and available
    total_required_gb: float
    available_gb: float
    
    # Context information
    context_length: int
    max_safe_context: Optional[int] = None
    
    # Model info
    params_billions: Optional[float] = None
    quantization: Optional[str] = None


def get_activation_overhead_gb(
    params_billions: float,
    is_moe: bool = False,
    backend: str = "mlx_lm"
) -> float:
    """
    Get fixed activation overhead in GB based on model size tier.
    
    These values account for:
    - Scratch buffers for matrix multiplication
    - Intermediate activation tensors
    - Runtime overhead (Python, CUDA context, etc.)
    - Graph compilation overhead (MLX)
    
    Args:
        params_billions: Model parameter count in billions
        is_moe: Whether this is a Mixture of Experts model
        backend: Inference backend ("mlx_lm", "vllm", "llama_cpp")
        
    Returns:
        Overhead in GB
    """
    # Fixed tiers based on empirical measurements
    if is_moe:
        base = 3.0  # MoE routing overhead
    elif params_billions < 10:
        base = 1.2  # Small models (7B-8B)
    elif params_billions <= 30:
        base = 2.5  # Medium models (13B-27B)
    else:
        base = 4.0  # Large models (70B+)
    
    # MLX has additional graph compilation ("wiring") overhead
    if backend == "mlx_lm":
        base *= 1.15  # +15% for wiring spike
    
    return base


def calculate_kv_cache_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_length: int,
    dtype_bytes: float = 2.0
) -> float:
    """
    Calculate KV cache memory in GB.
    
    Formula: 2 × layers × kv_heads × head_dim × context × dtype_bytes
    
    The factor of 2 accounts for both Keys (K) and Values (V).
    
    IMPORTANT:
    - Use num_kv_heads (GQA-aware), not num_attention_heads!
    - Use full context_length, ignore sliding window for memory validation
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (not attention heads!)
        head_dim: Dimension per head (hidden_size / num_attention_heads)
        context_length: Full context length to allocate
        dtype_bytes: Bytes per element (2 for FP16, 1 for FP8/Q8_0, 0.5 for Q4_0)
        
    Returns:
        KV cache size in GB
    """
    # K and V both stored
    kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * context_length * dtype_bytes
    return kv_cache_bytes / (1024**3)


def calculate_max_safe_context(
    available_budget_gb: float,
    metadata: ModelMetadata,
    safety_factor: float = 0.9
) -> int:
    """
    Calculate maximum safe context length given a memory budget.
    
    Solves the KV cache formula for context_length:
    context = budget_bytes / (2 × layers × kv_heads × head_dim × dtype_bytes)
    
    Args:
        available_budget_gb: Memory budget for KV cache in GB
        metadata: Model metadata with architecture info
        safety_factor: Apply safety margin (default 0.9 = 10% buffer)
        
    Returns:
        Maximum safe context length in tokens
    """
    if not all([metadata.num_layers, metadata.num_kv_heads, metadata.head_dim]):
        # Cannot calculate, return conservative default
        return 16384
    
    # Apply safety margin
    budget_bytes = available_budget_gb * (1024**3) * safety_factor
    
    # Bytes per token (FP16 cache)
    bytes_per_token = 2 * metadata.num_layers * metadata.num_kv_heads * metadata.head_dim * 2
    
    max_tokens = int(budget_bytes / bytes_per_token)
    
    # Clamp to reasonable bounds
    min_viable = 16384  # 16K minimum for agents
    max_supported = metadata.model_max_context or 131072  # Model's actual limit
    
    return max(min_viable, min(max_tokens, max_supported))


def estimate_llamacpp_memory(
    metadata: ModelMetadata,
    context_length: int
) -> MemoryEstimate:
    """
    Estimate memory requirements for llama.cpp backend.
    
    llama.cpp pre-allocates the entire KV cache at startup based on n_ctx.
    
    Args:
        metadata: Model metadata from inspector
        context_length: Requested context length
        
    Returns:
        MemoryEstimate with detailed breakdown
    """
    # Weights: Use exact size from GGUF if available, else estimate
    if metadata.exact_model_size_gb:
        weights_gb = metadata.exact_model_size_gb
    elif metadata.params_billions:
        weights_gb = (metadata.params_billions * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
    else:
        weights_gb = 8.0  # Conservative fallback
    
    # KV cache: Full context allocation
    if all([metadata.num_layers, metadata.num_kv_heads, metadata.head_dim]):
        kv_cache_gb = calculate_kv_cache_gb(
            num_layers=metadata.num_layers,
            num_kv_heads=metadata.num_kv_heads,
            head_dim=metadata.head_dim,
            context_length=context_length,
            dtype_bytes=2.0  # FP16
        )
    else:
        # Fallback estimation based on model size
        # Rough rule: KV cache ≈ 0.5 * (context / 4096) * (params / 7) GB
        kv_cache_gb = 0.5 * (context_length / 4096) * ((metadata.params_billions or 8) / 7)
    
    # Overhead: Fixed tier
    overhead_gb = get_activation_overhead_gb(
        params_billions=metadata.params_billions or 8,
        is_moe=metadata.is_moe,
        backend="llama_cpp"
    )
    
    total_required_gb = weights_gb + kv_cache_gb + overhead_gb
    
    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=0.0,  # Set by validator
        context_length=context_length,
        params_billions=metadata.params_billions,
        quantization=metadata.quantization_type
    )


def estimate_mlx_memory(
    metadata: ModelMetadata,
    context_length: int
) -> MemoryEstimate:
    """
    Estimate memory requirements for MLX backend.
    
    MLX uses unified memory on Apple Silicon. Graph compilation creates
    a temporary "wiring" spike that we account for in overhead.
    
    Args:
        metadata: Model metadata from inspector
        context_length: Requested context length
        
    Returns:
        MemoryEstimate with detailed breakdown
    """
    # Weights: Calculate from params + quantization
    if metadata.exact_model_size_gb:
        weights_gb = metadata.exact_model_size_gb
    elif metadata.params_billions:
        weights_gb = (metadata.params_billions * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
    else:
        weights_gb = 8.0  # Conservative fallback
    
    # KV cache: MLX uses FP16 for cache even with quantized weights
    if all([metadata.num_layers, metadata.num_kv_heads, metadata.head_dim]):
        kv_cache_gb = calculate_kv_cache_gb(
            num_layers=metadata.num_layers,
            num_kv_heads=metadata.num_kv_heads,
            head_dim=metadata.head_dim,
            context_length=context_length,
            dtype_bytes=2.0  # FP16 cache
        )
    else:
        # Fallback estimation
        kv_cache_gb = 0.5 * (context_length / 4096) * ((metadata.params_billions or 8) / 7)
    
    # Overhead: Fixed tier with MLX multiplier
    overhead_gb = get_activation_overhead_gb(
        params_billions=metadata.params_billions or 8,
        is_moe=metadata.is_moe,
        backend="mlx_lm"  # Applies 1.15x for wiring
    )
    
    total_required_gb = weights_gb + kv_cache_gb + overhead_gb
    
    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=0.0,  # Set by validator
        context_length=context_length,
        params_billions=metadata.params_billions,
        quantization=metadata.quantization_type
    )


def estimate_vllm_memory(
    metadata: ModelMetadata
) -> MemoryEstimate:
    """
    Estimate memory requirements for vLLM backend (load check only).
    
    vLLM uses a two-stage validation:
    1. Can it load? (weights + overhead + CUDA context)
    2. How much context? (remaining space for PagedAttention blocks)
    
    This function handles stage 1. Context is calculated separately.
    
    Args:
        metadata: Model metadata from inspector
        
    Returns:
        MemoryEstimate for stage 1 (load) check
    """
    # Weights: Usually FP16/BF16 for vLLM
    if metadata.exact_model_size_gb:
        weights_gb = metadata.exact_model_size_gb
    elif metadata.params_billions:
        # vLLM typically uses FP16 (2 bytes) unless AWQ/GPTQ
        dtype_bytes = 2.0
        if metadata.bits_per_weight < 16:
            dtype_bytes = metadata.bits_per_weight / 8
        weights_gb = (metadata.params_billions * 1e9 * dtype_bytes) / (1024**3)
    else:
        weights_gb = 16.0  # Conservative fallback
    
    # Overhead: Fixed tier + CUDA context
    base_overhead = get_activation_overhead_gb(
        params_billions=metadata.params_billions or 16,
        is_moe=metadata.is_moe,
        backend="vllm"
    )
    cuda_context_gb = 0.5  # NCCL/PyTorch context
    overhead_gb = base_overhead + cuda_context_gb
    
    # For vLLM stage 1, we don't include KV cache
    # (vLLM will fill remaining space with PagedAttention blocks)
    total_required_gb = weights_gb + overhead_gb
    
    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=0.0,  # Calculated separately for context
        overhead_gb=overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=0.0,  # Set by validator
        context_length=0,  # Determined in stage 2
        params_billions=metadata.params_billions,
        quantization=metadata.quantization_type
    )


# Validation data for testing (from expert research)
VALIDATION_BENCHMARKS = {
    # (model, context) -> expected_kv_cache_gb
    ("llama-3-8b", 8192): 1.0,
    ("llama-3-8b", 131072): 16.0,
    ("llama-3-70b", 8192): 2.5,
    ("llama-3-70b", 131072): 40.0,
    ("mistral-7b", 32768): 4.0,
}

