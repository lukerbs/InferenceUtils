"""
Memory Estimator for LLM Inference

Provides accurate memory estimation for different inference backends.
Uses fixed overhead tiers and precise KV cache calculations.
"""

from dataclasses import dataclass
from typing import Optional

from .metadata_parser import ModelMetadata
from .exceptions import PreflightValidationError


# =============================================================================
# ESTIMATION CONSTANTS
# =============================================================================

# MoE models keep router/gating at higher precision (Q6_K/Q8_0)
MOE_WEIGHT_OVERHEAD = 1.05

# KV cache dtype options (bytes per element)
KV_DTYPE_FP16 = 2.0   # Default - full precision cache
KV_DTYPE_FP8 = 1.0    # Halves KV cache size
KV_DTYPE_Q8 = 1.0     # 8-bit quantized cache
KV_DTYPE_Q4 = 0.5     # 4-bit quantized cache (experimental)

# Activation overhead tiers (GB) - empirically validated
OVERHEAD_MOE = 3.0           # MoE routing overhead
OVERHEAD_SMALL = 1.2         # Small models (<10B params)
OVERHEAD_MEDIUM = 2.5        # Medium models (10-30B params)
OVERHEAD_LARGE = 4.0         # Large models (>30B params)

# Model size thresholds (billions of parameters)
SMALL_MODEL_THRESHOLD = 10   # Below this = small model tier
MEDIUM_MODEL_THRESHOLD = 30  # Below this (but >= 10) = medium model tier

# Backend-specific multipliers
MLX_WIRING_MULTIPLIER = 1.15      # MLX graph compilation overhead (+15%)
VLLM_GPU_UTILIZATION = 0.90       # vLLM reserves 90% of VRAM by default
VLLM_CUDA_CONTEXT_GB = 0.5        # CUDA/PyTorch/NCCL overhead for vLLM

# Context calculation bounds
MIN_CONTEXT_BOUND = 16384         # Minimum viable for calculate_max_safe_context
MAX_CONTEXT_UPPER_BOUND = 131072  # Upper clamp (128K) when model max unknown

# KV cache calculation factors
KV_CACHE_KEY_VALUE_FACTOR = 2     # Factor of 2 for both Keys and Values


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
    # Theoretical maximum context at 100% memory utilization (informational only).
    # Note: This differs from usable_context in PreflightResult, which uses target utilization (default 85%).
    max_safe_context: Optional[int] = None
    
    # Model info
    params_billions: Optional[float] = None
    quantization: Optional[str] = None
    
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"MemoryEstimate(total={self.total_required_gb:.1f}GB, "
            f"weights={self.weights_gb:.1f}GB, kv={self.kv_cache_gb:.1f}GB, "
            f"overhead={self.overhead_gb:.1f}GB)"
        )


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
        base = OVERHEAD_MOE
    elif params_billions < SMALL_MODEL_THRESHOLD:
        base = OVERHEAD_SMALL
    elif params_billions <= MEDIUM_MODEL_THRESHOLD:
        base = OVERHEAD_MEDIUM
    else:
        base = OVERHEAD_LARGE
    
    # MLX has additional graph compilation ("wiring") overhead
    if backend == "mlx_lm":
        base *= MLX_WIRING_MULTIPLIER
    
    return base


def calculate_kv_cache_gb(
    metadata: ModelMetadata,
    context_length: int,
    dtype_bytes: float = 2.0
) -> float:
    """
    Calculate KV cache memory in GB using ModelMetadata.
    
    Supports:
    - Standard Dense Attention (all layers = full context)
    - DeepSeek MLA (compressed KV)
    - Hybrid / Sliding Window Attention (mix of global and windowed layers)
    
    Args:
        metadata: ModelMetadata object containing extracted architecture info
        context_length: Runtime context length to allocate
        dtype_bytes: Bytes per element (precision)
        
    Returns:
        KV cache size in GB
    """
    # 1. DeepSeek MLA (Compressed Cache) - Highest Priority
    if metadata.kv_lora_rank and metadata.num_layers:
        # No factor of 2. Latent vector (c_KV) + RoPE vector (k_R) are the only storage.
        if not metadata.head_dim:
            raise PreflightValidationError("Architecture missing: head_dim (required for RoPE)")
        rope_dim = metadata.head_dim
        kv_cache_bytes = metadata.num_layers * (metadata.kv_lora_rank + rope_dim) * context_length * dtype_bytes
        return kv_cache_bytes / (1024**3)

    # Validation for Standard/Hybrid models
    if not metadata.num_layers or not metadata.num_kv_heads or not metadata.head_dim:
        raise PreflightValidationError("Cannot calculate KV cache - missing architecture (layers, heads, dim)")

    # 2. Extract Architectural Counts (Safe Defaults provided by Inspector)
    # Inspector ensures these are populated or defaulted to Dense (Global=All)
    n_global = metadata.num_global_layers if metadata.num_global_layers is not None else metadata.num_layers
    n_local = metadata.num_local_layers if metadata.num_local_layers is not None else 0
    w_size = metadata.sliding_window_size if metadata.sliding_window_size else context_length

    # 3. Calculate Global Component
    # Global layers grow linearly with FULL context
    global_bytes = n_global * KV_CACHE_KEY_VALUE_FACTOR * metadata.num_kv_heads * metadata.head_dim * context_length * dtype_bytes

    # 4. Calculate Local Component
    # Local layers are capped at min(Context, Window)
    # This correctly handles the "capped" memory growth for Sliding Window layers
    effective_window = min(context_length, w_size)
    local_bytes = n_local * KV_CACHE_KEY_VALUE_FACTOR * metadata.num_kv_heads * metadata.head_dim * effective_window * dtype_bytes

    # 5. Total
    total_bytes = global_bytes + local_bytes
    return total_bytes / (1024**3)


def calculate_max_safe_context(
    available_budget_gb: float,
    metadata: ModelMetadata,
    safety_factor: float = 0.9
) -> int:
    """
    Calculate maximum safe context length given a memory budget.
    """
    # Handle negative budget
    if available_budget_gb <= 0:
        return 0
    
    budget_bytes = available_budget_gb * (1024**3) * safety_factor

    # MLA Case
    if metadata.kv_lora_rank and metadata.num_layers:
        if not metadata.head_dim:
            raise PreflightValidationError("Architecture missing: head_dim (required for RoPE)")
        rope_dim = metadata.head_dim
        bytes_per_token = metadata.num_layers * (metadata.kv_lora_rank + rope_dim) * KV_DTYPE_FP16
        return int(budget_bytes / bytes_per_token)

    # Standard/Hybrid Case
    if not metadata.num_kv_heads or not metadata.head_dim or not metadata.num_layers:
        raise PreflightValidationError("Cannot calculate context - missing architecture")

    # Determine Effective Linear Growth Cost
    # We only care about layers that grow linearly with context (Global Layers).
    # Local layers act as a fixed "tax" once we pass the window size.
    # For MAX context calculation (assuming context >> window), the Global layers dominate.
    
    n_global = metadata.num_global_layers if metadata.num_global_layers is not None else metadata.num_layers
    
    # Even purely local models (Mistral) consume *some* linear memory if 
    # context < window, but for MAX context calculation, we assume context >> window.
    # We treat at least 1 layer as global to prevent divide-by-zero or infinite context results.
    effective_growth_layers = max(1, n_global)

    bytes_per_token = KV_CACHE_KEY_VALUE_FACTOR * effective_growth_layers * metadata.num_kv_heads * metadata.head_dim * KV_DTYPE_FP16
    
    max_tokens = int(budget_bytes / bytes_per_token)
    
    # Clamp to reasonable bounds
    max_supported = metadata.model_max_context or MAX_CONTEXT_UPPER_BOUND
    return min(max_tokens, max_supported)


def calculate_mixed_precision_weights_gb(metadata: ModelMetadata) -> float:
    """
    Calculate weight size for mixed-precision models (e.g., GPT-OSS).
    
    GPT-OSS uses:
    - BF16 (16 bits) for embeddings and attention layers
    - MXFP4 (4.25 bits) for expert layers (MoE FFN)
    
    Args:
        metadata: ModelMetadata with quantization_exceptions and intermediate_size
        
    Returns:
        Total weight size in GB
    """
    if not metadata.hidden_size or not metadata.num_layers:
        raise PreflightValidationError("Cannot calculate mixed precision weights - missing architecture")
    
    hidden = metadata.hidden_size
    num_layers = metadata.num_layers
    
    if not metadata.vocab_size:
        raise PreflightValidationError("Cannot calculate mixed precision: vocab_size missing")
    vocab = metadata.vocab_size
    
    num_experts = metadata.num_experts or 1
    
    if not metadata.intermediate_size:
        # Note: GPT-OSS logic in parser should have already set this to hidden_size if needed.
        # If it's still None here, we cannot guess 4x safely.
        raise PreflightValidationError("Cannot calculate mixed precision: intermediate_size missing")
    intermediate = metadata.intermediate_size
    
    # BF16 bytes per element
    bf16_bytes = 2.0
    
    # 1. Embeddings (BF16) - kept in high precision
    # vocab_size * hidden_size * 2 bytes
    embedding_bytes = vocab * hidden * bf16_bytes
    
    # 2. Attention Layers (BF16) - kept in high precision
    # Q, K, V, O projections: 4 * hidden_size * hidden_size per layer
    # For GQA: K and V are scaled by num_kv_heads / num_attention_heads
    num_heads = metadata.num_attention_heads or (hidden // 64)
    num_kv_heads = metadata.num_kv_heads or num_heads
    gqa_scale = num_kv_heads / num_heads if num_heads > 0 else 1.0
    
    # Q: hidden * hidden, K: hidden * hidden * scale, V: hidden * hidden * scale, O: hidden * hidden
    attention_per_layer = hidden * hidden * (2.0 + 2.0 * gqa_scale)
    attention_bytes = num_layers * attention_per_layer * bf16_bytes
    
    # 3. Expert Layers (MXFP4) - quantized
    # SwiGLU: Gate, Up, Down projections
    # Each expert: 3 * hidden_size * intermediate_size
    mxfp4_bytes_per_element = 4.25 / 8.0
    expert_per_layer = num_experts * 3 * hidden * intermediate
    expert_bytes = num_layers * expert_per_layer * mxfp4_bytes_per_element
    
    # 4. Total
    total_bytes = embedding_bytes + attention_bytes + expert_bytes
    return total_bytes / (1024**3)


def calculate_architecture_params(metadata: ModelMetadata) -> float:
    """
    Calculate total parameters in billions from architectural dimensions.
    Used when explicit parameter count is missing.
    
    Formula sums:
    1. Embeddings (Vocab * Hidden)
    2. Attention (Layers * (Q + K + V + O))
    3. FFN (Layers * Experts * (Gate + Up + Down))
    4. Head (Vocab * Hidden)
    
    Returns:
        float: Parameters in Billions (B)
    """
    # Strict validation - no magic defaults
    if not metadata.num_layers or not metadata.hidden_size:
        raise PreflightValidationError("Cannot calculate params: missing layers/hidden_size")
    if not metadata.vocab_size:
        raise PreflightValidationError("Cannot calculate params: missing vocab_size")
    if not metadata.intermediate_size:
        raise PreflightValidationError("Cannot calculate params: missing intermediate_size")

    L = metadata.num_layers
    h = metadata.hidden_size
    v = metadata.vocab_size
    i = metadata.intermediate_size
    experts = metadata.num_experts or 1
    
    # GQA Scaling for Attention
    # Standard: Q(h*h), K(h*h), V(h*h), O(h*h) -> 4h^2
    # GQA: K and V are scaled down
    num_heads = metadata.num_attention_heads or (h // 64)
    num_kv = metadata.num_kv_heads or num_heads
    gqa_ratio = num_kv / num_heads if num_heads > 0 else 1.0
    
    # 1. Embeddings & Head
    # Counted twice: once for input embedding, once for output head (usually)
    # Even if tied, they represent weight usage.
    embeddings_params = 2 * v * h
    
    # 2. Attention Weights
    # Q + O = 2 * h^2
    # K + V = 2 * h^2 * gqa_ratio
    att_params_per_layer = (h * h * 2) + (h * h * 2 * gqa_ratio)
    total_att_params = L * att_params_per_layer
    
    # 3. FFN (MLP) Weights
    # Modern architectures (Llama/Mistral/Qwen) use SwiGLU: 3 matrices (Gate, Up, Down)
    # Size: h * i
    # MoE: multiplied by num_experts
    ffn_params_per_layer = experts * 3 * h * i
    total_ffn_params = L * ffn_params_per_layer
    
    total_params = embeddings_params + total_att_params + total_ffn_params
    
    return total_params / 1e9


def estimate_llamacpp_memory(
    metadata: ModelMetadata,
    context_length: int
) -> MemoryEstimate:
    """Estimate memory requirements for llama.cpp backend."""
    # Weights
    if metadata.exact_model_size_gb:
        weights_gb = metadata.exact_model_size_gb
    elif metadata.quantization_type == "mxfp4" or (metadata.quantization_exceptions and metadata.is_moe):
        # Mixed precision model (GPT-OSS)
        weights_gb = calculate_mixed_precision_weights_gb(metadata)
    elif metadata.params_billions and metadata.bits_per_weight:
        weights_gb = (metadata.params_billions * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
    else:
        # Fallback: Calculate params from architecture (The Recipe)
        try:
            calculated_params = calculate_architecture_params(metadata)
            weights_gb = (calculated_params * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
            # Store for reference
            metadata.params_billions = calculated_params
        except PreflightValidationError as e:
            # If we can't calculate from dimensions, THEN fail
            raise PreflightValidationError(f"Cannot estimate memory: {e}")
    
    if metadata.is_moe:
        weights_gb *= MOE_WEIGHT_OVERHEAD
    
    # KV Cache (Unified Logic)
    try:
        kv_cache_gb = calculate_kv_cache_gb(
            metadata=metadata,
            context_length=context_length,
            dtype_bytes=KV_DTYPE_FP16
        )
    except Exception as e:
        raise PreflightValidationError(f"KV cache calculation failed: {e}")
    
    # Overhead
    if not metadata.params_billions:
        raise PreflightValidationError("Cannot determine overhead - params unknown")
        
    overhead_gb = get_activation_overhead_gb(
        params_billions=metadata.params_billions,
        is_moe=metadata.is_moe,
        backend="llama_cpp"
    )
    
    total_required_gb = weights_gb + kv_cache_gb + overhead_gb
    
    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=0.0,
        context_length=context_length,
        params_billions=metadata.params_billions,
        quantization=metadata.quantization_type
    )


def estimate_mlx_memory(
    metadata: ModelMetadata,
    context_length: int
) -> MemoryEstimate:
    """Estimate memory requirements for MLX backend."""
    # Weights
    if metadata.exact_model_size_gb:
        weights_gb = metadata.exact_model_size_gb
    elif metadata.quantization_type == "mxfp4" or (metadata.quantization_exceptions and metadata.is_moe):
        # Mixed precision model (GPT-OSS)
        weights_gb = calculate_mixed_precision_weights_gb(metadata)
    elif metadata.params_billions and metadata.bits_per_weight:
        weights_gb = (metadata.params_billions * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
    else:
        # Fallback: Calculate params from architecture (The Recipe)
        try:
            calculated_params = calculate_architecture_params(metadata)
            weights_gb = (calculated_params * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
            # Store for reference
            metadata.params_billions = calculated_params
        except PreflightValidationError as e:
            # If we can't calculate from dimensions, THEN fail
            raise PreflightValidationError(f"Cannot estimate memory: {e}")
    
    if metadata.is_moe:
        weights_gb *= MOE_WEIGHT_OVERHEAD
    
    # KV Cache (Unified Logic)
    # MLX uses FP16 cache even with quantized weights
    try:
        kv_cache_gb = calculate_kv_cache_gb(
            metadata=metadata,
            context_length=context_length,
            dtype_bytes=KV_DTYPE_FP16
        )
    except Exception as e:
        raise PreflightValidationError(f"KV cache calculation failed: {e}")
    
    # Overhead
    if not metadata.params_billions:
        raise PreflightValidationError("Cannot determine overhead - params unknown")
        
    overhead_gb = get_activation_overhead_gb(
        params_billions=metadata.params_billions,
        is_moe=metadata.is_moe,
        backend="mlx_lm"
    )
    
    total_required_gb = weights_gb + kv_cache_gb + overhead_gb
    
    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=0.0,
        context_length=context_length,
        params_billions=metadata.params_billions,
        quantization=metadata.quantization_type
    )


def estimate_vllm_memory(
    metadata: ModelMetadata
) -> MemoryEstimate:
    """
    Estimate memory requirements for vLLM backend (Stage 1: Load Check).
    """
    # Weights
    if metadata.exact_model_size_gb:
        weights_gb = metadata.exact_model_size_gb
    elif metadata.quantization_type == "mxfp4" or (metadata.quantization_exceptions and metadata.is_moe):
        # Mixed precision model (GPT-OSS)
        weights_gb = calculate_mixed_precision_weights_gb(metadata)
    elif metadata.params_billions:
        dtype_bytes = 2.0
        if metadata.bits_per_weight < 16:
            dtype_bytes = metadata.bits_per_weight / 8
        weights_gb = (metadata.params_billions * 1e9 * dtype_bytes) / (1024**3)
    else:
        # Fallback: Calculate params from architecture (The Recipe)
        try:
            calculated_params = calculate_architecture_params(metadata)
            
            # Re-apply vLLM dtype logic
            dtype_bytes = 2.0
            if metadata.bits_per_weight < 16:
                dtype_bytes = metadata.bits_per_weight / 8
                
            weights_gb = (calculated_params * 1e9 * dtype_bytes) / (1024**3)
            # Store for reference
            metadata.params_billions = calculated_params
        except PreflightValidationError as e:
            raise PreflightValidationError(f"Cannot estimate memory: {e}")
    
    if metadata.is_moe:
        weights_gb *= MOE_WEIGHT_OVERHEAD
    
    # Overhead
    if not metadata.params_billions:
        raise PreflightValidationError("Cannot determine overhead - params unknown")
        
    base_overhead = get_activation_overhead_gb(
        params_billions=metadata.params_billions,
        is_moe=metadata.is_moe,
        backend="vllm"
    )
    
    # Calculate raw required (weights + base overhead)
    raw_required = weights_gb + base_overhead
    
    # Apply vLLM 90% GPU utilization rule
    # vLLM reserves 90% of VRAM, so we need to account for this reservation
    total_required_gb = raw_required / VLLM_GPU_UTILIZATION
    
    # Overhead now includes the implicit reservation space
    overhead_gb = total_required_gb - weights_gb
    
    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=0.0,
        overhead_gb=overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=0.0,
        context_length=0,
        params_billions=metadata.params_billions,
        quantization=metadata.quantization_type
    )
