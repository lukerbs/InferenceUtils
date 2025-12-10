"""
Quantization Maps for Memory Estimation

Provides Bits-Per-Weight (BPW) mappings for GGUF and MLX quantization types.
These values account for quantization overhead (scales, zeros, block structures).
"""

from typing import Optional, Tuple

# GGUF quantization types to Bits-Per-Weight
# Based on llama.cpp source and empirical benchmarks
# These are EFFECTIVE bits including quantization overhead
GGUF_BPW_MAP = {
    # Full precision
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    
    # Legacy quantization
    "Q4_0": 4.50,
    "Q4_1": 4.50,
    "Q5_0": 5.50,
    "Q5_1": 5.50,
    "Q8_0": 8.50,
    "Q8_1": 8.50,
    
    # K-quants (k-means quantization) - most common
    "Q2_K": 2.56,
    "Q3_K_S": 3.44,
    "Q3_K_M": 3.91,
    "Q3_K_L": 4.27,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.85,  # Most commonly used
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q6_K": 6.59,
    "Q8_K": 8.50,
    
    # I-quants (importance-weighted quantization)
    "IQ1_S": 1.56,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.50,
    "IQ3_XXS": 3.06,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    "IQ4_NL": 4.50,
    "IQ4_XS": 4.25,
}

# MLX quantization types to Bits-Per-Weight
# MLX uses group-wise quantization with FP16 scales
MLX_BPW_MAP = {
    "2bit": 2.25,
    "3bit": 3.75,
    "4bit": 4.25,  # Group size 32/64 with FP16 scales
    "8bit": 8.25,
    "fp16": 16.0,
    "bf16": 16.0,
}


def get_gguf_bpw(quant_type: str) -> float:
    """
    Get Bits-Per-Weight for a GGUF quantization type.
    
    Args:
        quant_type: GGUF quantization type string (e.g., "Q4_K_M")
        
    Returns:
        Bits per weight (e.g., 4.85 for Q4_K_M)
    """
    # Normalize the type string
    quant_upper = quant_type.upper().strip()
    
    # Direct lookup
    if quant_upper in GGUF_BPW_MAP:
        return GGUF_BPW_MAP[quant_upper]
    
    # Try without underscore variations
    for key, value in GGUF_BPW_MAP.items():
        if key.replace("_", "") == quant_upper.replace("_", ""):
            return value
    
    # Default to FP16 if unknown (conservative)
    return 16.0


def get_mlx_bpw(quant_type: str) -> float:
    """
    Get Bits-Per-Weight for an MLX quantization type.
    
    Args:
        quant_type: MLX quantization type string (e.g., "4bit")
        
    Returns:
        Bits per weight (e.g., 4.25 for 4bit)
    """
    quant_lower = quant_type.lower().strip()
    
    # Direct lookup
    if quant_lower in MLX_BPW_MAP:
        return MLX_BPW_MAP[quant_lower]
    
    # Try variations
    if "4" in quant_lower:
        return MLX_BPW_MAP["4bit"]
    elif "3" in quant_lower:
        return MLX_BPW_MAP["3bit"]
    elif "8" in quant_lower:
        return MLX_BPW_MAP["8bit"]
    elif "2" in quant_lower:
        return MLX_BPW_MAP["2bit"]
    
    # Default to FP16 if unknown (conservative)
    return 16.0


def parse_quantization_from_filename(filename: str) -> Tuple[Optional[str], float]:
    """
    Parse quantization type from a GGUF filename.
    
    Args:
        filename: GGUF filename (e.g., "llama-3-8b.Q4_K_M.gguf")
        
    Returns:
        Tuple of (quant_type, bpw) or (None, 16.0) if not found
    """
    filename_upper = filename.upper()
    
    # Check for each known quantization type in the filename
    # Order from most specific to least specific
    for quant_type in sorted(GGUF_BPW_MAP.keys(), key=len, reverse=True):
        if quant_type in filename_upper:
            return quant_type, GGUF_BPW_MAP[quant_type]
    
    return None, 16.0


def parse_quantization_from_model_id(model_id: str) -> Tuple[str, float]:
    """
    Parse quantization type from an MLX model ID.
    
    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Mistral-7B-4bit")
        
    Returns:
        Tuple of (quant_type, bpw) or ("fp16", 16.0) if not found
    """
    model_lower = model_id.lower()
    
    if "4bit" in model_lower or "4-bit" in model_lower:
        return "4bit", MLX_BPW_MAP["4bit"]
    elif "3bit" in model_lower or "3-bit" in model_lower:
        return "3bit", MLX_BPW_MAP["3bit"]
    elif "8bit" in model_lower or "8-bit" in model_lower:
        return "8bit", MLX_BPW_MAP["8bit"]
    elif "2bit" in model_lower or "2-bit" in model_lower:
        return "2bit", MLX_BPW_MAP["2bit"]
    
    return "fp16", 16.0

