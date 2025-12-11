"""
Model Inspector for Memory Validation

Extracts precise metadata from model files for accurate memory estimation.
Supports GGUF (llama.cpp), MLX, and standard Transformers models.
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from gguf import GGUFReader, GGMLQuantizationType

from .quantization_maps import (
    get_gguf_bpw,
    get_mlx_bpw_for_group_size,
    parse_quantization_from_filename,
    parse_quantization_from_model_id,
)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata for memory estimation."""
    
    # Model identification
    model_id: str
    model_type: str  # "gguf", "mlx", "transformers"
    
    # Size and quantization
    params_billions: Optional[float] = None
    bits_per_weight: float = 16.0
    quantization_type: Optional[str] = None
    exact_model_size_gb: Optional[float] = None  # From file or tensor calculation
    
    # Architecture (critical for KV cache calculation)
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None  # GQA-aware!
    head_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    
    # Context configuration
    base_context_length: Optional[int] = None  # From config
    rope_scaling_factor: float = 1.0  # RoPE scaling multiplier
    model_max_context: Optional[int] = None  # Actual supported context
    sliding_window: Optional[int] = None
    
    # Model type flags
    is_moe: bool = False
    num_experts: Optional[int] = None
    is_multimodal: bool = False
    
    # DeepSeek MLA (Multi-Head Latent Attention) - compressed KV cache
    # If set, KV cache uses low-rank latent compression (90%+ reduction)
    kv_lora_rank: Optional[int] = None
    
    # Additional metadata
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        size_info = f"{self.params_billions:.1f}B" if self.params_billions else "?B"
        quant_info = self.quantization_type or "fp16"
        return f"ModelMetadata({self.model_id!r}, {size_info}, {quant_info})"


def inspect_gguf_model(gguf_path: str, model_id: str) -> ModelMetadata:
    """
    Extract metadata from a local GGUF file using binary header inspection.
    
    This is the most accurate method as it reads directly from the file header
    rather than relying on filenames or external configs.
    
    Args:
        gguf_path: Path to the local GGUF file
        model_id: Original model ID (for identification)
        
    Returns:
        ModelMetadata with all extracted values
    """
    metadata = ModelMetadata(
        model_id=model_id,
        model_type="gguf"
    )
    
    # Parse filename for fallback info
    filename = os.path.basename(gguf_path)
    filename_quant, filename_bpw = parse_quantization_from_filename(filename)
    
    # Parse params from model name (fallback)
    params_from_name = _extract_params_from_name(model_id)
    
    try:
        # Read GGUF header (does NOT load weights into memory)
        reader = GGUFReader(gguf_path, mode='r')
        gguf_fields = {k: v for k, v in reader.fields.items()}
        
        # Extract quantization type from header
        if 'general.file_type' in gguf_fields:
            file_type_id = gguf_fields['general.file_type'].parts[0]
            try:
                quant_enum = GGMLQuantizationType(file_type_id)
                metadata.quantization_type = quant_enum.name
                metadata.bits_per_weight = get_gguf_bpw(quant_enum.name)
            except (ValueError, Exception):
                metadata.quantization_type = filename_quant
                metadata.bits_per_weight = filename_bpw
        else:
            metadata.quantization_type = filename_quant
            metadata.bits_per_weight = filename_bpw
        
        # Extract architecture parameters
        def get_field(key: str, default=None):
            """Helper to extract field value from GGUF metadata."""
            if key in gguf_fields:
                parts = gguf_fields[key].parts
                if len(parts) > 0:
                    return parts[0]
            return default
        
        # Get architecture prefix (e.g., "llama", "qwen2", "phi3", "deepseek2")
        # This is the "Source of Truth" per GGUF spec - all params are namespaced under it
        arch = get_field('general.architecture') or 'llama'
        
        def get_arch_field(suffix: str, default=None):
            """Get field with architecture prefix, falling back to llama prefix."""
            # Try architecture-specific key first, then llama as super-architecture fallback
            return (
                get_field(f'{arch}.{suffix}') or
                get_field(f'llama.{suffix}') or  # Many models use llama as super-architecture
                default
            )
        
        # Layers
        metadata.num_layers = get_arch_field('block_count')
        
        # Attention heads (total)
        metadata.num_attention_heads = get_arch_field('attention.head_count')
        
        # KV heads (GQA-aware - critical!)
        metadata.num_kv_heads = (
            get_arch_field('attention.head_count_kv') or
            metadata.num_attention_heads  # Fallback to MHA if not specified
        )
        
        # Hidden size
        metadata.hidden_size = get_arch_field('embedding_length')
        
        # Calculate head dimension (guard against division by zero)
        if metadata.hidden_size and metadata.num_attention_heads and metadata.num_attention_heads > 0:
            metadata.head_dim = metadata.hidden_size // metadata.num_attention_heads
        
        # Vocab size
        metadata.vocab_size = get_arch_field('vocab_size')
        
        # Context length - search for any key ending with ".context_length"
        context_length = None
        for key in gguf_fields:
            if key.endswith('.context_length'):
                context_length = get_field(key)
                if context_length:
                    break
        metadata.base_context_length = context_length
        metadata.model_max_context = metadata.base_context_length
        
        # Sliding window attention (for Mistral, Phi-3, etc.)
        sliding_window = get_arch_field('attention.sliding_window') or get_arch_field('rope.sliding_window')
        if sliding_window:
            metadata.sliding_window = sliding_window
        
        # MoE detection (architecture-agnostic)
        num_experts = get_arch_field('expert_count')
        if num_experts and num_experts > 1:
            metadata.is_moe = True
            metadata.num_experts = num_experts
        
        # DeepSeek MLA detection (kv_lora_rank indicates compressed KV cache)
        kv_lora_rank = get_arch_field('attention.kv_lora_rank')
        if kv_lora_rank:
            metadata.kv_lora_rank = kv_lora_rank
        
        # Calculate exact model size by iterating tensors (most accurate)
        if hasattr(reader, 'tensors'):
            total_bytes = 0
            for tensor in reader.tensors:
                try:
                    tensor_quant = GGMLQuantizationType(tensor.tensor_type)
                    tensor_bpw = get_gguf_bpw(tensor_quant.name)
                except (ValueError, Exception):
                    tensor_bpw = metadata.bits_per_weight
                
                # Calculate tensor size
                num_elements = 1
                for dim in tensor.shape:
                    num_elements *= dim
                
                tensor_bytes = (num_elements * tensor_bpw) / 8
                total_bytes += tensor_bytes
            
            metadata.exact_model_size_gb = total_bytes / (1024**3)
            
            # Estimate params from size (reverse calculation)
            if metadata.params_billions is None:
                # params ≈ total_bytes / (bpw / 8)
                estimated_params = (total_bytes * 8) / metadata.bits_per_weight
                metadata.params_billions = estimated_params / 1e9
        else:
            # Fallback: use file size as approximation
            if os.path.exists(gguf_path):
                metadata.exact_model_size_gb = os.path.getsize(gguf_path) / (1024**3)
        
        # Use params from name if not calculated
        if metadata.params_billions is None:
            metadata.params_billions = params_from_name
        
    except Exception:
        # Fallback to filename-based estimation
        metadata.quantization_type = filename_quant
        metadata.bits_per_weight = filename_bpw
        metadata.params_billions = params_from_name
        
        if os.path.exists(gguf_path):
            metadata.exact_model_size_gb = os.path.getsize(gguf_path) / (1024**3)
    
    return metadata


def inspect_transformers_model(model_cache_path: str, model_id: str) -> ModelMetadata:
    """
    Extract metadata from a cached Transformers/MLX model directory.
    
    Reads config.json with awareness of GQA and RoPE scaling.
    
    Args:
        model_cache_path: Path to the cached model directory
        model_id: Original model ID (for identification)
        
    Returns:
        ModelMetadata with all extracted values
    """
    metadata = ModelMetadata(
        model_id=model_id,
        model_type="mlx" if "mlx" in model_id.lower() else "transformers"
    )
    
    # Find config.json
    config_path = Path(model_cache_path) / "config.json"
    if not config_path.exists():
        # Try common subdirectories
        for subdir in ["", "model", "transformer"]:
            test_path = Path(model_cache_path) / subdir / "config.json"
            if test_path.exists():
                config_path = test_path
                break
    
    if not config_path.exists():
        # Fallback to name-based estimation
        metadata.params_billions = _extract_params_from_name(model_id)
        quant_type, bpw = parse_quantization_from_model_id(model_id)
        metadata.quantization_type = quant_type
        metadata.bits_per_weight = bpw
        return metadata
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        metadata.raw_config = config
        
        # Handle nested configs (common in multimodal models)
        text_config = config.get('text_config', config)
        
        # Number of layers
        metadata.num_layers = text_config.get('num_hidden_layers') or \
                              text_config.get('n_layer') or \
                              text_config.get('num_layers')
        
        # Hidden size
        metadata.hidden_size = text_config.get('hidden_size') or \
                               text_config.get('n_embd') or \
                               text_config.get('d_model')
        
        # Attention heads (total)
        metadata.num_attention_heads = text_config.get('num_attention_heads') or \
                                       text_config.get('n_head') or \
                                       text_config.get('num_heads')
        
        # KV heads (GQA-aware - critical!)
        # Must check multiple field names as they vary by model
        metadata.num_kv_heads = text_config.get('num_key_value_heads') or \
                                text_config.get('n_kv_heads') or \
                                text_config.get('num_kv_heads') or \
                                text_config.get('kv_heads') or \
                                metadata.num_attention_heads  # Fallback to MHA
        
        # Handle multi_query models (single KV head)
        if text_config.get('multi_query', False):
            metadata.num_kv_heads = 1
        
        # Calculate head dimension (guard against division by zero)
        if metadata.hidden_size and metadata.num_attention_heads and metadata.num_attention_heads > 0:
            metadata.head_dim = metadata.hidden_size // metadata.num_attention_heads
        
        # Vocab size
        metadata.vocab_size = text_config.get('vocab_size')
        
        # Context length (base, before RoPE scaling)
        # Search for context length using prioritized list of keys
        found_length = None
        for field in ['max_position_embeddings', 'sliding_window', 'n_positions', 'n_ctx', 'max_seq_len']:
            if field in text_config:
                found_length = text_config[field]
                break
        metadata.base_context_length = found_length
        
        # RoPE scaling (critical for real context length!)
        rope_scaling = text_config.get('rope_scaling')
        if rope_scaling and isinstance(rope_scaling, dict):
            scaling_type = rope_scaling.get('type', '')
            factor = rope_scaling.get('factor', 1.0)
            
            if scaling_type in ['linear', 'dynamic', 'yarn', 'su', 'longrope']:
                metadata.rope_scaling_factor = factor
        
        # Calculate actual max context
        if metadata.base_context_length:
            metadata.model_max_context = int(metadata.base_context_length * metadata.rope_scaling_factor)
        
        # Sliding window
        metadata.sliding_window = text_config.get('sliding_window')
        
        # MoE detection
        num_experts = text_config.get('num_local_experts') or text_config.get('num_experts')
        if num_experts and num_experts > 1:
            metadata.is_moe = True
            metadata.num_experts = num_experts
        
        # Multimodal detection
        if 'vision_config' in config or 'image_processor' in str(config):
            metadata.is_multimodal = True
        
        # Parameter count
        params = text_config.get('num_parameters') or text_config.get('n_params')
        if params:
            # If params > 1 million, assume it's in raw count and convert to billions
            metadata.params_billions = params / 1e9 if params > 1e6 else params
        else:
            metadata.params_billions = _extract_params_from_name(model_id)
        
        # Quantization detection with group size awareness
        # Check both nested 'quantization_config' and top-level 'quantization'
        quant_config = text_config.get('quantization_config') or config.get('quantization', {})
        if quant_config:
            bits = quant_config.get('bits', 16)
            group_size = quant_config.get('group_size', 64)  # MLX default is 64
            metadata.quantization_type = f"{bits}bit"
            # Use group-size-aware BPW calculation for accurate MLX estimation
            metadata.bits_per_weight = get_mlx_bpw_for_group_size(bits, group_size)
        else:
            # Parse from model ID
            quant_type, bpw = parse_quantization_from_model_id(model_id)
            metadata.quantization_type = quant_type
            metadata.bits_per_weight = bpw
        
        # Estimate model size
        if metadata.params_billions:
            metadata.exact_model_size_gb = (metadata.params_billions * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)
        
    except Exception:
        # Fallback to name-based estimation
        metadata.params_billions = _extract_params_from_name(model_id)
        quant_type, bpw = parse_quantization_from_model_id(model_id)
        metadata.quantization_type = quant_type
        metadata.bits_per_weight = bpw
    
    return metadata


def _extract_params_from_name(model_id: str) -> Optional[float]:
    """
    Extract parameter count from model name.
    
    Examples:
        - "Llama-3-70B" -> 70.0
        - "mistral-7b-instruct" -> 7.0
        - "Mixtral-8x7B" -> 46.7 (special handling for MoE)
        - "gemma-2b" -> 2.0
        
    Args:
        model_id: Model ID or name string
        
    Returns:
        Parameter count in billions, or None if not found
    """
    # Handle MoE patterns first (e.g., "8x7B" = ~46.7B)
    moe_match = re.search(r'(\d+)x(\d+\.?\d*)[bB]', model_id)
    if moe_match:
        num_experts = int(moe_match.group(1))
        expert_size = float(moe_match.group(2))
        # MoE models share attention layers, so total is roughly:
        # (num_experts * expert_size * 0.83) - accounts for shared layers
        total_params = num_experts * expert_size * 0.83
        return total_params
    
    # Standard patterns: "70B", "7b", "1.5B", "70-b"
    patterns = [
        r'(\d+\.?\d*)[bB](?!it)',  # 70B, 7b, 1.5B (not "bit")
        r'(\d+\.?\d*)-[bB]',        # 70-B, 7-b
        r'(\d+)billion',            # 70billion
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_id)
        if match:
            return float(match.group(1))
    
    return None

