"""
Metadata Parser for EdgeKit

Centralizes the logic for extracting ModelMetadata from raw configuration dictionaries
(Transformers/SafeTensors/MLX) and GGUF Key-Value pairs.

This module enforces DRY principles by ensuring that logic for detecting architectures
like DeepSeek MLA, MoE, and Hybrid SWA is defined in exactly one place.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

try:
    from gguf import GGMLQuantizationType
except ImportError:
    GGMLQuantizationType = None

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
    
    # Explicit layer types list (e.g., ["sliding", "full", "sliding"...])
    layer_types: Optional[List[str]] = None
    # Specific intermediate size (critical for gpt-oss "thin" experts)
    intermediate_size: Optional[int] = None
    # Modules kept in high precision (e.g. ["self_attn", "embed_tokens"])
    quantization_exceptions: List[str] = field(default_factory=list)
    
    # Context configuration
    base_context_length: Optional[int] = None  # From config
    rope_scaling_factor: float = 1.0  # RoPE scaling multiplier
    model_max_context: Optional[int] = None  # Actual supported context
    
    # Hybrid Attention / Sliding Window Architecture
    has_sliding_window: bool = False
    sliding_window_size: Optional[int] = None
    num_global_layers: Optional[int] = None
    num_local_layers: Optional[int] = None
    
    # Model type flags
    is_moe: bool = False
    num_experts: Optional[int] = None
    is_multimodal: bool = False
    
    # DeepSeek MLA (Multi-Head Latent Attention)
    kv_lora_rank: Optional[int] = None
    
    # Additional metadata
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        size_info = f"{self.params_billions:.1f}B" if self.params_billions else "?B"
        quant_info = self.quantization_type or "fp16"
        return f"ModelMetadata({self.model_id!r}, {size_info}, {quant_info})"


class MetadataParser:
    """
    Pure logic class for parsing raw configuration data into ModelMetadata.
    """

    @staticmethod
    def parse_gguf(gguf_fields: Dict[str, Any], model_id: str, file_size_gb: Optional[float] = None, filename: str = "") -> ModelMetadata:
        """
        Parse GGUF Key-Value pairs into ModelMetadata.
        """
        metadata = ModelMetadata(
            model_id=model_id,
            model_type="gguf"
        )
        
        # 1. Fallback Quantization (Filename)
        filename_quant, filename_bpw = parse_quantization_from_filename(filename)
        
        # 2. Extract Quantization from Header
        if 'general.file_type' in gguf_fields:
            ft = gguf_fields['general.file_type']
            # Access .parts[0] if it's a GGUFField, otherwise use directly
            file_type_id = ft.parts[0] if hasattr(ft, 'parts') else ft
            
            # Try to convert to enum and get quantization info
            if GGMLQuantizationType is not None:
                try:
                    quant_enum = GGMLQuantizationType(file_type_id)
                    metadata.quantization_type = quant_enum.name
                    metadata.bits_per_weight = get_gguf_bpw(quant_enum.name)
                except (ValueError, Exception):
                    # Fallback to filename parsing if enum conversion fails
                    metadata.quantization_type = filename_quant
                    metadata.bits_per_weight = filename_bpw
            else:
                # GGMLQuantizationType not available, use filename fallback
                metadata.quantization_type = filename_quant
                metadata.bits_per_weight = filename_bpw
        else:
            metadata.quantization_type = filename_quant
            metadata.bits_per_weight = filename_bpw

        # 3. Helpers
        def get_field(key: str, default=None):
            if key in gguf_fields:
                val = gguf_fields[key]
                return val.parts[0] if hasattr(val, 'parts') else val
            return default

        # 4. Architecture Prefix Detection
        arch = get_field('general.architecture') or 'llama'

        def get_arch_field(suffix: str, default=None):
            return (
                get_field(f'{arch}.{suffix}') or
                get_field(f'llama.{suffix}') or
                default
            )

        # 5. Extract Core Params
        metadata.num_layers = get_arch_field('block_count')
        metadata.num_attention_heads = get_arch_field('attention.head_count')
        metadata.num_kv_heads = get_arch_field('attention.head_count_kv') or metadata.num_attention_heads
        metadata.hidden_size = get_arch_field('embedding_length')
        metadata.vocab_size = get_arch_field('vocab_size')

        if metadata.hidden_size and metadata.num_attention_heads and metadata.num_attention_heads > 0:
            metadata.head_dim = metadata.hidden_size // metadata.num_attention_heads

        # 6. Context Length
        context_length = None
        for key in gguf_fields:
            if key.endswith('.context_length'):
                context_length = get_field(key)
                if context_length:
                    break
        metadata.base_context_length = context_length
        metadata.model_max_context = metadata.base_context_length

        # 7. Advanced Architectures
        
        # Sliding Window
        sliding_window = get_arch_field('attention.sliding_window') or get_arch_field('rope.sliding_window')
        if sliding_window:
            metadata.has_sliding_window = True
            metadata.sliding_window_size = sliding_window
            # GGUF Specific: Gemma 2 1:1 pattern check
            if arch == "gemma2" and metadata.num_layers:
                metadata.num_global_layers = metadata.num_layers // 2
                metadata.num_local_layers = metadata.num_layers - metadata.num_global_layers

        # MoE
        num_experts = get_arch_field('expert_count')
        if num_experts and num_experts > 1:
            metadata.is_moe = True
            metadata.num_experts = num_experts

        # DeepSeek MLA
        kv_lora_rank = get_arch_field('attention.kv_lora_rank')
        if kv_lora_rank:
            metadata.kv_lora_rank = kv_lora_rank

        # 8. Model Size / Params
        metadata.exact_model_size_gb = file_size_gb
        
        # Param estimation fallback
        if metadata.params_billions is None:
            metadata.params_billions = extract_params_from_name(model_id)
            
        return metadata

    @staticmethod
    def parse_transformers(config: Dict[str, Any], model_id: str, model_type_hint: str = "transformers") -> ModelMetadata:
        """
        Parse config.json (Transformers/MLX) into ModelMetadata.
        """
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type_hint
        )
        metadata.raw_config = config
        
        # Detect GPT-OSS models
        arch_name = config.get('architectures', [])
        is_gpt_oss = any('GptOss' in str(arch) for arch in arch_name)
        if is_gpt_oss:
            metadata.model_type = "gpt_oss"
        
        # Handle nested configs (common in multimodal models)
        text_config = config.get('text_config', config)

        # 1. Core Params
        metadata.num_layers = (
            text_config.get('num_hidden_layers') or
            text_config.get('n_layer') or
            text_config.get('num_layers')
        )
        metadata.hidden_size = (
            text_config.get('hidden_size') or
            text_config.get('n_embd') or
            text_config.get('d_model')
        )
        
        # Extract intermediate_size (critical for GPT-OSS "thin-deep" topology)
        metadata.intermediate_size = text_config.get('intermediate_size')
        # Critical: For GptOssForCausalLM, if missing, default to hidden_size (not 4x)
        # This handles "thin-deep" topology where intermediate == hidden
        if metadata.intermediate_size is None and metadata.hidden_size and is_gpt_oss:
            metadata.intermediate_size = metadata.hidden_size
        
        metadata.num_attention_heads = (
            text_config.get('num_attention_heads') or
            text_config.get('n_head') or
            text_config.get('num_heads')
        )
        # GQA / Multi-Query
        metadata.num_kv_heads = (
            text_config.get('num_key_value_heads') or
            text_config.get('n_kv_heads') or
            text_config.get('num_kv_heads') or
            text_config.get('kv_heads') or
            metadata.num_attention_heads
        )
        if text_config.get('multi_query', False):
            metadata.num_kv_heads = 1

        if metadata.hidden_size and metadata.num_attention_heads:
             metadata.head_dim = metadata.hidden_size // metadata.num_attention_heads

        metadata.vocab_size = text_config.get('vocab_size')

        # 2. Context Length & RoPE
        found_length = None
        for field in ['max_position_embeddings', 'sliding_window', 'n_positions', 'n_ctx', 'max_seq_len']:
            if field in text_config:
                found_length = text_config[field]
                break
        metadata.base_context_length = found_length

        rope_scaling = text_config.get('rope_scaling')
        if rope_scaling and isinstance(rope_scaling, dict):
            scaling_type = rope_scaling.get('type', '')
            factor = rope_scaling.get('factor', 1.0)
            if scaling_type in ['linear', 'dynamic', 'yarn', 'su', 'longrope']:
                metadata.rope_scaling_factor = factor
        
        if metadata.base_context_length:
            metadata.model_max_context = int(metadata.base_context_length * metadata.rope_scaling_factor)

        # Extract explicit layer types list (GPT-OSS uses this instead of patterns)
        metadata.layer_types = config.get('layer_types')

        # 3. Hybrid Attention (The "Gemma 3" / "Qwen 2" / "GPT-OSS" Logic)
        # Priority 1: Explicit layer_types list (GPT-OSS)
        if metadata.layer_types and isinstance(metadata.layer_types, list):
            num_global = sum(1 for lt in metadata.layer_types if lt == "full_attention" or lt == "full")
            num_local = sum(1 for lt in metadata.layer_types if lt == "sliding_attention" or lt == "sliding")
            if num_global > 0 or num_local > 0:
                metadata.has_sliding_window = True
                metadata.num_global_layers = num_global
                metadata.num_local_layers = num_local
                # Set sliding_window_size if not already set
                if metadata.sliding_window_size is None:
                    # Default to a reasonable value if not specified
                    metadata.sliding_window_size = 4096  # Conservative default
        
        # Priority 2: Pattern-based (Gemma 3, Qwen 2) - only if layer_types not set
        if metadata.num_global_layers is None:
            sw_size = text_config.get('sliding_window')
            if sw_size and isinstance(sw_size, int):
                metadata.has_sliding_window = True
                metadata.sliding_window_size = sw_size

            sw_pattern = text_config.get('sliding_window_pattern')
            if sw_pattern and isinstance(sw_pattern, int):
                if metadata.num_layers:
                    metadata.num_global_layers = metadata.num_layers // sw_pattern
                    metadata.num_local_layers = metadata.num_layers - metadata.num_global_layers

            use_sw = text_config.get('use_sliding_window', True)
            max_win_layers = text_config.get('max_window_layers')
            
            if metadata.has_sliding_window and use_sw and max_win_layers is not None:
                 metadata.num_global_layers = max_win_layers
                 if metadata.num_layers:
                     metadata.num_local_layers = metadata.num_layers - max_win_layers

            # Mistral fallback (Uniform SWA)
            if metadata.has_sliding_window and metadata.num_global_layers is None:
                metadata.num_global_layers = 0
                metadata.num_local_layers = metadata.num_layers

            # Safety fallback
            if metadata.num_global_layers is None and metadata.num_layers:
                 metadata.num_global_layers = metadata.num_layers
                 metadata.num_local_layers = 0

        # 4. Advanced Architectures
        num_experts = text_config.get('num_local_experts') or text_config.get('num_experts')
        if num_experts and num_experts > 1:
            metadata.is_moe = True
            metadata.num_experts = num_experts

        if 'vision_config' in config or 'image_processor' in str(config):
            metadata.is_multimodal = True

        kv_lora_rank = text_config.get('kv_lora_rank')
        if kv_lora_rank:
            metadata.kv_lora_rank = kv_lora_rank

        # 5. Params & Quantization
        params = text_config.get('num_parameters') or text_config.get('n_params')
        if params:
             metadata.params_billions = params / 1e9 if params > 1e6 else params
        else:
             metadata.params_billions = extract_params_from_name(model_id)

        # MLX Quantization Logic
        quant_config = (
            text_config.get('quantization_config') or
            config.get('quantization_config') or
            config.get('quantization', {})
        )
        if quant_config and isinstance(quant_config, dict):
            bits = quant_config.get('bits', 16)
            group_size = quant_config.get('group_size', 64)
            metadata.quantization_type = f"{bits}bit"
            metadata.bits_per_weight = get_mlx_bpw_for_group_size(bits, group_size)
            # Extract modules kept in high precision (BF16 for GPT-OSS)
            metadata.quantization_exceptions = quant_config.get('modules_to_not_convert', [])
        else:
            quant_type, bpw = parse_quantization_from_model_id(model_id)
            metadata.quantization_type = quant_type
            metadata.bits_per_weight = bpw

        # Size estimate
        if metadata.params_billions:
            metadata.exact_model_size_gb = (metadata.params_billions * 1e9 * metadata.bits_per_weight) / 8 / (1024**3)

        return metadata


def extract_params_from_name(model_id: str) -> Optional[float]:
    """
    Extract parameter count from model name.
    Moved here to be shared between inspectors.
    
    Note: MoE models are not handled here as we cannot accurately guess
    total params from filename (expert routing overhead is variable).
    """
    patterns = [
        r'(\d+\.?\d*)[bB](?!it)',
        r'(\d+\.?\d*)-[bB]',
        r'(\d+)billion',
    ]
    for pattern in patterns:
        match = re.search(pattern, model_id)
        if match:
            return float(match.group(1))
    return None

