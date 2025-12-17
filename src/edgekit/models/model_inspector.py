"""
Model Inspector for Memory Validation

Extracts precise metadata from model files for accurate memory estimation.
Supports GGUF (llama.cpp), MLX, and standard Transformers models.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from gguf import GGUFReader, GGMLQuantizationType

from .quantization_maps import get_gguf_bpw
from .metadata_parser import ModelMetadata, MetadataParser, extract_params_from_name


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
    # Parse filename for fallback info
    filename = os.path.basename(gguf_path)
    
    try:
        # Read GGUF header (does NOT load weights into memory)
        reader = GGUFReader(gguf_path, mode='r')
        gguf_fields = {k: v for k, v in reader.fields.items()}
        
        # Calculate exact model size by iterating tensors (most accurate)
        file_size_gb = None
        if hasattr(reader, 'tensors'):
            total_bytes = 0
            for tensor in reader.tensors:
                try:
                    tensor_quant = GGMLQuantizationType(tensor.tensor_type)
                    tensor_bpw = get_gguf_bpw(tensor_quant.name)
                except (ValueError, Exception):
                    # Fallback to a default if we can't determine tensor quantization
                    tensor_bpw = 16.0
                
                # Calculate tensor size
                num_elements = 1
                for dim in tensor.shape:
                    num_elements *= dim
                
                tensor_bytes = (num_elements * tensor_bpw) / 8
                total_bytes += tensor_bytes
            
            file_size_gb = total_bytes / (1024**3)
        else:
            # Fallback: use file size as approximation
            if os.path.exists(gguf_path):
                file_size_gb = os.path.getsize(gguf_path) / (1024**3)
        
        # Parse metadata using centralized parser
        metadata = MetadataParser.parse_gguf(gguf_fields, model_id, file_size_gb=file_size_gb, filename=filename)
        
        # Estimate params from size if not already set
        if metadata.params_billions is None and file_size_gb and metadata.bits_per_weight:
            # params ≈ total_bytes / (bpw / 8)
            total_bytes = file_size_gb * (1024**3)
            estimated_params = (total_bytes * 8) / metadata.bits_per_weight
            metadata.params_billions = estimated_params / 1e9
        
    except Exception:
        # Fallback to filename-based estimation
        # Create minimal metadata with fallback values
        metadata = ModelMetadata(
            model_id=model_id,
            model_type="gguf"
        )
        from .quantization_maps import parse_quantization_from_filename
        quant_type, bpw = parse_quantization_from_filename(filename)
        metadata.quantization_type = quant_type
        metadata.bits_per_weight = bpw
        metadata.params_billions = extract_params_from_name(model_id)
        
        if os.path.exists(gguf_path):
            metadata.exact_model_size_gb = os.path.getsize(gguf_path) / (1024**3)
    
    return metadata


def inspect_transformers_model(model_cache_path: str, model_id: str) -> ModelMetadata:
    """
    Extract metadata from a cached Transformers/MLX model directory.
    
    Reads config.json with awareness of GQA and RoPE scaling.
    Supports advanced hybrid attention detection (Gemma 3, Qwen 2).
    
    Args:
        model_cache_path: Path to the cached model directory
        model_id: Original model ID (for identification)
        
    Returns:
        ModelMetadata with all extracted values
    """
    # Determine model type hint
    model_type_hint = "mlx" if "mlx" in model_id.lower() else "transformers"
    
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
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type_hint
        )
        metadata.params_billions = extract_params_from_name(model_id)
        from .quantization_maps import parse_quantization_from_model_id
        quant_type, bpw = parse_quantization_from_model_id(model_id)
        metadata.quantization_type = quant_type
        metadata.bits_per_weight = bpw
        return metadata
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Parse metadata using centralized parser
        metadata = MetadataParser.parse_transformers(config, model_id, model_type_hint=model_type_hint)
        
    except Exception:
        # Fallback to name-based estimation
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type_hint
        )
        metadata.params_billions = extract_params_from_name(model_id)
        from .quantization_maps import parse_quantization_from_model_id
        quant_type, bpw = parse_quantization_from_model_id(model_id)
        metadata.quantization_type = quant_type
        metadata.bits_per_weight = bpw
    
    return metadata


