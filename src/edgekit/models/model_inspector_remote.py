"""
Remote Model Inspector for Memory Validation

Provides lightweight remote inspection capabilities for preflight checks
via HTTP Range requests without downloading full model files.

Bandwidth usage:
- GGUF: ~512KB (single range request)
- Safetensors/vLLM: ~50KB (config.json + header)
- MLX: ~5KB (config.json only)
"""

import json
import struct
import logging
from pathlib import Path
from typing import Dict, Any

import requests
from huggingface_hub import hf_hub_url, list_repo_files
from gguf import GGMLQuantizationType

from .quantization_maps import (
    get_gguf_bpw,
    parse_quantization_from_filename,
    parse_quantization_from_model_id,
)
from .metadata_parser import ModelMetadata, MetadataParser, extract_params_from_name
from ..utils import check_internet

logger = logging.getLogger(__name__)

# ============================================================================
# REMOTE INSPECTION CONSTANTS
# ============================================================================

# GGUF format constants
GGUF_MAGIC = b'GGUF'
GGUF_SPECULATIVE_FETCH_SIZE = 524288  # 512KB - covers header + KV block for 99% of models

# Security/sanity limits (DoS protection)
MAX_SAFETENSORS_HEADER_SIZE = 104857600  # 100MB
MAX_GGUF_KV_COUNT = 10000
MAX_GGUF_STRING_LENGTH = 65536  # 64KB

# GGUF value type enum (from GGUF spec)
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


# ============================================================================
# EXCEPTIONS
# ============================================================================

class RemoteInspectError(Exception):
    """
    Raised when remote inspection fails and fallback to full download is needed.
    
    This exception signals that the lightweight HTTP Range-based inspection
    could not complete, but the caller can fall back to downloading the
    full model for local inspection.
    """
    pass


# ============================================================================
# REMOTE INSPECTION HELPERS
# ============================================================================

def _get_hf_headers() -> Dict[str, str]:
    """
    Get HTTP headers with HuggingFace authentication token if available.
    
    Reads token from ~/.cache/huggingface/token (managed by huggingface_hub).
    
    Returns:
        Dict with Authorization header if token exists, empty dict otherwise.
    """
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    headers = {}
    
    if token_path.exists():
        try:
            token = token_path.read_text().strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        except Exception:
            pass  # No token available, proceed without auth
    
    return headers


def _fetch_range(url: str, start: int, end: int) -> bytes:
    """
    Fetch a byte range from a URL using HTTP Range request.
    
    Args:
        url: The URL to fetch from
        start: Start byte offset (inclusive)
        end: End byte offset (inclusive)
        
    Returns:
        The requested byte range as bytes
        
    Raises:
        RemoteInspectError: If the request fails or Range is not supported
    """
    headers = _get_hf_headers()
    headers["Range"] = f"bytes={start}-{end}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        # Check for auth errors
        if response.status_code in (401, 403):
            raise RemoteInspectError(
                f"Authentication required or access denied (HTTP {response.status_code}). "
                "Ensure you have accepted the model license and are logged in."
            )
        
        # Check for successful range response
        if response.status_code != 206:
            raise RemoteInspectError(
                f"Range request not supported or failed (HTTP {response.status_code})"
            )
        
        return response.content
        
    except requests.RequestException as e:
        raise RemoteInspectError(f"Network error during range fetch: {e}")


def _fetch_json_file(repo_id: str, filename: str) -> Dict[str, Any]:
    """
    Download and parse a small JSON file from a HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-3-8B")
        filename: Name of the JSON file (e.g., "config.json")
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        RemoteInspectError: If the file cannot be fetched or parsed
    """
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = _get_hf_headers()
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code in (401, 403):
            raise RemoteInspectError(
                f"Authentication required for {repo_id} (HTTP {response.status_code})"
            )
        
        if response.status_code == 404:
            raise RemoteInspectError(f"File not found: {filename} in {repo_id}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.RequestException as e:
        raise RemoteInspectError(f"Failed to fetch {filename}: {e}")
    except json.JSONDecodeError as e:
        raise RemoteInspectError(f"Invalid JSON in {filename}: {e}")


def _parse_gguf_kv_block(data: bytes, kv_count: int, offset: int = 24) -> Dict[str, Any]:
    """
    Parse GGUF key-value metadata block from binary data.
    
    Args:
        data: Raw bytes containing the GGUF header and KV block
        kv_count: Number of KV pairs to parse (from header)
        offset: Starting offset (default 24, after the header)
        
    Returns:
        Dictionary of parsed metadata key-value pairs
        
    Raises:
        RemoteInspectError: If parsing fails or sanity limits are exceeded
    """
    if kv_count > MAX_GGUF_KV_COUNT:
        raise RemoteInspectError(f"KV count {kv_count} exceeds safety limit {MAX_GGUF_KV_COUNT}")
    
    metadata = {}
    pos = offset
    
    def read_uint64() -> int:
        nonlocal pos
        if pos + 8 > len(data):
            raise RemoteInspectError("Buffer overflow reading uint64")
        val = struct.unpack('<Q', data[pos:pos+8])[0]
        pos += 8
        return val
    
    def read_uint32() -> int:
        nonlocal pos
        if pos + 4 > len(data):
            raise RemoteInspectError("Buffer overflow reading uint32")
        val = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        return val
    
    def read_int32() -> int:
        nonlocal pos
        if pos + 4 > len(data):
            raise RemoteInspectError("Buffer overflow reading int32")
        val = struct.unpack('<i', data[pos:pos+4])[0]
        pos += 4
        return val
    
    def read_float32() -> float:
        nonlocal pos
        if pos + 4 > len(data):
            raise RemoteInspectError("Buffer overflow reading float32")
        val = struct.unpack('<f', data[pos:pos+4])[0]
        pos += 4
        return val
    
    def read_string() -> str:
        nonlocal pos
        str_len = read_uint64()
        if str_len > MAX_GGUF_STRING_LENGTH:
            raise RemoteInspectError(f"String length {str_len} exceeds safety limit")
        if pos + str_len > len(data):
            raise RemoteInspectError("Buffer overflow reading string")
        s = data[pos:pos+str_len].decode('utf-8', errors='replace')
        pos += str_len
        return s
    
    def read_value(val_type: int) -> Any:
        """Read a value based on its type."""
        nonlocal pos
        
        if val_type == GGUF_TYPE_UINT8:
            if pos + 1 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = data[pos]
            pos += 1
            return val
        elif val_type == GGUF_TYPE_INT8:
            if pos + 1 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = struct.unpack('<b', data[pos:pos+1])[0]
            pos += 1
            return val
        elif val_type == GGUF_TYPE_UINT16:
            if pos + 2 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = struct.unpack('<H', data[pos:pos+2])[0]
            pos += 2
            return val
        elif val_type == GGUF_TYPE_INT16:
            if pos + 2 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = struct.unpack('<h', data[pos:pos+2])[0]
            pos += 2
            return val
        elif val_type == GGUF_TYPE_UINT32:
            return read_uint32()
        elif val_type == GGUF_TYPE_INT32:
            return read_int32()
        elif val_type == GGUF_TYPE_FLOAT32:
            return read_float32()
        elif val_type == GGUF_TYPE_BOOL:
            if pos + 1 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = data[pos] != 0
            pos += 1
            return val
        elif val_type == GGUF_TYPE_STRING:
            return read_string()
        elif val_type == GGUF_TYPE_UINT64:
            return read_uint64()
        elif val_type == GGUF_TYPE_INT64:
            if pos + 8 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = struct.unpack('<q', data[pos:pos+8])[0]
            pos += 8
            return val
        elif val_type == GGUF_TYPE_FLOAT64:
            if pos + 8 > len(data):
                raise RemoteInspectError("Buffer overflow")
            val = struct.unpack('<d', data[pos:pos+8])[0]
            pos += 8
            return val
        elif val_type == GGUF_TYPE_ARRAY:
            # Array: element_type (uint32) + count (uint64) + elements
            elem_type = read_uint32()
            arr_len = read_uint64()
            if arr_len > 10000:  # Sanity limit for arrays
                # Skip large arrays, we don't need them for metadata
                return None
            result = []
            for _ in range(arr_len):
                result.append(read_value(elem_type))
            return result
        else:
            raise RemoteInspectError(f"Unknown GGUF value type: {val_type}")
    
    # Parse all KV pairs
    for _ in range(kv_count):
        try:
            key = read_string()
            val_type = read_uint32()
            value = read_value(val_type)
            if value is not None:
                metadata[key] = value
        except RemoteInspectError:
            # If we hit a buffer overflow, we've extracted what we can
            logger.debug(f"GGUF parsing stopped early, extracted {len(metadata)} keys")
            break
    
    return metadata


# ============================================================================
# REMOTE INSPECTION FUNCTIONS
# ============================================================================

def inspect_gguf_remote(repo_id: str) -> "ModelMetadata":
    """
    Inspect a GGUF model remotely using HTTP Range requests.
    
    Fetches only the first 512KB of the GGUF file to extract metadata,
    avoiding the need to download the entire model (often 10-100GB).
    
    Args:
        repo_id: HuggingFace repository ID containing a GGUF file
        
    Returns:
        ModelMetadata with extracted architecture and quantization info
        
    Raises:
        RemoteInspectError: If inspection fails
    """
    # Find GGUF file in repository
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        raise RemoteInspectError(f"Failed to list repository files: {e}")
    
    gguf_files = [f for f in files if f.endswith('.gguf')]
    if not gguf_files:
        raise RemoteInspectError(f"No GGUF files found in {repo_id}")
    
    # Use first GGUF file (usually there's only one, or pick the main one)
    gguf_filename = gguf_files[0]
    
    # Get the CDN URL for the file
    try:
        url = hf_hub_url(repo_id=repo_id, filename=gguf_filename)
    except Exception as e:
        raise RemoteInspectError(f"Failed to resolve file URL: {e}")
    
    # Fetch first 512KB (speculative fetch)
    data = _fetch_range(url, 0, GGUF_SPECULATIVE_FETCH_SIZE - 1)
    
    # Validate magic bytes
    if len(data) < 24:
        raise RemoteInspectError("GGUF file too small for header")
    
    magic = data[0:4]
    if magic != GGUF_MAGIC:
        raise RemoteInspectError(f"Invalid GGUF magic: {magic!r}")
    
    # Parse header
    version, tensor_count, kv_count = struct.unpack('<IQQ', data[4:24])
    
    if version not in (2, 3):
        logger.warning(f"Unexpected GGUF version {version}, proceeding anyway")
    
    # Parse KV metadata block
    gguf_metadata = _parse_gguf_kv_block(data, kv_count, offset=24)
    
    # Parse metadata using centralized parser
    return MetadataParser.parse_gguf(gguf_metadata, repo_id, filename=gguf_filename)


def inspect_safetensors_remote(repo_id: str) -> "ModelMetadata":
    """
    Inspect a Safetensors model remotely using HTTP Range requests.
    
    Uses a dual-source approach:
    1. Fetch config.json for architecture parameters
    2. Fetch safetensors header for actual dtype (precision)
    
    Args:
        repo_id: HuggingFace repository ID containing Safetensors files
        
    Returns:
        ModelMetadata with extracted architecture and precision info
        
    Raises:
        RemoteInspectError: If inspection fails
    """
    # Fetch config.json for architecture parameters
    config = _fetch_json_file(repo_id, "config.json")
    
    # Parse base metadata from config using centralized parser
    metadata = MetadataParser.parse_transformers(config, repo_id)
    
    # Now fetch safetensors header to determine actual dtype
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        raise RemoteInspectError(f"Failed to list repository files: {e}")
    
    # Find safetensors file(s)
    safetensors_files = [f for f in files if f.endswith('.safetensors')]
    if not safetensors_files:
        raise RemoteInspectError(f"No safetensors files found in {repo_id}")
    
    # Check for sharded model
    index_file = "model.safetensors.index.json"
    if index_file in files:
        # Sharded model - just use first shard for dtype detection
        try:
            index = _fetch_json_file(repo_id, index_file)
            weight_map = index.get('weight_map', {})
            if weight_map:
                # Get first shard filename
                first_shard = list(set(weight_map.values()))[0]
                safetensors_file = first_shard
            else:
                safetensors_file = safetensors_files[0]
        except Exception:
            safetensors_file = safetensors_files[0]
    elif "model.safetensors" in safetensors_files:
        safetensors_file = "model.safetensors"
    else:
        safetensors_file = safetensors_files[0]
    
    # Fetch safetensors header
    url = hf_hub_url(repo_id=repo_id, filename=safetensors_file)
    
    # Step 1: Get header length (first 8 bytes)
    header_len_bytes = _fetch_range(url, 0, 7)
    header_len = struct.unpack('<Q', header_len_bytes)[0]
    
    if header_len > MAX_SAFETENSORS_HEADER_SIZE:
        raise RemoteInspectError(f"Safetensors header too large: {header_len}")
    
    # Step 2: Fetch header JSON
    header_bytes = _fetch_range(url, 8, 8 + header_len - 1)
    
    try:
        header = json.loads(header_bytes.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise RemoteInspectError(f"Invalid safetensors header JSON: {e}")
    
    # Extract dtype from a representative tensor
    # Skip __metadata__ key, find an actual tensor
    dtype = None
    for key, tensor_info in header.items():
        if key == '__metadata__':
            continue
        if isinstance(tensor_info, dict) and 'dtype' in tensor_info:
            dtype = tensor_info['dtype']
            break
    
    # Map dtype to bits_per_weight
    dtype_bpw = {
        'F32': 32.0, 'F16': 16.0, 'BF16': 16.0,
        'F8_E4M3': 8.0, 'F8_E5M2': 8.0,
        'I32': 32.0, 'I16': 16.0, 'I8': 8.0,
        'U8': 8.0, 'BOOL': 1.0,
    }
    
    if dtype and dtype in dtype_bpw:
        metadata.bits_per_weight = dtype_bpw[dtype]
        metadata.quantization_type = dtype
    else:
        # Fallback to config or name parsing
        quant_type, bpw = parse_quantization_from_model_id(repo_id)
        metadata.quantization_type = quant_type
        metadata.bits_per_weight = bpw
    
    # Estimate model size
    if metadata.params_billions:
        metadata.exact_model_size_gb = (
            metadata.params_billions * 1e9 * metadata.bits_per_weight
        ) / 8 / (1024**3)
    
    return metadata


def inspect_mlx_remote(repo_id: str) -> "ModelMetadata":
    """
    Inspect an MLX model remotely by fetching config.json.
    
    MLX models use quantization_config in config.json as the authority
    for quantization bits and group size. The safetensors dtype may show
    I32/U32 for packed 4-bit weights, which would be misleading.
    
    Args:
        repo_id: HuggingFace repository ID (typically from mlx-community)
        
    Returns:
        ModelMetadata with MLX-specific quantization handling
        
    Raises:
        RemoteInspectError: If inspection fails
    """
    # Fetch config.json
    config = _fetch_json_file(repo_id, "config.json")
    
    # Parse metadata using centralized parser with MLX type hint
    return MetadataParser.parse_transformers(config, repo_id, model_type_hint="mlx")


def inspect_model_remote(repo_id: str, engine: str) -> "ModelMetadata":
    """
    Inspect a model remotely using lightweight HTTP Range requests.
    
    This is the main entry point for remote inspection. It routes to the
    appropriate format-specific inspector based on the inference engine.
    
    Bandwidth usage:
    - GGUF: ~512KB (single range request)
    - Safetensors/vLLM: ~50KB (config.json + header)
    - MLX: ~5KB (config.json only)
    
    Args:
        repo_id: HuggingFace repository ID
        engine: Inference engine - "llama_cpp", "vllm", or "mlx_lm"
        
    Returns:
        ModelMetadata with architecture and quantization info
        
    Raises:
        RemoteInspectError: If inspection fails (caller should fall back to full download)
    """
    if not check_internet():
        raise RemoteInspectError("No internet connection")

    try:
        if engine == "llama_cpp":
            return inspect_gguf_remote(repo_id)
        elif engine == "vllm":
            return inspect_safetensors_remote(repo_id)
        elif engine == "mlx_lm":
            return inspect_mlx_remote(repo_id)
        else:
            raise RemoteInspectError(f"Unknown engine: {engine}")
    except RemoteInspectError:
        # Re-raise RemoteInspectError as-is
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise RemoteInspectError(f"Unexpected error during remote inspection: {e}")
