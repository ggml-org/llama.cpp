#!/usr/bin/env python3
"""
Model Architecture Inspector

This script inspects and visualizes LLM architecture from GGUF files.
Demonstrates:
- Reading GGUF metadata
- Extracting architecture parameters
- Calculating model dimensions
- Estimating memory requirements

Usage:
    python architecture_inspector.py model.gguf
"""

import sys
import struct
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelArchitecture:
    """Model architecture parameters"""
    name: str
    architecture: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    n_ff: int
    n_vocab: int
    n_ctx_train: int
    rope_freq_base: float

    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.n_embd // self.n_head

    @property
    def gqa_ratio(self) -> int:
        """Queries per KV head (GQA)"""
        return self.n_head // self.n_head_kv

    @property
    def param_count_approx(self) -> int:
        """Approximate parameter count"""
        # Embedding
        emb_params = self.n_vocab * self.n_embd

        # Per-layer parameters
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.n_embd * self.n_embd

        # FFN: Gate, Up, Down projections
        ffn_params = 3 * self.n_embd * self.n_ff

        # Layer norm (RMS): gamma only
        norm_params = 2 * self.n_embd

        layer_params = attn_params + ffn_params + norm_params
        total_layer_params = layer_params * self.n_layer

        # Output projection
        output_params = self.n_vocab * self.n_embd

        return emb_params + total_layer_params + output_params

    def estimate_kv_cache_size(self, n_ctx: int, bytes_per_elem: int = 2) -> int:
        """Estimate KV cache size in bytes"""
        # 2 (K and V) × n_layer × n_ctx × n_head_kv × head_dim × bytes
        return 2 * self.n_layer * n_ctx * self.n_head_kv * self.head_dim * bytes_per_elem

    def format_size(self, size_bytes: int) -> str:
        """Format size in human-readable form"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


class GGUFReader:
    """Simple GGUF metadata reader"""

    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 3

    # Value types
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

    def __init__(self, path: str):
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}

    def read(self):
        """Read GGUF metadata"""
        with open(self.path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: magic={hex(magic)}")

            version = struct.unpack('<I', f.read(4))[0]
            if version != self.GGUF_VERSION:
                raise ValueError(f"Unsupported GGUF version: {version}")

            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

    def _read_string(self, f) -> str:
        """Read a GGUF string"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, value_type: int) -> Any:
        """Read a value based on type"""
        if value_type == self.GGUF_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == self.GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == self.GGUF_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_BOOL:
            return struct.unpack('<?', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_STRING:
            return self._read_string(f)
        elif value_type == self.GGUF_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == self.GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == self.GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == self.GGUF_TYPE_ARRAY:
            # Read array type and length
            array_type = struct.unpack('<I', f.read(4))[0]
            array_length = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_length)]
        else:
            raise ValueError(f"Unknown value type: {value_type}")


def extract_architecture(metadata: Dict[str, Any]) -> ModelArchitecture:
    """Extract architecture from metadata"""
    arch = metadata.get('general.architecture', 'unknown')
    name = metadata.get('general.name', 'Unknown')

    # Get architecture-specific keys
    def get_key(key: str, default=None):
        return metadata.get(f'{arch}.{key}', default)

    return ModelArchitecture(
        name=name,
        architecture=arch,
        n_layer=get_key('block_count', 0),
        n_embd=get_key('embedding_length', 0),
        n_head=get_key('attention.head_count', 0),
        n_head_kv=get_key('attention.head_count_kv', get_key('attention.head_count', 0)),
        n_ff=get_key('feed_forward_length', 0),
        n_vocab=metadata.get(f'{arch}.vocab_size', 0),
        n_ctx_train=get_key('context_length', 0),
        rope_freq_base=get_key('rope.freq_base', 10000.0),
    )


def print_architecture_summary(arch: ModelArchitecture):
    """Print comprehensive architecture summary"""
    print("=" * 70)
    print(f"Model Architecture Summary: {arch.name}")
    print("=" * 70)

    print("\n📊 Core Dimensions:")
    print(f"  Architecture Type:    {arch.architecture}")
    print(f"  Number of Layers:     {arch.n_layer}")
    print(f"  Hidden Dimension:     {arch.n_embd:,}")
    print(f"  Vocabulary Size:      {arch.n_vocab:,}")
    print(f"  Training Context:     {arch.n_ctx_train:,} tokens")

    print("\n🎯 Attention Configuration:")
    print(f"  Query Heads:          {arch.n_head}")
    print(f"  KV Heads:             {arch.n_head_kv}")
    print(f"  Head Dimension:       {arch.head_dim}")
    print(f"  GQA Ratio:            {arch.gqa_ratio}:1 (Q:KV)")

    if arch.gqa_ratio == 1:
        attn_type = "Multi-Head Attention (MHA)"
    elif arch.n_head_kv == 1:
        attn_type = "Multi-Query Attention (MQA)"
    else:
        attn_type = f"Grouped-Query Attention (GQA, {arch.gqa_ratio} groups)"
    print(f"  Attention Type:       {attn_type}")

    print("\n🔧 Feed-Forward Network:")
    print(f"  FFN Hidden Dimension: {arch.n_ff:,}")
    print(f"  FFN Expansion Ratio:  {arch.n_ff / arch.n_embd:.1f}x")

    print("\n🧮 RoPE Configuration:")
    print(f"  Base Frequency:       {arch.rope_freq_base:,.0f}")

    print("\n💾 Memory Estimates:")
    params = arch.param_count_approx
    params_b = params / 1_000_000_000
    print(f"  Parameters:           ~{params_b:.1f}B ({params:,})")

    # Model size estimates (different quantizations)
    print(f"\n  Model Size (FP16):    {arch.format_size(params * 2)}")
    print(f"  Model Size (Q8_0):    {arch.format_size(params * 1)}")
    print(f"  Model Size (Q4_0):    {arch.format_size(params * 0.5)}")

    # KV cache sizes
    print(f"\n  KV Cache @ 2K ctx:    {arch.format_size(arch.estimate_kv_cache_size(2048))}")
    print(f"  KV Cache @ 4K ctx:    {arch.format_size(arch.estimate_kv_cache_size(4096))}")
    print(f"  KV Cache @ 8K ctx:    {arch.format_size(arch.estimate_kv_cache_size(8192))}")
    print(f"  KV Cache @ 32K ctx:   {arch.format_size(arch.estimate_kv_cache_size(32768))}")

    print("\n" + "=" * 70)


def compare_architectures(archs: list[ModelArchitecture]):
    """Compare multiple architectures"""
    print("\n📊 Architecture Comparison")
    print("=" * 100)

    # Header
    print(f"{'Model':<20} {'Layers':<8} {'Hidden':<10} {'Heads':<12} {'FFN':<10} {'Params':<12} {'KV Cache @4K':<15}")
    print("-" * 100)

    for arch in archs:
        params_b = arch.param_count_approx / 1_000_000_000
        kv_cache = arch.format_size(arch.estimate_kv_cache_size(4096))
        heads_str = f"{arch.n_head}/{arch.n_head_kv}"

        print(f"{arch.name:<20} {arch.n_layer:<8} {arch.n_embd:<10,} {heads_str:<12} {arch.n_ff:<10,} {params_b:<12.1f}B {kv_cache:<15}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python architecture_inspector.py <model.gguf> [model2.gguf ...]")
        sys.exit(1)

    architectures = []

    for model_path in sys.argv[1:]:
        print(f"\n🔍 Reading {model_path}...")

        try:
            reader = GGUFReader(model_path)
            reader.read()

            arch = extract_architecture(reader.metadata)
            architectures.append(arch)

            print_architecture_summary(arch)

        except Exception as e:
            print(f"❌ Error reading {model_path}: {e}")
            continue

    # If multiple models, compare them
    if len(architectures) > 1:
        compare_architectures(architectures)


if __name__ == "__main__":
    main()
