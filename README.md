# ThunderLLAMA

**Apple Silicon Paged Attention for llama.cpp**

> Enabling efficient KV cache management on M1/M2/M3/M4 GPUs

---

## Overview

ThunderLLAMA is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with enhanced Paged Attention support for Apple Silicon GPUs. It addresses a critical issue in the upstream implementation where paged attention was effectively disabled even when the `LLAMA_PAGED_ATTENTION` flag was set.

## Architecture

### Block Pool Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Block Pool Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   llama_kv_cache                                                │
│       │                                                         │
│       ├── block_pool: llama_block_pool                          │
│       │       ├── k_pool: vector<ggml_tensor*> (per layer)     │
│       │       ├── v_pool: vector<ggml_tensor*> (per layer)     │
│       │       ├── block_table_gpu: ggml_tensor*                │
│       │       ├── block_size: uint32_t                         │
│       │       └── n_blocks: uint32_t                           │
│       │                                                         │
│       └── llama_graph                                           │
│               └── build_attn_mha()                              │
│                       └── ggml_flash_attn_ext_set_paged()       │
│                               ├── cur (attention node)          │
│                               ├── block_table (actual tensor)   │
│                               ├── use_paged = 1                 │
│                               └── strides...                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `llama_block_pool` | `llama-block-pool.cpp/h` | Manages paged KV cache blocks |
| `ggml_flash_attn_ext_set_paged` | `ggml.c` | API to enable paged mode |
| `llama_kv_cache` | `llama-kv-cache.cpp/h` | Integrates block pool |
| `llama_context` | `llama-context.cpp` | Context-level paged params |
| `llama_graph` | `llama-graph.cpp` | Passes block_table to FA |

### Data Flow

```
1. Context Initialization
   llama_context::init()
       → llama_memory_params.use_paged_attention = true
       → llama_kv_cache::init() with block_pool

2. Block Pool Creation
   llama_block_pool::init()
       → create k_pool[n_layers], v_pool[n_layers]
       → create block_table_gpu tensor
       → allocate GPU memory

3. Graph Building
   llama_graph::build_attn_mha()
       → get block_pool from kv_cache
       → ggml_flash_attn_ext_set_paged(cur, block_table, 1, ...)

4. Inference
   Metal Flash Attention kernel
       → uses block_table for paged access
       → computes attention with block strides
```

## Performance

### Benchmark Results

#### Apple M4 - TinyLlama 1.1B Q4_K_M

| Mode | Prompt (pp512) | Generation (tg128) |
|------|----------------|---------------------|
| Contiguous + FA | 2907 t/s | 239 t/s |
| **Paged + FA** | 2830 t/s (-2.7%) | **247 t/s (+3.5%)** |

#### Apple M4 - Qwen3-30B-A3B MoE (17.28 GiB)

| Mode | Prompt (pp512) | Generation (tg128) |
|------|----------------|---------------------|
| Contiguous + FA | 714 t/s | 74.4 t/s |
| **Paged + FA** | 702 t/s (-1.7%) | 73.5 t/s (-1.2%) |

### Key Findings

1. **Performance Parity**: Paged attention matches contiguous mode (<2% difference)
2. **Small Models**: Slight generation improvement (+3.5%) on small models
3. **Large Models**: Near-identical performance on 30B MoE models

### Advantages of Paged Attention

- **Dynamic KV Cache**: Allocate memory on-demand
- **Multi-Sequence**: Support concurrent requests efficiently
- **Memory Efficiency**: Better utilization for long contexts
- **Scalability**: Foundation for advanced serving features

## Build Instructions

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- CMake >= 3.16

### Build

```bash
# Clone
git clone https://github.com/lisihao/ThunderLLAMA.git
cd ThunderLLAMA

# Build
cmake -B build
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### Run

```bash
# With Paged Attention + Flash Attention (recommended)
LLAMA_PAGED_ATTENTION=1 ./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -fa 1 \
  -ngl 99 \
  -p "Hello, world!"
```

## Usage Examples

### CLI Inference

```bash
# Paged attention mode
LLAMA_PAGED_ATTENTION=1 ./build/bin/llama-cli \
  -m model.gguf -fa 1 -ngl 99 -c 4096 \
  -p "Explain quantum computing in simple terms"
```

### Benchmark

```bash
# Compare Contiguous vs Paged
echo "=== Contiguous ===" && ./build/bin/llama-bench -m model.gguf -fa 1 -p 512 -n 128
echo "=== Paged ===" && LLAMA_PAGED_ATTENTION=1 ./build/bin/llama-bench -m model.gguf -fa 1 -p 512 -n 128
```

### Server Mode

```bash
# Start server with paged attention
LLAMA_PAGED_ATTENTION=1 ./build/bin/llama-server \
  -m model.gguf -fa 1 --port 8080
```

## Technical Details

### The Bug We Fixed

**Before (Upstream)**:
```cpp
// llama-graph.cpp:1816-1820 (old code)
if (use_paged) {
    ggml_flash_attn_ext_set_paged(
        cur,
        nullptr,  // ← block_table was null
        0,        // ← use_paged was 0
        0, 0, 0, 0, 0
    );
}
```

**After (ThunderLLAMA)**:
```cpp
// llama-graph.cpp (fixed)
if (use_paged) {
    const auto * block_pool = kv_ctx->get_block_pool();
    if (block_pool && !block_pool->k_pool.empty()) {
        ggml_tensor * block_table = kv_ctx->get_block_table();
        ggml_flash_attn_ext_set_paged(
            cur,
            block_table,  // ← actual tensor
            1,            // ← use_paged = 1
            block_pool->block_size,
            block_stride_k, block_stride_v,
            token_stride_k, token_stride_v
        );
    }
}
```

### k_pool/v_pool Design

**Before**: Single tensor per cache (wrong for multi-layer)
```cpp
ggml_tensor * k_pool;  // One tensor for all layers
ggml_tensor * v_pool;
```

**After**: Per-layer vectors
```cpp
std::vector<ggml_tensor *> k_pool;  // One tensor per layer
std::vector<ggml_tensor *> v_pool;
```

### Memory Layout

```
Block Pool Memory Layout:
─────────────────────────────────────────────────────────
│ Layer 0  │ Layer 1  │ ... │ Layer N-1 │
─────────────────────────────────────────────────────────
     │           │               │
     ▼           ▼               ▼
  k_pool[0]  k_pool[1]      k_pool[N-1]
  v_pool[0]  v_pool[1]      v_pool[N-1]

Block Table:
─────────────────────────────────────────────────────────
│ Block 0 │ Block 1 │ ... │ Block M-1 │
─────────────────────────────────────────────────────────
     │
     └── Maps logical → physical blocks
```

## Roadmap

### Phase 1: Core Implementation ✅
- [x] Block pool implementation
- [x] Paged attention API in ggml
- [x] Integration with llama_kv_cache
- [x] Flash attention support
- [x] Performance validation

### Phase 2: Optimization (Planned)
- [ ] Memory pre-allocation strategies
- [ ] Block defragmentation
- [ ] Multi-sequence scheduling
- [ ] Cache eviction policies

### Phase 3: Advanced Features (Future)
- [ ] vLLM-style continuous batching
- [ ] Prefix caching
- [ ] Speculative decoding integration
- [ ] Distributed inference support

### Phase 4: Production Readiness (Future)
- [ ] Comprehensive test suite
- [ ] Documentation and examples
- [ ] Performance profiling tools
- [ ] Integration with llama-server

## Contributing

Contributions are welcome! Please see:

1. **Issues**: Report bugs or request features
2. **Pull Requests**: Submit improvements
3. **Discussions**: Share ideas and use cases

### Development Setup

```bash
# Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

# Run tests
./build-debug/bin/test-backend-ops
```

## Related Projects

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Upstream project
- [vLLM](https://github.com/vllm-project/vllm) - Paged attention paper
- [ggml](https://github.com/ggml-org/ggml) - Tensor library

## References

1. [Paged Attention Paper](https://arxiv.org/abs/2309.06180) - vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
2. [Flash Attention](https://arxiv.org/abs/2205.14135) - Fast and Memory-Efficient Exact Attention
3. [Metal Performance Shaders](https://developer.apple.com/metal/) - Apple's GPU framework

## License

Same as llama.cpp (MIT License)

## Acknowledgments

- llama.cpp team for the excellent codebase
- vLLM team for the paged attention concept
- Apple for Metal framework and developer tools

---

**ThunderLLAMA** - Making Paged Attention roar on Apple Silicon 🍎⚡
