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

## The Right KPIs for Paged Attention

> **Paged Attention 的价值不是让单次推理更快，而是让系统更稳定、更可靠**

vLLM 的 PagedAttention 把它当成"KV cache 的 OS paging"，核心收益是：

### 正确的 KPI

| KPI | 说明 | Paged Advantage |
|-----|------|-----------------|
| **CAPACITY** | 同内存预算下的上下文长度 | 更长 context / 更多并发序列 |
| **OPERABILITY** | P95/P99 延迟抖动 | 更稳定，无 defrag 飙升 |
| **RELIABILITY** | 长时间运行稳定性 | **结构性移除 defrag 问题** |

### llama.cpp 的 defrag 问题

llama.cpp 有真实案例：**defrag 触发后输出乱码直到重启**

```
Contiguous KV Cache:
─────────────────────────────────────────────────────
时间 → 内存碎片积累 → 触发 defrag → 输出乱码 → 重启

Paged KV Cache:
─────────────────────────────────────────────────────
Block Pool → 按需分配 → 无碎片 → 无 defrag → 稳定运行
```

**Paged Attention 的价值 = 把 defrag 从系统里"结构性移除"**

### Performance Parity (基线验证)

虽然单次速度不是 KPI，但我们验证了性能对等：

| Model | Mode | pp512 | tg128 |
|-------|------|-------|-------|
| TinyLlama 1.1B | Contiguous | 2907 t/s | 239 t/s |
| TinyLlama 1.1B | **Paged** | 2830 t/s | 247 t/s |
| Qwen3-30B MoE | Contiguous | 714 t/s | 74.4 t/s |
| Qwen3-30B MoE | **Paged** | 702 t/s | 73.5 t/s |

**结论**: 性能差异 <3%，Paged 模式不牺牲单次性能

### 实测数据 (Qwen3-30B-A3B MoE, Apple M4)

| KPI | Contiguous | Paged | 结论 |
|-----|-----------|-------|------|
| **CAPACITY (pp8192)** | 465 t/s | **497 t/s (+6.8%)** | ✅ Paged 更快 |
| **OPERABILITY (jitter)** | 6.0% | **0.7%** | ✅ Paged 更稳定 8x |

### Benchmark Scripts

我们提供了正确 KPI 的测试脚本：

```bash
# 测试 CAPACITY / OPERABILITY / RELIABILITY
./benchmarks/paged-attention-kpi-v2.sh /path/to/model.gguf
```

### When to Use Paged Attention

| 场景 | 推荐 |
|------|------|
| 单用户短对话 | Contiguous (更简单) |
| 长上下文 (>16K) | **Paged** (内存效率) |
| 多并发请求 | **Paged** (序列隔离) |
| 生产环境服务 | **Paged** (稳定性) |
| 长时间运行 | **Paged** (无 defrag 风险) |

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

### Phase 2A: Prefix Caching (Copy-on-Write) 🎯 NEXT
- [ ] Block sharing for common prefixes
- [ ] Copy-on-Write for diverging suffix
- [ ] Agent/Tool-chain 场景 TTFT 大幅下降
- [ ] 参考: vLLM 技术报告 "共享前缀直接复用物理块"

### Phase 2B: Continuous Batching + 多序列调度
- [ ] 并发序列调度器
- [ ] 碎片/defrag/分配抖动控制
- [ ] 吞吐量优势体现

### Phase 2C: Chunked Prefill
- [ ] 长 prompt 切块处理
- [ ] 减少 prefill 带宽峰值
- [ ] P99 延迟改善

### Phase 3: Advanced Features (Future)
- [ ] Speculative decoding integration
- [ ] Distributed inference support
- [ ] Cache eviction policies

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
