# NVIDIA Blackwell GPU Architecture Support

This folder contains the implementation plan and development roadmap for adding comprehensive NVIDIA Blackwell GPU architecture support to llama.cpp.

## Contents

- `IMPLEMENTATION_PLAN.md` - Detailed implementation plan with phases, timelines, and technical specifications
- `TECHNICAL_SPECS.md` - Comprehensive technical specifications with code examples
- `README.md` - This overview document

## Overview

NVIDIA Blackwell architecture introduces several key features that can significantly improve AI inference performance:

- **Thread Block Clusters**: Enable cooperation between multiple thread blocks with distributed shared memory
- **Enhanced L2 Cache**: 126 MB L2 cache (vs 40 MB on Ada Lovelace) with persistence control
- **HBM3/HBM3e Memory**: Higher bandwidth memory subsystem
- **Increased Shared Memory**: Up to 228 KB per SM (vs 164 KB on previous architectures)

## Current Status

### Build Foundation: ✅ **COMPLETE** (via [PR #13360](https://github.com/ggml-org/llama.cpp/pull/13360))

- ✅ **CUDA 12.8 Support**: Required toolkit for Blackwell compilation
- ✅ **sm120 Architecture Target**: Compute capability 12.0 (CORRECTED from initial 10.0 assumption)
- ✅ **Build System Integration**: CMake and CI/CD ready for Blackwell

### Architecture Implementation: 🔄 **IN PROGRESS**

- 🔄 **Thread Block Clusters**: Advanced multi-block cooperation
- 🔄 **Flash Attention Optimizations**: Blackwell-specific kernel variants
- 🔄 **L2 Cache Management**: 126 MB cache utilization strategies
- 🔄 **Memory Optimizations**: HBM3/HBM3e bandwidth improvements

## Architecture Specifications

### Compute Capability: **12.0** (CORRECTED)

**Critical Update**: Blackwell GPUs use compute capability **12.0** (sm120), not 10.0 as initially assumed.

**Supported Hardware**:
- NVIDIA B100/B200 (Data Center)
- NVIDIA RTX 50 Series (Consumer)
- NVIDIA RTX 6000 Ada Generation successor

### Key Features

| Feature | Ada Lovelace | Blackwell | Improvement |
|---------|-------------|-----------|------------|
| **Compute Capability** | 8.9 | **12.0** | New Architecture |
| **L2 Cache** | 40 MB | 126 MB | **3.1x larger** |
| **Shared Memory/SM** | 164 KB | 228 KB | **39% increase** |
| **Memory Type** | GDDR6X | HBM3/HBM3e | **Higher bandwidth** |
| **Thread Block Clusters** | No | Yes | **New feature** |
| **Max Cluster Size** | N/A | 8 (portable), 16 (non-portable) | **New capability** |

## Implementation Roadmap

### ⚡ **Accelerated Timeline** (22 weeks, reduced from 24)

**Phase 1: Foundation** ✅ **ACCELERATED** (Week 1-2)
- Build system complete via [PR #13360](https://github.com/ggml-org/llama.cpp/pull/13360)
- Architecture detection (compute capability 12.0)
- Device capability enumeration

**Phase 2: Thread Block Clusters** (Week 2-4) ⚡
- Cluster launch framework
- Distributed shared memory
- L2 cache management APIs

**Phase 3: Flash Attention Optimizations** (Week 4-9) ⚡
- Enhanced MMA kernels for Blackwell
- Cluster-based attention computation
- L2 cache-aware KV access patterns

**Phase 4-7: Advanced Features & Validation** (Week 10-22) ⚡
- Multi-head attention clustering
- General kernel optimizations
- Performance validation and tuning
- Documentation and integration

## Performance Targets

Based on architectural improvements:

- **Flash Attention**: 20-40% improvement over Ada Lovelace
- **Overall Inference**: 15-30% improvement in tokens/second
- **Memory Efficiency**: Better utilization of 126 MB L2 cache
- **Scalability**: Improved performance on larger context lengths (8K+ tokens)

## Development Status

### Foundation Layer: ✅ **COMPLETE**

```cpp
// Blackwell detection (compute capability 12.0)
#define GGML_CUDA_CC_BLACKWELL 1200
#define GGML_CUDA_CC_IS_BLACKWELL(cc) (cc >= 1200 && cc < 1300)
```

### Current Development Priorities

1. **Enhanced Device Information** (Week 1)
   - Cluster capability detection
   - L2 cache size enumeration
   - HBM3 bandwidth detection

2. **Thread Block Clusters Framework** (Week 2-3)
   - Cluster launch utilities
   - Occupancy calculation
   - Distributed shared memory management

3. **Flash Attention Blackwell Kernels** (Week 4-7)
   - Enhanced tile sizes (228 KB shared memory)
   - Cluster-aware attention computation
   - L2 cache persistence for KV data

## Hardware Requirements

### Minimum Requirements
- NVIDIA Blackwell GPU (B100/B200/RTX50 series)
- CUDA Toolkit 12.8+
- CUDA Driver supporting compute capability 12.0

### Recommended Development Environment
- Multiple Blackwell GPUs for cluster testing
- High-memory configuration for large context validation
- CUDA Toolkit 12.8 or newer

## Quick Start

### Building with Blackwell Support

```bash
# Ensure CUDA 12.8+ is installed
cmake -B build -DGGML_CUDA=ON
cmake --build build
```

The build system automatically detects Blackwell capabilities and includes sm120 architecture if CUDA 12.8+ is available.

### Runtime Detection

```cpp
// Check for Blackwell support
const int cc = ggml_cuda_info().devices[0].cc;
if (GGML_CUDA_CC_IS_BLACKWELL(cc)) {
    // Blackwell optimizations available
    printf("Blackwell GPU detected (compute capability %.1f)\n", cc / 100.0);
}
```

## Contributing

### Development Focus Areas

1. **Thread Block Clusters**: Implementing cooperative multi-block kernels
2. **Flash Attention**: Optimizing attention computation for Blackwell
3. **Memory Management**: Leveraging 126 MB L2 cache effectively
4. **Performance Analysis**: Benchmarking and validation frameworks

### Testing Requirements

- Access to Blackwell hardware for validation
- Performance regression testing on older architectures
- Memory usage analysis for large context scenarios
- Cluster efficiency measurement tools

## Documentation

- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**: Comprehensive 22-week development roadmap
- **[TECHNICAL_SPECS.md](TECHNICAL_SPECS.md)**: Detailed technical specifications with code examples

## References

- [PR #13360](https://github.com/ggml-org/llama.cpp/pull/13360): CUDA 12.8 + sm120 build foundation
- [NVIDIA Blackwell Architecture Whitepaper](https://developer.nvidia.com/blackwell-architecture)
- [CUDA Programming Guide - Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters)
- [CUDA Toolkit 12.8 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

---

**Status**: Foundation complete via [PR #13360](https://github.com/ggml-org/llama.cpp/pull/13360) ✅  
**Timeline**: 22 weeks (accelerated from 24) ⚡  
**Architecture**: Compute Capability 12.0 (corrected) 🎯 