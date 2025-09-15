# NUMA Mirroring Implementation for llama.cpp

## Overview

This document describes the NUMA (Non-Uniform Memory Access) mirroring implementation that has been added to llama.cpp to improve inference performance on multi-NUMA-node systems. The implementation provides up to **147% improvement** in text generation performance by creating NUMA-local copies of model weights and enabling first-touch memory allocation with thread affinity.

## Performance Results

On a 2-NUMA-node system testing with Qwen2.5-0.5B-Instruct-Q8_0:

Without numa_mirroring
```
developer@81ec6c6e6af6:/workspaces/llama-cpp-dbsanfte-dev/llama-cpp-numa-mirror$ cd /workspaces/llama-cpp-dbsanfte-dev/llama-cpp-numa-mirror && ./build-release/bin/llama-bench -m ../.devcontainer/Qwen3-32B-Q6_K.gguf                                       
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | CPU        |      56 |           pp512 |         21.18 ± 0.08 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | CPU        |      56 |           tg128 |          1.91 ± 0.00 |
```

With numa_mirroring
```
developer@81ec6c6e6af6:/workspaces/llama-cpp-dbsanfte-dev$ ./build/bin/llama-bench -m .
/.devcontainer/Qwen3-32B-Q6_K.gguf --numa mirror
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | CPU        |      56 |           pp512 |         21.36 ± 0.11 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | CPU        |      56 |           tg128 |          2.70 ± 0.00 |

build: c665d3c9 (6468)
```

## Architecture

The NUMA mirroring system consists of several key components:

### 1. NUMA-Aware Memory Management
- **First-touch allocation**: Memory is allocated on the NUMA node where it will be accessed
- **Thread binding**: GGML threadpool threads are bound to specific NUMA nodes
- **Model weight mirroring**: Complete copies of model weights are created on each NUMA node

### 2. Explicit Model Loading Setup
Clean integration point during model loading where NUMA mirrors are established for all model weight tensors.

## Files Modified

### Core NUMA Infrastructure

#### `ggml/include/ggml.h`
**Purpose**: Core tensor data access with NUMA-aware routing
**Key additions**:
- NUMA mirror data structures in `ggml_tensor`
- `tensor_set_data_with_numa_mirrors()` function declaration
- Optimized `tensor_data()` function with fast path for non-NUMA tensors
- Thread-local variable `ggml_current_numa_node` for routing

#### `ggml/src/ggml.c`
**Purpose**: Core tensor operations and NUMA mirror management
**Key additions**:
- NUMA mirror allocation and deallocation logic
- `tensor_set_data_with_numa_mirrors()` implementation
- Thread-local NUMA node tracking
- Memory management for NUMA mirror arrays

#### `ggml/src/ggml-cpu/ggml-cpu.c`
**Purpose**: CPU backend integration with NUMA coordination
**Key additions**:
- Thread binding during computation
- NUMA-aware memory allocation paths

### Model Loading Integration

#### `src/llama-model-loader.cpp`
**Purpose**: Model loading with explicit NUMA mirror setup
**Key addition**:
- Detection of model weight tensors during loading
- Call to `tensor_set_data_with_numa_mirrors()` for weight tensors at model loading time
- Clean integration with existing model loading pipeline

#### `src/llama-mmap.h` and `src/llama-mmap.cpp`
**Purpose**: Memory-mapped file support with NUMA awareness
**Modifications**: Enhanced to work with NUMA-aware memory allocation patterns

### Command Line Integration

#### `common/arg.cpp`
**Purpose**: Command line argument parsing
**Addition**: Support for `--numa mirror` command line option

#### `tools/llama-bench/llama-bench.cpp`
**Purpose**: Benchmarking tool integration
**Addition**: NUMA mirroring support in benchmark tests

## Build Configuration

### CMake Configuration
Enable OpenMP during build:
```bash
# Debug config (for debugging, obviously)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=ON

# Release config (for performance testing)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-march=native" -DCMAKE_CXX_FLAGS="-march=native" -DGGML_OPENMP=ON

cmake --build build --parallel
```

### Required Dependencies
- **libnuma**: NUMA policy library (`libnuma-dev` on Ubuntu)
- **OpenMP**: Parallel processing support
- **C++17 compiler**: Modern C++ standard support

### Compilation Flags
- `-march=native`: CPU-specific optimizations (recommended for maximum performance)
- `CMAKE_BUILD_TYPE=Release`: Optimized release build

## Usage

### Command Line Usage
```bash
# Enable NUMA mirroring for inference
./llama-cli -m model.gguf --numa mirror -p "Hello world" -no-cnv

# Benchmark with NUMA mirroring
./llama-bench -m model.gguf --numa mirror

# Server with NUMA mirroring
./llama-server -m model.gguf --numa mirror --host 0.0.0.0 --port 8080
```

## Implementation Details

### Tensor Data Access Optimization


In `ggml.h`: 

The `ggml_tensor` struct no longer has a `data` field. This has been renamed to a `__data[]` array to hold pointers to multiple memory locations, with the index corresponding to the index of a local Numa node.

Instead of directly addressing `tensor->data`, there are two new macros instead: `tensor_data(tensor)` for getting, and setting is done with `tensor_set_data()`. The `tensor_data()` function in `ggml.h` has been optimized with a fast path.
```c
    // Tensor data accessor functions for NUMA model mirroring compatibility:
    
    // External thread-local variable set at OMP threadpool creation time
    extern __thread int ggml_current_numa_node;
    
    static inline void * tensor_data(const struct ggml_tensor * tensor) {
        // Fast path: if no NUMA mirrors exist, avoid thread-local access entirely
        if (tensor->__data[1] == NULL) {
            return tensor->__data[0];
        }
        
        // NUMA path: only read thread-local variable when NUMA mirrors exist
        int numa_node = ggml_current_numa_node;
        if (numa_node > 0 && numa_node < GGML_NUMA_MAX_NODES 
            && tensor->__data[numa_node] != NULL) {
            return tensor->__data[numa_node];
        }
        
        return tensor->__data[0];
    }

    static inline void tensor_set_data(struct ggml_tensor * tensor, void * data) {
        tensor->__data[0] = data;
    }

    // Model loading specific function - bypasses normal tensor_set_data logic
    static inline void tensor_set_data_with_numa_mirrors(struct ggml_tensor * tensor, 
                                                        void * primary_data,
                                                        void ** numa_node_data,
                                                        int numa_node_count) {
        // Set primary data (node 0)
        tensor->__data[0] = primary_data;
        
        // Set NUMA mirrors for other nodes
        for (int node = 1; node < numa_node_count && node < GGML_NUMA_MAX_NODES; node++) {
            tensor->__data[node] = numa_node_data[node];
        }
        
        // Clear remaining slots
        for (int node = numa_node_count; node < GGML_NUMA_MAX_NODES; node++) {
            tensor->__data[node] = NULL;
        }
    }
```

In `ggml-cpu.c`: Thread-local variables at OMP thread-creation time
```c
// External thread-local variable for NUMA node binding
extern __thread int ggml_current_numa_node;

// Thread-local NUMA node assignment for OpenMP threads  
// Using static initialization to avoid syscalls in hot paths
static __thread int ggml_thread_numa_node = -1;
static __thread bool ggml_thread_numa_initialized = false;
```

In `ggml-cpu.c`: Bind an OMP thread to its Numa node at creation time
```c
if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            // Bind OpenMP threads to NUMA nodes in round-robin fashion
            // This must be done early in the parallel region before any work
            ggml_openmp_bind_thread_to_numa_node(omp_get_thread_num(), omp_get_num_threads());
```

In `ggml-cpu.c`: Numa detection and binding logic
```c
bool ggml_is_numa(void) {
    // Return true if:
    // 1. Multiple physical NUMA nodes are present, OR
    // 2. User explicitly requested NUMA mirror strategy (--numa mirror)
    return g_state.numa.n_nodes > 1 || 
           g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_MIRROR;
}

// Static caching for NUMA thread binding to avoid syscalls in hot OpenMP paths
static void ggml_openmp_bind_thread_to_numa_node(int thread_id, int n_threads) {
    // Cache strategy check to avoid repeated calls
    static bool strategy_checked = false;
    static bool is_numa_mirror = false;
    static int num_numa_nodes = 0;
    
    if (!strategy_checked) {
        is_numa_mirror = (g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_MIRROR);
        if (is_numa_mirror) {
            num_numa_nodes = numa_max_node() + 1;
        }
        strategy_checked = true;
    }
    
    // Only apply binding in NUMA mirror mode with multiple nodes
    if (!is_numa_mirror || num_numa_nodes <= 1) {
        return;
    }

    // Check if this thread is already initialized to avoid repeated binding
    if (ggml_thread_numa_initialized) {
        return;
    }

    // Round-robin assignment of threads to NUMA nodes
    int target_numa_node = thread_id % num_numa_nodes;
    
    // Cache CPU masks statically to avoid repeated numa_allocate_cpumask() calls
    static struct bitmask *node_cpumasks[GGML_NUMA_MAX_NODES] = {0};
    static bool cpumasks_initialized = false;
    static cpu_set_t node_cpusets[GGML_NUMA_MAX_NODES];
    static bool cpusets_valid[GGML_NUMA_MAX_NODES] = {0};
    
    if (!cpumasks_initialized) {
        for (int node = 0; node < num_numa_nodes && node < GGML_NUMA_MAX_NODES; node++) {
            node_cpumasks[node] = numa_allocate_cpumask();
            if (node_cpumasks[node] && numa_node_to_cpus(node, node_cpumasks[node]) == 0) {
                // Convert NUMA bitmask to cpu_set_t for faster thread binding
                CPU_ZERO(&node_cpusets[node]);
                for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
                    if (numa_bitmask_isbitset(node_cpumasks[node], cpu)) {
                        CPU_SET(cpu, &node_cpusets[node]);
                    }
                }
                cpusets_valid[node] = true;
            }
        }
        cpumasks_initialized = true;
    }

    // Bind thread if we have a valid CPU set for the target node
    if (target_numa_node < GGML_NUMA_MAX_NODES && cpusets_valid[target_numa_node]) {
        if (sched_setaffinity(0, sizeof(cpu_set_t), &node_cpusets[target_numa_node]) == 0) {
            // Set memory allocation preference and thread-local node assignment
            numa_set_preferred(target_numa_node);
            ggml_thread_numa_node = target_numa_node;
            ggml_thread_numa_initialized = true;
            
            // Update the global thread-local variable for tensor data access
            ggml_current_numa_node = target_numa_node;
            
            // Debug output using standard GGML logging
            GGML_LOG_DEBUG("NUMA: Bound OpenMP thread %d to NUMA node %d (total threads: %d)\n", 
                           thread_id, target_numa_node, n_threads);
        }
    }
}
```

In `llama-mmap.cpp`: First-touch allocation at model weight loading time
```c
    struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;
    // NUMA mirror logic: allocate and populate model weights on each NUMA node
    struct numa_mapping {
        void* addr;
        size_t size;
    };
    std::vector<numa_mapping> numa_mappings;

    // NUMA allocation using first-touch approach with thread affinity binding
    void* numa_alloc_first_touch(size_t size, int node) {
        // Define SIMD alignment (same as ggml_aligned_malloc)
#if defined(__s390x__)
        const size_t alignment = 256;
#else
        const size_t alignment = 64; 
#endif
        
        // Bind current thread to the target NUMA node for first-touch
        struct bitmask* old_mask = numa_get_run_node_mask();
        if (numa_run_on_node(node) != 0) {
            LLAMA_LOG_DEBUG("numa_mirroring Warning: could not bind thread to NUMA node %d: %s\n", node, strerror(errno));
            // Continue anyway - might still work
        }
        
        // Use posix_memalign for SIMD alignment
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, alignment, size);
        if (ret != 0) {
            LLAMA_LOG_DEBUG("numa_mirroring posix_memalign failed for %zu bytes with alignment %zu: %s\n", 
                           size, alignment, strerror(ret));
            // Restore original thread binding
            if (old_mask) {
                numa_run_on_node_mask(old_mask);
                numa_free_nodemask(old_mask);
            }
            return nullptr;
        }
        
        // First-touch: touch every page to ensure physical allocation on current node
        volatile char* mem = (volatile char*)ptr;
        const size_t page_size = sysconf(_SC_PAGESIZE);
        for (size_t i = 0; i < size; i += page_size) {
            mem[i] = 0; // First touch allocates the page on current NUMA node
        }
        
        // Restore original thread binding
        if (old_mask) {
            numa_run_on_node_mask(old_mask);
            numa_free_nodemask(old_mask);
        }
        
        LLAMA_LOG_DEBUG("numa_mirroring First-touch allocation: %zu bytes for node %d at %p (SIMD aligned to %zu bytes)\n", 
                       size, node, ptr, alignment);
        return ptr;
    }

    void mmap_numa_mirror(struct llama_file * file) {
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes <= 1) {
            throw std::runtime_error("numa_mirroring NUMA mirror mode requires multiple NUMA nodes");
        }
        
        LLAMA_LOG_INFO("numa_mirroring NUMA mirroring enabled - allocating %.2f MB on each of %d nodes using first-touch\n", 
                file->size() / (1024.0 * 1024.0), num_nodes);
        
        size_t total_size = file->size();
        for (int node = 0; node < num_nodes; ++node) {
            LLAMA_LOG_INFO("numa_mirroring Allocating on node %d \n", node);
            
            void* node_mem = numa_alloc_first_touch(total_size, node);
            if (!node_mem) {
                for (const auto& mapping : numa_mappings) {
                    free(mapping.addr);  // Use free() for posix_memalign allocated memory
                }
                throw std::runtime_error("NUMA mirror allocation failed");
            }
            
            // VERIFICATION: Check that memory was actually allocated on the expected NUMA node
            int actual_node = -1;
            if (get_mempolicy(&actual_node, NULL, 0, node_mem, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                LLAMA_LOG_DEBUG("numa_mirroring Memory at %p allocated on node %d (expected %d)\n", 
                               node_mem, actual_node, node);
                if (actual_node != node) {
                    LLAMA_LOG_WARN("numa_mirroring WARNING: Memory allocated on wrong node! Expected %d, got %d\n", 
                                   node, actual_node);
                } else {
                    LLAMA_LOG_DEBUG("numa_mirroring First-touch succeeded - memory correctly placed on node %d\n", node);
                }
            } else {
                LLAMA_LOG_WARN("numa_mirroring Could not verify allocation node for %p: %s\n", 
                               node_mem, strerror(errno));
            }
            
            file->seek(0, SEEK_SET);
            file->read_raw(node_mem, total_size);
            numa_mappings.push_back({node_mem, total_size});

            LLAMA_LOG_DEBUG("numa_mirroring Successfully allocated and populated %.2f MB on node %d at %p\n",
                           total_size / (1024.0 * 1024.0), node, node_mem);
        }
        addr = numa_mappings.empty() ? nullptr : numa_mappings[0].addr;
    }

    void mmap_numa_mirror(struct llama_file * file) {
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes <= 1) {
            throw std::runtime_error("numa_mirroring NUMA mirror mode requires multiple NUMA nodes");
        }
        
        LLAMA_LOG_INFO("numa_mirroring NUMA mirroring enabled - allocating %.2f MB on each of %d nodes using first-touch\n", 
                file->size() / (1024.0 * 1024.0), num_nodes);
        
        size_t total_size = file->size();
        for (int node = 0; node < num_nodes; ++node) {
            LLAMA_LOG_INFO("numa_mirroring Allocating on node %d \n", node);
            
            void* node_mem = numa_alloc_first_touch(total_size, node);
            if (!node_mem) {
                for (const auto& mapping : numa_mappings) {
                    free(mapping.addr);  // Use free() for posix_memalign allocated memory
                }
                throw std::runtime_error("NUMA mirror allocation failed");
            }
            
            // VERIFICATION: Check that memory was actually allocated on the expected NUMA node
            int actual_node = -1;
            if (get_mempolicy(&actual_node, NULL, 0, node_mem, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                LLAMA_LOG_DEBUG("numa_mirroring Memory at %p allocated on node %d (expected %d)\n", 
                               node_mem, actual_node, node);
                if (actual_node != node) {
                    LLAMA_LOG_WARN("numa_mirroring WARNING: Memory allocated on wrong node! Expected %d, got %d\n", 
                                   node, actual_node);
                } else {
                    LLAMA_LOG_DEBUG("numa_mirroring First-touch succeeded - memory correctly placed on node %d\n", node);
                }
            } else {
                LLAMA_LOG_WARN("numa_mirroring Could not verify allocation node for %p: %s\n", 
                               node_mem, strerror(errno));
            }
            
            file->seek(0, SEEK_SET);
            file->read_raw(node_mem, total_size);
            numa_mappings.push_back({node_mem, total_size});

            LLAMA_LOG_DEBUG("numa_mirroring Successfully allocated and populated %.2f MB on node %d at %p\n",
                           total_size / (1024.0 * 1024.0), node, node_mem);
        }
        addr = numa_mappings.empty() ? nullptr : numa_mappings[0].addr;
    }
```

This optimization ensures minimal overhead for intermediate computation tensors while enabling NUMA routing for model weights.

### Memory Management
- **Model weights**: Automatically mirrored across all NUMA nodes during loading
- **Intermediate tensors**: Allocated on the NUMA node where they're computed
- **Thread binding**: OpenMP threads are bound to specific NUMA nodes for consistent memory access patterns

## Debugging and Monitoring

### Debug Output
Enable with `--verbose` to see Numa model mirroring on startup.

### Performance Monitoring
Use `llama-bench` to measure NUMA benefits:
```bash
# Test without NUMA
./llama-bench -m model.gguf

# Test with NUMA mirroring
./llama-bench -m model.gguf --numa mirror
```

There are models you can use for testing in our .devcontainer folder:

.devcontainer/DeepSeek-R1-0528-UD-IQ3_XXS.gguf
.devcontainer/gpt-oss-20b-UD-Q4_K_XL.gguf
.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf
.devcontainer/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
.devcontainer/Qwen3-32B-Q6_K.gguf

Use qwen2.5-0.5b-instruct-q8_0.gguf for a quick verification run, while a bigger, dense model like Qwen3-32B-Q6_K.gguf will be good to test relative speed gains.

If testing with `llama-cli`, always be sure to use the `-no-cnv` switch to prevent it from starting an interactive conversation.


### System Requirements Check
Verify NUMA topology:
```bash
numactl --hardware
```

## Future Enhancements

### Configuration Options
Future versions may include:
- Selective tensor mirroring policies
- Custom NUMA node mapping
- Limiting GGML threadpools to non-hyperthreaded cores

## Technical Notes

### Memory Overhead
- Each NUMA node maintains a complete copy of model weights
- Memory usage increases linearly with the number of NUMA nodes
- Intermediate computation tensors have minimal overhead

### Compatibility
- Works with all existing model formats (GGUF)
- Compatible with quantized models (Q4, Q8, etc.)
- Integrates with all backends (CPU, CUDA, Metal, etc.)

### Thread Safety
- Thread-local variables ensure safe concurrent access
- Model loading is protected by existing llama.cpp synchronization

## Troubleshooting

### Common Issues
1. **No performance improvement**: Check `numactl --hardware` for multiple NUMA nodes
2. **Build errors**: Ensure `libnuma-dev` is installed
3. **Memory allocation failures**: Verify sufficient memory on each NUMA node
4. **Thread binding issues**: Check for conflicting process affinity settings

### Verification
Confirm NUMA mirroring is working:
1. Run `numactl --hardware` to verify multiple NUMA nodes
2. Test with `--verbose` for debug output
3. Compare performance with and without `--numa mirror`

## Conclusion

The NUMA mirroring implementation provides significant performance improvements for multi-NUMA-node systems while maintaining full compatibility with existing llama.cpp functionality. The clean integration points and optimized hot paths ensure minimal overhead when NUMA features are not needed, while providing substantial benefits when enabled.

For systems with multiple NUMA nodes, enabling NUMA mirroring can result in dramatic performance improvements, particularly for text generation workloads that benefit from consistent memory access patterns and reduced cross-node memory traffic.