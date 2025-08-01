# NUMA Improvements and Development Container

This document describes the NUMA-aware improvements made to llama.cpp and how to use the development container to build and test them.

## 🚀 Quick Start with Dev Container

### Prerequisites
- **VS Code** with the "Dev Containers" extension
- **Docker Desktop** running on your system

### Setup Steps
1. **Open the project**: Open the llama.cpp folder in VS Code
2. **Start container**: Press `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
3. **Wait for build**: The container will build automatically (first time takes a few minutes)
4. **Build project**: Run `./build-numa.sh` or use VS Code tasks

### First Build
```bash
# Quick build and test
./build-numa.sh

# Or manual steps
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/bin/llama-server --cpu-topology
```

## 🧠 NUMA Improvements Overview

### Problem Solved
- **NUMA memory allocation broke** when users specified `--threads` argument
- **Hyperthreading assumptions were wrong** - code skipped hyperthreaded cores incorrectly
- **No user control** over hyperthreading and efficiency core usage

### Solutions Implemented

#### 1. Fixed NUMA Thread Assignment
**Before**: Threads were assigned to NUMA nodes using simple modulo arithmetic (`thread_id % num_numa_nodes`)
**After**: Proper CPU topology detection and NUMA-aware thread distribution

```cpp
// Old (broken) approach:
int numa_node = thread_id % numa_num_configured_nodes();

// New (correct) approach:
int numa_node = get_numa_node_for_cpu(assigned_cpu_id);
```

#### 2. Improved CPU Topology Detection
**Before**: Naive assumptions about CPU ID pairing for hyperthreading
**After**: Reading actual Linux `/sys/devices/system/cpu/` topology information

```cpp
// New CPU topology detection
struct cpu_topology_info {
    int total_logical_cpus;
    int total_physical_cores;
    std::vector<std::vector<int>> core_siblings; // Actual HT groups
    std::vector<int> performance_cpus;          // P-cores
    std::vector<int> efficiency_cpus;           // E-cores
};
```

#### 3. Configurable Hyperthreading Usage
**Before**: Hyperthreading disabled by default, no user control
**After**: Hyperthreading enabled by default, user can disable with `--cpu-no-hyperthreading`

```bash
# Default behavior (hyperthreading enabled)
./llama-server --model model.gguf

# Disable hyperthreading
# Test without hyperthreading
./llama-server --model model.gguf --cpu-no-hyperthreading

# Test with efficiency cores disabled  
./llama-server --model model.gguf --cpu-no-efficiency-cores
```

#### 4. Environment Variable Support
```bash
# Use environment variables
LLAMA_CPU_NO_HYPERTHREADING=1 ./llama-server --model model.gguf

# Disable efficiency cores via environment
LLAMA_CPU_NO_EFFICIENCY_CORES=1 ./llama-server --model model.gguf
```

## 🔧 Technical Details

### NUMA Memory Allocation
The NUMA mirroring system (`GGML_NUMA_MIRROR`) duplicates model weights across NUMA nodes for optimal memory access:

```cpp
// Each thread accesses memory from its local NUMA node
void * numa_ptr = numa_alloc_onnode(size, ggml_current_numa_node);
```

### CPU Affinity Assignment
Threads are now assigned to specific CPUs based on topology:

```cpp
static int ggml_graph_compute_thread(void * data) {
    // ... existing code ...
    
    // Assign thread to specific CPU for NUMA locality
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(assigned_cpu_id, &mask);
    pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    
    // ... computation code ...
}
```

### Intel Hybrid CPU Support
Detects P-cores vs E-cores using CPUID instructions:

```cpp
static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}
```

## 🧪 Testing the Improvements

### 1. CPU Topology Information
```bash
# View detailed CPU topology
./build/bin/llama-server --cpu-topology

# Check NUMA hardware
numactl --hardware

# View system CPU info
lscpu
```

### 2. Performance Testing
```bash
# Benchmark with default settings
./build/bin/llama-bench -m model.gguf

# Benchmark without hyperthreading
./build/bin/llama-bench -m model.gguf --cpu-no-hyperthreading

# Test different thread counts
for threads in 4 8 16; do
    echo "Testing with $threads threads:"
    ./build/bin/llama-bench -m model.gguf --threads $threads
done
```

### 3. NUMA Binding Tests
```bash
# Run on specific NUMA node
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server --model model.gguf

# Check memory allocation patterns
numastat -p $(pgrep llama-server)
```

### 4. Memory Access Patterns
```bash
# Monitor NUMA memory access with perf
perf stat -e node-loads,node-stores,node-load-misses,node-store-misses \
    ./build/bin/llama-bench -m model.gguf

# Use hwloc to visualize topology
hwloc-info --topology --of console
```

## 📊 Expected Performance Improvements

### NUMA Systems
- **Better memory locality**: Reduced cross-NUMA memory access
- **Consistent performance**: No degradation when using `--threads`
- **Scalability**: Better performance on multi-socket systems

### Hyperthreading
- **Default enabled**: Better utilization of available cores
- **User control**: Can disable if workload doesn't benefit
- **Hybrid CPU support**: Proper handling of P-cores vs E-cores

### Benchmarking Results
Test on your system and compare:

```bash
# Before improvements (simulation)
LLAMA_CPU_NO_HYPERTHREADING=1 ./llama-bench --threads $(nproc --ignore=1)

# After improvements (default)
./llama-bench --threads $(nproc)
```

## 🐛 Troubleshooting

### Container Issues
```bash
# Rebuild container
# In VS Code: Ctrl+Shift+P → "Dev Containers: Rebuild Container"

# Check container status
docker ps
docker logs <container-id>
```

### Build Issues
```bash
# Clean build
rm -rf build
./build-numa.sh

# Verbose build
cmake --build build --verbose

# Check dependencies
apt list --installed | grep -E "(numa|hwloc|cmake)"
```

### Runtime Issues
```bash
# Check NUMA availability
numactl --show

# Test basic functionality
./build/bin/llama-server --help | grep -E "(hyperthreading|efficiency|topology)"

# Debug CPU assignment
strace -e sched_setaffinity ./build/bin/llama-server --cpu-topology
```

### Performance Issues
```bash
# Check CPU frequency scaling
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Monitor during execution
htop -H  # Show threads
numastat -p $(pgrep llama)  # NUMA stats
```

## 🔬 Development Notes

### Code Organization
- `common/common.cpp`: CPU topology detection, NUMA functions
- `common/common.h`: CPU parameter structures
- `common/arg.cpp`: Command-line argument parsing
- `ggml-cpu.c`: Thread computation and NUMA assignment (in ggml submodule)

### Key Functions
- `detect_cpu_topology()`: Reads Linux CPU topology
- `cpu_count_math_cpus()`: Counts available CPUs with options
- `cpu_print_topology_info()`: Debug information display
- `ggml_graph_compute_thread()`: Thread computation with NUMA awareness

### Testing Guidelines
1. **Always test on actual NUMA hardware** for real performance validation
2. **Compare before/after** using environment variables to simulate old behavior
3. **Test various thread counts** to ensure no regression
4. **Monitor memory access patterns** with NUMA tools
5. **Validate on different CPU architectures** (Intel, AMD, hybrid)

This development container provides everything needed to build, test, and validate these NUMA improvements in a consistent Ubuntu 24.04 environment.
