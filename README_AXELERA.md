# Axelera Backend for llama.cpp

This document provides an overview of the Axelera inference accelerator backend integration for llama.cpp.

## Overview

The Axelera backend enables llama.cpp to offload computation to Axelera AI accelerator hardware. The implementation follows GGML's backend architecture and includes support for external compilation via Axelera's toolchain.

## Project Structure

```
llama.cpp/
├── CUSTOM_BACKEND_IMPLEMENTATION_PLAN.md  # Implementation roadmap
├── AXELERA_COMPILER_INTEGRATION.md        # Compiler integration guide
├── README_AXELERA.md                      # This file
├── ggml/
│   ├── include/
│   │   └── ggml-axelera.h                 # Public API header
│   └── src/
│       └── ggml-axelera/
│           ├── CMakeLists.txt             # Build configuration
│           ├── ggml-axelera.cpp           # Main implementation
│           └── ops/                       # Operation kernels (future)
└── ax-tests/
    ├── test-backend-axelera.cpp           # Basic test program
    ├── example-compiler-integration.cpp   # Compiler integration skeleton
    ├── CMakeLists.txt                     # Test build config
    └── build-and-run.sh                   # Convenience script
```

## Building

### Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler
- (Optional) Axelera SDK for hardware support

### Build with Axelera Backend

```bash
mkdir -p build
cd build
cmake .. -DGGML_AXELERA=ON
cmake --build . -j8
```

### Build with Axelera SDK

If you have the Axelera SDK installed:

```bash
export AXELERA_SDK_PATH=/path/to/axelera/sdk
cmake .. -DGGML_AXELERA=ON
cmake --build . -j8
```

## Testing

### Basic Backend Test

```bash
# Run the test to see graph debug output
GGML_LOG_LEVEL=info ./build/bin/test-backend-axelera
```

This test creates a simple matrix multiplication graph and shows:
- Operations in the computation graph
- Tensor shapes and data types
- Memory layout information
- Source tensor relationships

### Testing with llama-cli

```bash
# The backend will be initialized automatically
./build/bin/llama-cli --version

# You should see:
# Initializing Axelera backend registry with 1 devices
```

## Current Implementation Status

### ✅ Completed

- [x] Backend directory structure
- [x] Public API header (`ggml-axelera.h`)
- [x] Device interface implementation
- [x] Backend interface implementation
- [x] Buffer type implementation
- [x] Backend registration with GGML
- [x] CMake build integration
- [x] Graph debugging output
- [x] Test infrastructure

### 🚧 In Progress

- [ ] Graph compilation integration
- [ ] Operation kernels (matrix multiplication, etc.)
- [ ] Actual hardware execution
- [ ] Quantization support

### 📋 Planned

- [ ] Multi-device support
- [ ] Async operations
- [ ] Performance optimization
- [ ] Comprehensive testing

## Architecture

### Backend Registration

The Axelera backend registers with GGML on startup via `ggml_backend_axelera_reg()`. This makes it available to llama.cpp without code changes.

### Graph Compilation Flow

For accelerators with external compilers (like Axelera), the recommended flow is:

1. **Graph Plan Create** - Compile graph using Axelera toolchain (once)
2. **Graph Plan Compute** - Execute pre-compiled graph (many times)
3. **Graph Plan Free** - Clean up resources

See `AXELERA_COMPILER_INTEGRATION.md` for detailed implementation guide.

### Typical Execution Flow

```
llama.cpp creates GGML graph
          ↓
Backend checks if operations are supported
          ↓
graph_plan_create() - Compile with Axelera
          ↓
Check cache for compiled graph
          ↓
    ┌─────┴─────┐
    ↓           ↓
Cache hit   Cache miss
Load it     Compile it
    └─────┬─────┘
          ↓
graph_plan_compute() - Execute on Axelera hardware
          ↓
Return results to llama.cpp
```

## Integration Points

### 1. Backend Registration
**File**: `ggml/src/ggml-backend-reg.cpp`

The Axelera backend is registered conditionally:
```cpp
#ifdef GGML_USE_AXELERA
register_backend(ggml_backend_axelera_reg());
#endif
```

### 2. CMake Configuration
**Files**: `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`

Build option:
```cmake
option(GGML_AXELERA "ggml: enable Axelera backend" OFF)
```

### 3. Public API
**File**: `ggml/include/ggml-axelera.h`

Key functions:
- `ggml_backend_axelera_reg()` - Get backend registration
- `ggml_backend_axelera_init(device_id)` - Initialize backend
- `ggml_backend_axelera_buffer_type(device_id)` - Get buffer type
- `ggml_backend_axelera_get_device_count()` - Query devices

## Environment Variables

### Build Time
- `AXELERA_SDK_PATH` - Path to Axelera SDK installation

### Runtime
- `GGML_LOG_LEVEL` - Set to `info` or `debug` for verbose output
- `GGML_AXELERA_NDEV` - Number of Axelera devices to use (default: 1)
- `AXELERA_CACHE_DIR` - Where to cache compiled graphs (default: `/tmp/axelera_cache`)
- `AXELERA_COMPILER` - Path to Axelera compiler executable

## Adding Compiler Integration

See `AXELERA_COMPILER_INTEGRATION.md` for comprehensive guide on:

1. Implementing graph compilation
2. Serializing GGML graphs to Axelera format
3. Invoking the Axelera compiler
4. Caching compiled graphs
5. Executing on hardware

A complete skeleton implementation is provided in `ax-tests/example-compiler-integration.cpp`.

## Debugging

### Enable Verbose Logging

```bash
# Show backend initialization and graph info
GGML_LOG_LEVEL=info ./build/bin/llama-cli --version

# Show detailed debug information
GGML_LOG_LEVEL=debug ./build/bin/llama-cli --version
```

### Graph Debugging

The current implementation prints detailed graph information when `graph_compute()` is called:

```
=== Axelera Graph Compute ===
Graph has 1 nodes

Node 0: result_C
  Operation: MUL_MAT
  Type: f32
  Shape: [3, 2, 1, 1]
  Strides (bytes): [4, 12, 24, 24]
  Src[0]: matrix_A, type=f32, shape=[4, 3, 1, 1]
  Src[1]: matrix_B, type=f32, shape=[4, 2, 1, 1]
  Is empty: no
  Is contiguous: yes
```

This helps understand what operations and tensor formats llama.cpp sends to your backend.

## Next Steps

### 1. Implement Graph Compilation

Add the graph planning functions to `ggml-axelera.cpp`:
- `graph_plan_create()` - Compile graph
- `graph_plan_compute()` - Execute compiled graph
- `graph_plan_free()` - Clean up

See `AXELERA_COMPILER_INTEGRATION.md` for details.

### 2. Add Operation Support

Implement operation kernels starting with:
- Matrix multiplication (`GGML_OP_MUL_MAT`)
- Element-wise operations
- Activation functions

### 3. Integrate Axelera SDK

Replace placeholder code with actual SDK calls:
- Device initialization
- Memory management
- Graph execution

### 4. Test with Real Models

```bash
# Test with a small model
./build/bin/llama-cli -m models/tinyllama.gguf -p "Hello"
```

## Resources

- **Implementation Plan**: `CUSTOM_BACKEND_IMPLEMENTATION_PLAN.md`
- **Compiler Integration**: `AXELERA_COMPILER_INTEGRATION.md`
- **Example Code**: `ax-tests/example-compiler-integration.cpp`
- **Test Program**: `ax-tests/test-backend-axelera.cpp`

## Support

For questions about:
- GGML backend architecture: See llama.cpp documentation
- Axelera SDK: Refer to Axelera documentation
- This integration: Review the implementation plan and example code

## License

This backend implementation follows llama.cpp's license (MIT).
