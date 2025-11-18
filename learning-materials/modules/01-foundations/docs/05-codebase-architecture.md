# Codebase Architecture

**Learning Module**: Module 1 - Foundations
**Estimated Reading Time**: 14 minutes
**Prerequisites**: C/C++ programming knowledge, understanding of llama.cpp basics
**Related Content**:
- [What is llama.cpp?](./01-what-is-llama-cpp.md)
- [Building from Source](./03-building-from-source.md)
- [Lab 4: Codebase Exploration](../../labs/lab-04/)

---

## Repository Overview

llama.cpp is organized into a clean, modular structure that separates concerns and enables easy navigation and contribution.

### High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│              llama.cpp Architecture                   │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │         Application Layer                   │     │
│  │  (llama-cli, llama-server, tools)          │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │         llama Library (libllama)            │     │
│  │  - Model loading & management               │     │
│  │  - Tokenization                             │     │
│  │  - Inference coordination                   │     │
│  │  - Sampling                                 │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │         ggml Tensor Library                 │     │
│  │  - Tensor operations                        │     │
│  │  - Computation graphs                       │     │
│  │  - Memory management                        │     │
│  │  - Backend abstraction                      │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                   │
│  ┌────────────────▼───────────────────────────┐     │
│  │         Backend Implementations             │     │
│  │  CPU │ CUDA │ Metal │ Vulkan │ HIP │ ...  │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Root Directory Layout

```
llama.cpp/
├── .github/              # GitHub workflows, issue templates
├── .devops/              # Docker files, CI/CD configurations
├── common/               # Shared utilities across tools
├── docs/                 # Official documentation
├── examples/             # Example applications
├── ggml/                 # Core tensor library (submodule/integrated)
│   ├── include/          # ggml header files
│   └── src/              # ggml implementation
├── gguf-py/              # Python utilities for GGUF
├── grammars/             # Grammar files for constrained generation
├── include/              # Public API headers
│   ├── llama.h           # Main llama.cpp API
│   └── llama-cpp.h       # C++ wrapper API
├── src/                  # Core llama.cpp implementation
│   ├── llama.cpp         # Main implementation
│   ├── llama-vocab.cpp   # Tokenization
│   ├── llama-model.cpp   # Model loading/management
│   ├── llama-quant.cpp   # Quantization
│   └── llama-sampling.cpp # Sampling algorithms
├── tools/                # Command-line tools and utilities
│   ├── main/             # llama-cli
│   ├── server/           # llama-server (HTTP API)
│   ├── quantize/         # Model quantization tool
│   ├── perplexity/       # Perplexity measurement
│   └── ...
├── tests/                # Unit and integration tests
├── scripts/              # Build and utility scripts
├── convert_hf_to_gguf.py # Model conversion script
└── CMakeLists.txt        # Build configuration
```

### Key Directories Explained

#### `/include/` - Public API Headers

The public interface to llama.cpp:

```
include/
├── llama.h          # C API (primary interface)
│                    # - Model loading
│                    # - Inference functions
│                    # - Tokenization API
│                    # - Sampling functions
│
└── llama-cpp.h      # C++ wrapper (convenience)
                     # - RAII wrappers
                     # - STL integration
                     # - Exception safety
```

#### `/src/` - Core Implementation

The heart of llama.cpp:

```
src/
├── llama.cpp             # Main inference engine
│                         # - Model initialization
│                         # - Forward pass orchestration
│                         # - KV cache management
│
├── llama-vocab.cpp       # Tokenization
│                         # - BPE, SPM, WPM tokenizers
│                         # - Encoding/decoding
│
├── llama-model.cpp       # Model management
│                         # - GGUF loading
│                         # - Architecture detection
│                         # - Layer initialization
│
├── llama-quant.cpp       # Quantization
│                         # - Quantization algorithms
│                         # - Dequantization
│
├── llama-sampling.cpp    # Sampling strategies
│                         # - Temperature
│                         # - Top-k, top-p
│                         # - Repetition penalty
│
└── unicode*.cpp          # Unicode handling
                          # - UTF-8 operations
                          # - Normalization
```

#### `/ggml/` - Tensor Library

The low-level computational engine:

```
ggml/
├── include/
│   ├── ggml.h            # Core tensor API
│   ├── ggml-backend.h    # Backend abstraction
│   ├── ggml-alloc.h      # Memory allocator
│   ├── gguf.h            # GGUF format
│   ├── ggml-cuda.h       # CUDA backend
│   ├── ggml-metal.h      # Metal backend
│   ├── ggml-vulkan.h     # Vulkan backend
│   └── ...               # Other backends
│
└── src/
    ├── ggml.c            # Core implementation
    ├── ggml-backend.c    # Backend system
    ├── ggml-alloc.c      # Memory management
    ├── gguf.cpp          # GGUF I/O
    ├── ggml-quants.c     # Quantization kernels
    ├── ggml-cuda/        # CUDA implementation
    ├── ggml-metal/       # Metal implementation
    ├── ggml-sycl/        # SYCL implementation
    └── ...               # Other backends
```

#### `/tools/` - Applications

User-facing tools built on llama.cpp:

```
tools/
├── main/                 # llama-cli (interactive)
│   ├── main.cpp          # Entry point
│   └── README.md         # Usage documentation
│
├── server/               # llama-server (HTTP API)
│   ├── server.cpp        # HTTP server implementation
│   ├── webui/            # Web interface
│   └── README.md         # API documentation
│
├── quantize/             # Model quantization
│   └── quantize.cpp      # Quantization tool
│
├── perplexity/           # Model evaluation
│   └── perplexity.cpp    # Perplexity calculation
│
├── llama-bench/          # Benchmarking
├── imatrix/              # Importance matrix
└── ...                   # Other utilities
```

#### `/common/` - Shared Utilities

Reusable components across tools:

```
common/
├── common.h              # Common definitions
├── common.cpp            # Shared utilities
├── arg.h/arg.cpp         # Argument parsing
├── sampling.h/cpp        # Sampling logic
├── console.h/cpp         # Console utilities
├── log.h/log.cpp         # Logging system
└── ...
```

---

## Core Components

### 1. llama.h - Public API

The primary interface for applications:

```c
// Key structures
struct llama_model;          // Loaded model
struct llama_context;        // Inference context
struct llama_batch;          // Input batch

// Model loading
llama_model* llama_load_model_from_file(
    const char* path,
    struct llama_model_params params
);

// Context creation
llama_context* llama_new_context_with_model(
    llama_model* model,
    struct llama_context_params params
);

// Inference
int llama_decode(
    llama_context* ctx,
    llama_batch batch
);

// Get output
float* llama_get_logits(llama_context* ctx);
float* llama_get_logits_ith(llama_context* ctx, int32_t i);

// Tokenization
int llama_tokenize(
    const llama_model* model,
    const char* text,
    int text_len,
    llama_token* tokens,
    int n_max_tokens,
    bool add_bos,
    bool special
);

// Cleanup
void llama_free(llama_context* ctx);
void llama_free_model(llama_model* model);
```

### 2. ggml - Tensor Operations

The computational backbone:

```c
// Tensor structure
struct ggml_tensor {
    enum ggml_type type;     // Data type (F32, Q4_K, etc.)
    int64_t ne[4];           // Number of elements per dimension
    size_t nb[4];            // Stride in bytes
    void* data;              // Actual data pointer
    char name[64];           // Tensor name
    // ... more fields
};

// Create tensors
struct ggml_tensor* ggml_new_tensor_2d(
    struct ggml_context* ctx,
    enum ggml_type type,
    int64_t ne0,
    int64_t ne1
);

// Operations
struct ggml_tensor* ggml_mul_mat(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    struct ggml_tensor* b
);

struct ggml_tensor* ggml_add(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    struct ggml_tensor* b
);

struct ggml_tensor* ggml_rope(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    int n_past,
    int n_dims,
    int mode,
    int n_ctx
);
```

### 3. Backend System

Hardware abstraction layer:

```c
// Backend interface
struct ggml_backend_i {
    const char* (*get_name)(ggml_backend_t backend);

    void (*free)(ggml_backend_t backend);

    ggml_backend_buffer_type_t (*get_default_buffer_type)(ggml_backend_t backend);

    void (*set_tensor_async)(ggml_backend_t backend, struct ggml_tensor* tensor,
                             const void* data, size_t offset, size_t size);

    void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor* tensor,
                             void* data, size_t offset, size_t size);

    void (*synchronize)(ggml_backend_t backend);

    ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph* cgraph);

    // ... more methods
};

// Example backends
ggml_backend_t ggml_backend_cpu_init(void);
ggml_backend_t ggml_backend_cuda_init(int device);
ggml_backend_t ggml_backend_metal_init(void);
```

---

## Data Flow

### Complete Inference Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Inference Data Flow                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. INPUT                                                │
│     User prompt: "Hello, world!"                         │
│              ↓                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llama_tokenize()                                  │  │
│  │ - Loads tokenizer from model                      │  │
│  │ - Encodes text to tokens                          │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│     Token IDs: [156, 1245, 235, ...]                    │
│              ↓                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llama_batch_init()                                │  │
│  │ - Creates batch structure                         │  │
│  │ - Fills token, pos, seq_id arrays                │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│  2. PROCESSING                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llama_decode()                                    │  │
│  │  ├─ Build computation graph (ggml)                │  │
│  │  ├─ Embed tokens                                  │  │
│  │  ├─ For each layer:                               │  │
│  │  │   ├─ Attention (with KV cache)                 │  │
│  │  │   ├─ Feed-forward                              │  │
│  │  │   └─ Residual + norm                           │  │
│  │  ├─ Final norm                                    │  │
│  │  └─ LM head (project to vocab)                    │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ggml_backend_graph_compute()                      │  │
│  │ - Schedule operations                             │  │
│  │ - Execute on backend (CPU/GPU)                    │  │
│  │ - Return results                                  │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│  3. OUTPUT                                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llama_get_logits()                                │  │
│  │ - Get output logits (vocab_size floats)           │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│     Logits: [0.2, -1.5, 2.3, ..., 0.8]                  │
│              ↓                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llama_sampler_sample()                            │  │
│  │ - Apply temperature, top-k, top-p                 │  │
│  │ - Sample next token                               │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│     Next token: 235                                      │
│              ↓                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llama_token_to_piece()                            │  │
│  │ - Decode token to text                            │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     ↓                                    │
│  4. RESULT                                               │
│     Output text: " I"                                    │
│                                                          │
│  Repeat steps 2-4 for each generated token              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Key Files and Their Purposes

### Essential Headers

| File | Purpose | Key Contents |
|------|---------|--------------|
| `include/llama.h` | Public API | Model/context management, inference, tokenization |
| `ggml/include/ggml.h` | Tensor ops | Tensor structures, operations, graph API |
| `ggml/include/ggml-backend.h` | Backend abstraction | Backend interface, buffer management |
| `ggml/include/gguf.h` | GGUF format | File I/O, metadata access |
| `common/common.h` | Utilities | Shared helper functions |

### Core Implementation Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/llama.cpp` | Main inference | `llama_load_model_from_file()`, `llama_decode()` |
| `src/llama-vocab.cpp` | Tokenization | `llama_tokenize()`, `llama_detokenize()` |
| `src/llama-model.cpp` | Model loading | GGUF parsing, layer initialization |
| `src/llama-sampling.cpp` | Sampling | Temperature, top-k, top-p, penalties |
| `ggml/src/ggml.c` | Tensor ops | Operation implementations, graph building |
| `ggml/src/ggml-backend.c` | Backend system | Backend registration, scheduling |
| `ggml/src/gguf.cpp` | GGUF I/O | Reading/writing GGUF files |

### Application Entry Points

| File | Purpose | Description |
|------|---------|-------------|
| `tools/main/main.cpp` | llama-cli | Interactive CLI interface |
| `tools/server/server.cpp` | llama-server | HTTP API server |
| `tools/quantize/quantize.cpp` | Model quantization | Convert model formats |
| `tools/perplexity/perplexity.cpp` | Quality measurement | Compute perplexity scores |
| `tools/llama-bench/llama-bench.cpp` | Benchmarking | Performance testing |

---

## Where to Find What

### Common Tasks and File Locations

#### Understanding Model Loading

```
Start here:
1. src/llama.cpp → llama_load_model_from_file()
2. src/llama-model.cpp → llama_model_load_internal()
3. ggml/src/gguf.cpp → gguf_init_from_file()

Key concepts:
- GGUF file parsing
- Tensor mapping
- Architecture detection
```

#### Understanding Inference

```
Start here:
1. src/llama.cpp → llama_decode()
2. src/llama.cpp → llama_build_graph()
3. ggml/src/ggml.c → ggml_graph_compute()
4. ggml/src/ggml-backend.c → ggml_backend_graph_compute()

Key concepts:
- Graph construction
- KV cache management
- Backend execution
```

#### Understanding Tokenization

```
Start here:
1. src/llama-vocab.cpp → llama_tokenize()
2. src/llama-vocab.cpp → llama_vocab (struct)

Key concepts:
- BPE/SPM/WPM tokenizers
- Special token handling
- Unicode normalization
```

#### Understanding Quantization

```
Start here:
1. src/llama-quant.cpp → quantization types
2. ggml/src/ggml-quants.c → dequantize_row_*()
3. tools/quantize/quantize.cpp → main()

Key concepts:
- Quantization formats
- Block structures
- Dequantization
```

#### Understanding Sampling

```
Start here:
1. src/llama-sampling.cpp → llama_sampler_*()
2. common/sampling.h → sampling parameters

Key concepts:
- Temperature scaling
- Top-k/top-p filtering
- Repetition penalties
```

#### Understanding Backends

```
Start here:
1. ggml/include/ggml-backend.h → Backend interface
2. ggml/src/ggml-cpu.c → CPU implementation
3. ggml/src/ggml-cuda/ → CUDA implementation
4. ggml/src/ggml-metal.m → Metal implementation

Key concepts:
- Backend abstraction
- Kernel implementations
- Memory management
```

---

## Navigation Tips

### Finding Implementation of a Feature

1. **Start with the API**: Look in `include/llama.h` for the public function
2. **Follow to implementation**: Check `src/llama.cpp` for the definition
3. **Dive into ggml**: If it's computation-heavy, check `ggml/src/`
4. **Check backends**: For hardware-specific code, check `ggml/src/ggml-*/`

### Understanding a Tool

1. **Read the tool's README**: Each tool has documentation in its directory
2. **Check main()**: Look at the entry point in the tool's .cpp file
3. **Trace API calls**: Follow calls to llama.h functions
4. **Check common/**: Many tools share utilities from common/

### Debugging Issues

1. **Enable logging**: Set `LLAMA_LOG_LEVEL` environment variable
2. **Check error messages**: Look in `src/llama.cpp` for error handling
3. **Verify tensor shapes**: Use `ggml_tensor` print functions
4. **Profile performance**: Use built-in timer functions

---

## Build System

### CMake Structure

```
CMakeLists.txt (root)
├─ Project configuration
├─ Compiler flags
├─ Option definitions (GGML_CUDA, etc.)
├─ Include directories
├─ ggml library target
├─ llama library target
└─ Tool targets (llama-cli, llama-server, etc.)

Key variables:
- GGML_CUDA: Enable CUDA
- GGML_METAL: Enable Metal
- GGML_NATIVE: Enable native optimizations
- BUILD_SHARED_LIBS: Build as shared library
```

### Adding a New Tool

```cmake
# In CMakeLists.txt

# 1. Create executable
add_executable(my-tool tools/my-tool/my-tool.cpp)

# 2. Link against llama library
target_link_libraries(my-tool PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})

# 3. Link against common utilities
target_link_libraries(my-tool PRIVATE common)

# 4. Install target
install(TARGETS my-tool RUNTIME DESTINATION bin)
```

---

## Code Conventions

### Naming Conventions

```c
// Public API (C-style)
llama_*            // Functions
llama_*_t          // Types
LLAMA_*            // Constants

// Internal (C++)
llama::*           // Classes/namespaces

// ggml (C-style)
ggml_*             // Functions
ggml_*_t           // Types
GGML_*             // Constants
```

### File Organization

```c
// Header file structure
#pragma once / #ifndef guards
#include dependencies
#define constants
struct declarations
function declarations

// Implementation file structure
#include corresponding header
#include dependencies
static helper functions
public function implementations
```

### Memory Management

```c
// Rule 1: Who allocates, deallocates
// If llama_* allocates, llama_free_* deallocates

// Rule 2: Use RAII in C++
// Wrap resources in classes with destructors

// Rule 3: Document ownership
// Clearly state who owns memory in comments
```

---

## Extension Points

### Adding Support for New Model Architecture

1. **Update tokenizer** (if needed): `src/llama-vocab.cpp`
2. **Add architecture enum**: `src/llama.cpp`
3. **Implement graph builder**: `llama_build_graph()` in `src/llama.cpp`
4. **Update model loader**: `src/llama-model.cpp`
5. **Test thoroughly**: Add tests in `tests/`

### Adding a New Backend

1. **Create header**: `ggml/include/ggml-mybackend.h`
2. **Implement interface**: `ggml/src/ggml-mybackend.cpp`
3. **Register backend**: In `ggml_backend_register_mybackend()`
4. **Add CMake option**: `option(GGML_MYBACKEND ...)`
5. **Document usage**: Add to `docs/backend/MYBACKEND.md`

### Adding a New Quantization Format

1. **Define type**: Add to `ggml_type` enum in `ggml/include/ggml.h`
2. **Implement quantize**: In `ggml/src/ggml-quants.c`
3. **Implement dequantize**: In `ggml/src/ggml-quants.c`
4. **Add kernel support**: Update backend implementations
5. **Update tools**: Modify `tools/quantize/quantize.cpp`

---

## Interview Questions

**Q: "Explain the relationship between llama.cpp and ggml."**

**A**: Discuss:
- ggml is the low-level tensor operation library
- llama.cpp builds on ggml for LLM-specific functionality
- ggml provides: tensors, operations, backends
- llama.cpp provides: model loading, inference, tokenization, sampling
- ggml is reusable for other ML applications

**Q: "How does llama.cpp achieve cross-platform GPU support?"**

**A**: Cover:
- Backend abstraction layer (ggml-backend)
- Multiple backend implementations (CUDA, Metal, Vulkan, etc.)
- Common interface for all backends
- Runtime backend selection
- Compile-time backend enabling

**Q: "Walk me through what happens when you call llama_decode()."**

**A**: Explain:
1. Build computation graph (llama_build_graph)
2. Allocate buffers for intermediate tensors
3. Schedule operations on backend
4. Execute graph (backend-specific)
5. Copy results back (logits)
6. Update KV cache

---

## Further Reading

### Official Documentation
- [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
- [GGML Repository](https://github.com/ggml-org/ggml)
- [How to Add a Model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)

### Related Content
- [What is llama.cpp?](./01-what-is-llama-cpp.md)
- [Building from Source](./03-building-from-source.md)
- [Inference Fundamentals](./04-inference-fundamentals.md)
- [Lab 4: Codebase Exploration](../../labs/lab-04/)

### Code Reading
- Start with: `examples/simple/simple.cpp` - Minimal example
- Then read: `tools/main/main.cpp` - Full-featured CLI
- Deep dive: `src/llama.cpp` - Core implementation

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Feedback**: [Submit feedback](../../../feedback/)
