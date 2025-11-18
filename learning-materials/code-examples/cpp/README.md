# llama.cpp C++ API Examples

This directory contains practical C++ examples demonstrating how to use the llama.cpp API for various tasks. Each example is self-contained, well-documented, and follows modern C++ best practices.

## Examples Overview

### 1. Simple Inference (`01-simple-inference.cpp`)

**Purpose**: Basic text generation using the llama.cpp API.

**What you'll learn**:
- Loading GGUF models
- Initializing contexts
- Tokenizing text
- Generating text with greedy sampling
- Proper resource management with RAII
- Performance tracking

**Key concepts**:
- Model loading and parameters
- Context creation and sizing
- Batch processing
- Sampler chains
- Token decoding loop

### 2. Embeddings (`02-embeddings.cpp`)

**Purpose**: Generate text embeddings for RAG (Retrieval-Augmented Generation) and semantic search applications.

**What you'll learn**:
- Configuring embedding mode
- Processing multiple texts in batches
- Different pooling strategies (mean, CLS, last)
- Normalizing embeddings
- Calculating similarity metrics
- Performance optimization for batch processing

**Key concepts**:
- Embedding model configuration
- Sequence-based batch processing
- Pooling types and their uses
- Vector normalization
- Cosine similarity

### 3. Custom Sampling (`03-custom-sampling.cpp`)

**Purpose**: Advanced sampling techniques and custom sampler implementation.

**What you'll learn**:
- Building custom sampler chains
- Combining multiple sampling strategies
- Implementing custom samplers from scratch
- Temperature, top-k, top-p, min-p sampling
- Repetition penalties
- Comparing sampling strategies

**Key concepts**:
- Sampler interface (`llama_sampler_i`)
- Sampler chains
- Token probability manipulation
- Creative vs. conservative sampling
- Custom sampling logic

## Prerequisites

### System Requirements

- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.14 or later
- llama.cpp built and available
- (Optional) CUDA/ROCm/Metal for GPU acceleration

### Models

You'll need GGUF format models. Download from:
- [Hugging Face](https://huggingface.co/models?library=gguf)
- [TheBloke's models](https://huggingface.co/TheBloke)

For embeddings, use specialized embedding models like:
- `nomic-embed-text-v1.5.Q4_K_M.gguf`
- `all-minilm-l6-v2.gguf`

For inference/sampling, use any LLM model like:
- `llama-2-7b.Q4_K_M.gguf`
- `mistral-7b-instruct-v0.2.Q4_K_M.gguf`

## Building the Examples

### Method 1: Standalone Build

If you have llama.cpp already built:

```bash
# Navigate to this directory
cd learning-materials/code-examples/cpp

# Create build directory
mkdir build
cd build

# Configure (point to your llama.cpp installation)
cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp

# Build
cmake --build . -j

# The executables will be in the build directory
ls -l 01-simple-inference 02-embeddings 03-custom-sampling
```

### Method 2: Build from llama.cpp Root

If you're building as part of the llama.cpp project:

```bash
# From llama.cpp root directory
mkdir build
cd build

# Configure
cmake .. -DBUILD_SHARED_LIBS=OFF

# Build main library
cmake --build . -j

# Build learning examples
cd ../learning-materials/code-examples/cpp
mkdir build
cd build
cmake ..
cmake --build . -j
```

### Build Options

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build (optimized)
cmake .. -DCMAKE_BUILD_TYPE=Release

# With GPU support (if llama.cpp was built with CUDA)
# Just link against the GPU-enabled llama library
```

## Running the Examples

### 1. Simple Inference

```bash
# Basic usage
./01-simple-inference -m /path/to/model.gguf

# Custom prompt
./01-simple-inference -m model.gguf -p "The future of AI is"

# Generate more tokens
./01-simple-inference -m model.gguf -p "Once upon a time" -n 100

# Use GPU acceleration
./01-simple-inference -m model.gguf -ngl 35

# Show help
./01-simple-inference --help
```

**Options**:
- `-m, --model`: Path to GGUF model (required)
- `-p, --prompt`: Text prompt (default: "Hello, my name is")
- `-n, --n-predict`: Number of tokens to generate (default: 32)
- `-ngl, --n-gpu-layers`: GPU layers to offload (default: 99)

### 2. Embeddings

```bash
# Generate embeddings for single text
./02-embeddings -m embedding-model.gguf -t "Hello world"

# Multiple texts
./02-embeddings -m embedding-model.gguf -t "Hello" "World" "AI is amazing"

# Without normalization
./02-embeddings -m embedding-model.gguf -t "Text" --no-normalize

# With GPU acceleration
./02-embeddings -m embedding-model.gguf -t "Text 1" "Text 2" -ngl 35

# Show help
./02-embeddings --help
```

**Options**:
- `-m, --model`: Path to embedding model (required)
- `-t, --texts`: Texts to embed (required, space-separated)
- `--normalize`: Normalize embeddings (default: true)
- `--no-normalize`: Don't normalize embeddings
- `-ngl, --n-gpu-layers`: GPU layers to offload (default: 99)

**Output**: The program displays:
- First 10 dimensions of each embedding
- L2 norm of embeddings
- Pairwise cosine similarity between texts
- Performance statistics

### 3. Custom Sampling

```bash
# Compare all sampling strategies
./03-custom-sampling -m model.gguf -p "Once upon a time"

# Use only conservative sampling
./03-custom-sampling -m model.gguf -p "The capital of France" -s conservative

# Use creative sampling for stories
./03-custom-sampling -m model.gguf -p "In a galaxy far away" -s creative -n 100

# Use custom sampler implementation
./03-custom-sampling -m model.gguf -p "Hello" -s custom

# Different random seed
./03-custom-sampling -m model.gguf -p "Test" --seed 12345

# Show help
./03-custom-sampling --help
```

**Options**:
- `-m, --model`: Path to GGUF model (required)
- `-p, --prompt`: Text prompt (default: "Once upon a time")
- `-n, --n-predict`: Tokens to generate (default: 50)
- `-s, --strategy`: Sampling strategy (default: all)
  - `conservative`: Low temperature, narrow focus, factual
  - `creative`: High temperature, wide selection, imaginative
  - `custom`: Uses custom sampler implementation
  - `all`: Run all strategies for comparison
- `--seed`: Random seed (default: 42)
- `-ngl, --n-gpu-layers`: GPU layers to offload (default: 99)

**Output**: For each strategy, displays:
- Generated text
- Sampling performance statistics

## Code Structure and Best Practices

### Memory Management

All examples use RAII (Resource Acquisition Is Initialization) with custom deleters:

```cpp
struct ModelDeleter {
    void operator()(llama_model* model) const {
        if (model) llama_model_free(model);
    }
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
```

This ensures automatic cleanup even if exceptions occur.

### Error Handling

Each example includes comprehensive error checking:

```cpp
if (!model) {
    std::cerr << "Error: Failed to load model" << std::endl;
    return 1;
}
```

### Performance Tracking

All examples enable performance tracking:

```cpp
ctx_params.no_perf = false;
// ... later ...
llama_perf_context_print(ctx.get());
```

## Common Patterns

### Loading a Model

```cpp
ggml_backend_load_all();

llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 35;  // GPU offloading

llama_model* model = llama_model_load_from_file(path.c_str(), model_params);
```

### Creating a Context

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;      // Context size
ctx_params.n_batch = 512;     // Batch size
ctx_params.embeddings = true; // For embeddings

llama_context* ctx = llama_init_from_model(model, ctx_params);
```

### Tokenization

```cpp
const llama_vocab* vocab = llama_model_get_vocab(model);

// Get token count
const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(),
                                      nullptr, 0, true, true);

// Tokenize
std::vector<llama_token> tokens(n_tokens);
llama_tokenize(vocab, text.c_str(), text.size(),
               tokens.data(), tokens.size(), true, true);
```

### Batch Processing

```cpp
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

if (llama_decode(ctx, batch)) {
    // Handle error
}
```

### Sampling

```cpp
// Create sampler chain
auto params = llama_sampler_chain_default_params();
llama_sampler* sampler = llama_sampler_chain_init(params);

// Add samplers
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

// Sample
llama_token token = llama_sampler_sample(sampler, ctx, -1);
```

## Troubleshooting

### "Unable to load model"

- Check the model path is correct
- Ensure the model is in GGUF format
- Verify the model file isn't corrupted
- Check you have read permissions

### "Failed to create context"

- Reduce `n_ctx` if you're running out of memory
- Reduce `n_gpu_layers` if GPU memory is insufficient
- Check available RAM/VRAM

### "Out of memory" errors

- Use a smaller model
- Reduce `n_ctx` (context size)
- Reduce `n_batch` (batch size)
- Offload fewer layers to GPU (`-ngl`)

### Slow performance

- Enable GPU offloading with `-ngl`
- Use quantized models (Q4_K_M, Q5_K_M)
- Reduce batch size if using CPU
- Use Release build (`CMAKE_BUILD_TYPE=Release`)

### Linking errors

- Ensure llama.cpp is built before building examples
- Check `LLAMA_CPP_DIR` points to the correct location
- Verify the llama library exists in the expected location
- Try rebuilding llama.cpp

## Advanced Usage

### Custom Sampler Implementation

See `03-custom-sampling.cpp` for a complete example of implementing a custom sampler:

```cpp
struct llama_sampler_i custom_sampler_iface = {
    /* .name   = */ custom_sampler_name,
    /* .accept = */ nullptr,
    /* .apply  = */ custom_sampler_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ custom_sampler_clone,
    /* .free   = */ custom_sampler_free,
};

struct llama_sampler* my_sampler = llama_sampler_init(&custom_sampler_iface, ctx);
```

### Batch Embeddings Processing

For efficient RAG pipelines:

```cpp
// Create batch with multiple sequences
llama_batch batch = llama_batch_init(max_tokens * n_sequences, 0, n_sequences);

// Add sequences with different IDs
for (size_t i = 0; i < n_sequences; i++) {
    batch_add_seq(batch, tokens[i], i);
}

// Process all at once
llama_decode(ctx, batch);

// Extract per-sequence embeddings
for (size_t i = 0; i < n_sequences; i++) {
    const float* embd = llama_get_embeddings_seq(ctx, i);
    // Use embedding
}
```

## Performance Tips

1. **Use GPU offloading**: Set `-ngl` to offload layers to GPU
2. **Quantize models**: Use Q4_K_M or Q5_K_M for best speed/quality tradeoff
3. **Batch processing**: Process multiple sequences together for embeddings
4. **Context sizing**: Set `n_ctx` to just what you need
5. **Release builds**: Always use Release builds for production

## Further Reading

### Documentation
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama.h API Reference](../../include/llama.h)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

### Related Examples
- [Official examples](../../../examples/)
- [Python bindings](../python/)
- [CUDA examples](../cuda/)

### Concepts
- [Sampling strategies](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p)
- [Text embeddings](https://www.pinecone.io/learn/vector-embeddings/)
- [RAG patterns](https://www.anthropic.com/index/retrieval-augmented-generation)

## Contributing

Found an issue or want to improve these examples? Contributions are welcome!

## License

These examples follow the same license as llama.cpp (MIT License).
