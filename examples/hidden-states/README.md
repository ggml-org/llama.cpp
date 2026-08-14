# Hidden States Example

Minimal example demonstrating hidden state extraction from llama.cpp models.

## Overview

This example shows how to:
1. Load a GGUF model
2. Tokenize a prompt
3. Run forward pass
4. Extract residual stream activations from intermediate layers

## Building

```bash
cd llama.cpp
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

## Usage

```bash
build/bin/llama-hidden-states path/to/model.gguf
```

## Expected Output

```
Prompt: "Hello, world!"
Tokens: 4
Hidden states extracted from 32 layers (4096-dim)
Layer 0: [0.123, -0.456, 0.789, ...]
Layer 1: [0.234, -0.567, 0.890, ...]
...
```

## Code Structure

```cpp
// 1. Load model
llama_model* model = llama_model_load_from_file(model_path, params);

// 2. Create context with hidden states enabled
llama_context_params ctx_params = llama_context_default_params();
ctx_params.extract_hidden_states = true;
llama_context* ctx = llama_init_from_model(model, ctx_params);

// 3. Tokenize and run forward pass
std::vector<llama_token> tokens = tokenize(ctx, prompt);
llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

// 4. CRITICAL: Synchronize before reading (CUDA writes are async)
llama_synchronize(ctx);

// 5. Extract hidden states from each layer
for (int layer = 0; layer < n_layers; layer++) {
    float* hidden = llama_get_hidden_state(ctx, layer);
    int n_tokens = llama_get_hidden_state_n_tokens(ctx);
    // hidden contains n_tokens * hidden_dim floats
}
```

## Key Points

### CUDA Synchronization

Hidden states are written asynchronously by CUDA kernels. **Always call `llama_synchronize()` after `llama_decode()` and before reading hidden states**, otherwise you'll get garbage data.

### API Functions

- `llama_get_hidden_state(ctx, layer)` - Get hidden states for a specific layer
- `llama_get_hidden_state_n_tokens(ctx)` - Get number of tokens with hidden states
- `llama_get_hidden_state_ith(ctx, layer, token_idx)` - Get hidden state for specific token

### Memory Layout

Hidden states are stored in row-major order:
```
hidden[token * hidden_dim + dim]
```

For example, with 4 tokens and 4096 hidden dimensions:
- Token 0: `hidden[0..4095]`
- Token 1: `hidden[4096..8191]`
- Token 2: `hidden[8192..12287]`
- Token 3: `hidden[12288..16383]`

## Use Cases

- **Mechanistic interpretability** - Analyze how information flows through the network
- **Probing classifiers** - Train linear probes on intermediate representations
- **Representation similarity** - Compare activations across prompts/models
- **Feature extraction** - Use hidden states as input to downstream tasks

## Related Tools

- [hs-extract](../../tools/hs-extract/) - Single-prompt extraction to JSON
- [hs-extract-batch](../../tools/hs-extract-batch/) - High-performance batch processing
