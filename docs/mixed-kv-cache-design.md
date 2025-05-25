# Mixed KV Cache Design Document

## Overview

This document describes the new mixed precision KV cache implementation for llama.cpp, which stores recent tokens in FP16 precision and automatically quantizes older tokens to save memory.

## Architecture

### Core Design Principle

Instead of using two separate unified caches (hot and cold), the new design implements mixed precision directly within each `kv_layer`:

```cpp
struct kv_layer_mixed {
    // FP16 tensors for recent tokens
    ggml_tensor * k_fp16;
    ggml_tensor * v_fp16;
    
    // Quantized tensors for old tokens
    ggml_tensor * k_quant;
    ggml_tensor * v_quant;
    
    // Dequantized views (for returning FP16 to attention)
    ggml_tensor * k_dequant;
    ggml_tensor * v_dequant;
    
    // Token counts
    uint32_t n_fp16_tokens = 0;
    uint32_t n_quant_tokens = 0;
};
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mixed KV Cache Layer                        │
│  ┌─────────────────┐                   ┌─────────────────┐     │
│  │   FP16 Buffer   │ ──quantize──────▶ │ Quantized Buffer│     │
│  │  (recent tokens)│                   │  (old tokens)   │     │
│  └─────────────────┘                   └─────────────────┘     │
│         │                                       │               │
│         └───────────── dequantize ─────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                    Merged FP16 View                            │
│                  (returned to attention)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Transparent FP16 Interface

The cache always returns FP16 tensors to the attention mechanism, regardless of internal storage:

```cpp
ggml_tensor * llama_kv_cache_mixed::get_k(ggml_context * ctx, int32_t il) const {
    // Returns merged FP16 view (includes both FP16 and dequantized data)
    return get_merged_k(ctx, il);
}
```

### 2. Automatic Quantization

When the number of FP16 tokens exceeds a threshold, the cache automatically quantizes them:

```cpp
void llama_kv_cache_mixed::commit() {
    if (config.enable_quantization) {
        for (auto & layer : layers) {
            if (layer.n_fp16_tokens >= config.quantization_threshold) {
                quantize_tokens(layer.il);
            }
        }
    }
}
```

### 3. Configurable Quantization

The cache supports various configuration options:

```cpp
struct llama_kv_cache_mixed_config {
    bool     enable_quantization = true;    // Enable per-channel quantization
    uint32_t quantization_threshold = 32;   // Number of tokens before quantization
    uint32_t group_size = 16;               // Number of tokens to quantize at once
    
    // Cache types
    ggml_type hot_type_k  = GGML_TYPE_F16;  // Recent tokens (FP16)
    ggml_type hot_type_v  = GGML_TYPE_F16;
    ggml_type cold_type_k = GGML_TYPE_Q4_0; // Old tokens (quantized)
    ggml_type cold_type_v = GGML_TYPE_Q4_0;
};
```

## Benefits

1. **Memory Efficiency**: Old tokens use ~8x less memory when quantized to Q4_0
2. **Quality Preservation**: Recent tokens remain in full FP16 precision
3. **Transparent to Model**: Attention always sees FP16 data via automatic dequantization
4. **Flexible Configuration**: Quantization thresholds and types can be adjusted

## Usage Example

```cpp
// Create mixed cache with automatic quantization
llama_kv_cache_mixed_config config;
config.enable_quantization = true;
config.quantization_threshold = 32;  // Quantize after 32 tokens
config.cold_type_k = GGML_TYPE_Q4_0;
config.cold_type_v = GGML_TYPE_Q4_0;

auto cache = std::make_unique<llama_kv_cache_mixed>(
    model, 
    filter,
    false,   // v_trans
    false,   // offload
    1024,    // kv_size
    4,       // n_seq_max
    8,       // n_pad
    config
);
```

## Quantization Process Visualization

### Step 1: Initial State (all FP16)
```
FP16:  [T0][T1][T2][T3][T4][T5][T6][T7]
Quant: [  ][  ][  ][  ][  ][  ][  ][  ]
```

### Step 2: After Quantization Threshold
```
FP16:  [  ][  ][  ][  ][T4][T5][T6][T7]
Quant: [T0][T1][T2][T3][  ][  ][  ][  ]
        └── Quantized to Q4_0 ──┘
```

### Step 3: Merged View (always FP16)
```
Merged: [T0'][T1'][T2'][T3'][T4][T5][T6][T7]
         └─ Dequantized Q4_0→FP16 ─┘
```

## Future Enhancements

1. **Per-channel Quantization**: Implement custom per-channel quantization for better quality
2. **Dynamic Thresholds**: Adjust quantization threshold based on available memory
3. **Multiple Quantization Levels**: Support gradual quantization (FP16 → Q8_0 → Q4_0)
4. **Selective Layer Quantization**: Different quantization strategies for different layers

## Testing

The implementation includes comprehensive tests:

- `test-mixed-kv-cache.cpp`: Verifies basic functionality
- `test-unified-cache-copy.cpp`: Tests move/copy operations between caches
- `test-kv-cache-unified.cpp`: Tests unified cache with mixed precision support

Run tests with:
```bash
cmake --build build --target test-mixed-kv-cache
./build/bin/test-mixed-kv-cache
``` 