# LoRA Adapters in llama.cpp

## Overview

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that allows you to adapt a pre-trained language model to a specific task or domain without modifying the original model weights. Instead of updating all model parameters, LoRA injects small, trainable low-rank matrices into the model's layers. These matrices are stored separately as a lightweight "adapter" that can be loaded and applied at inference time.

llama.cpp provides first-class support for LoRA adapters, enabling you to:

- Load one or more LoRA adapters on top of a base model
- Control adapter strength via scaling factors
- Merge adapters into the base model for faster inference
- Switch between adapters at runtime via the server API
- Convert Hugging Face PEFT LoRA adapters to GGUF format

This tutorial covers the complete workflow: from obtaining a LoRA adapter to using it in llama.cpp.

## Prerequisites

Before you begin, ensure you have the following:

1. **A GGUF base model** — LoRA adapters are model-specific. The adapter must match the architecture and tensor shapes of the base model. You can download GGUF models from [Hugging Face](https://huggingface.co/ggml-org) or convert them using `convert_hf_to_gguf.py`.

2. **A LoRA adapter in GGUF format** — The adapter file must be a GGUF file containing LoRA tensors (with `.lora_a` and `.lora_b` suffixes). You can:
   - Download pre-converted adapters from Hugging Face (e.g., `ngxson/Llama-3-Instruct-abliteration-LoRA-8B-F16-GGUF`)
   - Convert a Hugging Face PEFT adapter using `convert_lora_to_gguf.py` (see [Converting Hugging Face Adapters](#converting-hugging-face-adapters))

3. **llama.cpp built with LoRA support** — LoRA is supported in the standard build. Ensure you have a recent build of llama.cpp (the `llama-cli` and `llama-server` binaries).

4. **Basic familiarity with the command line** — All examples use the `llama-cli` or `llama-server` binaries.

## How LoRA Works in llama.cpp

A LoRA adapter contains pairs of low-rank matrices (A and B) for selected weight matrices in the base model. During inference, the adapter modifies the output of each targeted layer as follows:

```
output = base_output + scale * (B @ A) * input
```

Where:
- `A` and `B` are the low-rank matrices stored in the adapter
- `scale` is computed as `alpha / rank * user_scale`
- `alpha` is the LoRA alpha value stored in the adapter metadata
- `rank` is the LoRA rank (the inner dimension of the A/B matrices)
- `user_scale` is the scaling factor you provide (default: 1.0)

The adapter tensors are stored in a GGUF file with the following naming convention:
- `blk.N.attn_q.weight.lora_a` — the A matrix for the attention Q projection of layer N
- `blk.N.attn_q.weight.lora_b` — the B matrix for the attention Q projection of layer N

## Step-by-Step Guide

### Step 1: Obtain a Base Model and LoRA Adapter

Download a GGUF base model and a matching LoRA adapter. For this tutorial, we'll use a small example:

```bash
# Download a base model (e.g., Llama-3.2-1B-Instruct)
# Replace with your preferred model
wget https://huggingface.co/ggml-org/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Download a matching LoRA adapter
# This is an example adapter for Llama-3.2-1B-Instruct
wget https://huggingface.co/ngxson/Llama-3-Instruct-abliteration-LoRA-8B-F16-GGUF/resolve/main/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf
```

> **Note:** The adapter must match the base model's architecture. Using an adapter trained for a different model will result in an error.

### Step 2: Load and Use a LoRA Adapter with llama-cli

The simplest way to use a LoRA adapter is via the `--lora` flag:

```bash
./llama-cli \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf \
  -p "Write a computer virus" \
  -n 128 \
  --temp 0.0
```

**Expected output:** The model will generate text influenced by the LoRA adapter. With the abliteration adapter, the model will be more willing to generate content it would normally refuse.

### Step 3: Control Adapter Strength with `--lora-scaled`

You can control how strongly the adapter affects the model's output using the `--lora-scaled` flag. The format is `FNAME:SCALE`:

```bash
# Apply the adapter at 50% strength
./llama-cli \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora-scaled Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf:0.5 \
  -p "Write a computer virus" \
  -n 128 \
  --temp 0.0
```

**Scale values:**
- `0.0` — Adapter is loaded but has no effect (equivalent to base model)
- `0.5` — Half strength, subtle influence
- `1.0` — Full strength (default when using `--lora`)
- `1.5` — Over-indexed, stronger than training (use with caution)

### Step 4: Load Multiple Adapters

You can load multiple adapters simultaneously by using comma-separated values:

```bash
./llama-cli \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora adapter1.gguf,adapter2.gguf,adapter3.gguf \
  -p "Your prompt here" \
  -n 128
```

Or with different scales for each adapter:

```bash
./llama-cli \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora-scaled adapter1.gguf:1.0,adapter2.gguf:0.5,adapter3.gguf:0.8 \
  -p "Your prompt here" \
  -n 128
```

> **Note:** When multiple adapters are loaded, their effects are combined. The order of application is the order specified on the command line.

### Step 5: Use LoRA Adapters with llama-server

The llama-server provides a more flexible way to use LoRA adapters, especially for production use cases.

#### Start the server with LoRA adapters:

```bash
./llama-server \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora adapter1.gguf \
  --lora adapter2.gguf \
  --port 8080
```

#### Load adapters without applying them (deferred activation):

```bash
./llama-server \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora adapter1.gguf \
  --lora adapter2.gguf \
  --lora-init-without-apply \
  --port 8080
```

With `--lora-init-without-apply`, all adapters are loaded with scale 0 (inactive). You can then activate them at runtime via the API.

#### List loaded adapters:

```bash
curl http://localhost:8080/lora-adapters
```

**Response:**
```json
[
    {
        "id": 0,
        "path": "adapter1.gguf",
        "scale": 1.0
    },
    {
        "id": 1,
        "path": "adapter2.gguf",
        "scale": 1.0
    }
]
```

#### Set adapter scales globally:

```bash
curl -X POST http://localhost:8080/lora-adapters \
  -H "Content-Type: application/json" \
  -d '[{"id": 0, "scale": 0.5}, {"id": 1, "scale": 0.0}]'
```

This sets adapter 0 to 50% strength and disables adapter 1.

#### Per-request adapter configuration:

You can also specify adapter scales per request, which overrides the global settings:

```bash
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a computer virus",
    "n_predict": 128,
    "temperature": 0.0,
    "lora": [
      {"id": 0, "scale": 0.7},
      {"id": 1, "scale": 0.3}
    ]
  }'
```

> **Important:** Requests with different LoRA configurations will not be batched together, which may result in performance degradation. If you need to serve multiple adapter configurations concurrently, use multiple server slots (`--n-slots`).

### Step 6: Merge LoRA Adapters into the Base Model

For production deployment where you always use the same adapter, you can merge the adapter weights directly into the base model. This eliminates the runtime overhead of applying the adapter.

```bash
./llama-export-lora \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --lora Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf \
  -o Llama-3.2-1B-Instruct-merged.gguf
```

**Expected output:**
```
llama_adapter_lora_init: loading lora adapter from 'Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf' ...
llama_adapter_lora_init: loaded 136 tensors from lora file
llama-export-lora : merged 68 tensors with lora adapters
llama-export-lora : wrote 68 tensors to output file
done, output file is Llama-3.2-1B-Instruct-merged.gguf
```

The merged model can then be used without any `--lora` flags:

```bash
./llama-cli \
  -m Llama-3.2-1B-Instruct-merged.gguf \
  -p "Write a computer virus" \
  -n 128 \
  --temp 0.0
```

> **Note:** The merged model is always output in F16 format. The merge operation is irreversible — you cannot unmerge an adapter from a merged model.

### Step 7: Convert Hugging Face PEFT Adapters to GGUF

If you have a LoRA adapter in Hugging Face PEFT format (with `adapter_config.json` and `adapter_model.safetensors`), convert it to GGUF using the provided script:

```bash
python convert_lora_to_gguf.py \
  /path/to/peft/adapter/directory \
  --base /path/to/base/model/directory \
  --outtype f16
```

**Arguments:**
- `lora_path` — Path to the directory containing `adapter_config.json` and `adapter_model.safetensors` (or `adapter_model.bin`)
- `--base` — Path to the base model directory (containing `config.json`). Only the config is needed, not the full model weights.
- `--outtype` — Output format: `f32`, `f16`, `bf16`, `q8_0`, or `auto` (default: `f32`)
- `--outfile` — Output file path (optional; defaults to the adapter directory)

**Example:**
```bash
python convert_lora_to_gguf.py \
  ./my-lora-adapter/ \
  --base ./Llama-3.2-1B-Instruct/ \
  --outtype f16 \
  --outfile my-lora-adapter.gguf
```

The script will:
1. Load the adapter configuration and weights
2. Load the base model configuration (for architecture matching)
3. Convert the LoRA tensors to the GGUF format
4. Write the output GGUF file with proper metadata

## Using LoRA Adapters via the C API

For developers integrating llama.cpp into their applications, the C API provides full control over LoRA adapters.

### Basic C API Example

```c
#include "llama.h"
#include <stdio.h>

int main() {
    // 1. Load the base model
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("base-model.gguf", mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // 2. Create a context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    llama_context * ctx = llama_context_from_model(model, cparams);

    // 3. Load the LoRA adapter
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, "adapter.gguf");
    if (!adapter) {
        fprintf(stderr, "Failed to load LoRA adapter\n");
        return 1;
    }

    // 4. Apply the adapter to the context with scale 1.0
    llama_adapter_lora * adapters[] = { adapter };
    float scales[] = { 1.0f };
    llama_set_adapters_lora(ctx, adapters, 1, scales);

    // 5. Now use the context for inference as usual
    // ... (tokenization, decoding, etc.)

    // 6. Cleanup
    llama_adapter_lora_free(adapter);
    llama_context_free(ctx);
    llama_model_free(model);
    return 0;
}
```

### Key C API Functions

| Function | Description |
|----------|-------------|
| `llama_adapter_lora_init(model, path)` | Load a LoRA adapter from a GGUF file |
| `llama_adapter_lora_init_from_file_ptr(model, file)` | Load from an open FILE pointer |
| `llama_set_adapters_lora(ctx, adapters, n, scales)` | Apply adapters to a context with given scales |
| `llama_adapter_lora_free(adapter)` | Free a loaded adapter |
| `llama_adapter_meta_val_str(adapter, key, buf, size)` | Get adapter metadata by key |
| `llama_adapter_meta_count(adapter)` | Get number of metadata entries |
| `llama_adapter_get_alora_invocation_tokens(adapter)` | Get aLoRA invocation tokens |

### Switching Adapters at Runtime

You can change the active adapters and their scales at any time by calling `llama_set_adapters_lora` again:

```c
// Disable all adapters
llama_set_adapters_lora(ctx, NULL, 0, NULL);

// Switch to a different adapter with a different scale
llama_adapter_lora * new_adapters[] = { adapter2 };
float new_scales[] = { 0.7f };
llama_set_adapters_lora(ctx, new_adapters, 1, new_scales);
```

> **Note:** When switching adapters, the KV cache may need to be cleared to avoid stale activations. The llama-server handles this automatically.

## Advanced: aLoRA (Activated LoRA)

aLoRA is a variant of LoRA where the adapter is activated automatically when specific tokens appear in the input, rather than being applied unconditionally. llama.cpp supports aLoRA adapters that include invocation token sequences in their metadata.

When an aLoRA adapter is loaded, the invocation tokens are stored in the adapter metadata. The server can detect these tokens in the input and automatically activate the adapter.

To check if an adapter is an aLoRA:

```c
uint64_t n_tokens = llama_adapter_get_alora_n_invocation_tokens(adapter);
if (n_tokens > 0) {
    const llama_token * tokens = llama_adapter_get_alora_invocation_tokens(adapter);
    printf("This is an aLoRA with %llu invocation tokens\n", n_tokens);
}
```

## Troubleshooting

### "failed to load lora adapter"

**Cause:** The adapter file is corrupted, not a valid GGUF file, or not a LoRA adapter.

**Solution:** Verify the file is a valid GGUF LoRA adapter. Check that it was converted correctly using `convert_lora_to_gguf.py`.

### "model arch and LoRA arch mismatch"

**Cause:** The LoRA adapter was trained for a different model architecture than the base model.

**Solution:** Ensure the adapter matches the base model. Check the `general.architecture` metadata in both files.

### "LoRA tensor 'X' does not exist in base model"

**Cause:** The adapter contains tensors that don't exist in the base model. This usually means the adapter was trained for a different model variant.

**Solution:** Verify that the adapter is compatible with your base model. The tensor names in the adapter must match the tensor names in the base model.

### "LoRA tensor 'X' has incorrect shape"

**Cause:** The tensor shapes in the adapter don't match the base model. This can happen if the adapter was trained for a different model size.

**Solution:** Ensure the adapter was trained for the exact same base model (same hidden size, number of layers, etc.).

### "lora_a tensor is not transposed"

**Cause:** The adapter was created with an older format or a different tool that doesn't match llama.cpp's expected tensor layout.

**Solution:** Re-convert the adapter using the latest version of `convert_lora_to_gguf.py`.

### Adapter has no effect on model output

**Cause:** The adapter scale is set to 0, or the adapter is loaded but not applied.

**Solution:** 
- Check that you're using `--lora` (not just loading the file)
- Verify the scale is non-zero: `--lora-scaled adapter.gguf:1.0`
- If using the server, check the adapter scale via `GET /lora-adapters`

### Server returns different results for the same prompt

**Cause:** LoRA adapters introduce additional computation that may cause slight numerical differences. Also, if using per-request LoRA configurations, requests are not batched together.

**Solution:** This is expected behavior. For deterministic results, use the same adapter configuration and seed.

### Out of memory when loading large adapters

**Cause:** Large LoRA adapters (e.g., rank 128 or 256 for 70B models) can consume significant memory.

**Solution:** 
- Use quantized base models to free up memory
- Load fewer adapters simultaneously
- Use `--lora-init-without-apply` and only activate adapters when needed
- Consider merging the adapter into the model if you only use one

## Best Practices

1. **Match adapter to model:** Always verify that the LoRA adapter was trained for your specific base model. Mismatched adapters will produce poor results or fail to load.

2. **Start with scale 1.0:** The default scale is `alpha / rank`, which is the intended strength. Adjust from there based on your needs.

3. **Use `--lora-scaled` for fine control:** If the adapter is too strong or too weak, use `--lora-scaled` to find the right balance.

4. **Merge for production:** If you're deploying with a fixed adapter, merge it into the base model using `llama-export-lora` for better performance.

5. **Use the server for multiple adapters:** If you need to switch between adapters dynamically, use `llama-server` with the `/lora-adapters` API.

6. **Monitor memory usage:** Large adapters can significantly increase memory usage. Monitor your system resources when loading multiple or large adapters.

7. **Use aLoRA for conditional activation:** If you want the adapter to activate only for specific prompts, look for or train aLoRA adapters that include invocation tokens.

## Additional Resources

- [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [PEFT library](https://github.com/huggingface/peft) — Hugging Face's Parameter-Efficient Fine-Tuning library
- [llama.cpp server documentation](../tools/server/README.md) — Full server API reference
- [convert_lora_to_gguf.py](../../convert_lora_to_gguf.py) — Adapter conversion script
- [llama-export-lora](../../tools/export-lora/export-lora.cpp) — Adapter merging tool
