# SDXL-Turbo → GGUF conversion

Converts a [HuggingFace SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)
model directory into one or more GGUF files, packing all four components into
each output file.

## Components packed

| Prefix | Component | Source dir |
|--------|-----------|------------|
| `te1.*` | CLIP ViT-L/14 text encoder | `text_encoder/` |
| `te2.*` | OpenCLIP ViT-bigG/14 text encoder | `text_encoder_2/` |
| `vae.*` | Variational Autoencoder | `vae/` |
| `unet.*` | Denoising U-Net | `unet/` |

## Requirements

```bash
pip install torch safetensors
```

For K-quant types you also need the `llama-quantize` binary:

```bash
cmake -B build && cmake --build build --target llama-quantize
```

## Download the model

```bash
pip install huggingface_hub
huggingface-cli download stabilityai/sdxl-turbo --local-dir ./sdxl-turbo
```

## Usage

```
python examples/sdxl/convert-sdxl-to-gguf.py \
    -m MODEL_DIR [-o OUTPUT_DIR] [--q-type TYPE [TYPE ...]] \
    [--llama-quantize PATH] \
    [--no-text-enc-1] [--no-text-enc-2] [--no-vae] [--no-unet]
```

### Weight types

| Flag value | Implemented in | Notes |
|------------|---------------|-------|
| `f32` | Python | Full precision |
| `f16` | Python | Default |
| `q8_0` | Python | 8-bit, ~2× smaller than f16 |
| `Q4_K_S` / `Q4_K_M` | C++ (`llama-quantize`) | 4-bit K-quant |
| `Q5_K_S` / `Q5_K_M` | C++ (`llama-quantize`) | 5-bit K-quant |
| `Q2_K` | C++ (`llama-quantize`) | 2-bit, smallest |
| `Q3_K_S/M/L` | C++ (`llama-quantize`) | 3-bit K-quant |
| `Q6_K` | C++ (`llama-quantize`) | 6-bit K-quant |

**Per-tensor dtype policy** (applied regardless of `--q-type`):

| Tensor shape | Dtype |
|---|---|
| 1-D (biases, norms) | always `f32` |
| 4-D (conv kernels) | always `f16` |
| 2-D (weight matrices) | `--q-type` value; `q8_0` falls back to `f16` if row width is not a multiple of 32 |

### Single variant

```bash
# Default (f16)
python examples/sdxl/convert-sdxl-to-gguf.py -m ./sdxl-turbo -o ./output

# Q4_K_M directly (writes temp f16, quantizes, removes temp)
python examples/sdxl/convert-sdxl-to-gguf.py -m ./sdxl-turbo -o ./output \
    --q-type Q4_K_M
```

### Multiple variants in one pass

The model is **loaded once**; all requested types are written simultaneously.
For K-quants the f16 GGUF is reused as the quantization source if it is also
requested, otherwise a temporary f16 file is created and removed automatically.

```bash
# Four output files, one model load
python examples/sdxl/convert-sdxl-to-gguf.py -m ./sdxl-turbo -o ./output \
    --q-type f16 q8_0 Q4_K_M Q5_K_M
```

Output:
```
output/sdxl-turbo-f16.gguf
output/sdxl-turbo-q8_0.gguf
output/sdxl-turbo-q4_k_m.gguf
output/sdxl-turbo-q5_k_m.gguf
```

### Partial conversion (skip components)

```bash
# Text encoders only
python examples/sdxl/convert-sdxl-to-gguf.py -m ./sdxl-turbo \
    --no-vae --no-unet

# UNet only, Q4_K_M
python examples/sdxl/convert-sdxl-to-gguf.py -m ./sdxl-turbo \
    --no-text-enc-1 --no-text-enc-2 --no-vae --q-type Q4_K_M
```

### Custom llama-quantize path

```bash
python examples/sdxl/convert-sdxl-to-gguf.py -m ./sdxl-turbo \
    --q-type Q4_K_M --llama-quantize ./build/bin/llama-quantize
```

## Inspecting the output

```bash
python gguf-py/scripts/gguf-dump.py output/sdxl-turbo-f16.gguf | head -60
```

## GGUF metadata keys written

| Key | Source |
|-----|--------|
| `sdxl.te1.vocab_size` / `context_length` / `embedding_length` / … | `text_encoder/config.json` |
| `sdxl.te2.*` | `text_encoder_2/config.json` |
| `sdxl.vae.in_channels` / `latent_channels` / `scaling_factor` / … | `vae/config.json` |
| `sdxl.unet.in_channels` / `cross_attention_dim` / `block_out_channels` / … | `unet/config.json` |
