# llama-bnb-compat

Experimental GGML-side compatibility tool for porting the portable parts of
bitsandbytes 4-bit quantization into the THOTH `llama.cpp-1-bit-turbo` build.

What it does:

- ports the bitsandbytes blockwise FP4 and NF4 codebooks
- ports the simple blockwise absmax + nearest-codebook quantize/dequantize path
- compares that reconstruction error against native GGML:
  - `IQ4_NL` for NF4-like behavior
  - `MXFP4` for FP4-like behavior

What it does not do:

- it does not embed the Python bitsandbytes runtime into `llama.cpp`
- it does not add a new serving-time backend dependency
- it does not make the audio server use bitsandbytes kernels directly

Build target:

- `llama-bnb-compat`

Example:

```bash
./build-thoth-audio/bin/llama-bnb-compat --rows 64 --cols 4096 --blocksize 64
```

Optional input:

- `--input file.bin` expects raw little-endian `float32` values
- `--cols` stays required so rows can be inferred from the file length
