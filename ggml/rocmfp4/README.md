# ROCmFP4

ROCmFP4 adds two experimental 4-bit GGUF tensor layouts intended for compact
AMD-oriented inference formats:

- `Q4_0_ROCMFP4`: 32 weights per block, packed 4-bit values, and two finite
  unsigned E4M3 scale bytes, one scale per 16 weights. The block size is
  18 bytes, or 4.50 bits per weight.
- `Q4_0_ROCMFP4_FAST`: 32 weights per block, packed 4-bit values, and one
  finite unsigned E4M3 scale byte for the full block. The block size is
  17 bytes, or 4.25 bits per weight.

The 4-bit values use a small signed codebook with levels up to `5.0` after
the decoded scale is applied. Quantization searches all finite E4M3 scale
candidates and keeps the lowest-error assignment. Invalid scale bytes are
rejected during row validation so malformed tensors fail early.

This directory contains the format-specific CPU reference helpers. This initial
change adds the GGUF tensor types, row validation, quantization, and
dequantization needed to create and inspect ROCmFP4 files from the normal
`llama-quantize` workflow.

- CPU reference quantization, dequantization, row validation, and
  `llama-quantize` support.

The feature is additive. Existing tensor types, file types, and backend
dispatch paths are unchanged unless a tensor is explicitly stored as
`Q4_0_ROCMFP4` or `Q4_0_ROCMFP4_FAST`.

Accelerated ROCm/HIP and Vulkan execution paths can be added in follow-up
changes after the tensor formats and CPU reference behavior are reviewed.

Example quantization:

```sh
./llama-quantize model-f16.gguf model-rocmfp4.gguf Q4_0_ROCMFP4
./llama-quantize model-f16.gguf model-rocmfp4-fast.gguf Q4_0_ROCMFP4_FAST
```

Importance matrices use the existing quantize interface:

```sh
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-rocmfp4.gguf Q4_0_ROCMFP4
```

Advanced mixed recipes can be expressed with the existing
`--tensor-type` and `--tensor-type-file` options instead of adding extra
public file-type presets.
