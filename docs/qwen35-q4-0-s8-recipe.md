# Qwen3.6-35B-A3B Q4_0-S8 recipe

The first implementation is intentionally script-only. It does not add `GGML_TYPE_Q4_0_S8` or patch the public quantization API.

`scripts/build-qwen35-q4-0-s8.sh` performs:

1. raw HF directory -> BF16 main GGUF, without `--no-mtp`;
2. raw HF directory -> direct Q8_0 vision projector;
3. a `llama-quantize --dry-run ... Q4_K_M` pass as the live sensitivity oracle;
4. a complete TSV tensor plan and estimated sizes;
5. a pure Q4_0 quantization pass with exact Q8_0/F16 overrides;
6. optionally, an existing llama.cpp imatrix during the final pass.

The resulting file contains only normal existing GGUF types. MTP tensors matching the exact `.nextn.` naming used by llama.cpp are forced to Q4_0. Shape-incompatible rows fall back to F16. The final type validator rejects Q4_K/Q5_K/Q6_K in the output.

## Usage

Build `llama-quantize` first, then run:

```bash
cd /home/edwin/llama.cpp-rdna2
scripts/build-qwen35-q4-0-s8.sh \
  --input /home/edwin/models/Qwen3.6-35B-A3B-raw \
  --out-dir /home/edwin/models/qwen35-q4-0-s8 \
  --threads 24 \
  --imatrix /home/edwin/models/qwen35-imatrix/imatrix_unsloth.gguf_file
```

Use `--plan-only` to inspect the plan before quantizing. The script refuses already-quantized input and retains the BF16 intermediate by default; use `--remove-bf16` only after validation.

The downloaded imatrix is auto-detected at `$HOME/models/qwen35-imatrix/imatrix_unsloth.gguf_file` and passed only to the final quantization pass. Use `--no-imatrix` to force the original plain-RTN behavior, or `--imatrix PATH` to select another file. Q8_0 ignores the imatrix; Q4_0 uses it for weighted quantization.

The script aborts by default if Q8_0 exceeds half of planned quantized bytes. This is a guard against an unexpectedly broad sensitivity policy; `--allow-large-q8` explicitly overrides it after inspecting `q4_0-s8-plan/summary.txt` and `tensor-plan.tsv`.

## Why the first version is script-only

A separate Q4_K_M dry run uses llama.cpp's current selection logic directly, avoiding a manually maintained tensor list and avoiding a second counter state inside the quantizer. The final pass uses existing `--pure` and `--tensor-type-file` features, so an unmodified llama.cpp can load the output. If this plan is validated and worth keeping, a later `--s8-fast` native option can replace the two-pass orchestration without changing the GGUF format.