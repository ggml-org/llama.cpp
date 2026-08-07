# Qwen3.6-35B-A3B stock-map S8 recipes

The recipe remains script-only and uses existing GGUF tensor types. It now starts from the exact tensor-type map of the existing stock Q4_0 GGUF rather than treating every quantizable tensor as Q4_0.

`scripts/build-qwen35-q4-0-s8.sh` performs:

1. raw HF directory -> BF16 main GGUF, retaining embedded MTP;
2. raw HF directory -> direct Q8_0 vision projector;
3. a live `llama-quantize --dry-run ... Q4_K_M` sensitivity pass;
4. a stock Q4_0 tensor-map plan with exactly 29 Q4_0 -> Q8_0 promotions;
5. either the quality-isolation `fixed` stage or the V620-native `native` stage;
6. optional imatrix-weighted quantization.

## Stages

### `fixed` (default)

This preserves the stock Q4_0 recipe exactly and adds only the 29 Q4_0 -> Q8_0 promotions selected by the Q4_K_M oracle. It intentionally retains the stock Q5_0, Q4_1, Q6_K, and BF16 entries so its KLD can isolate the sensitivity promotions.

### `native`

This starts from the same fixed map and maps the remaining Q5_0, Q4_1, and Q6_K tensors to Q8_0. BF16 entries become F32. The expected final type set is F32, Q4_0, and Q8_0 only, approximately:

```text
F32    370  # 368 stock F32 plus two stock BF16 MTP tensors
Q4_0   179
Q8_0   204
```

The native stage may require `--allow-large-q8`; inspect the generated summary first.

## Usage

Build `llama-quantize` first. The stock Q4_0 map is auto-detected at `$HOME/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`:

```bash
cd /home/edwin/llama.cpp-rdna2
scripts/build-qwen35-q4-0-s8.sh \
  --input /home/edwin/models/Qwen3.6-35B-A3B-raw \
  --out-dir /home/edwin/models/qwen35-q4-0-s8-fixed \
  --stage fixed \
  --threads 24 \
  --imatrix /home/edwin/models/qwen35-imatrix/imatrix_unsloth.gguf_file
```

Use `--stage native` for the V620-native build after validating the fixed stage. Use `--stock-q4-0 PATH` to select another stock-map GGUF. Use `--plan-only` to inspect the map before quantizing.

The imatrix is auto-detected at `$HOME/models/qwen35-imatrix/imatrix_unsloth.gguf_file`; use `--no-imatrix` to disable it. It changes quantization values but not tensor-type selection.

The script refuses already-quantized input, retains BF16 by default, and validates the final stage-specific tensor set. Use `--remove-bf16` only after validation.