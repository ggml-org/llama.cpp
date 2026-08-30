# llama.cpp/examples/compare-single-and-block

Diagnostic tool `llama-compare-single-and-block`: given identical token conditioning, does a W-token batch decode produce the same logits as W single-token steps?

Prefix KV is rebuilt with the same batch shape on both paths, so the remaining gap is a floating-point batch-shape effect — not trajectory drift. All comparisons are teacher-forced along a greedy AR trajectory.

Background (DFlash motivating example, gap hypothesis): [README.md](../../README.md).

### What this tool measures

Three logit sources on the same tokens:

```
AR greedy (ground truth):
    prefill(prompt); decode 1 token/step, greedy argmax
    → trajectory AR[0..n-1] and logits l_ar

Single:
    for each window start t:
      wipe KV, re-prefill prompt+AR[0..t-1]
      decode W inputs [last, AR[t], ..., AR[t+W-2]] one token per llama_decode
    → logits l_single at each block offset j

Block:
    same prefix and same W inputs, one llama_decode of all W tokens (all logits=true)
    → logits l_block at each offset j
```

The tool reports `dlogit = l_single - l_block` and records every **argmax flip** between single and block. Pairwise stats also include `l_ar - l_single` (prefill-vs-incremental drift) and `l_ar - l_block` (the mix of both).

## How to run

The tool builds as the `llama-compare-single-and-block` target.

### Step 1: Convert model to GGUF

```bash
MODEL_HF="${MODELS_DIR}/Qwen3-4B"
MODEL_GGUF="${MODELS_DIR}/Qwen3-4B.gguf"

python convert_hf_to_gguf.py \
    "${MODEL_HF}" \
    --outtype bf16 \
    --outfile "${MODEL_GGUF}"
```

Use bf16/f16/f32 for the cleanest signal. Quantization adds its own rounding on top of the batch-shape effect.

### Step 2: Build

```bash
cmake -B build                 # add -DGGML_CUDA=ON for CUDA
cmake --build build --target llama-compare-single-and-block -j
```

### Step 3: Run

```bash
./build/bin/llama-compare-single-and-block \
  -m "${MODEL_GGUF}" \
  -p "Explain the Pythagorean theorem" \
  -ngl 0 -fa off                 # CPU; use -ngl 99 for GPU
```

Each window re-prefills the full prefix on two contexts. On Qwen3-4B CPU, `-n 100` is typically **minutes**.

Exit codes: `0` no argmax mismatch, `1` mismatch found, `2` error.

### Options

All common llama.cpp flags apply (`-m`, `-p`, `-n`, `-c`, `-b`, `-ub`, `-t`, `-tb`, `-ngl`, `-fa`, ...; `--help` for the full list). Defaults: `-n 100 -c 4096 -b 512 -ub 512 -np 1`.

| Flag | Description |
|------|-------------|
| `--block-width N` | Tokens per block decode (default `5`). Must be `<= -b`. |
| `--json PATH` | Write a JSON report (config, AR trajectory, every cell, all flips). |
| `--verbose`, `--full` | AR trajectory, per-window lines, dlogit matrix, per-cell table. Default stdout is run config + pairwise stats + flip details. |
| `--help-extra` | Tool-specific flags only. |

Notes:

- **Chat template (Qwen-oriented):** the prompt is wrapped with the model's chat template, thinking disabled. To pass a raw string, include `<|im_start|>` — that substring is the bypass check, not a generic `--no-chat-template` flag. Other families still get the template unless the prompt contains that marker.
- Use `-ub N` with `N < --block-width` to force physical ubatch splits inside the block decode.
- Keep `-t`/`-b`/`-ub`/`-fa` fixed across runs you want to compare: absolute diffs are only comparable under identical backend settings.
- `-c` must satisfy `prompt_tokens + max(n_predict, block_width) <= n_ctx`; the tool checks this up front.
