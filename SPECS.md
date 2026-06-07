Below is a `SPECS.md` you can drop into `https://github.com/djinn/bluellama.cpp`.

````md
# BlueLlama.cpp SPEC

## Goal

Modify llama.cpp to test whether INT4 KV cache with deterministic blue-noise-style dithering improves long-context quality versus normal INT4 KV cache.

Starting repo:

```bash
git clone https://github.com/djinn/bluellama.cpp
cd bluellama.cpp
````

Model:

```text
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Target GPU:

```text
NVIDIA Quadro P2000, 5 GB VRAM
```

## Core hypothesis

Normal INT4 KV cache creates structured quantization error.

Blue-noise-style dither should make error less correlated.

We are not trying to make inference faster first.

We are testing whether quality falls apart more slowly.

## Required modes

Implement KV cache modes:

```text
f16              baseline
q4_0             existing INT4 KV cache
q4_0_blue        INT4 KV cache with deterministic blue-noise dither
q4_1             INT4 KV cache with asymmetric range
q4_1_blue        INT4 KV cache with asymmetric range + blue-noise dither
q2_K             2-bit K-quant KV cache
q2_K_blue        2-bit K-quant KV cache with blue-noise dither
q3_K             3-bit K-quant KV cache
q3_K_blue        3-bit K-quant KV cache with blue-noise dither
```

Expose through CLI:

```bash
--cache-type-k q4_0_blue
--cache-type-v q4_0_blue
--cache-type-k q2_K_blue
--cache-type-k q3_K_blue
```

Also allow mixed mode:

```bash
--cache-type-k q4_0_blue --cache-type-v q4_0          # blue K, normal V (most important)
--cache-type-k q4_0    --cache-type-v q4_0_blue        # normal K, blue V
--cache-type-k q2_K_blue --cache-type-v q2_K           # blue K (2-bit), normal V (2-bit)
--cache-type-k q3_K    --cache-type-v q3_K_blue        # normal K (3-bit), blue V (3-bit)
--cache-type-k q2_K_blue --cache-type-v q3_K_blue      # blue K (2-bit), blue V (3-bit)
```

Keys steer attention. Values carry payload.

Testing lower-bit types (Q2_K, Q3_K) helps determine whether blue-noise dithering has a proportionally larger effect at lower bit depths where the quantization grid is coarser.

## Implementation plan for Cline

### 1. Inspect cache type plumbing

Find:

```text
llama_kv_cache_init
cache-type-k
cache-type-v
GGML_TYPE_Q4_0
ggml_type
```

Map how CLI string becomes `ggml_type`.

Do not rewrite KV cache.

Add the smallest possible extension.

### 2. Add new type or experimental flag

Preferred:

```text
GGML_TYPE_Q4_0_BLUE
```

Fallback if adding a full type is too invasive:

```bash
-DGGML_Q4_0_BLUE_DITHER=ON
```

But CLI-visible type is better.

## Dither design

Use deterministic tiled blue-noise-style offsets.

Start with a 64-value table:

```c
static const float blue_noise_64[64] = {
    -0.484375f,  0.265625f, -0.109375f,  0.453125f,
     0.078125f, -0.359375f,  0.328125f, -0.234375f,
    -0.046875f,  0.390625f, -0.421875f,  0.171875f,
     0.234375f, -0.296875f,  0.015625f, -0.171875f,

     0.484375f, -0.265625f,  0.109375f, -0.453125f,
    -0.078125f,  0.359375f, -0.328125f,  0.234375f,
     0.046875f, -0.390625f,  0.421875f, -0.171875f,
    -0.234375f,  0.296875f, -0.015625f,  0.171875f,

    -0.375000f,  0.125000f, -0.500000f,  0.312500f,
    -0.187500f,  0.437500f, -0.062500f,  0.250000f,
     0.500000f, -0.125000f,  0.375000f, -0.312500f,
     0.187500f, -0.437500f,  0.062500f, -0.250000f,

    -0.218750f,  0.406250f, -0.343750f,  0.031250f,
     0.156250f, -0.468750f,  0.281250f, -0.093750f,
     0.218750f, -0.406250f,  0.343750f, -0.031250f,
    -0.156250f,  0.468750f, -0.281250f,  0.093750f
};
```

Apply before rounding:

```c
float strength = 0.25f;
float noise = blue_noise_64[(i + seed) & 63] * strength;
int q = nearest_int(x / d + noise);
```

Seed must be deterministic.

Use available indexes where possible:

```text
layer
token position
element index
head index if available
```

If only element index is available, start there.

## Build commands

CPU baseline:

```bash
cmake -B build -DGGML_CUDA=OFF
cmake --build build -j
```

CUDA build for P2000:

```bash
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j
```

## Test matrix

Use fixed prompt files:

```text
tests/prompts/needle_4k.txt
tests/prompts/needle_8k.txt
tests/prompts/summarise_long.txt
tests/prompts/math_reasoning.txt
tests/prompts/repetition_stress.txt
```

Use fixed seed:

```bash
--seed 42
```

### KV cache type combinations to test

Each mode is tested with all prompt files.

```text
# Baseline
K=f16,  V=f16

# Existing INT4
K=q4_0, V=q4_0
K=q4_1, V=q4_1

# Blue-noise INT4
K=q4_0_blue, V=q4_0        # Blue K only
K=q4_0,      V=q4_0_blue   # Blue V only
K=q4_0_blue, V=q4_0_blue   # Both blue
K=q4_1_blue, V=q4_1        # Blue K (asymmetric)
K=q4_1,      V=q4_1_blue   # Blue V (asymmetric)
K=q4_1_blue, V=q4_1_blue   # Both blue (asymmetric)

# 3-bit (Q3_K)
K=q3_K,      V=q3_K        # Normal 3-bit
K=q3_K_blue, V=q3_K        # Blue K (3-bit)
K=q3_K,      V=q3_K_blue   # Blue V (3-bit)
K=q3_K_blue, V=q3_K_blue   # Both blue (3-bit)

# 2-bit (Q2_K)
K=q2_K,      V=q2_K        # Normal 2-bit
K=q2_K_blue, V=q2_K        # Blue K (2-bit)
K=q2_K,      V=q2_K_blue   # Blue V (2-bit)
K=q2_K_blue, V=q2_K_blue   # Both blue (2-bit)

# Cross-bit mixed (blue K at lower bit, blue V at different bit)
K=q2_K_blue, V=q3_K_blue   # Blue K (2-bit), Blue V (3-bit)
K=q3_K_blue, V=q2_K_blue   # Blue K (3-bit), Blue V (2-bit)
```

Example invocation:

```bash
./build-cuda/bin/llama-cli \
  -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -ngl 999 -c 4096 \
  --cache-type-k q3_K_blue \
  --cache-type-v q3_K \
  --seed 42 \
  -f tests/prompts/needle_4k.txt \
  -n 256
```

## Validation tests

### 1. Build test

Passes if:

```bash
cmake --build build-cuda -j
```

finishes without warnings related to enum fallthrough, unsupported type, or CUDA compile failure.

### 2. CLI recognition test

Passes if:

```bash
./build-cuda/bin/llama-cli --help | grep q4_0_blue
```

finds the new cache type.

### 3. Determinism test

Run same command twice.

Passes if output is identical with:

```bash
--seed 42 --temp 0
```

### 4. Difference test

Run:

```text
q4_0
q4_0_blue
```

Passes if token output or logits differ.

If output is identical, dither is not active.

### 5. Sanity quality test

Needle prompt should ask for one hidden fact placed deep in the context.

Passes if `q4_0_blue` retrieves the same fact as `f16`.

### 6. Regression guard

Passes if `q4_0_blue` is not worse than `q4_0` on more than 2 out of 5 prompts.

Metric:

```text
exact answer success
output length
repetition count
obvious hallucination
tokens/sec
peak VRAM
```

### 7. Perplexity test

Run:

```bash
./build-cuda/bin/llama-perplexity \
  -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -f tests/data/wiki_small.txt \
  -c 4096 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

Then compare with:

```bash
--cache-type-k q4_0_blue --cache-type-v q4_0
```

Passes if perplexity is within 3 percent of normal q4_0.

Better if lower.

## Output report

Cline must create:

```text
reports/kv_dither_results.md
```

With table:

```text
mode
context
prompt
tokens/sec
peak VRAM
exact success
perplexity
notes
```

## Acceptance criteria

The experiment is successful when:

1. `q4_0_blue` builds.
2. CLI accepts it.
3. Deterministic runs match.
4. It produces different behavior from normal `q4_0`.
5. It does not crash on P2000.
6. It is not clearly worse than normal INT4 KV cache.
7. At least one long-context test shows equal or better retrieval than normal INT4.

## Cline workflow

Cline should work in this order:

1. Create branch:

```bash
git checkout -b feature/q4-blue-kv-cache
```

2. Locate KV cache type parsing.
3. Locate q4_0 quantization function.
4. Add `q4_0_blue`.
5. Add dither table.
6. Add tests.
7. Build CPU.
8. Build CUDA.
9. Run smoke tests.
10. Run quality tests.
11. Write report.
12. Commit.

Commit message:

```text
Add experimental blue-noise dithered INT4 KV cache
```

## Do not do

Do not optimize CUDA first.

Do not rewrite attention.

Do not add Python dependency for the core path.

Do not use random noise.

Do not claim quality improvement from one prompt.

This is a measurement project.

The product is evidence.

```

The key move: make `q4_0_blue` a first-class cache type. Not a hidden hack. If it happens twice, it becomes a system.
::contentReference[oaicite:0]{index=0}
```
