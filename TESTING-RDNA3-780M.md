# Test-replay protocol — RDNA3 iGPU (AMD Radeon 780M, Mesa/radv)

> Reference branch only. NOT meant to be merged into llama.cpp.
> Purpose: reproduce the validation of "ggml-vulkan : tune L-tile warp
> micro-dimension for RDNA3 iGPUs" (branch `ggml-vk-rdna3-igpu-ltile`) on
> known-good hardware.

## Baseline hardware

- Minisforum UM790 Pro, Ryzen 9 7940HS, Radeon 780M (gfx1151 / RDNA3, UMA)
- Ubuntu Server, Mesa/radv Vulkan driver (device string "RADV PHOENIX")
- Model: `models/Qwen3.8-9B-Q4_K_M.gguf` (cross-check with a 27B quant if desired)

## Reference numbers (measured 2026-09-07, master `0cae43063` vs tuned `60b099ca4`)

| test | master | tuned | delta |
|---|---|---|---|
| pp512 (llama-bench `-p 512 -n 64 -r 3`, fa on, q8_0 KV) | 249.42 ± 0.04 t/s | 401.16 ± 0.88 t/s | **+60.8%** |
| tg64 | 14.39 t/s | 14.39 t/s | flat |
| greedy text (`-temp 0`, 9B Q4_K_M, identical flags/prompt) | byte-identical through same stop token | | |

If a re-run of the tuned branch deviates by >5% from reference, or the greedy
outputs no longer match the master build's, something upstream (driver, model,
or ggml-vulkan master state) has changed — check `git log --oneline
TESTING-rdna3..origin/master` and driver version before trusting the numbers.

## Manual steps (the script wrapper is `test-rdna3-r780m.sh`)

### 1. Two fresh builds, adjacent commits
```bash
git fetch
git checkout -q -B base-replay ggml-vk-rdna3-igpu-ltile~1   # upstream state under test
cmake -B build-base -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build-base --config Release -j

git checkout -q ggml-vk-rdna3-igpu-ltile                     # candidate
cmake -B build-tune -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build-tune --config Release -j
ls build-base/bin/llama-cli build-tune/bin/llama-cli         # both must exist
```

### 2. Correctness — greedy byte-identical text
Same flags, `-temp 0`, both binaries, same prompt. Compare the generated text
(timing/timestamp lines differ and are expected):
```bash
A="-m models/Qwen3.8-9B-Q4_K_M.gguf -ngl 99 -c 4096 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 -temp 0 -n 60"
P="Continue in exactly two short sentences: The morning mist always hung lowest over the village of Oakhaven. Pip, the small green dragon, lived alone in his cave at the top of the peak."
build-base/bin/llama-cli $A -p "$P" 2>&1 | tee out-base.log
build-tune/bin/llama-cli $A -p "$P" 2>&1 | tee out-tune.log
diff <(sed -n '/\[Prompt/,$p' out-base.log) <(sed -n '/\[Prompt/,$p' out-tune.log) || true   # expect only the timing line to differ
```
Pass = generated tokens identical up to the same stop token.

### 3. Speed — llama-bench (NO `-st` / no flags that one binary rejects — keep both commands byte-identical)
```bash
build-base/bin/llama-bench -m models/Qwen3.8-9B-Q4_K_M.gguf -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 -p 512 -n 64 -r 3 2>&1 | tee bench-base.log
build-tune/bin/llama-bench -m models/Qwen3.8-9B-Q4_K_M.gguf -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 -p 512 -n 64 -r 3 2>&1 | tee bench-tune.log
```

### 4. Context to capture with every run
```bash
git -C . log -1 --oneline; git -C . rev-parse HEAD
vulkaninfo --summary 2>/dev/null || echo "vulkaninfo not available"
```
Build ID banner inside llama-cli/llama-bench output (`build : bXXXXX-<sha>`) is the
authoritative "which tree is this binary" proof.

### 5. Uploading results (keep PR branch clean)
```bash
git checkout -b results-$(date +%Y%m%d)
git add out-base.log out-tune.log bench-base.log bench-tune.log
git commit -m "test results $(date +%Y%m%d): R780M, master vs RDNA3-tuned branch"
git push -f <your-fork-url> HEAD
```

## Pitfalls learned (do not repeat)
- `-st` is NOT a valid llama-bench flag (llama-cli tolerates it — asymmetric parsers); keep flag strings byte-identical between the two binaries.
- `-temp 1` makes outputs differ every run — never use it for the correctness check.
- Fork `master` lags upstream `master`; always baseline against the tuned branch's
  *parent commit* (step 1), not the fork's master.
- Old build dirs (`build/`) can serve stale binaries — verify `build : bXXXXX-<sha>`
  before trusting any number.
