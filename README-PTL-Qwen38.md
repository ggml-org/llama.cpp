# Qwen3.8-Flash-Next on Intel Panther Lake (llama.cpp SYCL)

This fork runs Qwen3.8-Flash-Next (Q2_0, 177B MoE) with MTP speculative decoding on the Arc B390 iGPU of Panther Lake. Upstream base: tag `b11836` (commit `f5027fb21`). Branch: `ptl-qwen38`.

## Results (PTL368H, 61 GiB LPDDR5X-8533)

Decode (generation) speed as reported by the server.

| Setup | Decode tok/s | Agents4PLC strict (normalised) |
|---|---|---|
| upstream b11836, MTP 5 drafts | ~12-13 | 65.0 % (65.8 %) |
| 7 Nov: Q2_0 kernel + small-batch MoE + direct IO, MTP (`NEW=0`) | 27.7 over Agents4PLC | 64.1 % (65.0 %) |
| **8 Nov: + Q8_0 hyper-connections, prefill on XMX, sparse attention (start script default)** | **30.1 over Agents4PLC** | **66.7 % (67.5 %)** |

Accuracy differences of 1-2 tasks are within run-to-run variation (5 tasks better, 3 worse than upstream in the 8 Nov run); the output text is not deterministic between runs even at temperature 0.

Long contexts (opencode-style CODESYS sessions), synthetic ST prompt, MTP, 8 Nov setup:

| Prompt | Prefill tok/s | Decode tok/s (without sparse attention) |
|---|---|---|
| 7.8k tokens | 232 | 25.6 (25.1) |
| 14k tokens | 238 | 28.0 (27.3) |
| 34k tokens | 224 | 28.3 (23.8) |

Without the two prefill switches, prefill was 139-152 tok/s at 23-34k tokens.

## What the branch changes

Every change is behind an environment variable and is off by default. Without the variables the build behaves like upstream.

| Variable | Files | Effect |
|---|---|---|
| `GGML_SYCL_Q2_0_FAST=1` | `ggml/src/ggml-sycl/mmvq.cpp`, `mmvq.hpp`, `ggml-sycl.cpp` | Faster Q2_0 MoE expert kernel (2.6-2.8x in microbenchmarks): one lane per pair of Q2_0 blocks with dword loads, SoA q8_1 activations. `=file` switches it on only while `/tmp/q2_0_fast` exists (for A/B tests inside one server run). |
| `GGML_SYCL_MMID_SMALL_BATCH=8` | `ggml/src/ggml-sycl/ggml-sycl.cpp` | MoE with up to 8 tokens (MTP verify) runs the fused one-token expert path per token instead of the generic path with a host round trip. |
| `LLAMA_DIO_BOUNCE=1` | `src/llama-mmap.cpp` | With `-lm dio`: direct IO into the pinned SYCL staging buffer fails with EFAULT, and upstream then falls back to buffered IO for the rest of the file. With this variable the load keeps direct IO and reads through a 64 MiB private buffer. |
| `GGML_SYCL_BF16_FAST=1` | `ggml/src/ggml-sycl/dmmv.cpp`, `dmmv.hpp`, `ggml-sycl.cpp` | BF16 mat-vec for 1-8 tokens with 16-byte loads and split-K over sub-groups (the skinny hyper-connection shapes). |
| `GGML_SYCL_TOP_K_MASK=1` | `ggml/src/ggml-sycl/ggml-sycl.cpp` | SYCL port of the fused sparse-attention mask op (`GGML_OP_TOP_K_MASK`). |
| `GGML_SYCL_XMX_Q2_0=1` with `GGML_SYCL_DYNAMIC_PRECISION=BF16` | `ggml/src/ggml-sycl/fused-gemm.cpp`, `fused-gemm.hpp` | Prefill: Q2_0 experts go through the grouped XMX dequant-GEMM (one launch for all experts, bf16 operands, f32 accumulate) instead of dequantizing every expert to f32 for a library GEMM. This build has `GGML_SYCL_F16=OFF`, so the XMX paths are off unless `GGML_SYCL_DYNAMIC_PRECISION` is set. F16 is slightly faster but changed tokens; use BF16. Keep `-ub 2048`. |
| `GGML_SYCL_DNNL_FPMATH=bf16` | `ggml/src/ggml-sycl/gemm.hpp` | Prefill: oneDNN f32 GEMMs use bf16 math on the XMX units (f32 accumulate), ~1.4x. |
| `GGML_SYCL_SPARSE_FA=1`, `GGML_SYCL_SPARSE_FA_ROWS=8` | `ggml/src/ggml-sycl/fattn-sparse.cpp` | Decode: attention runs only over the KV cells the sparse indexer picked (2048 per token) instead of the whole context. `SPARSE_FA` existed for single-token decode; `SPARSE_FA_ROWS` extends it to the MTP verify rows (union of their cells). Needs `GGML_SYCL_FA_SKIP_MASKED=1`. |
| `GGML_SYCL_FA_SKIP_MASKED=1` | `ggml/src/ggml-sycl/fattn-tile.hpp` | Flash attention skips KV tiles whose mask is -inf for every row of the block. |
| (tests) | `tests/test-backend-ops.cpp` | Perf and eval cases for the model's shapes: Q2_0 experts 640x2560 / 2560x640 (512 experts / 10 used, 1-2048 tokens), BF16 hyper-connection shapes, sparse attention with 2-8 rows. |
| (tool) | `scripts/ptl/bf16_to_q8.cpp` | Rewrites the first model split with the hyper-connection and router tensors (240 tensors) in Q8_0 instead of BF16. |

Why `-lm dio` + `LLAMA_DIO_BOUNCE=1` matters: the model is ~62 GB on a 61 GiB machine. An mmap load fills the page cache, the kernel reclaims 6-15 M pages during the load, and the GPU weight buffer, which is already allocated, comes out of that with scattered small pages and reads up to 3x slower (26-60 GB/s instead of ~110 GB/s). That made some server launches 2x slower than others. With direct IO the load does not touch the page cache: 12 of 12 launches were fast, also after a reboot, and the load takes ~30 s instead of 2-5 min. No root setting is needed.

## Requirements

Tested with:

- Intel Panther Lake with Arc B390 iGPU, 61 GiB usable RAM
- Ubuntu 26.04, kernel 7.0.0-42-generic (xe driver)
- Intel compute runtime 26.31.39395.13 (`intel-opencl-icd`, `libze-intel-gpu1`), Level Zero loader 1.33.1
- Intel oneAPI DPC++/C++ Compiler 2026.1.1 (with oneMKL and oneDNN), installed under `/opt/intel/oneapi`
- ~70 GB free disk for the model files

Check that the GPU is visible:

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls          # should list a level_zero:gpu device (Intel Arc B390)
```

## 1. Get the code

Option A, clone the fork (recommended):

```bash
git clone https://github.com/shailesh837/llama.cpp.git
cd llama.cpp
git checkout ptl-qwen38
```

Option B, apply the patch to upstream b11836:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b11836
git apply --check /path/to/qwen38-ptl.patch && git apply /path/to/qwen38-ptl.patch
```

The patch file is `scripts/ptl/qwen38-ptl.patch` in the fork. It may not apply to newer upstream versions without changes.

## 2. Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build \
    -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-server test-backend-ops
```

Defaults used (no need to set them): `GGML_SYCL_F16=OFF`, `GGML_SYCL_GRAPH=ON`, `GGML_SYCL_DNN=ON`, `GGML_SYCL_DEVICE_ARCH` empty (kernels are JIT-compiled by the driver), `GGML_NATIVE=ON`, `BUILD_SHARED_LIBS=ON`.

Optional check that the branch is in the binary:

```bash
strings build/bin/libggml-sycl.so | grep -c GGML_SYCL_Q2_0_FAST     # 1
strings build/bin/libggml-sycl.so | grep -c GGML_SYCL_SPARSE_FA_ROWS # 1
strings build/bin/libllama.so | grep -c "using a bounce buffer"      # 1
```

## 3. Model files

Put the files in one directory (default `~/models`):

```
models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
models/mtp/mtp-Qwen3.8-Flash-Next-Q4_0.gguf
```

Then make the Q8_0 variant the start script uses (`models/Q2_0_hcq8/`). Only the first split changes (240 hyper-connection and router tensors, BF16 -> Q8_0); the second split is linked. Needs ~35 GB free disk.

```bash
source /opt/intel/oneapi/setvars.sh
icpx -std=c++17 -O2 -Iggml/include scripts/ptl/bf16_to_q8.cpp -o build/bin/bf16_to_q8 \
    -Lbuild/bin -lggml-base -Wl,-rpath,$PWD/build/bin
mkdir -p ~/models/Q2_0_hcq8
build/bin/bf16_to_q8 --dry-run ~/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
build/bin/bf16_to_q8 --write ~/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    ~/models/Q2_0_hcq8/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
ln ~/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf ~/models/Q2_0_hcq8/
```

`ln` makes a hard link (same file system); use `cp` otherwise. `--stats` prints the round-trip error per tensor group without writing anything.

## 4. Run

```bash
MODEL_DIR=~/models HOST=0.0.0.0 PORT=8601 scripts/ptl/start-qwen38-server.sh
```

The script prints the PID and the log file and returns when the server is ready (about 30 s). It is an OpenAI-compatible endpoint: `http://<host>:8601/v1`. Stop it with `kill <pid>` and wait at least 30 s before starting it again (fast restarts can cause a GPU reset).

What the script sets, if you want to start the server yourself:

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_LOADER_USE_LEVEL_ZERO_V2=0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export GGML_SYCL_Q2_0_FAST=1 GGML_SYCL_MMID_SMALL_BATCH=8 LLAMA_DIO_BOUNCE=1
export GGML_SYCL_BF16_FAST=1 GGML_SYCL_TOP_K_MASK=1
export GGML_SYCL_XMX_Q2_0=1 GGML_SYCL_DYNAMIC_PRECISION=BF16 GGML_SYCL_DNNL_FPMATH=bf16
export GGML_SYCL_SPARSE_FA=1 GGML_SYCL_SPARSE_FA_ROWS=8 GGML_SYCL_FA_SKIP_MASKED=1
taskset -c 0-3 build/bin/llama-server \
    -m ~/models/Q2_0_hcq8/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    -lm dio -lzm on -ngl 99 -fa on --jinja --reasoning off --no-reasoning-preserve \
    -c 65536 -np 1 -b 4096 -ub 2048 \
    -md ~/models/mtp/mtp-Qwen3.8-Flash-Next-Q4_0.gguf -ngld 99 \
    --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.5 \
    -cram 0 --alias qwen38-flash-next-q2_0-mtp --host 0.0.0.0 --port 8601
```

Notes:

- `taskset -c 0-3` pins the server to the P-cores. Unpinned runs varied 4-10 tok/s because the host thread landed on slow E-cores.
- `-lm dio` must go together with `LLAMA_DIO_BOUNCE=1`. The log then shows `direct IO into this buffer failed (Bad address), using a bounce buffer` once per model file; that is expected.
- `-lzm on` reads the 27 GB n-gram embedding table (`per_layer_token_embd`) from disk on demand instead of keeping it in RAM.
- `-c 65536` leaves ~14 GB of RAM free after the load on a 61 GiB machine. The model is served as `qwen38-flash-next-q2_0-mtp`.
- Prompt processing: ~225 tok/s on long prompts (34k tokens in ~150 s). Follow-up turns reuse the cached prefix, so only new content (file reads, tool results) is processed.
- `-cram 0` turns off the host-RAM prompt cache, which grew 120-230 MB per request until 8 GB and pushed the machine into swap with `-c 65536`. SYCL graphs (`GGML_SYCL_ENABLE_GRAPH`) stay off: with MTP no graph is replayed and the graph cache grows ~110 MB per request.
- Run only one GPU job (server, benchmark, test) at a time.

## 5. Check correctness (optional)

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_Q2_0_FAST=1 GGML_SYCL_MMID_SMALL_BATCH=8
export GGML_SYCL_BF16_FAST=1 GGML_SYCL_XMX_Q2_0=1 GGML_SYCL_DYNAMIC_PRECISION=BF16 GGML_SYCL_DNNL_FPMATH=bf16
export GGML_SYCL_SPARSE_FA=1 GGML_SYCL_SPARSE_FA_ROWS=8 GGML_SYCL_FA_SKIP_MASKED=1
build/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0 -p "type_a=q2_0"
build/bin/test-backend-ops test -o MUL_MAT -b SYCL0 -p "type_a=(q3_K|iq4_xs|q4_K|q5_K|q6_K|q8_0|bf16|f32),type_b=f32"
build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b SYCL0 -p "hsk=256"
```

All cases should pass against the CPU backend, except two `FLASH_ATTN_EXT` cases that also fail with every switch off (not caused by this branch's switches) (`nr23=[16,1],kv=1025,nb=64` with q8_0 cache and `nr23=[12,2],kv=8192,nb=67`).

## Known issues

- The first server launch after a reboot was ~17 % slower in one test (plain decode 11.3 vs 13.5 tok/s with the Q2_0 kernel); later launches were at full speed. Not yet explained.
- Output text differs between runs even at temperature 0 (non-deterministic GPU reductions), so compare accuracy with a benchmark, not with identical text.
