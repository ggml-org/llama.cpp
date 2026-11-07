# Qwen3.8-Flash-Next on Intel Panther Lake (llama.cpp SYCL)

This fork runs Qwen3.8-Flash-Next (Q2_0, 177B MoE) with MTP speculative decoding on the Arc B390 iGPU of Panther Lake. Upstream base: tag `b11836` (commit `f5027fb21`). Branch: `ptl-qwen38`.

## Results (PTL368H, 61 GiB LPDDR5X-8533)

Decode (generation) speed as reported by the server. Prompt processing is not included.

| Setup | Decode tok/s |
|---|---|
| upstream b11836, MTP 5 drafts | ~12-13 |
| this branch, no MTP | 13.5 |
| this branch, MTP 5 drafts (start script) | 24-26 (one long ST prompt), **27.7 over Agents4PLC** |

Accuracy on Agents4PLC (117 tasks, strict pass): 64.1 % (normalised 65.0 %) vs 65.0 % (65.8 %) for upstream. The difference is one task, which is within run-to-run variation; the output text is not deterministic between runs even at temperature 0.

## What the branch changes

Every change is behind an environment variable and is off by default. Without the variables the build behaves like upstream.

| Variable | Files | Effect |
|---|---|---|
| `GGML_SYCL_Q2_0_FAST=1` | `ggml/src/ggml-sycl/mmvq.cpp`, `mmvq.hpp`, `ggml-sycl.cpp` | Faster Q2_0 MoE expert kernel (2.6-2.8x in microbenchmarks): one lane per pair of Q2_0 blocks with dword loads, SoA q8_1 activations. `=file` switches it on only while `/tmp/q2_0_fast` exists (for A/B tests inside one server run). |
| `GGML_SYCL_MMID_SMALL_BATCH=8` | `ggml/src/ggml-sycl/ggml-sycl.cpp` | MoE with up to 8 tokens (MTP verify) runs the fused one-token expert path per token instead of the generic path with a host round trip. |
| `LLAMA_DIO_BOUNCE=1` | `src/llama-mmap.cpp` | With `-lm dio`: direct IO into the pinned SYCL staging buffer fails with EFAULT, and upstream then falls back to buffered IO for the rest of the file. With this variable the load keeps direct IO and reads through a 64 MiB private buffer. |
| (tests) | `tests/test-backend-ops.cpp` | Perf and eval cases for the model's shapes (Q2_0 experts 640x2560 / 2560x640, 512 experts / 10 used, 1-6 tokens; BF16 hyper-connection shapes). |

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
strings build/bin/libllama.so | grep -c "using a bounce buffer"      # 1
```

## 3. Model files

Put the files in one directory (default `~/models`):

```
models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
models/mtp/mtp-Qwen3.8-Flash-Next-Q4_0.gguf
```

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
taskset -c 0-3 build/bin/llama-server \
    -m ~/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    -lm dio -lzm on -ngl 99 -fa on --jinja --reasoning off --no-reasoning-preserve \
    -c 65536 -np 1 -b 4096 -ub 2048 \
    -md ~/models/mtp/mtp-Qwen3.8-Flash-Next-Q4_0.gguf -ngld 99 \
    --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.5 \
    --alias qwen38-flash-next-q2_0-mtp --host 0.0.0.0 --port 8601
```

Notes:

- `taskset -c 0-3` pins the server to the P-cores. Unpinned runs varied 4-10 tok/s because the host thread landed on slow E-cores.
- `-lm dio` must go together with `LLAMA_DIO_BOUNCE=1`. The log then shows `direct IO into this buffer failed (Bad address), using a bounce buffer` once per model file; that is expected.
- `-lzm on` reads the 27 GB n-gram embedding table (`per_layer_token_embd`) from disk on demand instead of keeping it in RAM.
- `-c 65536` leaves ~14 GB of RAM free after the load on a 61 GiB machine. The model is served as `qwen38-flash-next-q2_0-mtp`.
- Prompt processing is the slow part: ~160 tok/s on a 24k-token prompt (~150 s before the first token). Follow-up turns reuse the cached prefix.
- Run only one GPU job (server, benchmark, test) at a time.

## 5. Check correctness (optional)

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_Q2_0_FAST=1 GGML_SYCL_MMID_SMALL_BATCH=8
build/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0
```

All Q2_0 `MUL_MAT_ID` cases should pass against the CPU backend.

## Known issues

- The first server launch after a reboot was ~17 % slower in one test (plain decode 11.3 vs 13.5 tok/s with the Q2_0 kernel); later launches were at full speed. Not yet explained.
- Output text differs between runs even at temperature 0 (non-deterministic GPU reductions), so compare accuracy with a benchmark, not with identical text.
