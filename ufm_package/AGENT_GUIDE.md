# UFM Fixes — Agent Integration Guide
### For VS Code Copilot · Cursor · Google Gemini CLI · Any AI Agent

---

## What this guide does

An AI agent following this guide will:
1. Pull a fresh `llama.cpp` at the pinned commit
2. Apply fixes from the `fixes/` folder **one at a time**
3. Run a validation check after **each addition**
4. Compile the final result
5. Run a quick sanity benchmark

Every step has a **VALIDATE** block. The agent must confirm the validation
passes before moving to the next step. If a step fails, instructions for
diagnosis are provided inline.

---

## Prerequisites (confirm before starting)

```
Windows 11 with:
  - Git installed             (winget install Git.Git)
  - CMake 3.26+               (winget install Kitware.CMake)
  - VS 2022 Build Tools       (C++ workload, x64)
  - Vulkan SDK 1.3+           (https://vulkan.lunarg.com → glslc must be on PATH)
  - Python 3.10+
  - ROCm 7.2.1 installed

Needs installed for compiling :
  - Git
  - CMake
  - VS 2022 Build Tools  (or MSVC)
  - Vulkan SDK           (for glslc shader compiler)
  - Python 3.10+

Does NOT need for compiling :
  - ROCm
  - HIP SDK
  - AMD drivers of any kind

RX 9070 XT machine (running)
Needs installed:
  - AMD Adrenalin driver  (ships with Vulkan support built-in)
  - The compiled llama.cpp binary from the Intel laptop
  - Models (.gguf files)

Does NOT need:
  - ROCm (unless you want the HIP backend instead)
  - Vulkan SDK
  - glslc

GPU: AMD Radeon RX 9070 XT (gfx1201)
VRAM: 16GB
```

**Agent: verify these before Step 1:**
```bat
git --version
cmake --version
glslc --version
python --version
```
All must return without error. If glslc is missing, Vulkan SDK is not installed
or not on PATH. Fix before continuing.

---

## Folder layout you should have

```
fixes/                          ← this folder, alongside AGENT_GUIDE.md
├── shaders/
│   ├── flash_attention_kv_quant.glsl
│   ├── flash_attention_paged.glsl
│   ├── kvcache_update_q8.glsl
│   ├── linear_coop_32.glsl
│   ├── linear_coop_fp8.glsl
│   ├── linear_coop_q4k.glsl
│   ├── linear_coop_q4k_silu.glsl
│   ├── linear_coop_q4k_w32.glsl       ← wave32 A/B variant
│   ├── linear_coop_q8.glsl
│   └── linear_coop_q8_w32.glsl        ← wave32 A/B variant
├── src/
│   ├── CMakeLists_append.cmake
│   ├── ggml_fp8.patch
│   ├── ggml_vulkan_custom_kernels.patch
│   ├── gguf_kv_cache_meta.txt
│   ├── paged_kv_cache.h
│   ├── paged_kv_eviction.h
│   └── paged_kv_upload.h
├── validation/
│   ├── validate_fp8.py
│   └── validate_shaders.py
└── scripts/
    ├── compile_shaders.bat
    ├── convert_to_fp8.py
    ├── ingest.py
    ├── query.py
    └── update.py
```

---

## STEP 0 — Clone llama.cpp at pinned commit

```bat
cd C:\dev
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b8665
```

**VALIDATE Step 0:**
```bat
git log -1 --format="%H %s"
```
Expected: line starting with `9c69907`

If wrong commit: `git checkout b8665` again. If b8665 not found:
`git fetch origin && git checkout b8665`

---

## STEP 1 — Copy the fixes folder into llama.cpp

Copy the entire `fixes/` folder (containing shaders, src, validation, scripts)
into the llama.cpp root directory.

Result:
```
llama.cpp/
├── ggml/
├── src/
├── fixes/        ← newly added
└── ...
```

**VALIDATE Step 1:**
```bat
python fixes\validation\validate_shaders.py
```
Expected output: `ALL PASS ✓` with 19 checks passing.

If any check fails, do NOT continue. The fix file is corrupt or missing.
Re-copy the fixes folder and retry.

---

## STEP 2 — Compile GLSL shaders to SPIR-V

```bat
cd C:\dev\llama.cpp
fixes\scripts\compile_shaders.bat fixes\
```

This compiles all 10 shaders and embeds them as C headers in:
`ggml\src\ggml-vulkan\spv_out\`

**VALIDATE Step 2:**
```bat
dir ggml\src\ggml-vulkan\spv_out\
```
Expected: 10 files ending in `_spv.h`

```
flash_attention_kv_quant_spv.h
flash_attention_paged_spv.h
kvcache_update_q8_spv.h
linear_coop_32_spv.h
linear_coop_fp8_spv.h
linear_coop_q4k_spv.h
linear_coop_q4k_silu_spv.h
linear_coop_q4k_w32_spv.h
linear_coop_q8_spv.h
linear_coop_q8_w32_spv.h
```

If any are missing: check that glslc is on PATH and re-run.
If a shader fails to compile: the error message names the file and line.
Do NOT continue until all 10 are present.

---

## STEP 3 — Apply ggml-vulkan dispatch patch

```bat
cd C:\dev\llama.cpp
git apply fixes\src\ggml_vulkan_custom_kernels.patch
```

**VALIDATE Step 3:**
```bat
git diff --stat
```
Expected: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` listed as modified.

If the patch rejects (hunk failures):
1. Open `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in VS Code
2. Open `fixes/src/ggml_vulkan_custom_kernels.patch` alongside
3. Each hunk has an `--- Anchor:` comment — search for that text in the .cpp
4. Apply the `+` lines manually at the anchor location
5. After manual application, verify: `git diff ggml/src/ggml-vulkan/ggml-vulkan.cpp`
   must show changes

---

## STEP 4 — Add custom kernel includes to ggml-vulkan.cpp

Open `ggml/src/ggml-vulkan/ggml-vulkan.cpp`.
Find the block where other SPV headers are included (search for `#include "vulkan_shaders/`).
Add these lines in the same block:

```cpp
#if defined(UFM_CUSTOM_KERNELS)
#include "spv_out/flash_attention_kv_quant_spv.h"
#include "spv_out/flash_attention_paged_spv.h"
#include "spv_out/kvcache_update_q8_spv.h"
#include "spv_out/linear_coop_32_spv.h"
#include "spv_out/linear_coop_fp8_spv.h"
#include "spv_out/linear_coop_q4k_spv.h"
#include "spv_out/linear_coop_q4k_silu_spv.h"
#include "spv_out/linear_coop_q4k_w32_spv.h"
#include "spv_out/linear_coop_q8_spv.h"
#include "spv_out/linear_coop_q8_w32_spv.h"
#endif
```

**VALIDATE Step 4:**
```bat
findstr /c:"linear_coop_q4k_spv.h" ggml\src\ggml-vulkan\ggml-vulkan.cpp
```
Expected: one line printed showing the include.

---

## STEP 5 — Append CMakeLists.txt

Open `ggml/src/ggml-vulkan/CMakeLists.txt` in VS Code.
Scroll to the very bottom. Paste the contents of `fixes/src/CMakeLists_append.cmake`.

That is exactly these 6 lines:
```cmake
set(UFM_SPV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spv_out")
if(EXISTS "${UFM_SPV_DIR}")
    target_compile_definitions(ggml-vulkan PRIVATE UFM_CUSTOM_KERNELS=1)
    target_include_directories(ggml-vulkan PRIVATE "${UFM_SPV_DIR}")
    message(STATUS "[UFM] Custom kernels ENABLED — ${UFM_SPV_DIR}")
else()
    message(WARNING "[UFM] spv_out/ not found — run scripts/compile_shaders.bat first")
endif()
```

**VALIDATE Step 5:**
```bat
findstr /c:"UFM_CUSTOM_KERNELS" ggml\src\ggml-vulkan\CMakeLists.txt
```
Expected: one line containing `UFM_CUSTOM_KERNELS`.

---

## STEP 6 — Copy paged KV cache headers

```bat
copy fixes\src\paged_kv_cache.h     ggml\src\ggml-vulkan\
copy fixes\src\paged_kv_upload.h    ggml\src\ggml-vulkan\
copy fixes\src\paged_kv_eviction.h  ggml\src\ggml-vulkan\
```

**VALIDATE Step 6:**
```bat
dir ggml\src\ggml-vulkan\paged_kv*.h
```
Expected: 3 files listed:
```
paged_kv_cache.h
paged_kv_eviction.h
paged_kv_upload.h
```

---

## STEP 7 — Apply FP8 type registration patch (optional — for FP8 models)

Skip this step if you do not plan to use FP8-quantised models.
It registers `GGML_TYPE_F8_E4M3FN` (type id 40) in ggml.

**The patch is a set of instructions, not a git patch file.**
Open `fixes/src/ggml_fp8.patch` and follow each numbered hunk:

- **Hunk 1**: `ggml/include/ggml.h` — add enum entry after `GGML_TYPE_IQ4_NL`
- **Hunk 2**: `ggml/include/ggml.h` — add traits table entry
- **Hunk 3**: `ggml/src/ggml-cpu/ggml-quants.h` — add function declarations
- **Hunk 4**: `ggml/src/ggml-cpu/ggml-quants.c` — add encode/decode implementation
- **Hunk 5**: `gguf-py/gguf/constants.py` — add Python type registry entry

**VALIDATE Step 7:**
```bat
python fixes\validation\validate_fp8.py
```
Expected: `ALL TESTS PASSED ✓` (12 tests)

This validation runs a pure Python roundtrip test — it does NOT require a
compiled binary. If it fails, check that the FP8 patch overflow boundary
is `exp8 > 15` (not `>= 15`) in ggml-quants.c.

---

## STEP 8 — Verify op_params offsets (one-time calibration)

This step confirms two byte offsets that can shift between llama.cpp commits.
It takes 5 minutes and is required for correct dispatch.

**8a.** Open `ggml/src/ggml-vulkan/ggml-vulkan.cpp`.
Find the function that handles `GGML_OP_FLASH_ATTN_EXT`.
Add this temporary printf at its start:

```cpp
// TEMP CALIBRATION — remove after Step 8
for(int _i=0;_i<8;_i++)
  fprintf(stderr,"[UFM-CAL] FA op_params[%d]=%d\n",_i,((const int32_t*)node->op_params)[_i]);
```

**8b.** Do a quick build (just enough to run):
```bat
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-cli
```

**8c.** Run with any small model:
```bat
build\bin\Release\llama-cli.exe -m any_small_model.gguf -p "test" -n 3 2>&1 | findstr UFM-CAL
```

**8d.** Check the output:
- Find the parameter index where value = `1` (the causal flag). Note this index.
- In the patch, find `((const int32_t*)node->op_params)[4]` and confirm index 4 is correct.
  If the causal flag is at a different index, update the patch hunk that reads the causal flag.

**8e.** Remove the temporary printf. Save the file.

**VALIDATE Step 8:**
```bat
findstr /c:"UFM-CAL" ggml\src\ggml-vulkan\ggml-vulkan.cpp
```
Expected: no output (printf removed).

---

## STEP 9 — Final build

```bat
cd C:\dev\llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2>&1 | tee build_log.txt
```

**VALIDATE Step 9a — CMake configure:**
```bat
findstr /c:"UFM" build_log.txt | findstr /c:"ENABLED"
```
Expected: `[UFM] Custom kernels ENABLED — ...spv_out`

If it says `WARNING: spv_out/ not found`: Step 2 failed. Re-run compile_shaders.bat.

**VALIDATE Step 9b — Compilation:**
```bat
findstr /c:"error" build_log.txt
```
Expected: no C++ compiler errors. CMake may print "error" in other contexts
(e.g. error handling function names) — look specifically for lines with
`error C` (MSVC) or `error:` (clang-cl).

If there are errors: see Troubleshooting section below.

---

## STEP 10 — Smoke test

```bat
build\bin\Release\llama-bench.exe -m YOUR_MODEL.gguf -ngl 99 -r 3 -p 128 -n 128
```

**VALIDATE Step 10:**
Output should show tokens/sec. Compare against a baseline (build without the patch).
Expected improvement: 10-40% on Q4_K_M models depending on matrix sizes.

If tokens/sec is LOWER than baseline: one of the kernel dispatches is likely
hitting a wrong path. Check `build_log.txt` for `[UFM]` log lines at runtime.

---

## STEP 11 — Wave32 A/B test (optional but recommended)

The package includes wave32 variants of the Q4K and Q8 shaders.
These may be faster on RDNA4 due to single-issue execution and dual-issue
parallelism. Test both and keep whichever is faster.

**11a.** Build with wave32 dispatch enabled:
In `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, find where the custom kernels
are dispatched (search for `UFM_CUSTOM_KERNELS`).
There is a `#define UFM_USE_WAVE32 0` — change it to `1`.

**11b.** Rebuild:
```bat
cmake --build build --config Release -j
```

**11c.** Benchmark:
```bat
build\bin\Release\llama-bench.exe -m YOUR_MODEL.gguf -ngl 99 -r 5 -p 512 -n 128
```

**11d.** Compare wave64 vs wave32 results.
If wave32 is faster: keep `UFM_USE_WAVE32 1`.
If wave64 is faster: revert to `UFM_USE_WAVE32 0`.

**VALIDATE Step 11:**
Both builds must complete without errors. The output tokens/sec number is
the validation — whichever is higher wins.

---

## Troubleshooting

### Patch rejects (git apply fails)
The patch was written against b8665. If your checkout differs, apply manually:
1. Open the .patch file
2. Find each `--- Anchor:` comment
3. Search for that text in the target .cpp file
4. Apply the `+` lines at that location

### Undefined symbol errors at compile
Most likely the spv_out headers are not being included.
Check: `findstr /c:"UFM_CUSTOM_KERNELS" ggml\src\ggml-vulkan\ggml-vulkan.cpp`
Must return the include block from Step 4.

### Shader compile errors (glslc fails)
Run glslc manually on the failing shader:
```bat
glslc --target-env=vulkan1.2 -fshader-stage=compute ^
      fixes\shaders\failing_shader.glsl -o NUL
```
The error message will point to the line. Do NOT ignore shader errors —
a broken shader silently produces wrong results, it doesn't crash.

### Push constant struct mismatch
If you see garbage output or crashes in attention layers:
The PC struct in ggml-vulkan.cpp must match the `layout(push_constant)` block
in the shader exactly — same fields, same order, same types.
Print `sizeof(pc_struct)` and compare to the shader's push constant range.

### validate_shaders.py reports failures after Step 2
The wave32 shaders are only validated for structure, not compiled output.
If the check `Wave32 shader must use local_size_x = 32` fails:
The file was not copied correctly. Recheck fixes\shaders\.

---

## Post-build: RAG setup for coding assistance

After the build succeeds, index the technical docs to get accurate answers
about ROCm, Vulkan, and llama.cpp internals:

```bat
pip install chromadb requests beautifulsoup4 ollama pyyaml
ollama pull nomic-embed-text
ollama pull gemma4:26b
python fixes\scripts\ingest.py
```

Query:
```bat
python fixes\scripts\query.py --interactive
```

Update docs weekly:
```bat
python fixes\scripts\update.py
```

---

## Fix summary — what each file does

| File | What changed | Why |
|---|---|---|
| `flash_attention_kv_quant.glsl` | subgroupMax/Add for softmax tile | Was 128×16 serial ops, now 1 |
| `flash_attention_paged.glsl` | Same fix + block table indirection | Enables 32K+ context |
| `linear_coop_q4k.glsl` | Scale hoisted outside inner loop | 16x fewer header reads |
| `linear_coop_q4k_silu.glsl` | Same scale fix, fused SiLU gate | 1 dispatch instead of 3 |
| `linear_coop_q4k_w32.glsl` | Wave32 variant | A/B test — may be faster |
| `linear_coop_q8.glsl` | fp32 coopmat accumulator | Registers not shared mem |
| `linear_coop_q8_w32.glsl` | Wave32 + fp32 accumulator | A/B test |
| `linear_coop_32.glsl` | 32×32 tiled FP16 GEMM | Better matrix core use |
| `linear_coop_fp8.glsl` | FP8 decoder (exp>15 overflow fix) | Was encoding 256 as 448 |
| `kvcache_update_q8.glsl` | INT8 KV write | Online KV compression |
| `paged_kv_cache.h` | O(log n) LRU heap | Was O(n) scan |
| `paged_kv_eviction.h` | Async eviction ring (4 fences) | Was blocking per copy |
| `paged_kv_upload.h` | Block table upload bridge | CPU→GPU each layer |
| `ggml_fp8.patch` | FP8 type id 40 in ggml | For FP8 GGUF models |
| `CMakeLists_append.cmake` | UFM_CUSTOM_KERNELS flag | Activates all of the above |

---

## All-RX-9070-XT-are-the-same note

Every RX 9070 XT ships as gfx1201, 64 CUs, 16GB GDDR6, 640 GB/s bandwidth.
The wave32 vs wave64 decision and tile sizing are hardware-specific to this die.
TILE_SIZE=16 matches the hardware WMMA instruction size exactly (confirmed from
AMD GPUOpen RDNA4 article). There is no 32×32 native WMMA — that would require
four 16×16 operations, which `linear_coop_32.glsl` already does.
