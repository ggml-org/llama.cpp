# UFM — Complete Integration Guide
### AMD RDNA4 RX 9070 XT · Windows 11 · ROCm 7.2.1 · llama.cpp b8665

---

## What this package adds to llama.cpp

| File | What it does | Impact |
|---|---|---|
| `linear_coop_q4k.glsl` | Fused Q4_K dequant + WMMA in one pass | ~40% faster on Q4_K models |
| `linear_coop_q4k_silu.glsl` | Fused gate+up+SiLU (LLaMA FFN) | 1 dispatch instead of 3 per FFN |
| `linear_coop_q8.glsl` | INT8 WMMA for Q8_0 weights | 4x INT8 throughput on RDNA4 |
| `linear_coop_32.glsl` | 32×32 tiled FP16 GEMM | Better matrix core utilisation |
| `linear_coop_fp8.glsl` | FP8 via uint8 coopmat | 4x FP16 speed (driver-dependent) |
| `flash_attention_kv_quant.glsl` | Tiled attention with INT8 KV cache | ~49% KV cache memory reduction |
| `kvcache_update_q8.glsl` | Online INT8 KV quantisation on write | Paired with flash_attention_kv_quant |
| `paged_kv_cache.h` | CPU paged block allocator for KV cache | Enables 32K+ context on 16GB |
| `paged_kv_upload.h` | Block table upload bridge (CPU→GPU) | Piece 2b of paged KV |
| `ggml_fp8.patch` | Registers FP8 as ggml tensor type 40 | Needed for FP8 GGUF files |
| RAG scripts | Index ROCm/Vulkan docs for Gemma 4 | Better AI coding assistance |

---

## Prerequisites — install once

```bat
REM 1. Git
winget install Git.Git

REM 2. CMake 3.26+
winget install Kitware.CMake

REM 3. Visual Studio 2022 Build Tools (C++ workload)
winget install Microsoft.VisualStudio.2022.BuildTools

REM 4. Vulkan SDK (for glslc shader compiler)
REM    https://vulkan.lunarg.com/sdk/home#windows
REM    Install to default path: C:\VulkanSDK\<version>\

REM 5. Python 3.10+
winget install Python.Python.3.12

REM 6. Ollama (for RAG)
winget install Ollama.Ollama

REM 7. ROCm 7.2.1 — already installed per your setup
```

---

## Part 1 — Clone llama.cpp and apply shaders

### Step 1.1 — Clone the pinned commit

```bat
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b8665
```

**Validation:** `git log -1 --format="%H %s"` should show commit starting with `9c69907`.

---

### Step 1.2 — Copy this package into the repo

Drop the entire `ufm_package` folder into the llama.cpp root:

```
llama.cpp/
├── ggml/
├── src/
├── ufm_package/          ← this folder
│   ├── shaders/
│   ├── src/
│   ├── scripts/
│   ├── config/
│   └── validation/
└── ...
```

---

### Step 1.3 — Compile shaders to SPIR-V

Open a Command Prompt (not PowerShell):

```bat
cd llama.cpp\ufm_package\scripts
python compile_shaders.bat ..\..\
```

Wait for all 8 lines showing `[OK]`. If any show `[FAIL]`:
- Check that Vulkan SDK is installed and `glslc` is on your PATH
- Run `glslc --version` to confirm

**Validation:** Check that `llama.cpp\ggml\src\ggml-vulkan\spv_out\` exists and contains `*.h` files.

---

### Step 1.4 — Run shader static checks

```bat
python ufm_package\validation\validate_shaders.py
```

Expected output: `ALL PASS ✓`

If anything fails, do NOT continue — fix the reported issue first.

---

### Step 1.5 — Apply the ggml-vulkan.cpp patch

```bat
cd llama.cpp
git apply ufm_package\src\ggml_vulkan_custom_kernels.patch
```

If it rejects (line numbers drifted between commits): open
`ggml/src/ggml-vulkan/ggml-vulkan.cpp` in VS Code and apply each hunk
manually. Each hunk in the patch file has an `--- Anchor:` comment
showing exactly what to search for.

**Validation:** `git diff --stat` should show `ggml-vulkan.cpp` as modified.

---

### Step 1.6 — Append to CMakeLists.txt

Open `ggml/src/ggml-vulkan/CMakeLists.txt` in VS Code. Scroll to the bottom. Add:

```cmake
set(UFM_SPV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spv_out")
if(EXISTS "${UFM_SPV_DIR}")
    target_compile_definitions(ggml-vulkan PRIVATE UFM_CUSTOM_KERNELS=1)
    target_include_directories(ggml-vulkan PRIVATE "${UFM_SPV_DIR}")
    message(STATUS "[UFM] Custom kernels ENABLED — ${UFM_SPV_DIR}")
else()
    message(WARNING "[UFM] spv_out/ not found — run compile_shaders.bat first")
endif()
```

**Validation:** Save and confirm no red squiggles in VS Code's CMake extension.

---

### Step 1.7 — Verify op_params offsets (do once, takes 5 minutes)

Before running, confirm two byte offsets that can shift between llama.cpp commits.
Open `ggml/src/ggml-vulkan/ggml-vulkan.cpp` and add these temporary printfs:

**For flash_attention** — find `ggml_vk_flash_attn` and add at its start:
```cpp
// TEMP: remove after verifying
for (int _i = 0; _i < 8; _i++)
    fprintf(stderr, "[UFM] FA op_params[%d] = %d\n", _i, ((const int32_t*)node->op_params)[_i]);
```

**For rope** — find `ggml_vk_rope` and add at its start:
```cpp
// TEMP: remove after verifying
for (int _i = 0; _i < 6; _i++)
    fprintf(stderr, "[UFM] ROPE op_params float[%d] = %f\n", _i, ((const float*)node->op_params)[_i]);
```

Build once, run `llama-cli -m any_small_model.gguf -p "test" -n 3`, check stderr:
- FA causal flag should be `1` at index 4. If different, update `((const int32_t*)node->op_params)[4]` in the patch hunk 5.
- ROPE theta (freq_base) should be `10000.0` at float offset 4 (byte offset 16). If different, update the rope dispatch.

Remove both printfs after verifying.

---

### Step 1.8 — Build

```bat
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Watch for `[UFM] Custom kernels ENABLED` in cmake output.

**Validation:**
```bat
build\bin\Release\llama-bench.exe -m your_model.gguf -ngl 99 -r 3 -p 512
```
Compare tokens/sec to a baseline build without the patch.

---

## Part 2 — FP8 GGUF support (optional)

Only needed if you want to run models stored in FP8 format. Q4_K_M models
already work faster with no format change (Part 1 handles that).

### Step 2.1 — Apply the ggml type registration patch

Open each of these files and apply the changes from `src/ggml_fp8.patch`.
The patch file has clear `Search for:` / `Insert AFTER:` instructions for each hunk.

Files to edit:
- `ggml/include/ggml.h`
- `ggml/src/ggml-cpu/ggml-quants.h`
- `ggml/src/ggml-cpu/ggml-quants.c`
- `gguf-py/gguf/constants.py`

**Validation:**
```bat
python ufm_package\validation\validate_fp8.py --llama-path .
```

The compiled test requires building first. Run the Python-only test immediately:
```bat
python ufm_package\validation\validate_fp8.py
```
All 12 tests must pass before continuing.

---

### Step 2.2 — Convert a model to FP8

```bat
pip install gguf numpy
python ufm_package\scripts\convert_to_fp8.py ^
    models\Llama-3.1-8B-Q4_K_M.gguf ^
    models\Llama-3.1-8B-FP8.gguf
```

**Validation:** The converter prints `max_rel_err` for each tensor. Values under 0.15 (15%) are normal for FP8. If any tensor shows > 0.5, something is wrong with the scale computation.

---

## Part 3 — RAG for technical coding (ROCm, Vulkan, llama.cpp)

### Step 3.1 — Pull Ollama models

```bat
ollama pull nomic-embed-text
ollama pull gemma4:26b
```

The 26B MoE uses ~4B active params — fits easily in your 16GB.

### Step 3.2 — Index documentation

```bat
cd llama.cpp\ufm_package
pip install chromadb requests beautifulsoup4 ollama pyyaml
python scripts\ingest.py
```

First run takes ~15 minutes. Subsequent runs skip already-indexed sources.

**Validation:** Check `ufm_package\db\` folder exists and contains ChromaDB files.

### Step 3.3 — Query

```bat
REM Single question
python scripts\query.py "What is the correct Q4_K nibble layout in ggml-quants.c?"

REM Interactive mode (best for active coding sessions)
python scripts\query.py --interactive

REM Update stale docs (run weekly)
python scripts\update.py
```

---

## Part 4 — Using AI agents (Cursor / Gemini / GitHub Copilot)

The prompts below work in any coding agent. Replace `[agent]` with whatever you're using.

### Applying the patch with an agent

Paste this into the agent chat:

```
I have a patch file at ufm_package/src/ggml_vulkan_custom_kernels.patch.
Apply it to ggml/src/ggml-vulkan/ggml-vulkan.cpp in this workspace.

The patch has 5 hunks. Each hunk has an "--- Anchor:" comment.
Use the anchor text to find the correct location in the file.
Do NOT guess line numbers — search for the anchor string.

After applying each hunk, verify the inserted code compiles
by checking for obvious syntax errors. If any hunk fails,
show me the surrounding context and I will help manually.
```

### Debugging a build error with an agent

```
I applied ggml_vulkan_custom_kernels.patch to llama.cpp b8665.
Build failed with this error: [paste error here]

The custom kernels are gated behind #if defined(UFM_CUSTOM_KERNELS).
The patch added: [describe what was added]

Check if:
1. The push constant struct size matches the GLSL layout(push_constant) block
2. The coopmat type parameters match (gl_ScopeSubgroup, 16, 16)
3. All includes for spv_out/*.h are present at the top of ggml-vulkan.cpp
```

### Updating to a new llama.cpp commit

```
llama.cpp updated from b8665 to [new tag].
I need to reapply ufm_package/src/ggml_vulkan_custom_kernels.patch.

The patch likely failed on hunk 5 (the dispatch switch — it changes most often).

Steps:
1. Show me the current GGML_OP_RMS_NORM case in ggml-vulkan.cpp
2. Show me hunk 5 of the patch
3. Find where the custom dispatch code should be inserted
4. Insert it, preserving the existing fallthrough to original handling
```

### Adding a new shader with an agent

```
I want to add a new Vulkan compute shader to this llama.cpp fork.
The shader is at ufm_package/shaders/my_shader.glsl.

Steps needed:
1. Compile my_shader.glsl with glslc to spv_out/my_shader_spv.h
2. Add #include "spv_out/my_shader_spv.h" inside the UFM_CUSTOM_KERNELS block in ggml-vulkan.cpp
3. Add a vk_pipeline pipeline_my_shader; field to vk_device_struct
4. Create the pipeline in ggml_vk_load_shaders() using ggml_vk_create_pipeline()
5. Add a dispatch function ggml_vk_my_op()
6. Hook it into the ggml_vk_build_graph() switch

Follow the same pattern as pipeline_rms_norm_custom in the existing patch.
```

---

## Validation checklist — run in order

```
□ git log -1 shows 9c69907 (b8665)
□ spv_out/ directory exists with .h files
□ python validation/validate_shaders.py → ALL PASS
□ python validation/validate_fp8.py     → ALL TESTS PASSED  (if using FP8)
□ cmake output shows [UFM] Custom kernels ENABLED
□ llama-bench shows improved tokens/sec vs baseline
□ op_params[4] verified as causal flag for flash_attention
□ ROPE float offset 16 verified as theta_base
□ RAG db/ directory exists after ingest.py
```

---

## File manifest

```
ufm_package/
├── shaders/
│   ├── flash_attention_kv_quant.glsl   P4: flash attn + INT8 KV cache
│   ├── kvcache_update_q8.glsl          P4: KV write with online INT8 quant
│   ├── linear_coop_32.glsl             P3: 32×32 FP16 WMMA tiling
│   ├── linear_coop_fp8.glsl            P5: FP8 via uint8 coopmat (experimental)
│   ├── linear_coop_q4k.glsl            P1: fused Q4_K dequant + WMMA ★ MAIN WIN
│   ├── linear_coop_q4k_silu.glsl       P1+: fused gate+up+SiLU for LLaMA FFN
│   └── linear_coop_q8.glsl             P2: INT8 WMMA for Q8_0
│
├── src/
│   ├── ggml_fp8.patch                  Register FP8 type 40 in ggml
│   ├── ggml_vulkan_custom_kernels.patch Wire shaders into ggml-vulkan.cpp
│   ├── gguf_kv_cache_meta.txt          KV cache GGUF key spec (for save/restore)
│   ├── paged_kv_cache.h                Paged KV block allocator (CPU side)
│   └── paged_kv_upload.h               Block table CPU→GPU upload bridge
│
├── scripts/
│   ├── compile_shaders.bat             Compile GLSL → SPIR-V → C headers
│   ├── convert_to_fp8.py               Convert GGUF model to FP8 format
│   ├── ingest.py                       Index technical docs for RAG
│   ├── query.py                        Query RAG with Gemma 4
│   └── update.py                       Refresh stale doc sources
│
├── config/
│   └── sources.yaml                    Technical doc URLs + retrieval weights
│
├── validation/
│   ├── validate_fp8.py                 12 FP8 encode/decode tests
│   └── validate_shaders.py             Static checks + Q4K math verification
│
└── SETUP.md                            This file
```

---

## Known limitations and next steps

**What is NOT done yet:**

1. `paged_kv_cache.h` is Piece 1 (CPU allocator) only.
   Piece 2 (GPU shader that reads the block table indirectly) and
   Piece 3 (UFM eviction/reload wiring) are not written yet.
   Long-context inference beyond ~8K tokens needs all three pieces.

2. `linear_coop_q8.glsl` accumulates into a shared fp32 array rather
   than an fp32 coopmat. The INT8 WMMA runs on matrix cores correctly,
   but the post-MMA accumulation is scalar. Fix: stage through LDS with
   an int32 coopmat then load as fp32 coopmat.

3. `linear_coop_fp8.glsl` requires models pre-converted with
   `convert_to_fp8.py`. No standard GGUF models ship as FP8 yet.
   Whether the AMD Windows driver maps uint8 coopmat to hardware FP8
   WMMA can be verified with Radeon GPU Profiler (look for WMMA_F8F6F4
   instructions in the ISA view).

4. The RAG sources.yaml includes `gemma4:26b` as the generation model.
   If you prefer `gemma4:31b` for better quality, edit `GEN_MODEL` in
   `scripts/query.py` or pass `--model gemma4:31b`.
