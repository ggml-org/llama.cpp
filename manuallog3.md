# manuallog: Project6 - Expert-Granular MoE Tiering (GPU Hot Store)

> Session log + design/state reference for the agent working in
> `/run/media/miltos/Linuxaddon/AI_Workspace/Opencode/Project6/repo2`.
> Read AGENTS.md first for the authoritative rule set and current build/test
> commands. This file is the long-form reference: state, architecture, why past
> decisions failed, and the launch/benchmark etiquette. ASCII only.
> Last updated: 2026-08-05 (section 14 CLEANED: corruption root cause = 3-node
> cold path; fix = port fused MOE_COLD op. Section 14.7 is the current truth).

## 0. One-line status (2026-08-05)

Single-stream MoE decode accelerated ~2.5x via an expert-granular GPU hot
store (S=96 slots) + a dedicated `MUL_MAT_ID_COLD` CPU op. Verified at ~40
tok/s on Qwen3.6-35B-A3B IQ2_M (vs ~16 stock -cmoe) on CUDA. Multi-slot
server batches freeze the hot store (committed). **VULKAN SUPPORT DROPPED**
(2026-08-05): after a full investigation session, all tier implementations
(repo2, v2, v3.bak, v5) corrupt on Vulkan - there is no known-good Vulkan
tier to mirror. Working tree is clean at commit 1a850ec17.
MULTI-SLOT (2C gate, n_tokens>1) IS DROPPED FOR NOW: the mmid count+rank
CUDA port remains reverted, and -np>1 server batches fall back to stock.
CUDA single-stream is the supported tier path. See section 12 for eliminated
theories and the one remaining (unconfirmed) Vulkan lead.

## 1. Project context & constraints

- **Repo**: `Project6/repo2` (a fork of llama.cpp, merge-request targeted).
  Keep the upstream diff minimal and reviewable. Read `repo2/AGENTS.md` too.
- **Goal (SCOPE - do not overextend)**: MANUAL expert-granular GPU hot store
  for MoE. User sets `--expert-hot-s N` (0 = disabled). We:
  1. Find expert weight tensors per layer, compute bytes/slot.
  2. Allocate N GPU slot buffers at context init (VRAM committed at init).
  3. After the heatmap warms, copy the top-N hottest expert weight slices
     from CPU to the GPU hot store (one-shot first fill).
  4. Periodically re-sync the GPU slots to mirror the current top-N
     (plain top-N mirror on a token cadence, NO hysteresis gate / NO dwell).
  5. Hook the MoE ffn graph so hot experts compute from the GPU hot store,
     cold experts from a dedicated CPU op, summed per routed expert.
- **Strict constraint - Rule 6 new-file isolation**: all new logic lives in
  new .cpp/.h/.c files. Edits to upstream files stay minimal: single-line
  hooks, struct-member or CLI-flag additions, or (where unavoidable) dropping
  `static` to share a helper. The cold kernel is the only exception allowed
  and it now lives in its OWN file (see Architecture).

### Out of scope / NOT in scope
- Hysteresis ratio gate + dwell (Trick 5/6): DEFERRED. (v3.bak/v5 DO use
  hysteresis + dwell internally, but repo2 deliberately keeps it simple.)
- Cross-session warm start (Trick 21): repo2 fresh heatmap each session.
  (v3.bak/v5 have a `.tier` sidecar warm seed - see section 6.J.)
- Prompt harvesting / MOE_COUNT op / TMAX gate (Trick 11, 24).
- Perf tuning tricks (9/10/15) beyond what shipped.
- mmid.cu count+rank CUDA port: ATTEMPTED and REVERTED (see 6.H) - it made
  multi-slot output fully corrupt. The 2C gate stays.

If the user asks for any of the above, treat it as a new phase with its own
scope discussion. oldtricks.md is a REFERENCE for in-scope tricks, not a
feature list.

## 2. Environment & paths (ABSOLUTE)

- Project root: `/run/media/miltos/Linuxaddon/AI_Workspace/Opencode/Project6`
- Active source: `Project6/repo2`
- Build dir (CUDA): `Project6/build` (`-DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release`)
- Build dir (Vulkan): `Project6/build_vulkan` (`-DGGML_VULKAN=ON -DGGML_CUDA=OFF`)
- Reference / pre-fix tree: `Project5/repo2`
- Colibri original: `Project1/folder2/llama.cpp`
- **wackMall family (CRITICAL REFERENCES for the Vulkan question)**:
  - `Project3/llama-wackMall_v3.bak` - "the one that does not complicate
    things". Tier is auto-enabled (no CLI flag), auto-fit S, native 2D LUT,
    mask on CPU buffer, discards w_s on tiered path. WORKS on Vulkan.
  - `Project3/llama-wackMall_v5` - v3 + more features (RAM pool, sidecar,
    autofit ALWAYS ON, hysteresis env vars). WORKS on Vulkan, faster than v3
    (IQ2 22.35 vs 18.65 tok/s on this machine).
  - Both have `build_vulkan/` dirs created 2026-08-04 for this investigation.
- Models: `/run/media/miltos/Boost drive/Models/`
  - Qwen3.6-35B-A3B IQ2_M (`Qwen3.6-35B-A3B-abliterated-MAX.i1-IQ2_M.gguf`):
    40 MoE layers (0-39, ALL MoE), n_expert=256, n_expert_used=8. Has per-expert
    scale tensors (`*.scale` in GGUF) - relevant to the scale bug.
  - Qwen3.6-35B-A3B Q5_K_P (`.../BoostHome/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf`):
    bigger quant, per-expert scale too. Has a `.tier` sidecar (v5 warm seed).
  - Non-MoE guard: Bonsai-27B-Q1_0 (3.8GB) - heatmap+hotstore stay inert.

## 3. Architecture (current, accurate)

### 3.1 Files we touched (in repo2)
```
src/llama-expert-heatmap.{h,cpp}     - per-layer expert usage tracking + log
src/llama-expert-hotstore.{h,cpp}   - GPU hot store: alloc, copy_top_s, resync, LUTs
src/llama-expert-tier.{h,cpp}       - in-graph dual-path build hook
ggml/src/ggml-cpu/ggml-cpu-mul-mat-id-cold.{c,h}  - the MUL_MAT_ID_COLD kernel + shared mmid helpers
common/arg.cpp                      - --expert-hot-s / -las / --expert-heat-* flags
common/common.{h,cpp}               - param plumbing
include/llama.h                     - context_params fields
src/llama-context.cpp               - hotstore/heatmap init + post-compute hook
src/llama-graph.{cpp,h}             - moe_sel_experts capture + tier dispatch hook
ggml/include/ggml.h                 - GGML_OP_MUL_MAT_ID_COLD enum + ctor decl
ggml/src/ggml.c                     - op name/symbol + constructor + OP_COUNT 102
ggml/src/ggml-cpu/ggml-cpu.c        - 3 switch hooks (forward/n_tasks/work_size)
ggml/src/ggml-cpu/CMakeLists.txt    - new source added
```

### 3.2 Data flow
1. At context construction (gated: `hparams.n_expert > 0 && !warmup &&
   hot_s != 0`), build heatmap + hotstore. Hotstore picks the first non-CPU
   backend's default buffer type for the GPU hot store.
2. After each ubatch's graph compute (process_ubatch), if heatmap is set,
   `synchronize()` then `update_from_graph(res->moe_sel_experts)` reads the
   selected-experts tensors back and bumps per-(layer,expert) heat.
3. First fill: `copy_top_s` copies the top-N hottest expert weight slices
   CPU->GPU, builds hot_lut/cold_mask per layer, registers each expert
   weight tensor with the tier hook.
4. Cadence re-sync: `maybe_resync` mirrors the current top-N into the slots.
   NEW (committed 5c5020d80): when the ubatch has n_tokens>1 (multi-slot
   server batch), `maybe_resync` is called with multi_slot=true and SKIPS
   swapping - the hot store is frozen/static for the batch.
5. At graph build (`build_lora_mm_id`, called by build_moe_ffn), if `w` is
   registered with the tier, `llama_expert_tier_build` constructs the dual path:
     - hot:  remap real ids -> slot ids via hot_lut (sentinel S for cold),
             `ggml_mul_mat_id(dst_hot, cur, ids_hot)` on GPU.
     - cold: `ggml_mul_mat_id_cold(w, cur, ids, cold_mask)` on CPU - skips
             hot experts (cold_mask==0), computes only cold-selected rows.
     - result = `add(hot, cold)`.
   Returns nullptr -> caller falls back to stock `ggml_mul_mat_id`.
   NOTE: the per-expert w_s scale was REMOVED from this path (see 3.5).

### 3.3 The MUL_MAT_ID_COLD op
- Declared in `ggml.h` after `GGML_OP_GLU`; constructor `ggml_mul_mat_id_cold`
  takes `(ctx, as, b, ids, cold_mask)` - 5 sources.
- `cold_mask` is f32[n_experts], 1.0f=cold/compute, 0.0f=hot/skip; the kernel
  reads it as int32 zero-check.
- CPU-only. CUDA never sees it. The hot path uses stock `ggml_mul_mat_id`.
- Lives in `ggml/src/ggml-cpu/ggml-cpu-mul-mat-id-cold.c`. Uses the SAME
  chunking helper as stock mul_mat_id (`ggml_compute_forward_mul_mat_id_one_chunk`),
  exposed via `ggml-cpu-mul-mat-id-cold.h`. Reads type info via the public
  `ggml_get_type_traits_cpu()` accessor.

### 3.4 Working-tree changes (REVERTED 2026-08-05 - Vulkan attempt dropped)
Two candidate Vulkan fixes (Option A: removed the in-graph `w_s` scale block;
LUT native 2D) were applied and tested but were INSUFFICIENT to fix Vulkan
corruption, and the Vulkan path is now dropped entirely. They have been
REVERTED; the working tree is clean at 1a850ec17. See section 12.

### 3.5 v3.bak reference architecture (what "works")
- Tier auto-enabled (no CLI flag; unconditional init), auto-fit S from free VRAM.
- `s.lut` native 2D `ggml_new_tensor_2d(g_ctx_gpu, I32, 1, n_expert)` on GPU.
- `s.mask` `ggml_new_tensor_1d(g_ctx_cpu, I32, n_expert)` on CPU (i32, 1=cold).
  IMPORTANT: mask is on a CPU buffer - read directly by the CPU cold op.
- `s.w_hot` 3D on g_ctx_gpu (no_alloc ctx + alloc_ctx_tensors_from_buft, WEIGHTS usage).
- `build_mul_mat_id`: same remap -> get_rows(lut) -> ids_hot -> mul_mat_id(w_hot)
  -> mul_mat_id_cold(w, ids, mask, ptrs) -> add. Discards w_s.
- Cold op takes an extra `s.ptrs` (i64 host weight addresses for RAM pool).
- Gate: `ids->ne[1] > g_tmax` (g_tmax default 16 in v3, 1 in v5).
- v5 adds: RAM pool (SR slots), sidecar warm seed, LLAMA_EXPERT_DECAY (default
  1.0) / LLAMA_EXPERT_HYSTERESIS (default 1.5) env vars, dwell>=32, autofit
  ALWAYS ON, LLAMA_EXPERT_S/HOT/ADAPT env vars.

## 4. Build & test (authoritative; AGENTS.md is the source of truth)

### Configure from clean (CUDA)
```sh
cd /run/media/miltos/Linuxaddon/AI_Workspace/Opencode/Project6
mkdir -p build && cd build
cmake ../repo2 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release
```
### Configure from clean (Vulkan)
```sh
mkdir -p build_vulkan && cd build_vulkan
cmake ../repo2 -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
```
### Build (incremental)
```sh
cd /run/media/miltos/Linuxaddon/AI_Workspace/Opencode/Project6/build
cmake --build . -j$(nproc) --target llama-completion llama-server
# Vulkan:
cd /run/media/miltos/Linuxaddon/AI_Workspace/Opencode/Project6/build_vulkan
cmake --build . -j$(nproc) --target llama-completion
```
### Device selection (Vulkan - IMPORTANT, differs from vulkaninfo)
`./bin/llama-completion --list-devices` is authoritative. In llama.cpp's
enumeration order on this machine:
- Vulkan0 = NVIDIA GeForce RTX 3070 Laptop GPU (the FAST one)
- Vulkan1 = AMD Radeon RX 570 (RADV POLARIS10)
This is OPPOSITE to `vulkaninfo` (which shows AMD as GPU0). Use
`GGML_VK_VISIBLE_DEVICES=0` for the RTX, `=1` for the AMD. A default run
picks Vulkan0 = RTX.
### PASS/FAIL test (CUDA single-stream, the headline case)
```sh
cd /run/media/miltos/Linuxaddon/AI_Workspace/Opencode/Project6/build
ulimit -c 0
./bin/llama-completion \
  -m "/run/media/miltos/Boost drive/Models/Qwen3.6-35B-A3B-abliterated-MAX.i1-IQ2_M.gguf" \
  -ngl 99 -cmoe -c 8000 -ctk q8_0 -ctv q8_0 -b 256 -ub 256 -t 6 \
  --kv-offload --flash-attn on -fit off --temp 0 -no-cnv \
  --expert-hot-s 96 \
  -p "Write a comprehensive technical guide to setting up a home Linux server" \
  -n 256 </dev/null
```
PASS = coherent output, eval ~25 ms/token (40 tok/s), "Expert hotstore
sizing (S=96)" and "re-sync swapped" log lines present, exit 0.
FAIL = no hotstore lines (tier inactive), eval >60 ms/token (stock ~16 tok/s),
crash, or incoherent output.
### NOTE ON --no-mmap (REMOVED PERMANENTLY, 2026-08-04)
Do NOT pass `--no-mmap` to test commands anymore. The user said it "hurts us"
permanently. Reason observed: with --no-mmap each server reads the FULL model
into RAM; two servers = ~23GB, maxed 31GB RAM + 17GB swap (looked like a leak,
was not - killing the servers freed all of it instantly). Drop it.

## 5. Launch & benchmark etiquette (READ BEFORE ANY TEST)

1. **Check for concurrent builds/tests first** (`ps -eo pid,pmem,comm --sort=-pmem | head`).
   RAM/CPU pressure causes OOMs and spurious timeouts. Wait for it.
2. **Minimum -n 128** for generation tests. -n 16 masks bugs and is noisy.
   For the wackMall/v5 caching tier use -n 256+ (it has not warmed up until then).
3. **Non-interactive**: pass -no-cnv and redirect stdin from /dev/null
   (`< /dev/null`) so the test exits on its own.
4. **-st exits cleanly** on llama-cli (single-turn). Keep `timeout 60` as a
   hard safety net for llama-completion (it has no -st; it exits after -n).
5. **Do NOT change user-specified flags** without asking. -ngl 99 -cmoe etc.
6. **Never use decay 0.9 for "diagnostic" tests** unless explicitly told.
7. **Use LLAMA_LOG, not LLAMA_LOG_INFO**, for any line you must see.
8. **Use the grep tool narrow**: scope `path` to a dir or a file.
9. **Quote model paths** with spaces.
10. **Do NOT pass -ngl to the CPU-only build**.
11. **Per backend build dirs**: build/ = CUDA, build_vulkan/ = Vulkan. Do not
    confuse binaries.
12. **Launching a server without softlocking the harness**: do NOT use
    `pkill -f llama-server` (kills your own shell). Use `pkill -x llama-server`.
    To launch detached WITHOUT the harness killing it on timeout: use
    `setsid --fork ./bin/llama-server ... </dev/null >log 2>&1 < /dev/null`
    then poll /health in a SEPARATE command. The earlier nohup+disown+same-cmd
    poll softlocks the harness (the timeout then SIGTERMs the process group).
13. **Commit style**: user's name (Miltos22), lowercase, no period, no
    Co-authored-by/Assisted-by tags. Draft the message WITH the user before
    committing. Local commits only, no push, no PR.
14. **No git mutations** (commit/push/amend/rebase) unless explicitly asked.
    The user explicitly asked for the -las commit amend twice.
15. **push needs a token**: `git push https://miltos22:TOKEN@github.com/...`
    (username/password auth is rejected by GitHub). The remote is the user's
    fork `miltos22/llama.cpp-wackMall-merge-request.git`. Only push when asked.

## 6. Debugging history - WHY past decisions didn't work

### A. The `ggml_backend_tensor_get` signature trap
5 args vs real 4: `ggml_backend_tensor_get(tensor, data, offset, size)`.

### B. OOM on CPU-only inference
35B fully on CPU (-ngl 0) triggers kernel OOM killer during PP graph alloc.
Fix: reduce -c. Watch RAM before builds/tests.

### C. LLAMA_LOG_INFO is invisible (CRITICAL QUIRK)
LLAMA_LOG_INFO -> LOG_LEVEL_TRACE(4) filtered by INFO(3). Use LLAMA_LOG (0).

### D. "Hysteresis" naming confusion
Heatmap accumulation rate vs swap-policy ratio gate (Trick 6). repo2 uses pure
additive counting + decay only; no ratio gate / dwell (deferred).

### E. CLI float lambda
`common_arg` lambdas only accept `int` and `std::string`. Use std::stof.

### F. Fabricated test data - NEVER LIE ABOUT OUTPUT
NEVER invent log lines or test output. AGENTS.md Rule 1: evidence or it did
not happen.

### G. GPU tensor readback bug (the big one)
Async CUDA + recycled scratch buffers -> garbage expert ids. Fix:
`synchronize()` before readback AND `ggml_set_output(selected_experts)`.

### H. The 4-step cold chain REGRESSION -> MUL_MAT_ID_COLD
Original chain: 5-node remap x 4 x 3 matrices x 40 layers = ~3,500 extra
graph nodes + dummy expert-0 rows. 1.3 tok/s. Ported GGML_OP_MUL_MAT_ID_COLD
(colibri Trick 7 Path A): single skip-hot CPU op. 40 tok/s.
Then moved the kernel to its own file (`d1cdf1b35`, pushed via sync-fork).

### I. The cold kernel from_float NULL segfault
`type_traits_cpu[type].from_float` is NULL for IQ2_M. Fix: fetch from
`type_traits_cpu[vec_dot_type].from_float` (matches stock mul_mat_id).

### J. The "self-contained new-file kernel" rewrite attempt
The working version REUSES stock chunking helpers; a fresh self-contained
rewrite segfaulted (bug I). Do not re-attempt without a new decision.

### K. **mmid.cu count+rank port ATTEMPTED AND REVERTED (2026-08-04)**
Goal: lift the 2C gate (n_tokens==1) so the tier works at n_tokens>1
(multi-slot server). Ported the colibri count+rank fix into BOTH the generic
and templated paths of repo2's mmid.cu (kept write_inverse), sized shared
memory for the worst case (n_tokens*n_expert_used), and REMOVED the gate
`if (cur->ne[2] > 1) return nullptr;` in tier.cpp.
RESULT: **fully corrupt multi-slot output** - worse than before (before = slow
but correct stock fallback; after = garbage). User: "the slots thing does not
work for multiple ones, it is actually worse than before."
REVERTED both files to committed state (git checkout). The 2C gate stays.
The user decided to instead FREEZE the hot store during multi-slot batches
(see 3.2, committed 5c5020d80). The mmid count+rank fix may be revisited only
as a separate deliberate effort; the corrupt result suggests our sentinel
duplicate handling interacts badly with the CUDA MMQ path beyond what colibri's
fix covers (or the port had a subtle bug).

### L. Freeze-exchange analysis (and why it was chosen)
Earlier analysis said freezing does NOT fix multi-slot speed (the gate is the
cause, not resync churn). But the mmid port made things CORRUPT, so the user
chose freeze + keep the gate: multi-slot falls back to stock (correct, slow)
and the hot store stops swapping during multi-slot batches. Committed as
`5c5020d80`. Tested: multi-slot -np 8 coherent (stock speed ~2.45 t/s/slot),
single-stream 38-41 tok/s no regression. This is the current committed state.

### M. Emoji-loop degenerate output is NOT the tier
--ignore-eos + temp-0 + this Qwen3 model drifts into repetition loops.
Isolated: happens with tier ON and OFF; only --ignore-eos triggers it.
Do not blame the tier; check --ignore-eos first.

### N. The earlier "random server crash" was a self-inflicted kill
`pkill -f llama-server` killed the launching shell. Use `pkill -x`.

### O. Plant_static and log_hit_rate are diagnostic, not core
Diagnostic helpers, not part of the core feature path.

### P. **Vulkan device ordering trap (2026-08-04)**
`vulkaninfo` shows AMD as GPU0, NVIDIA as GPU1. But llama.cpp's OWN
`--list-devices` shows Vulkan0 = NVIDIA RTX 3070, Vulkan1 = AMD RX 570.
A test with GGML_VK_VISIBLE_DEVICES=1 silently ran on the AMD (9.94 tok/s,
"it still used the 570"). Correct: GGML_VK_VISIBLE_DEVICES=0 = RTX.

### Q. **RAM was NOT leaking - it was --no-mmap (2026-08-04)**
Two --no-mmap servers held 2x 11.6GB model in RAM -> 30Gi used, 17Gi swap,
user thought it was a leak. Killing them freed 23GB instantly. Also: removing
`-c` from a run let context default huge and eat RAM alongside the v5 RAM pool
(18Gi). Always keep `-c` on multi-GB model tests.

### R. **repo2 tier corrupts on Vulkan; v3.bak/v5 are coherent (2026-08-04)**
This is the OPEN issue. Full detail in section 9.

### S. **v5 "autofit always on" gotcha (2026-08-04)**
v3.bak/v5 enable the tier UNCONDITIONALLY (no CLI flag needed) and autofit S
from free VRAM. So EVERY v3/v5 run is tiered - there is no "stock" v3/v5 run
to compare against without disabling the tier. My early "v5 stock-ish" 23.52
tok/s number was tiered too (run-to-run variance / tuning churn).

### T. **v5 cold first-run is slow (2026-08-04)**
v5 Q5 Vulkan first run 2.96 tok/s (init churn: RAM pool, sidecar build).
Warm run (2nd, sidecar present) 11.72 tok/s. First-run numbers are meaningless
for v5; use warm runs. Also the `.tier` sidecar file is a warm seed - deleting
it forces cold start. User asked to delete sidecar + LLAMA_EXPERT_DECAY=0.999
LLAMA_EXPERT_HYSTERESIS=1.3 + no -cmoe/-ngl/-t for a "clean" v5 test: result
was coherent, 1.61 tok/s with RAM pool 18Gi (slow but works).

## 7. CLI flags (repo2)

| Flag | Type | Default | Env | Purpose |
|------|------|---------|-----|---------|
| `-las`, `--expert-hot-s N` | int | 0 | `LLAMA_ARG_EXPERT_HOT_S` | top-S expert slots for GPU hot store (0=disabled). -las added this session (pushed) |
| `--expert-heat-decay F` | float | 0.99 | `LLAMA_ARG_EXPERT_HEAT_DECAY` | multiplicative decay per update |
| `--expert-heat-log-period N` | int | 100 | `LLAMA_ARG_EXPERT_HEAT_LOG_PERIOD` | log + re-sync cadence in updates |

There is NO `LLAMA_EXPERT_S` env fallback - removed earlier, CLI only.
(v3.bak/v5 DO use LLAMA_EXPERT_S / LLAMA_EXPERT_DECAY / LLAMA_EXPERT_HYSTERESIS
/ LLAMA_EXPERT_ADAPT / LLAMA_EXPERT_HOT - but those are THEIR flags.)

## 8. Commits & roadmap

### Commits on this branch (origin/master has them up to 6e75753ab)
- `fcaac3d75` expert heatmap: decay-tracked usage counters per layer
- `610802380` expert heatmap: add top-S ranking, log top-8 per layer
- `21537c16f` expert heatmap: add --expert-hot-s flag for GPU hot store slot count
- `b27d2f59e` expert heatmap: fix GPU tensor readback via ggml_set_output and synchronize
- `413e65dde` expert heatmap: move readback logic into heatmap module
- `3fd577828` Merge remote-tracking branch 'origin/master'
- `23ff80ec3` expert hotstore: add per-layer expert slot sizing
- `289c41ef5` expert hotstore: allocate GPU hot store buffers for S slots
- `284215c04` expert hotstore: reduced cross contamination when expert args off
- `8e955700f` expert heatmap: count updates in tokens not layers
- `08e657d41` expert heatmap: fix decay and log trigger
- `1aefbbe58` expert hotstore: copy top-S experts to GPU after first ubatch
- `cee1740bb` expert hotstore: re-sync hot slots on cadence (stable slots)
- `a3c442540` expert hotstore: add sentinel slot for zero-contribution routing
- `904090276` expert hotstore: per-layer LUTs and masks for in-graph routing
- `8d1a68e5b` Merge remote-tracking branch 'origin/master'
- `3c9616b3a` expert tier: hook GPU hot store into graph with MUL_MAT_ID_COLD cold op
- `d1cdf1b35` ggml-cpu: move MUL_MAT_ID_COLD kernel to its own file
- `8687ee2b6` common: added -las short flag for expert hot store slots (alias of --expert-hot-s)  [PUSHED]
- `6e75753ab` Merge branch 'ggml-org:master' into master  [on origin/master - created by GitHub "Sync fork" button, upstream pulled in; local fast-forwarded after tar backup]
- `5c5020d80` expert hotstore: freeze swapping during multi-slot batches  [LOCAL, NOT pushed, 1 ahead]
- `1a850ec17` llama : gate expert hot store to CUDA only, with force override  [LOCAL, NOT pushed, 2 ahead] - CUDA-only guard + LLAMA_EXPERT_HOT_FORCE

### Working tree (CLEAN as of 2026-08-05)
- Clean at `1a850ec17`. The Option A (scale removed) + LUT 2D Vulkan-fix
  changes were reverted, and the CUDA-only guard commit is in. Nothing
  uncommitted.

### DEFERRED / NOT DONE
- Step 3f: auto-S via native fit (`--expert-hot-s -1`). NOT DONE in repo2.
- Hysteresis ratio gate + dwell (Trick 5/6): deferred in repo2.
- mmid.cu count+rank CUDA port: ATTEMPTED, CORRUPT, REVERTED. Do not re-approach without a new decision.
- Per-backend gate (option B): let Vulkan/Metal enable tier at n_tokens>1 - irrelevant now (gate is by design after the mmid failure).
- Fused MoE cold (GELU/gate_up): not ported.

## 9. KNOWN LIMITATIONS + THE OPEN VULKAN ISSUE

### 9.1 The 2C gate (n_tokens==1) - MULTI-SLOT DROPPED FOR NOW
Tier only engages at single-token decode. Multi-slot server batches fall back
to stock. This is now BY DESIGN after the mmid port failed (see 6.K/6.L).
DECISION (2026-08-05): multi-slot (n_tokens>1) tiered support is DROPPED FOR
NOW. The mmid.cu count+rank port remains reverted; do not re-approach without
an explicit new decision. -np>1 server batches use stock CPU MoE speed.
Single-stream CUDA is the supported tier path.

### 9.2 ~~OPEN: repo2 tier corrupts on Vulkan~~ RESOLVED BY DROPPING VULKAN (2026-08-05)
Historical record only. The Vulkan tier path is DROPPED and the CUDA-only
guard (1a850ec17) prevents accidental use. Facts below are preserved for
reference; see section 12 for the eliminated theories and the sole remaining
lead. FACTS (all verified 2026-08-04):
- repo2 tier + Vulkan (RTX) + IQ2_M or Q5_K_P => `<think>` loop / garbage.
- repo2 tier + CUDA + same models => fully coherent.
- v3.bak tier + Vulkan + same models => coherent (IQ2 18.65, Q5 5.29 tok/s).
- v5 tier + Vulkan + same models => coherent (IQ2 22.35, Q5 warm 11.72).
- repo2 stock (tier disabled) + Vulkan => coherent.
- repo2 tier + Vulkan + `-ngl 0` => still `<think>` loop (rules out GPU->CPU
  cur sync for the cold op).
- repo2 tier + Vulkan + S=1 => coherent (but ends early, not a full test).
- v3.bak tier + Vulkan + `-ngl 0` => coherent (the decisive A/B: same model,
  same flags, same Vulkan backend, v3 works / repo2 loops).

ROOT CAUSE STATUS: two candidate fixes applied but NOT sufficient:
1. Scale removed (Option A) - fixed binary garbage, still loops.
2. LUT native 2D - still loops.
Both were structurally necessary to match v3.bak but the corruption persists.

NEXT LEAD (2026-08-04, highest priority to investigate):
**cold_mask buffer placement.** v3.bak allocates the cold `mask` on a CPU
buffer (`ggml_new_tensor_1d(g_ctx_cpu, I32, n_expert)`). repo2 allocates
`cold_mask` in the SAME ggml context as dst_hot/hot_lut, which is allocated
with the GPU buft - so `cold_mask` lives on the GPU buffer. The CPU cold op
reads `mask->data` DIRECTLY (`const int32_t * cold_mask = (const int32_t *)
mask->data;`). On Vulkan, a GPU-buffer tensor's `->data` is NOT host-readable
=> garbage mask => wrong skip/compute decisions => degenerate output. On CUDA
it apparently works (host-mapped or scheduler copies). THE FIX would be to
allocate cold_mask on a CPU/host buffer (like v3.bak), or read the mask via a
host-accessible path. VERIFY THIS FIRST in the next session.
Secondary candidates if the mask fix is not sufficient:
- The `ggml_mul_mat_id` hot call at n_tokens==1 uses the Vulkan VEC-id path
  (`mul_mat_vec_id`); upstream `ggml_vk_get_dequantize_mul_mat_vec_id` omits
  GGML_TYPE_IQ2_M from its supported list (has IQ2_XXS/XS/S but not IQ2_M).
  v3.bak runs IQ2_M on the same backend, so likely not the blocker, but worth
  confirming repo2 does not take a different dispatch.
- `ggml_mul_mat_id_cold` signature differs: v3 takes `(mask, ptrs)`, repo2
  takes `(cold_mask)` only. The ptrs arg is a RAM-pool optimization, likely
  irrelevant to correctness.
- The `dst_hot` buffer content after `ggml_backend_tensor_set` copies on
  Vulkan - verify by dumping dst_hot vs source on both backends.

### 9.3 Hotstore segfault without -cmoe
repo2 tier requires the MoE expert weights to stay on CPU (needs `-cmoe` or
`-ncmoe`). Running `--expert-hot-s 96` WITHOUT -cmoe segfaulted on Vulkan
(2026-08-04): the hotstore copy reads `w->data` as a host pointer, but
without -cmoe the experts are GPU-offloaded. This is a latent bug - the tier
should reject or handle non-CPU experts. Not the Vulkan corruption (CUDA needs
-cmoe too and works).

### 9.4 IQ2_M missing from Vulkan mul_mat_vec_id supported list
Upstream `ggml_vk_get_dequantize_mul_mat_vec_id` lists IQ2_XXS/XS/S but not
IQ2_M. Not the primary suspect (v3.bak runs IQ2_M cleanly on the same backend).

### 9.5 Server -np>1 = stock speed
Multi-slot decode batches (n_tokens>1) fall back to stock CPU MoE via the
gate - ~20 tok/s aggregate regardless of S. Single-stream is where the
40 tok/s lives. This is by design after the mmid failure.

### 9.6 Hot store competes for VRAM
8GB VRAM + 11GB IQ2_M model + S=96 hotstore barely coexist. Q5_K_P needs S<=48
(96 slots would be 9424 MiB, exceeds 8GB). Larger GPU = more headroom.

### 9.7 Stale comments / minor
- `ggml.h` cold_mask comment calls it "i32" but the tensor is f32 (read as int32).
- `ggml-rpc.h:14` static_assert 101 vs ggml.c 102 (harmless, RPC off).
- `llama-expert-tier.h` header comment describes the OLD dual-mul_mat_id design
  (cold_lut/hot_mask) - stale, code is correct. Low priority.
- v3.bak/v5 ggml-vulkan.cpp matches upstream - their Vulkan success is NOT a
  backend patch; it is their tier code.

## 10. Reference docs

- `oldtricks.md` - 25 tricks from colibri/wackMall. In scope: Trick 13
  (WEIGHTS usage tag), Trick 18 (non-owning ggml contexts / no_alloc batch),
  Trick 23 (create-then-allocate), Trick 7 Path A (MUL_MAT_ID_COLD, ported).
  Others OUT OF SCOPE unless re-discussed.
- wackMall references: `Project3/llama-wackMall_v3.bak` (simple, works on
  Vulkan) and `Project3/llama-wackMall_v5` (feature-rich, faster, works on
  Vulkan). Their tier files are the reference for the Vulkan fix.
- Colibri mmid.cu count+rank diff (the "revisit later" reference): the change
  to `mm_ids_helper` replacing `iex_used` (one hit) with `cnt` + inclusive
  warp scan `rank` + `n_hit`, so every duplicate sentinel hit gets its own
  compact row. `Project1/folder2/llama.cpp/ggml/src/ggml-cuda/mmid.cu`.
  NOTE: attempted in repo2 2026-08-04 and produced corrupt output - see 6.K.
- `Project6/repo2/AGENTS.md` - llama.cpp contributor rules.

## 11. Vulkan benchmark table (2026-08-04, all RTX 3070 unless noted)

| Build | Model | Flags | Result | tok/s |
|-------|-------|-------|--------|-------|
| repo2 CUDA | IQ2 | -cmoe -ngl99 S96 | coherent | 26.2-41 |
| repo2 Vulkan | IQ2 | -cmoe -ngl99 S96 | GARBAGE-><think> loop | 14-16 |
| repo2 Vulkan | Q5 | -cmoe -ngl99 S48 | GARBAGE | 5.24 |
| repo2 Vulkan | IQ2 | -cmoe -ngl0 S96 | <think> loop | 4.66 |
| repo2 Vulkan | IQ2 | -cmoe -ngl0 S1 | coherent (short) | 5.9 |
| v3.bak Vulkan | IQ2 | -cmoe -ngl99 autoS | coherent | 16.4-18.7 |
| v3.bak Vulkan | Q5 | -cmoe -ngl99 autoS=51 | coherent | 5.29 |
| v3.bak Vulkan | IQ2 | -ngl0 autoS | coherent | 4.35 |
| v5 Vulkan | IQ2 | -cmoe -ngl99 autoS | coherent | 13.96-22.35 |
| v5 Vulkan | Q5 | -cmoe -ngl99 autoS=51 warm | coherent | 11.72 |
| v5 Vulkan | Q5 | cold first run | coherent | 2.96 |
| v5 Vulkan | Q5 | decay.999 hyst1.3 no-cmoe c8000 n256 | coherent (slow, RAM pool 18Gi) | 3.92 |
| v5 Vulkan | IQ2 | AMD RX570 (=1) | coherent | 9.94 |

Takeaways: v3/v5 coherent on Vulkan; repo2 not. v5 faster than v3 (IQ2 22 vs 19,
Q5 11.7 vs 5.3). v3/v5 autofit S; repo2 manual S. The v3/v5 `.tier` sidecar is
a warm seed (seed coverage 84.6% warm vs 0% cold).

## 12. VULKAN DECISION - DROPPED (2026-08-05)

The Vulkan tier attempt is DROPPED after a full session of investigation.
CUDA tier (40 tok/s, coherent) remains the supported path. This section
records the eliminated theories and remaining leads so future sessions do not
re-litigate.

### ELIMINATED THEORIES (do not re-test without new evidence)

1. **IQ2_M unsupported type theory - DEAD.** The model file is named
   "i1-IQ2_M" but the ACTUAL expert tensor types are `iq2_s` (gate/up, 82MiB)
   and `iq3_s` (down, 110MiB), verified from loader logs. GGML_TYPE_IQ2_M does
   NOT exist in this tree's ggml.h enum (lines 400-430). IQ2_S/IQ3_S ARE in
   both the Vulkan dmmv getter and supports_op lists. So the hot node runs the
   vec-id path fine (no abort, exit 0). Type was never the problem.
2. **supports_op probe as a discriminator - DEAD.** CUDA's MUL_MAT_ID
   supports_op also lacks whatever (its list ends at IQ2_XXS/XS/S, IQ3_*, IQ4_*,
   BF16) yet CUDA works. A supports_op probe would disable the working CUDA
   path too.
3. **Mask buffer placement / dtype - DEAD.** cold_mask being F32-on-GPU (repo2)
   vs I32-on-CPU (v3) is irrelevant: the scheduler copies the mask to CPU
   correctly (seen as CPU#leaf_1012#0 in dumps) and the F32 zero-check is
   equivalent to I32.
4. **tensor_set / buffer_clear asyncness - DEAD.** Both ggml_vk_buffer_write_2d
   and ggml_vk_buffer_memset are synchronous (host-visible memcpy or
   transfer+fence-wait). No race.
5. **IQ2_M missing from Vulkan vec-id pipeline - DEAD.** Irrelevant since the
   type is actually IQ2_S/IQ3_S, and v3/v5 also lack IQ2_M in that getter yet
   (see #6) none of them are actually coherent on Vulkan anyway.
6. **"v3/v5 work on Vulkan" - FALSE PREMISE, DEAD.** Directly tested 2026-08-05:
   v2 (Project1/llama-wackMall_v2, its OWN vulkan build), v3.bak, and v5 ALL
   corrupt on Vulkan with the tiered IQ2 config (same `<think>` loop) including
   with `-ngl 99 -cmoe`. The manuallog's earlier "coherent on Vulkan" rows for
   v3/v5 do not reproduce under these exact commands and were likely measured
   under different conditions (Q5 model, or runs that terminated early).
   Conclusion: there is NO known-good Vulkan tier implementation to mirror.
7. **Per-matrix cold+ADD on Vulkan vs fused cold - DEAD.** v2/v3 share
   byte-identical build_mul_mat_id + begin/end_moe_cold + ggml_moe_cold fused
   cold, and v2 corrupts while the "fix" claim for v3 is void (both corrupt).
   The v2-vs-v3 code diff is exclusively the prefetch/predict/poolB feature,
   which is inactive by default; replacing v2's 3 differing files with v3's
   changed nothing (still corrupt). So the graph structure is NOT the cause.
8. **Scheduler "1.wgt" weights-rule difference - DEAD.** The rule is
   identical across repo2/v3/v5. All assign the hot node to Vulkan.

### REMAINING LEAD (unconfirmed, low priority since Vulkan is dropped)

The tiered hot node (`mul_mat_id(dst_hot[S+1 planes], cur, ids_hot)` with
slot indices 0..S incl. the zeroed sentinel plane) runs on Vulkan's vec-id
path and corrupts on ALL implementations (repo2/v2/v3/v5). Stock Vulkan (tier
off, MoE on CPU) is coherent. So the bug is in how Vulkan's `mul_mat_id`
handles the slot-indexed S+1-plane hot tensor with the sentinel plane, NOT in
repo2's tier code. Candidate mechanisms, unverified:
   - coopmat2 path on Ampere (RTX 3070) mishandling the slot tensor
   - vec-id shader plane indexing with a zeroed sentinel plane
   - row_ids/result-tile shared memory sizing for S+1 planes
If Vulkan support is ever re-attempted, start here with a minimal standalone
repro (mul_mat_id on a 3-plane slot tensor vs the full tensor) before touching
the tier.

### ACTIONS TAKEN
- Reverted uncommitted Vulkan fix attempts (Option A scale removal + LUT 2D)
  in tier.cpp/hotstore.cpp -> working tree back to clean committed state
  (5c5020d80). CUDA path untouched and verified.
- Added CUDA-only guard in llama-context.cpp (2026-08-05): the hotstore now
  only allocates into a backend whose device name starts with "CUDA". On Vulkan
  (bugged) it logs a WARN and skips -> tier off -> coherent stock output.
  On CPU the existing non-CPU check already skips. LLAMA_EXPERT_HOT_FORCE=1
  overrides the guard (re-enables the tier on any non-CUDA GPU for testing/
  emergency only). Verified: Vulkan skips+coherent, CUDA enables+coherent
  (re-sync swapped 2471), Vulkan+HOT_FORCE engages tier. COMMITTED as
  `1a850ec17`. HEAD is now 1a850ec17 (2 ahead of origin, nothing uncommitted).

## 13. Next actions (for the next session)

State as of 2026-08-05: ALL WORK COMMITTED. Working tree clean, 5 commits
ahead of origin. Feature set complete for this pass:
- Auto-S via fit (`--expert-hot-s -1`) - DONE (f201ef52a)
- Hysteresis gate (--expert-hyst, default 1.3) - DONE (ca624bc06)
- decay default 0.999 - DONE (ca624bc06)
- --expert-dwell exists (default 0 = off) - DONE but OFF by default

Committed this session (oldest->newest):
- `1a850ec17` llama : gate expert hot store to CUDA only, with force override
- `b18c4266c` common: expert hot store manual slots activate --cmoe
- `f201ef52a` common: autofit expert hot store slots via --expert-hot-s -1
- `ca624bc06` expert hot store: hysteresis gate for slot swaps

Verified on Qwen3.6-35B-A3B IQ2_M (255-run decode): stock autofit 33.25
tok/s, autofit tier (S=133) 41 tok/s, hyst gate on 34.6-35.1 tok/s
(hyst=1.3 dwell=0: 35.1, 2600 swaps vs 2985 no-gate). Dwell default 0
(user decision: not worth the speed cost). Dwell aging counts real tokens
and initial fill is eligible (else first sync defers -> speed crash).

## 14. CPU OVERHEAD + CORRUPTION INVESTIGATION (2026-08-05) - CLEANED REFERENCE

This section is the authoritative record. It is CHRONOLOGICAL and marks
superseded conclusions explicitly. The final truth (14.7) is the current
state. All measurements on CUDA build, Qwen3.6-35B-A3B IQ2_M, -c 8000,
--temp 0, --fit-target as noted, unless stated. Corruption judgements are
DIRECT READS of full output (see AGENTS.md rule 10), never grep counts.

### 14.1 Phase A - the "5-10% CPU overhead" (t6) is CLOSED as a non-issue

The user-reported +5-10% CPU vs stock was investigated thoroughly and
proved to be a 256-token short-run artifact of the tier's initial
convergence burst, not a real overhead.

Key measurements (n=256 burst vs n=1024/4096 real):
- -t 6, n=256: ours 178.6 ms CPU/token vs stock 160.6 = +11%. Real per
  token, but ONLY during the first ~100-token convergence burst.
- -t 6, n=4096 (VALID, fresh hotset): dynamic 60.87 tok/s / 98.1 ms
  CPU/token vs static 58.45 / 101.3. Dynamic is FASTER and uses LESS CPU
  than a frozen set. Both crush stock (34 tok/s, 160.6 ms/token).
- Verdict: dynamic tier at real lengths = ~1.8x faster and ~39% less CPU
  per token than stock. CLOSED. Benchmark at -n 1024+ only.

Superseded intermediate claims (for history only): 14.9 "ROOT CAUSE =
heatmap bookkeeping" was WRONG (conflated readback+resync+fill quality;
controlled test showed readback only ~1.8 ms/token); 14.11 "swap moves
~20ms/token" was the 256-burst artifact; 14.7 PASSIVE / 14.8 GGML_OPENMP=OFF
were dead ends (spin is load-bearing; native pool identical cost).

### 14.2 Phase B - the -t 12 collapse is REAL and structural

The real remaining issue is -t 12 (ours 11-19 tok/s vs stock 25). Isolated:
- Not heatmap (static also collapses), not swaps (0 in static), not
  convergence (n=1024), not cold-gpu. It is the tier graph at 12 threads.
- Energy (perf power/energy-pkg, t12, n=1024): stock 3604 J / 3.48 J/tok
  vs ours 5435 J / 5.25 J/tok. Ours does 51% MORE real work, not spin.
- Root cause: repo2 emits 3 separate MUL_MAT_ID_COLD CPU nodes per MoE
  layer (gate, up, down) + 3 mul + 1 add = ~120 CPU nodes/token, each with
  a full ggml_barrier (ggml-cpu.c:3116). v3/v5 use ONE fused GGML_OP_MOE_COLD
  node per layer computing the whole chain under ONE barrier (~40
  nodes/token, 3x fewer barriers).

### 14.3 Phase C - corruption discovered, mechanism LOCALIZED

While tuning repo2's swap cadence for the t12 fix, output CORRUPTION was
found (mid-word splits: "Here 's", "summar izing", "Back ups", "/ usr",
"x 8 6 _ 6 4"). This VOIDED the tuning speed gains (they were measured on
corrupt output).

Localization (direct read of interleaved stdout): a resync marker lands
BETWEEN the halves of a generated word:
  ...using a **De=== re-sync swapped X ===bian-based** system...
The model emitted "De", a resync fired, then "bian" - the expert's weights
changed between two tokens of one word. Cadence sensitivity: cadence-1 =
pervasive, cadence-2 = intermittent, cadence-20 = rare (1/~700 words),
cadence-100 = unobserved. Resyncs cluster in the initial convergence burst.

GRAPHLOG experiment (ggml_cuda_graph_compute instrumentation): the corrupt
token is produced in DIRECT mode (no CUDA graph active yet) exactly at a
resync marker. This RULED OUT the CUDA-graph-replay theory.

### 14.4 Phase D - candidate fixes (ALL SUPERSEDED by 14.7)

Multiple fixes reduced corruption frequency but NONE reached zero:
1. host-mask (cold_mask on host/CPU buffer) - closed a cross-stream race
   on the mask write but corruption persisted.
2. post-resync ggml_backend_sched_synchronize - orders LUT copies but not
   the logical hot/cold flip.
3. active-expert guard (don't evict experts routed this token) - helped
   only partially (protects THIS token, not the word's prefix).
4. PP-priming (fill at first decode) - reasonable perf idea, not a fix.
5. rate-limit bursts (suppress swaps N tokens after a big resync) - capped
   the burst but residual corruption remained.
6. word-boundary gating (Enhancement B: only resync when the just-sampled
   token is ENTIRELY whitespace/punct, no cap) - reduced to rare, NOT zero.
7. atomic-pair LUT+mask write ordering - helped, not zero.
8. CUDA set_tensor device-wide sync (option 1) - helped, not zero.

None achieved the required ZERO. All are superseded by 14.7.

### 14.5 Phase E - static store is 100% clean (the control)

LLAMA_EXPERT_STATIC_FILE (planted fixed set, heatmap+resync OFF, 0 swaps):
t6, n=1024, 30.36 tok/s, 755 words DIRECT READ = COMPLETELY CLEAN. Zero
mid-word splits. PROVES: zero swaps = zero corruption. The swap itself is
the sole corruption source. (Autofit S is nondeterministic and can OOM
graph capture; pin S manually when testing.)

### 14.6 Phase F - v3 aggressive-repin: fused op is STRUCTURALLY immune

To discriminate "v3 clean because swaps are RARE" (frequency) vs "v3 clean
because fused op is EXACT" (structural): forced v3's repin gate to be
aggressive (dwell 0, ratio 0.3 instead of 32/1.5). Repins = 10,280
(massive). t6, n=1024, cold start, 57.56 tok/s. DIRECT READ of 656 words:
COMPLETELY CLEAN. "iptables", "systemd-resolved", "/etc/netplan/
01-netcfg.yaml", "SSH: 22" all correct. ZERO artifacts at ~10 swaps/token.

CONCLUSION: frequency theory WRONG. The fused single-node MOE_COLD op is
structurally immune to the mid-word corruption regardless of swap rate.
repo2's 3-node separate path is the corruption source.

### 14.7 FINAL TRUTH (current state) - port the fused op

The fix is to PORT v3/v5's fused GGML_OP_MOE_COLD op into repo2:
- One graph node computes the entire cold MoE chain (gate->silu/gelu->
  up->down) under ONE threadpool barrier, with a fixed intermediate
  quantization contract (act_q) that matches what the hot path produces,
  so hot<->cold flips are numerically identical (no divergence to corrupt).
- Repo2's 3-node separate MUL_MAT_ID_COLD path is the REGRESSION.
- Porting it fixes BOTH the -t 12 collapse (3x fewer barriers) AND the
  corruption (structural immunity). Genuine cadence-1/2 becomes possible.
- v3 tree reverted to original gate after the experiment (git diff clean).

FILES for the port (see AGENTS.md "Files you may touch" + approved
llama-graph.cpp):
- ggml/include/ggml.h: GGML_OP_MOE_COLD enum + ggml_moe_cold() decl
- ggml/src/ggml.c: op name, constructor, shape inference
- ggml/src/ggml-cpu/ggml-cpu.c: ggml_compute_forward_moe_cold kernel +
  dispatch registration (~250 lines, the core)
- src/llama-expert-tier.h/.cpp: begin_moe_cold / end_moe_cold + g_hot_only
- src/llama-graph.cpp (APPROVED): cold_ok gate + begin/end calls in
  build_moe_ffn (v5 lines 1974/2111)
- src/llama-context.cpp: wire the fused update path

## 15. Current state and next actions (for the next session)

STATE (2026-08-05): working tree has UNCOMMITTED experimental changes from
the corruption investigation (host-mask, word-boundary gating, rate-limit,
PP-priming, atomic-pair, active-guard, device-sync in ggml-cuda.cu,
graphlog instrumentation, diagnostic env hooks). These are all SUPERSEDED
by the fused-op port decision (14.7) and should be DISCARDED before
porting (git checkout -- the dirty files) to keep the port diff clean.

COMMITTED (unpushed, keep): 
- `5425deb6b` sentinel autofit S-1 fix
- `919168e1e` MSVC Interlocked fallback in cold op

OPEN THREADS:
1. ACTIVE: port the fused GGML_OP_MOE_COLD op (14.7). Clean base first.
2. Decide whether to push the 2 committed commits (needs token).
3. v3 tree is untouched (reverted); v3/v5 remain reference trees for the
   fused op implementation.
4. SYCL backend port of the tier exists (user's agent); the CUDA-only
   guard (strncmp "CUDA") is more restrictive than the backend-agnostic
   code requires - consider an allowlist if shipping multi-backend.
