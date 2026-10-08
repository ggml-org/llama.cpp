# TurboQuant+ fork — upstream catch-up merge notes

Merges `upstream/master` (ggml-org/llama.cpp) into
`feature/turboquant-kv-cache` (the fork's active branch). This is a **merge**
(not a rebase): every fork commit, hash, and author is retained, and the merge
commit keeps `feature/turboquant-kv-cache` as its first parent.

Base: `feature/turboquant-kv-cache` (the canonical branch, 262 fork commits +
a prior partial upstream sync). Note: `master` is a stale snapshot (2 months
behind feature); do not catch that up instead.

## Verified on M5 Max (Metal)

- Build: green (full `cmake --build`, llama-cli + llama-quantize).
- Turbo KV A/B vs f16 baseline (gemma-4-12B-it-Q8_0): turbo4/turbo3 match f16,
  turbo2 coherent. The attn-rotation default-off policy holds (no double-rotate).

## Contributor work preserved (verified in the merged tree)

- **CUDA TurboQuant (Gabe Ortiz / signalnine, 27 commits):** turbo2/3/4 CUDA
  kernels, TQ4_1S/TQ3_1S native + fused mul_mat_vec, warp-cooperative dequant,
  WHT, InnerQ, sparse-V dequant, MLA fixes, cross-type VEC FA, D=640 MMA FA.
- **HIP/ROCm:** variadic `__shfl_*_sync` macros, HIP VEC-force for quantized KV,
  the #176 graph-capture turbo-KV decode crash fix (in auto-merged regions).
- **Metal TurboQuant:** TQ3-rotated mul_mm path + turbo_wht pipeline + bookends.
- **Core KV cache:** auto-asymmetric turbo-K upgrade (GQA >= 6 -> K to q8_0),
  empirically-tuned attention-rotation policy (default OFF + per-side
  `LLAMA_ATTN_ROT_K/V_OVERRIDE` env knobs), turbo head zero-padding, +3 rotation
  tensor overhead, `n_layer_kv()`.
- **MTP / draft:** gemma4-assistant + masked-embd tensors, draft-MTP server
  multimodal processing (`[TAG_MTMD_DRAFT_PROCESSING]`), `llama_get_ctx_other`,
  speculative impls.
- **Fork server features:** `get_slot_by_cache_key` / cache-key slot binding
  (unioned with upstream's new `get_slot_by_cmpl_id`).
- **Vulkan turbo3 KV cache:** dequant/get_rows/set_rows/cpy pipelines (the FA
  fast-path is deferred, see below).

## Notable resolutions

- **`llama-kv-cache.cpp` constructor:** unioned upstream's shared-cells refactor
  (`other`/`v_cells_impl`/`v_cells`) with the fork's auto-asymmetric turbo block
  and `n_layer_kv`.
- **attn-rotation:** kept the fork's default-off + per-side-override policy,
  grafted in upstream's DeepSeek-V3.2 DSA lightning-indexer force (a model
  requirement, guarded by `LLAMA_ATTN_ROT_DISABLE`).
- **`fattn-common.cuh`:** took upstream's `f16_extra` refactor (graph-allocated
  f16 KV scratch); it supersedes the fork's HIP pool workaround and is
  graph-capture-safe by design.
- **server `get_available_slot`:** kept the fork signature
  (`allow_prompt_similarity`) that the shared body relies on, alongside the new
  `get_slot_by_cmpl_id`.
- Took upstream for build/UI refactor (LLAMA_BUILD_APP, llama_ui assets, xxd
  removal), model evolution (n_layer -> n_layer_all rename, gated_delta_net
  signature, nextn/MTP additions), tokenizer/vocab/normalizer additions.

## Build artifacts fixed (auto-merge dual-additions / API drift)

- `ggml.c`: `GGML_OP_COUNT` static_assert 97 -> 98 (fork's `TURBO_WHT` op).
- `llama.h` + `common.h`: duplicate `n_outputs_max` member (both sides added it
  in different field order); kept one, upstream order.
- `llama-context.cpp`: matching duplicate `n_outputs_max` initializer removed;
  added the missing `const auto n_embd = hparams.n_embd;` that upstream's
  layer-input-embeddings code needs.
- `llama-vocab.h`: duplicate `get_suppress_tokens` decl.
- `clip.cpp` / `models.h`: duplicate `PROJECTOR_TYPE_GEMMA4UA` case +
  `clip_graph_gemma4uv` struct.

## DEFERRED — Vulkan turbo3 flash-attention re-port (NOT lost)

Upstream evolved the Vulkan flash-attention stack further than the fork's last
sync. The fork's turbo3 Vulkan FA (`flash_attn_cm1.comp`,
`flash_attn_dequant.glsl`, `ggml-vulkan.cpp` dispatch) conflicted with upstream's
newer FA changes; per decision, upstream's FA was taken and the turbo3 Vulkan FA
must be re-ported and validated on the AMD RDNA4 box. The turbo3 KV-cache Vulkan
pipelines are preserved; only the FA fast-path needs restoring.

Source to re-port from (present in `feature/turboquant-kv-cache`):
- `a09bafedd` vulkan: restore turbo_wht op + turbo3/4 FA dispatch
- `ff8bb7394` (Simon Gardling) vulkan: fix and complete turbo3 KV cache support
- `a494833d0` / `0198d5819` (Tuklus-Labs) Vulkan turbo3 KV + coopmat FA

Note: the Vulkan backend cannot be built on the M5 (no Vulkan); shader-gen still
emits turbo3 FA SPIR-V, so the Vulkan build will need reconciliation on the AMD
box (this is expected and tracked).

## TODO before relying on this merge

- [ ] AMD RDNA4 box: build Vulkan, reconcile shader-gen, re-port turbo3 FA,
      smoke-test turbo KV.
- [ ] 5090: build CUDA, run turbo KV correctness + perf tests.
- [ ] M3 GGUF Config-I: with this catch-up + the MiniMax-M3 support PR
      (upstream #24523), the fork can quantize M3 to Config-I in GGUF.
