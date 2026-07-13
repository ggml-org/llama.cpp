# DeepSeek-V4 MTP for llama.cpp — project plan (started 2026-07-12)

**Goal:** first working V4-Flash MTP (nextn) on llama.cpp: converter mapping + graph
implementation → ~2× decode (proven 2.2× on Qwen3.6-27B with `--spec-type draft-mtp`).
Ends in a potential upstream PR + the first public V4-MTP GGUF (rogerai variant).

**Why it doesn't exist:** conversion/deepseek.py drops `mtp.*` ("conversion v0"),
`src/models/deepseek4.cpp` has zero nextn code (mainline/taco/cchuter all lack it).
V3.2 (`deepseek32.cpp`) loads nextn "preserved but unused"; qwen35/glm4/step35 have
working driver integration via `common/speculative.cpp` (`draft-mtp`).

## Authoritative reference (ON DISK)
`/home/luis/ai/models-hf/deepseek-v4-flash-orig/inference/model.py` — DeepSeek's own
PyTorch. `class MTPBlock(Block)` (line ~738):
```
e = enorm(embed(input_ids))          # embed SHARED with main model
x = hnorm(x)                          # x = main hidden states [b,s,hc,d]
x = e_proj(e).unsqueeze(2) + h_proj(x)
x = Block.forward(x, ...)             # FULL standard V4 layer (MLA attn + MoE + hc)
logits = head(x, hc_head_fn/scale/base, norm)   # head SHARED; hc_head mixes hc dim
```
- MTP layer_id = n_layers (43). NO indexer tensors → uses the non-indexer attn path.
- embed + lm head shared with main model (no mtp embed/head weights).
- `e_proj(e) + h_proj(x)` ≡ `eh_proj(concat[e;x])` with W_eh = [W_e | W_h]
  → pack into existing LLM_TENSOR_NEXTN_EH_PROJ {2*n_embd, n_embd}.

## Source tensors (1,575 under `mtp.0.*`) → GGUF mapping design

| source (mtp.0.) | GGUF name | notes |
|---|---|---|
| attn.* (wq_a/wq_b/wkv/wo_a/wo_b, q_norm/kv_norm, attn_sink) | `blk.43.attn_*` (standard V4 names: attn_q_a, attn_q_b, attn_kv, attn_output_a, attn_output_b, attn_q_a_norm, attn_kv_a_norm, attn_sinks) | reuse regular-layer mapping at bid=43 |
| ffn.experts.N.w1/w2/w3 (+scale) | `blk.43.ffn_{gate,down,up}_exps` | merge 256 experts + FP8-dequant + MXFP4 repack (existing code paths) |
| ffn.shared_experts.w1/w2/w3 | `blk.43.ffn_{gate,down,up}_shexp` | standard |
| ffn.gate.weight/bias | `blk.43.ffn_gate_inp` (+ tid2eid?) | match regular layer handling |
| attn_norm / ffn_norm | `blk.43.attn_norm` / `blk.43.ffn_norm` | standard |
| hc_attn_* / hc_ffn_* | `blk.43.hc_attn_*` / `blk.43.hc_ffn_*` | standard V4 hc names |
| e_proj.weight + h_proj.weight | `blk.43.nextn.eh_proj.weight` = concat(W_e, W_h) dim-wise | EXISTING tensor type; FP8: dequant both, concat, requant |
| enorm.weight | `blk.43.nextn.enorm.weight` | existing |
| hnorm.weight | `blk.43.nextn.hnorm.weight` | existing |
| norm.weight | `blk.43.nextn.shared_head_norm.weight` | existing (pre-head norm) |
| hc_head_fn / hc_head_base / hc_head_scale | `blk.43.nextn.hc_head_{fn,base,scale}` | NEW tensor types (3, small F32) |

## Phases
- **B (converter)**: conversion/deepseek.py — remove the `mtp.` filter for a
  `--keep-mtp` mode (or always), rename mtp.0.→model.layers.43. style mapping into
  the existing modify_tensors flow, special-case the 6 glue tensors + eh_proj concat.
  gguf-py: add the 3 new NEXTN_HC_HEAD_* tensor types (constants.py, tensor_mapping.py).
  Verify: converted GGUF has ~1370 tensors incl. blk.43.* + nextn.*.
- **C (C++)** — detailed design (recon done 2026-07-12 evening):
  Template = `src/models/qwen35.cpp`, the complete working pattern: `load_block_mtp(il)`
  at il=n_layer; in build_graph `if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP)
  return graph_mtp(...)`; graph_mtp contract: input `llm_graph_input_embd_h`
  {tokens, embd, h (ggml name "mtp_h_input")}; front-end enorm(tok_embd) + hnorm(h)
  → ggml_concat(dim 0) → eh_proj → layer body → `res->t_h_nextn` (pre-head hidden)
  → get_rows(inp_out_ids) → shared head → `res->t_logits`. Driver =
  common/speculative.cpp draft-mtp "single trained head" mode (same as qwen35).
  V4 specifics (established from reference inference/model.py):
  1. **MTP layer attention = pure SWA MLA**: no compressor, no indexer
     (`compress_ratio=None`), base rope_theta, NO YaRN, window=128. deepseek4.cpp
     already builds this variant for uncompressed trunk layers → reuse at il=43.
  2. **Hyper-connections**: MTP block is a full V4 Block → hc_attn/hc_ffn machinery
     (already per-layer in deepseek4.cpp). Tail uses its OWN nextn.hc_head_fn/base/
     scale to mix hc streams → 1, then nextn.shared_head_norm → shared model.output.
     (ParallelHead.hc_head: flatten hc → rsqrt-RMS ×; mixes = hc_fn·x; pre =
     sigmoid(mixes*scale+base)+eps; y = Σ_hc pre·x.)
  3. **Width wrinkle**: V4 trunk state is hc_mult×n_embd wide (hc streams). MTPBlock
     takes x [b,s,hc,d]; h_proj applies per-stream (last dim), e_proj(e) broadcast-
     adds across hc. So: (a) target graph must expose t_h_nextn = pre-hc_head trunk
     state (hc*d wide), (b) driver width assert
     (`n_embd == llama_model_n_embd(tgt)`) needs hc-aware width, (c) in the MTP
     graph, eh_proj applies per-hc-stream (3D view [d, hc, n_tokens]).
  4. **KV cache filter**: deepseek4's llama_kv_cache_dsv4 block in llama-model.cpp
     needs the STEP35-style ctx_type filter (MTP ctx → il >= n_layer only) so
     ctx_dft allocates cells for just the SWA MTP layer (tiny: window 128).
  5. **Loader**: read LLM_KV_NEXTN_PREDICT_LAYERS; loop one extra il with the
     uncompressed-layer tensor set (compressor/indexer TENSOR_NOT_REQUIRED at
     il>=n_layer) + glue: eh_proj {2*n_embd,n_embd}, enorm/hnorm/shared_head_norm
     {n_embd}, hc_head_fn {hc_dim, hc_mult}, hc_head_base {hc_mult},
     hc_head_scale {1}. llama-arch.cpp: 3 new nextn.hc_head_* names +
     LLM_TENSOR_INFOS entries.
  6. **Main-graph addition**: set res->t_h_nextn in deepseek4's trunk graph
     (pre-hc_head state) so the driver can capture verify_h rows.
- **D (validate/ship)**: draft-mtp acceptance rate + output-identity checks,
  quantize Q4_K-MTP (pin nextn/indexer/attn Q8: llama-quantize --tensor-type),
  benchmark tok/s vs 16.5 no-spec, upstream PR, rogerai HF GGUF.

## Workspace
- Dev tree: `/home/luis/ai/build/llama.cpp-mtp-dev` (branch `dsv4-mtp`, worktree of
  the pr25545 clone, base e624d0f = master+#25545).
- Source weights: `/home/luis/ai/models-hf/deepseek-v4-flash-orig` (149GiB — KEEP).
- No-MTP convert output `/home/luis/ai/models/deepseek-v4-flash-q8-mtp.gguf` (156GB,
  MXFP4 experts + Q8, NO nextn) — reproducible; delete if disk needed.
- Convert cmd: `PYTHONPATH=gguf-py TMPDIR=/home/luis/ai/models/.convtmp
  vllm-venv/bin/python convert_hf_to_gguf.py <src> --outfile <out> --outtype q8_0
  --use-temp-file` (/tmp is 8G tmpfs; converter OOMs without temp-file).
- Serve/fit recipe for 175GB-class: see `~/ai/build/serve-q4.sh` + `bench-env.sh down`.
- MTP flags: `--spec-type draft-mtp --spec-draft-n-max 3 --parallel 1`.

## Key files
- conversion/deepseek.py (`DeepseekV4Model`, `filter_tensors` drops mtp.*, line ~509)
- gguf-py/gguf/constants.py + tensor_mapping.py (tensor enums/names)
- src/llama-arch.cpp (~line 492: NEXTN names; per-arch usage via llama-model.cpp)
- src/models/deepseek4.cpp (no nextn yet), deepseek32.cpp (loads nextn, lines 141-150),
  qwen35.cpp (working MTP graph), common/speculative.cpp (~line 1200: draft-mtp driver)
- DeepSeek ref: models-hf/deepseek-v4-flash-orig/inference/model.py (MTPBlock ~738)
