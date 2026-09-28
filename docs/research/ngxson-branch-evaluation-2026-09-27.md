# ngxson/llama.cpp branch evaluation (2026-09-27)

Branch: `ngxson-featureset` off `origin/master` 589bf18cb. Fork merge-base with ggml-org master:
81bc6b83f (2026-09-26 08:14 UTC). All figures below come from the GitHub compare API
(`repos/ggml-org/llama.cpp/compare/master...ngxson:<branch>`), `gh pr view`, and
`git diff --no-index` of fork files against the upstream blob at 81bc6b83f, taken on 2026-09-27.

## Summary

| Branch | Head | Upstream PR | PR state | In fork tree | Verdict |
|---|---|---|---|---|---|
| `xsn/llama_batch_ext` | 40704338f | #24669 | merged 2026-09-24 (fc343a84b) | yes | already present |
| `xsn/server_spipe_improve` | 92fe35e5e | #25541 | merged 2026-07-11 (ea1f7bbb5) | yes | already present |
| `xsn/server_tools_improve` | ef498290b | #25498 | merged 2026-07-10 (c4ae9a88f) | yes | already present |
| `xsn/server_tools_shell_stream` | 6a0f329d1 | #25526 | merged 2026-07-11 (c92e806d1) | yes | already present |
| `xsn/mul_mat_id_skip` | 23d8ac073 | #26631 | open, CI 9 failures | no | ported (CPU, SYCL, Vulkan) |
| `xsn/llama_batch_ext_2` | 28ce6edab | #29385 | open, mergeable, CI green | no | ported |
| `xsn/llama_batch_ext_mtp` | a748c2705 | none | WIP | no | deferred |
| `xsn/remote_server` | 7c30fcd13 | #24577 (+ #28277) | open, CONFLICTING | no | rejected |

For the four merged branches the PR head SHA equals the branch head SHA, so nothing on those
branches is missing from the merged result. The branch heads are 86 to 1279 commits behind
upstream master and were not used as a source.

## Already present (verify only)

### xsn/llama_batch_ext (#24669)

`include/llama.h` carries the `llama_batch_ext_*` API (23 references), `src/llama-batch.cpp` is
byte-identical to upstream at the merge-base, `common/common.h` has `common_prompt_batch_decode`.
`tests/test-batch-alloc.cpp` is identical to upstream and enabled in `tests/CMakeLists.txt:190`.
Fork delta at these files: `include/llama.h` +36 lines (fork additions), `src/llama-batch.h` +1/-1.

### xsn/server_spipe_improve (#25541)

`tools/server/server-stream.{h,cpp}` are byte-identical to upstream at the merge-base and carry
`server_res_spipe` with the `stream_pipe_producer spipe` member introduced by the PR.
`server-http.cpp` has a fork delta of +44/-12 unrelated to the pipe.

### xsn/server_tools_improve (#25498)

`tools/server/server-tools.cpp` has `class tools_io` (line 147), `tools_io_basic` (298) and the
isolate-backed variant (521); no `apply_diff` remains. Fork delta +92/-39 in server-tools.cpp,
+9/-2 in the header, on top of the merged PR.

### xsn/server_tools_shell_stream (#25526)

`tools/server/server-tools.h` has `support_stream` (line 16), `struct stream` (23) and the
`invoke(json, stream *)` signature (29); `tools/server/tests/unit/test_tools_builtin.py` exists
(fork delta +243/-41, additional tests).

## Ported: xsn/mul_mat_id_skip (#26631)

Semantics: an id of `-1` in `MUL_MAT_ID` skips the slot and zeroes the matching dst row; in
`ADD_ID` it adds nothing. Upstream PR state on 2026-09-27: open since 2026-08-05, +692/-76 over 53
files, 44 checks green, 9 failed (gpu-cuda, gpu-rocm, gpu-vulkan-nvidia-cm, gpu-webgpu-apple,
gpu-webgpu-nvidia, cpu-arm64-graviton4 x2, gpu-vulkan-apple, build-cmake-pkg), 3 cancelled; the
author tested on Metal only and marks the code as fully AI-generated.

What was taken: `git diff f46bc30cb 23d8ac073` restricted to the files this tree still has:
`ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml-cpu/{ggml-cpu.c,ops.cpp,repack.cpp,spacemit/ime.cpp}`,
`ggml-sycl/{add-id.cpp,ggml-sycl.cpp,mmvq.cpp}`, `ggml-vulkan/{ggml-vulkan.cpp,ggml-vulkan-push-constants.h,
vulkan-shaders/add_id.comp,count_experts.comp,mul_mat_vec_base.glsl}`, `tests/test-backend-ops.cpp`.
The CUDA/HIP/MUSA, Metal, OpenCL, WebGPU, Hexagon and zendnn hunks were dropped (backends deleted
from this fork). Everything applied with `git apply -3`; only `tests/test-backend-ops.cpp` needed a
hand merge (fork-added W4A8/W4A4 test structs sit right above `init_mul_mat_id_ids`).

Fork-specific review:

- Fused SYCL decode path `ggml_sycl_mul_mat_id_mmvq_fused` hands `ids->data` to the mmvq MoE
  kernels, so the `mmvq.cpp` hunks (warp returns and writes 0 before touching `vx`) cover it, for
  both the plain and the reordered Q4_K kernels. `check_graph_compatibility` is untouched.
- SYCL sorted path: upstream adds a `stream->wait()` after `k_zero_dst_rows` because the zero
  kernel's row list is a local host vector handed to an asynchronous `memcpy`. The fork drops that
  wait and keeps the list in `ctx.mmid_skipped_row_host`, the way the routed-row list already lives
  in `ctx.mmid_row_mapping_host`: the wait at the top of the next call drains both before reuse,
  and the device copy is a pool allocation reused only in stream order. Without this every MoE
  layer with a skipped slot would have paid a host sync in the batched path, which matters once
  expert partitioning makes `-1` ids routine.
- Master's #67 (grouped dequant XMX GEMM for IQ weights, merged into this branch) consumes the same
  sorted rows: skipped slots own no slice, so its row-count argument is `n_valid_rows`, and the
  skip cases include IQ4_NL so the grouped path meets skipped slots in the tests on devices where
  it dispatches. On this A770 it never does (`fused_gemm_f16_supported()` rejects the reported
  matrix combinations, as #67 documents; a `SYCL_UR_TRACE` run creates no grouped kernel), so the
  change is verified by reading here and the IQ4_NL cases run through the library fallback. The scheduler's
  host-weight expert copy in `ggml-backend.cpp` ignores `-1` ids and copies nothing for an
  all-skipped node; it was the last host-side reader of MoE ids that asserted `id >= 0`.
- MoE cache providers need no change: `moe-cache.cpp` `plan()` (SYCL) and
  `ggml-vulkan-moe-cache.cpp` (Vulkan) preset every slot index to -1 and `continue` on
  `expert < 0`, so a skipped slot is a miss owned by the CPU fallback, and the CPU hook in
  `ggml-cpu.c` zeroes `-1` rows before consulting the slot index. Both providers' `fused_begin`
  are stubs. `topk-moe` fusion emits ids >= 0.
- OpenVINO: `translate_mul_mat_id` and `translate_add_id` lower through Gather, where a negative
  index counts from the end, so a `-1` id would silently pick the last expert (or add the last
  bias row). Not fixable at graph build time; comments at both converters document the gap (same
  as upstream). No model in this tree emits `-1`.
- Test deviation: the PR's new `MUL_MAT_VEC_FUSION` skip cases include the `with_lane_scale`
  variant, whose graph routes ids through `GGML_OP_GET_ROWS`. That op has no `-1` semantics and
  the CPU reference aborts (`ops.cpp: GGML_ASSERT(i01 >= 0 && i01 < ne01)`); this is the likely
  cause of the PR's own CPU CI failures. The fork leaves `with_lane_scale` out of the skip cases.

Why the fork wants it (analytic, not measured): with `-1` skip, a MoE layer's experts can be split
into two `MUL_MAT_ID` ops over disjoint expert subsets (a VRAM slab and a host-USM or CPU set)
using a static I32 id remap (`ggml_get_rows` on an I32 table; SYCL supports I32 rows), which is
the static-placement variant of the expert paging plan implemented at graph level, capturable by
the SYCL graph. This is the design argument for carrying the change ahead of upstream.

Verification (Arc A770, JIT SYCL build `~/build-pr61-sycl`, production `llama-gpu@` running on the
same GPU, correctness only):

| Check | Backend | Result |
|---|---|---|
| `test-backend-ops -o 'MUL_MAT.*' -p skip=1` (170 MUL_MAT_ID + 1 MUL_MAT_ID_FUSION + 154 MUL_MAT_VEC_FUSION) | SYCL0 (A770) | 325/325 |
| `test-backend-ops -o ADD_ID` (36 plain + 36 skip) | SYCL0 | 72/72 |
| `test-backend-ops -o MUL_MAT_ID -p 'type_a=(f32\|f16\|bf16\|q4_0\|q4_K\|q8_0\|mxfp4),'` | SYCL0 | 668/671, the 3 failures are the pre-existing q8_0 amax=1e5 NaN cases (below) |
| `test-backend-ops -o 'MUL_MAT.*' -p skip=1` | Vulkan1 (A770, ANV) | 325/325 |
| `test-backend-ops -o ADD_ID` | Vulkan1 | 72/72 |
| `test-backend-ops -o MUL_MAT_ID -p 'type_a=(f32\|f16\|bf16\|q4_0\|q4_K\|q8_0\|mxfp4),'` | Vulkan1 | 665/665 (6 shapes reported unsupported and skipped) |
| `test-sycl-turbo-correctness` default sweep | SYCL | 0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP |
| `test-batch-alloc` | CPU | 50 tests, 327 assertions, 0 failures |
| `llama-completion` Qwen3-Coder-30B-A3B UD-Q3_K_XL, `--cpu-moe --moe-cache 2048 -lv 4`, `GGML_SYCL_ENABLE_GRAPH=1`, 96 tokens | SYCL + CPU hook + provider | session ready, q3_K and q4_K pools, hits 110715/218347 (50.7%), dispatch-fail 0, collect-fail 0, graphs reused 94, coherent output |

CPU is the reference backend in every test-backend-ops row, so the CPU hunks are exercised by all of them.

Pre-existing issues met on the way, unrelated to the port (each reproduced on a pre-port binary):

- `test-backend-ops -o MUL_MAT_ID` on SYCL aborts at `convert.cpp:792: unsupport data type=tq3_1s`
  before reaching the new cases (already flagged in PR #62); the runs above use a `-p` filter.
- `MUL_MAT_ID(type_a=q8_0, n_mats=8, n_used=2, m=512, n=16|32|64, k=256, amax=100000)` returns
  NaN on SYCL intermittently (n=16: 2/4 pass on the pre-port binary, 3/4 on the ported one; n=32
  and n=64 fail on both).
- With `GGML_SYCL_ENABLE_GRAPH=1`, `test-backend-ops -o MUL_MAT_ID` (same `-p` subset) aborts
  after the `q4_K, n_mats=4, n_used=1, b=0, n=129` case with `Graph nodes cannot depend on events
  from outside the graph`, caught by the function-level handler in `ggml_sycl_mul_mat_id`. The
  next case (`b=1, n=1`) passes in isolation, so the failure depends on the sequence of graphs in
  one process (the SYCL graph record/replay state), and the pre-port binary from master aborts at
  the same case with the same message. The MUL_MAT_ID regression rows above were run with graphs
  off; the skip-case and ADD_ID rows pass with graphs on as well.
- The Vulkan backend of this fork did not build on this box: shaderc 2026.3's spirv-opt rejects
  the OCP FP4 shader variants (`Invalid capability operand: 4229`, 26 shaders, master's untouched
  source fails identically), and the four `ggml_backend_vk_get_*` handle accessors added by the
  2026-09-05 TheTom sync compiled with C++ linkage and hidden visibility, so `libggml-vulkan.so`
  could not resolve them for the Vulkan cache provider (`ggml-vulkan.h` declares them only once
  `VK_VERSION_1_0` is defined, and `ggml-vulkan-types.h` included it before the Vulkan header).
  Both fixed on this branch so the Vulkan hunks could be exercised; the accessor fix ended up as an
  include-order swap in `ggml-vulkan-types.h`, replacing an earlier redeclaration block.

## Ported: xsn/llama_batch_ext_2 (#29385)

Upstream state on 2026-09-27: open, mergeable, review required, 12 checks green, head 28ce6edab
(2026-09-25), 12 commits on top of b248f4a3c. It adds `common_batch` (a wrapper over
`llama_batch_ext` with a token mirror and `add` / `add_embd` / `set_embd` / `set_output`),
`common_batch_get_one`, `common_batch_from_llama_batch`, removes `string_from(ctx, llama_batch)`,
and migrates `server_batch` (renders a sub-batch into a `common_batch` view), every
`common_speculative_impl::process()` (new `const common_batch &` overload; the `llama_batch`
overload stays as a converting shim) and the mtmd helpers.

Applied with `git apply -3` from `git diff b248f4a3c 28ce6edab` over the 11 files. Ten files
applied cleanly (mtmd helpers, `common/speculative.h`, `speculative-simple`, `common/common.{h,cpp}`,
`tools/server/server-context.cpp`); `common/speculative.cpp` had ten conflict blocks, all inside
the DFlash and MTP drafts where the fork carries its own logic on top of upstream:

- DFlash: kept the fork's rule of skipping embedding (image) batches entirely and its NaN /
  f16-overflow sanitising of the gathered target features, then hand the sanitised rows to
  `batch_inject.add_embd()` and `llama_process()` as the PR does. The PR's M-RoPE pinned-position
  skip is unreachable under the fork rule and was not taken. The PR's new `features_buf` member
  duplicated the fork's; one declaration kept.
- MTP: kept the fork's stale-KV trim, deferred catch-up rows (`defer`, `flush_deferred`,
  `drop_deferred_from`, `defer_capacity`), chained drafting (`chain_graph`, `llama_set_mtp_chain`)
  and adaptive depth (`n_cap`), and moved every raw `llama_batch` construction in them to
  `common_batch::add()` + `set_embd()` (rows are copied into the batch by
  `llama_batch_ext::set_token_embd`, so the fork's clear-the-deferred-buffers-before-decode order
  is unchanged). The chained rows that used to be `memset` to zero now point at a per-draft
  `zeros` row. `llama_decode` calls in the fork paths became `llama_process(...,
  LLAMA_PROCESS_TYPE_DECODE, batch.get())`. `batch_capacity` / `ubatch_capacity` stay, they size
  the deferral window.
- Fork-only callers elsewhere (`tools/server/server-context.cpp` and `speculative-simple`) already
  used the API the PR migrates, or keep compiling through the retained `llama_batch` overload.
- Review follow-ups on top of the PR: the legacy `llama_batch` shim
  (`common_batch_from_llama_batch`) now rebuilds the positions a null-position batch was decoded
  with (pos_max counted back over the rows the batch holds) instead of handing out the next
  positions, and skips the memory lookup for an out-of-range sequence id so `add()` reports it;
  the M-RoPE position loops in the mtmd helper and the server's mtmd callback assert
  `n_pos <= GGML_MROPE_SECTIONS` instead of silently truncating. No in-tree caller passes null
  positions to the shim; `speculative-simple` and the server set them explicitly.
- Second review round (unverified external pass, checked here against the code): `common_batch::add()`
  aborts when `llama_batch_ext::set_token_id` rejects an id at or above the draft vocab, and the
  compatibility check tolerates a vocab size difference of 128, so a target token in that gap took
  the server down where the old `llama_decode` path returned an error. `draft-simple` now checks
  target ids against the draft vocab before adding them (`process()` fails like the old decode
  error, `draft()` skips that sequence for the round) and fails `process()` when the target
  embedding cannot be attached, instead of silently decoding the token row alone. The EAGLE3
  encoder passed a scalar position where `add_embd()` reads `n_pos` entries (4 on an M-RoPE
  draft); it now passes a full position row. The legacy shim warns once when a token carries
  several sequence ids, since the draft mirror keeps only the first. Findings about the mtmd
  callback copying embeddings per sub-batch and the shim duplicating `llama_batch_compat` are
  upstream #29385 design and left as they are.
- Third round (re-review of the second, then a third pass on the result): the guard fixed one call
  site, not the class, and the first class-level cut used a per-sequence flag that the third pass
  took apart (not part of the saved draft state, missing on the MTP deferred and chain paths,
  re-enabled by `begin()`, which the server calls after the prefill inside the same request).
  Final shape: `common_batch::add()` and `add_embd()` return the `llama_batch_ext` error code
  (batch full, token outside the vocab, bad seq id or row width) and leave the batch unchanged;
  `llama_batch_ext_add_token` / `add_embd` roll their entry back on a rejected id or width, where
  they used to leave a phantom row that failed the next decode with "all entries in the batch must
  have the same content types" (a #24669 bug as well). Every caller propagates: `common_batch_get_one`
  returns an empty batch and its five callers fail or skip, `common_replay_last_token` and
  `common_prompt_batch_decode` return false, `llama-mtmd-cli` exits, the mtmd helper's `render()`
  returns null and `mtmd_helper_eval_chunk_single` returns an error where both asserted, the
  server's mtmd speculative callback returns the error code, the server's own `render()` asserts
  (its view is sized from the context and its tokens are validated on input). A new `llama_batch_ext_remove_last()` (and `common_batch::remove_last()`) makes the paired
  token-plus-embedding add transactional: when the embedding is rejected the token row is removed
  again, in `draft_add()`, in the legacy shim and inside the C API's own rollback. The SYCL sorted
  path skips its mapping copy when every slot was skipped.
  Drafts: `draft-simple` is the only draft with an independent vocab, so it decides per sequence
  from the draft memory itself, no stored state: a sequence is mirrored only when its incoming rows
  continue `pos_max + 1` of its draft memory; a row the draft cannot take (token outside a smaller
  draft vocab) is logged once per token, the rows before it are decoded, and the lagging sequence
  is skipped by `process()` and `draft()` until the server resets its draft memory for the next
  request, after which it rejoins by itself. Prompt-cache restores carry the draft memory, so the
  rule holds across them. EAGLE3, DFlash and MTP share the target vocab; their remaining rejection
  class, a draft context smaller than the target's batch or sequence count, is checked once at
  construction (`spec_check_draft_ctx`) and fails init with a clear message, so a rejected row at
  run time is an internal error and fails the round or `process()` instead of being swallowed.
  `draft()` never submits an empty batch in any draft.
  The legacy shim (`common_batch_from_llama_batch`) mirrors `llama_batch_compat::init` for
  null-position batches: rows count against their first sequence id only, since that allocator
  advances only that counter (the CodeRabbit thread asked for every id; that was wrong and is
  corrected on the thread), the result is clamped at zero, `n_seq_id` without `seq_id` is treated
  as one sequence, `llama_batch_ext_add_seq` is checked, and an unconvertible row yields an empty
  batch that the legacy `common_speculative_process` overload reports as failure.
  Fifth pass: EAGLE3 and MTP now require the draft vocab to cover the target's at init (EAGLE3
  decodes target token ids in `process()`; the EAGLE3 and DFlash converters inherit the target
  tokenizer but pad to the draft config's `vocab_size`, so only a misconverted or differently
  padded draft trips it); DFlash needs no such rule. Draft-simple's continuity check follows the
  draft allocator: an M-RoPE draft accepts forward position jumps, stale draft rows are trimmed
  with a warning, and the rejection log no longer promises recovery on the next request (a
  prompt-cache entry that stores a lagging draft memory keeps that prefix undrafted). MTP keeps
  its deferred catch-up rows until the whole batch is built; DFlash clamps `n_max` so one noise
  block per sequence fits the draft batch; the draft context is sized to at least the target's
  `n_batch`.
  Left as upstream #29385 design: the mtmd callback copying embeddings per sub-batch (every
  current draft ignores or zero-substitutes embedding batches, so the copy is wasted, noted for
  upstream), the legacy conversion living beside `llama_batch_compat`, and one seq id per draft
  mirror row (the pre-PR drafts asserted it; the shim warns once).
  Repro of the original finding (Qwen2.5-Coder-7B target, vocab 152064, Qwen2.5-0.5B draft, vocab
  151936, same tokenizer, gap 128 accepted by the compatibility check; `/completion` with a token
  array containing id 152000, then a normal request, twice; Qwen3 drafts do not qualify, the
  compatibility check rejects their tokenizer): the PR head before this round aborted the server
  in `common_batch::add()`; the installed pre-PR build returns HTTP 500 "failed to process
  speculative batch" and stays up; this branch returns 200 with the right answer and no drafts for
  the bad request, logs one warning, and the following request drafts normally again (6 of 8
  accepted, the same as a clean run on both builds).

Verification (same box and sharing rules as above; `-ngl 0` for the 9B because production holds
the GPU; `/completion` with `n_predict 48`, `temperature 0`, `seed 1`):

| Check | Result |
|---|---|
| SYCL build of `llama-server`, `llama-completion`, `llama-mtmd-cli`, `test-batch-alloc` | 0 errors |
| `test-batch-alloc` | 50 tests, 327 assertions, 0 failures |
| `llama-server --spec-type draft-mtp`, Ornith-1.5-9B Q4_K_M on CPU, ported build | 48 tokens, drafted 57, accepted 27 |
| same, installed pre-port build (`/usr/bin/llama-server`, pkg `b12275.bb6908513`) | 48 tokens, drafted 54, accepted 28; output byte-identical to the ported build |
| same model, ported build, no draft | 48 tokens; differs from the drafted output after the first sentence (expected temperature-0 drift under speculation) |
| `llama-server --spec-type draft-simple --spec-draft-n-max 8`, Qwen3-Coder-30B-A3B UD-Q3_K_XL `--cpu-moe` + Qwen3-1.7B Q8_0 draft on CPU, ported build | 48 tokens, drafted 131, accepted 28 (21%); shares a 195 / 232 character prefix with the no-draft output of the same build |
| same, installed pre-port build | 48 tokens, drafted 142, accepted 26 (18%); diverges from the no-draft reference after 22 characters; the process dumped core (exit 139) at forced shutdown after a second interrupt, once the request had completed. The ported build exited cleanly. Not investigated. |
| same model, ported build, no draft | 48 tokens, 47 graphs reused |

Speculative decoding changing temperature-0 output is expected on this backend (kernels are not
batch-invariant), so the comparison is on completion, draft statistics and prefix agreement, not
on exact hashes.

## Deferred: xsn/llama_batch_ext_mtp

Head a748c2705 (2026-09-12), no upstream PR, merge-base with upstream 41fc7584f (2026-09-10),
327 commits behind. The branch is the older lineage of `xsn/llama_batch_ext` plus an earlier cut of
`xsn/llama_batch_ext_2`, so everything except one commit is superseded by #24669 (merged) and
#29385 (ported here). The one unique commit, a748c2705 "support vision input for mtp", does:

- split a per-token *state* embedding from the token embedding in `llama_batch_ext`:
  `llama_batch_ext_set_embd_state()` gains a return value, the allocator and `llama_ubatch` get
  `n_embd_state` / `embd_state`, `llm_graph_input_embd_h` takes `(n_embd_inp, n_embd_state)`;
- lets each model's `graph_mtp` accept embedding inputs (vision tokens) instead of only token ids,
  touching 12 `src/models/*.cpp` including `qwen35moe.cpp`, plus `llama-graph.{h,cpp}` and
  `llama-kv-cache-dsv4.cpp`;
- adds `common_batch::set_embd_state()` and switches the MTP draft to it;
- disables `test-batch-alloc` in `tests/CMakeLists.txt` with a "fix this before merging" TODO.

Nothing in it is needed for text-only MTP drafting (the fork's `common_speculative_impl_draft_mtp`
already handles qwen35moe text, and production serves Ornith without a draft). It is WIP quality
(disabled test, no PR, no CI). Deferred until ngxson opens the PR; re-check then.

## Rejected: xsn/remote_server (#24577) and llama-connect (#28277)

#24577 "ui: (demo) access server remotely via webrtc": head 7c30fcd13, 16 commits, +1229/-34,
all under `tools/ui/` (WebRTC tunnel that intercepts `fetch()`, signalling over public WebTorrent
trackers, a pass-code splash and a remote-server registry). The author calls it a PoC that is
"NOT intended to be production-ready"; it is CONFLICTING with upstream master as of 2026-09-27
and caps at the frontend (cannot upgrade the PWA remotely). The server-side follow-up #28277
`llama-server --connect` pulls a prebuilt Rust `llama-connect` binary from a separate repository
at build time (no macOS x64 build, 8 CI failures), which the Arch packaging under
`packaging/arch/` cannot carry and which adds a supply-chain surface for a box that already
reaches `llama-server` remotely over Tailscale (see `/etc/systemd/system/llama-gpu@.service`
and the nginx-to-Tailscale pattern in the user's notes).

Rejected. Re-evaluate only if upstream merges #28277 with a source build of the connector.

## Not claimed

- No benchmark or timing claim for either port. Every GPU run above shared the A770 with the
  production `llama-gpu@` instance, so throughput figures in the logs are noise, not data.
- The `-1` skip is verified by test-backend-ops against the CPU reference and by an end-to-end run
  that never emits `-1`. No model in this tree produces `-1` ids, so the feature has not been
  exercised by a real graph; the expert-partitioning use is a design argument, not a measurement.
- OpenVINO `MUL_MAT_ID` and `ADD_ID` with `-1` ids stay wrong (documented, same as upstream).
- #29385 is unmerged upstream. The next upstream sync may land a revised version and need a
  re-merge of `common/speculative.cpp`, whose fork-specific MTP and DFlash logic sits exactly where
  the PR changes the batch handling.
- Speculative decoding was smoke-tested only through `draft-mtp` on a CPU-resident 9B and
  `draft-simple` with a CPU-resident draft; `draft-eagle3`, `draft-dflash`, `draft-dspark` and the
  chained MTP path (`--spec-chain`) compile but were not run (no draft models on disk; chained MTP
  needs flash attention and the GPU, which production occupies).
- The mtmd helper migration was built (`llama-mtmd-cli`) but not run: no vision projector on disk.
- The server Python test suite (`tools/server/tests`) was not run: it downloads models through
  `--hf-repo`, and the fork's build recipe sets `LLAMA_CURL=OFF`.
- The two Vulkan build fixes were verified only by a Vulkan build and test-backend-ops on this
  box (shaderc 2026.3, ANV on DG2); other shaderc versions were not tried.
- The M-RoPE branch of draft-simple's continuity check, the EAGLE3/MTP vocab rule and the DFlash
  block clamp are verified by reading and building only: no vision draft pair, EAGLE3 or DFlash
  model is on disk. An M-RoPE draft cannot tell a lag from an image jump and keeps mirroring across
  the gap after a rejected token (degraded drafts, no failure).
- The scheduler's `-1` handling in the host-weight expert-copy path is verified by build only; no
  graph in this tree reaches that path with `-1` ids.
- `test-backend-ops -o MUL_MAT_ID` with SYCL graphs enabled aborts in graph capture on this fork
  before and after the port (pre-existing, sequence-dependent); the MUL_MAT_ID regression tables
  are graphs-off runs.
