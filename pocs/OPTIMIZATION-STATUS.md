# Optimization status master list

Scope: Radeon Pro V620 / gfx1030 work around llama.cpp, Qwen3.8/Qwen3.6, DSV4, DFlash/MTP, RCCL, and related inference paths. This is a concise decision index; detailed raw evidence remains in the linked canonical reports and run directories. No production dispatch is changed by this document.

Status vocabulary:

- **RETAINED** — implemented, validated, and intentionally kept (usually opt-in or narrowly dispatched).
- **REJECTED** — implemented/tested, then removed or disabled because a correctness, stability, resource, or performance gate failed.
- **DEFERRED** — not implemented or not promoted because the required mechanism/consumer proof is missing.
- **DIAGNOSTIC** — useful tooling or an accepted non-performance fix, not a speed path.

## Retained / implemented

| Area | What exists | Result / scope |
|---|---|---|
| gfx1030 native umbrella | `GGML_HIP_GFX1030_NATIVE=1`; validated Q4_0 DOT8/MMVQ path | 0/512 A/B mismatches; final `27.60 us`, `max_abs 0.000244141`; opt-in, not a universal Q8 DOT8 path. |
| gfx1030 FlashAttention | Native tiled `fdot2` and wave reductions | `2920/2920` focused cases; end-to-end stock/native stayed within ±1% (`-0.639%` worst reported TG case). Retained. |
| Muse Q8_0 eight-wave MMVQ | Exact shape selector, `K=6656,N=128` only | Muse TG128 `42.057 -> 42.992 tok/s`, `+2.22%`; outputs tolerance-correct. `6656x2048` remains stock. |
| Routed Q4_K/Q6_K six-row MMVQ | Conservative top-k/row selector | 48-case operation sweep `+15.9%` to `+61.2%`; top-k 6/8 regressions excluded. Top-k 8 only uses an explicit validated advisory hint. |
| GDN chunked prefill | gfx1030 chunked scalar/broadcast loads | Direct kernel `+7.9%` at 256 tokens and `+17.7%` at 512; all 36 backend cases passed. Full-model gain is not claimed. |
| Standard Q8_1 graph cache | Exact graph-owned activation reuse plus dual RMSNorm producer | 20 staging launches removed by reuse, 10 more by producer; TG128 `86.2102 -> 86.7479`, `+0.62%`; exact bytes and top-10 checks. Narrow opt-in. |
| Routed SwiGLU → Q8_1 staging | Prompt-only fused staging | Target sequence `41.5 -> 11.6 us`; full PP effect about `+0.3%` (earlier runs near 1%); 330 MB / 280 dispatch byte comparison exact. Narrow opt-in. |
| GDN sibling projection fusion | Packed `[wqkv|z]` and `[beta|alpha]` siblings | Exact hashes for 181 tensors and 32-token output; PP `+1.38%`, TG `+4.39%`; costs +780 MiB total / ~190 ms load. Retained only for eligible Qwen35MoE models. |
| Vocabulary-sharded output | Qwen3.8/DSV4 output-head vocabulary sharding | Full Qwen3.8 Q8 logits byte-identical (`993,280` values); TG `+8.582%` in matched Q8 TP4. DSV4 bounded A/B also exact and ~`19.0 -> 20.3 t/s`. |
| Qwen3.8 DFlash/native MTP | Q4/Q6 rows2, MXFP4 rows2, NVFP4 exact scale decode, grammar fallback, recurrent safe state I/O | Retained production changes include Q4/Q6 rows2 `+7.17%` integrated, MXFP4 `+2.45%`, NVFP4 native-MTP `+7.48%` / external-DFlash `+2.54%`, and exact grammar/recurrent fixes. These are separate from the rejected QPN layout. |
| gfx1030 TP4 greedy speculative target sampling | Automatic backend target sampling for exact structural MTP `n_max=4` and DFlash `n_max=5|7`, large-vocabulary, neutral temperature-zero cases; explicit request choice and auto-disable paths are preserved | MTP A/B/B/A: `65.7854 -> 75.9478 tok/s` (**+15.45%**). DFlash2 n7: `62.3902 -> 69.7006 tok/s` (**+11.72%**). Fresh normal n5 reproduction: `64.68 -> 73.17 tok/s` (**+13.1%**). Exact hashes/acceptance and long-output, prompt-cache, reasoning, stochastic/fallback, grammar, probabilities, cancellation, and parallel-slot checks passed. Q4_0 target verified; mechanism is quant-independent. |
| DFlash2 width-six rows/block=2 | Default-on gfx1030 Q4_0/Q4_K/Q6_K width-six MMVQ route; `GGML_HIP_GFX1030_DFLASH_MMVQ_ROWS2=0|off|false`, unsupported shapes/types, and global auto-off retain prior dispatch | With backend sampling active: `72.717 -> 77.783 tok/s` (**+6.97%**), cycle `51.872 -> 48.554 ms`; target `-2.317 ms`, draft `-1.064 ms`. Raw-F32 synthetic gates and Q4/Q8 long-output, reasoning, grammar, cache, probabilities, parallel, cancellation, and repeated-request checks passed on TP4. Retained as the default for the validated V620 path; TP2 performance is unclaimed. Representative mean `83.85 vs 44.74` AR (`1.87x`), not a sustained 2x claim. |
| Q4_0 DFlash2 M8 rows/block=4 | Automatic native gfx1030 dispatch for certified standard-Q8_1, non-routed Q4_0 width-eight shapes; `GGML_HIP_GFX1030_MMVQ_W8_ROWS4=0` and global auto-off fall back safely | All eight direct shapes and integrated 1,024-token, varied-prompt, cache, graph on/off, fallback, and two-slot checks exact. Locked A/B/B/A `69.7819 -> 70.8576 tok/s` (**+1.5414%**); retained as a safe incremental win, not a >=5% material claim. Resource `128 VGPR / 128 SGPR` |
| Recurrent state/checkpoint handling | Safe transactional state I/O, rollback, exact-prefix reuse | Checkpoint stress `155/155`, restore `152`; full state path retained. Sequence-only restore is not retained (see rejected list). |
| DSV4 TOP_K large-ncols | Upstream bitonic/hipCUB portability fix isolated on a branch | Compile-reviewed for normal and forced non-CUB builds. On the production hipCUB build the new gate is effectively inert; keep only if supporting non-hipCUB environments. |
| RCCL/topology policy | Narrow automatic RCCL selection and fallback policy | Retained from prior Q4/MTP work; forced protocol/peer shortcuts are not retained. |
| TP4 ordinary 5120-FP32 host-snapshot expansion + consumer fusion | With RDNA2 Auto enabled, unset/`auto` selects the validated expanded host-snapshot policy for ordinary `linear_attn_out-*`, `ffn_out-*`, and `attn_output-*` with exact contiguous `[5120,1,1,1]` F32 and `RESHAPE -> ADD -> RMS_NORM -> MUL` graph-prefix guards; `auto-basic` restores the former automatic control, and `0`/`off` opts out; unsupported TP counts/topologies fall back to RCCL | Fresh host expansion: `50.645675 -> 53.552075 tok/s` (`+5.7387%`). Integrated consumer-fused candidate versus production control: `53.4202 -> 53.8177 tok/s` (`+0.744%` mean, `+0.742%` median); deterministic long/cache/grammar/FA/graph/fallback matrix exact/healthy. Ordinary-only; MTP/DFlash are structurally unaffected because the fused prefix requires `ne[1]=1`. Evidence: `/home/edwin/.ralph/gfx1030-host-fused-poc/FINAL-REPORT.md`. |

## Rejected / removed

| Area | Finding | Why it was rejected |
|---|---|---|
| QPN/C2 permanent Q4_0 layout | Exact C2 readers and PF2/PF4 policy; `0/979,740` raw-float differences and no occupancy loss | Q4 subgraph projections were large (`+10.386%` external DFlash, `+3.337%` native MTP) but optimistic end-to-end ceilings were only `+1.764%` request wall and `+0.581%` respectively. Consumer-complete loader/prefill/dequant/copy integration was not worth it. **Production integration: NO-GO.** The raw POC archive was later removed; this decision index is the durable record. |
| QPN activation prepack B | Exact but hot-kernel M5/M8 regressed (`+5.26%` / `+15.42%`) | Setup did not compensate for register/load schedule; retained only as evidence. |
| QPN broad prefetch variants | M2/PF2 `-0.315%`, M3/PF3 `-0.835%`, M6 about `-4%`; PF8 reached 75 VGPR | Width-specific register pressure; only narrow isolated readers were proven, and the production integration was rejected. |
| Native gfx1030 MMVQ nwarps policy | Initial native nwarps source path and simple/K sweeps | Removed at user request; Q4_0/Q8_0 synthetic end-to-end was slower and packed K layouts had no uniform winner. Default `calc_nwarps()` restored. |
| Q4_K/Q5_K/Q6_K new unpack/feed path | Source/ISA inspection found no justified new permutation | No exact arithmetic/layout proof and no measured uniform gain; no code retained. Existing feed paths remain. |
| Q4_0 row sibling fusion | Deterministic output/logit gate failed | Changed first greedy token; `max_abs=0.535`, minimum correlation `0.99852`; reverted. |
| Qwen27 dense MMVQ `rpb2` | Exact kernel and model tokens, but | Kernel shape reduction `7.561%`; TG128 only `+0.066%`, TG512 `+0.207%`, MTP `+0.613%`; failed required model-level gain. `w2v8` diverged token 2. Removed. |
| Qwen27 MTP five-block/per-column collective | Byte-exact controls | Reduction-only grid-five regressed integrated throughput `1.545%`; other publication/accepted-prefix/catch-up shortcuts were unsafe or incomplete. Frozen winner retained, candidates rejected. |
| Direct P2P / mapped-host / device-barrier shortcuts | Some exact microbench controls, no valid production publication or slower path | Cross-root synchronization, fences, or host staging erased the theoretical gain; no unsafe publication shortcut retained. |
| TP4 persistent communication / barrier redesign | Isolated no-op/barrier-noop ceiling and current exact host-snapshot path | Ordinary TP4 server is `+21.46%` over TP2; existing host-snapshot contributes `+2.78–2.95%` over RCCL-only; profiling no-op removes most block reductions but invalidates outputs and yields `+10.06%` target ceiling, while host barriers regress `13.76–13.83%`. Residual TP-sharded LM-head reduction remains mandatory. **Persistent worker: DEFERRED/NO-GO pending a different exact consumer-preserving design.** No production dispatch changed. |
| `hipExtLaunchMultiKernelMultiDevice` P2P path | Exact microbench | Regressed to about `79.40 us`; removed. |
| TP2×PP2 replacing TP4 | Functionally valid hybrid topology | Generation stayed `9.98%` behind TP4 with backend sampling; optimistic lower bound still `8.49%` behind. Prompt concurrency can improve, but no decode win. Experimental only. |
| TP1×PP4 / graph-overlap / internal-allreduce TP alternatives | No generation crossover | TP1×PP4 about `16.1 TG t/s`; graph/internal-allreduce and boundary sweeps were flat/slower. Not retained. |
| RCCL HIP graph capture | Multi-device graph instantiate failed `invalid device ordinal`; per-device replay was `-0.6%` to `-1.8%` | `NCCL_GRAPH_REGISTER=1` was flat; architecture keeps grouped allreduce host-side. No production graph change. |
| Forced BF16 small allreduces | Target ~20.4→20.1 t/s; MTP ~38.6→28.5 t/s | Verification logits changed enough to reduce acceptance; current FP32-small policy is correct. |
| DSV4 indexer TP / candidate-row sharding | Focused unit tests passed, deterministic tensor e2e diverged after 8 tokens | Requires global top-k candidate-ID merge and routing protocol; reverted. |
| Qwen3.8 AutoRound fastest Q4_0 / Ring-LL controls | Fastest Q4 `45.257 t/s`, only `98.4375%` top-1 agreement; Ring/LL `50.0689 t/s`, stream diverged at token 120 | Exact output/token gate failed. Controls are not production settings. |
| DFlash mirrored/shared output-head experiments | Large raw-logit divergence (`19,833,837 / 29,798,400` values) and shared-head placement failures | Output sharing/layout assumptions were invalid for the external draft model; reverted. |
| Fixed MXFP4 lookup-table specialization | Exact (`0/34,816`) but `57.07` vs `57.36 us` | Neutral within noise; static certified rows2 path was retained instead. |
| NVFP4 width-eight rows2 | Correctness passed, but roughly `48.8%–55.4%` slower at ~211 VGPR | Register/occupancy failure; rows1 PF path retained, rows2 rejected. |
| Raw MXFP4/NVFP4 `udot8/sdot8` substitutions | Not a valid representation of the formats | Arithmetic/packing mismatch; explicitly not implemented. |
| DSV4 BF16 source candidate | Argmax sometimes matched, but violations `35,662/83,471/24,359/81,175`, RMSE up to `0.18557` | Failed strict numerical gate; no long run authorized. |
| Q4S8 source/imatrix experiment | Detached `q4s8-src` worktree had an uncommitted `tools/imatrix/imatrix.cpp` change | No retained runtime result or production commit; worktree removed during cleanup. Non-Q4S8 model files were preserved outside the worktree. |
| Qwen27 GDN stateful POC | Decode diverged at generated token 19 | Recurrent/state publication semantics were not exact; reverted. |
| Sequence-only recurrent restore | `516,983 / 516,900 / 516,930` tolerance violations at 2K/3K/16K; max error up to `4.118656` | Full context state is required; sequence-only reuse rejected. |
| DFlash2 2x sprint alternatives | Width-six rows/block=3/4, Q4_0 eight-wave, generic helper/shape-gated rows2, `p_split` tuning, forced RCCL settings, host-snapshot width changes, and gate+up+GLU fusion | Rows3 was exact but dominant Q4_K regressed `16.48%`; rows4 was slower; eight-wave failed `4916/24576` raw-F32 values; helpers erased the gain; `p_split` was neutral; RCCL overrides regressed. The old width-two P2P POC targeted a nonexistent payload; a corrected exact precompiled TP4 width-six route saves only `0.347 ms/cycle` (`+0.738%`) and is retained behind startup RCCL exactness validation. Width-six GLU fusion raised VGPR `64 -> 96` and regressed DFlash `4.63%`. Removed or never promoted. |

## Deferred / deliberately not implemented

- Permanent C2 loader integration for prompt prefill/MMQ, dequant/conversion, views, copies/offload, CPU/backend fallbacks, and graph widths beyond M8. The isolated readers remain archive-only.
- Q4_K/Q5_K/Q6_K DOT8 and a separate Q8 DOT8 implementation. Existing exact SDOT4/DP4A paths are already native enough; no proof justified a new format-specific path.
- DSV4 expert ownership, two-exchange/layer redesign, candidate-row global top-k merge, and communication fusion. These are scheduler/protocol projects, not safe local kernel substitutions.
- Generic RCCL graph capture for the current single-process multi-device scheduler.
- Hidden-axis output sharding, unvalidated on-device vocabulary-parallel sampling, and any optimization that changes reduction order without a raw-logit gate.
- DFlash2 sustained 2x remains open. The retained n5 sampler+rows2 candidate measures `77.783 tok/s` at `48.554 ms/cycle` versus a `89.784 tok/s` / `41.477 ms` target. The terminal current-rows2 census plus nwarps2 exactness screen found no remaining local candidate with a >=5% engine ceiling; further work requires a new overlap-preserving design rather than another local row/warp tweak. Continue only exact TP4 M6 work with enough ceiling to remove roughly `7 ms/cycle`: overlap-preserving skinny-GEMM/weight reuse, consumer-fused LM-head reduction, or GPU-side acceptance. Do not retry single-GPU verification, naïve chunking, new direct width-six host-snapshot variants, gate/up serialization, the rejected RCCL settings above, or sub-5% orchestration micro-optimizations.

- FFN/attention activation-residency fusion across the `ffn_out` → next projection boundary. The measured hidden partial is 20,480 bytes while candidate packed next QKV/GDN outputs are approximately 57,344–65,536 bytes before synchronization/extra computation; the initial ceiling did not justify changing activation ownership or reduction order. Defer unless a new packed-layout proof beats the exact narrow host-snapshot path.

## Canonical evidence and cleanup

Keep these canonical records:

- `/home/edwin/llama.cpp-rdna2/pocs/OPTIMIZATION-STATUS.md` — this concise decision index.
- QPN/C2 raw archive — deleted at the user's request after its findings were condensed here; no production code depended on it. The absent archive is not a verification artifact and must not be reconstructed without explicit authorization.
- `/home/edwin/.ralph/*.md` — execution ledgers; retain until the project is formally closed.
- `/home/edwin/models/*-runs/`, `/home/edwin/hybrid-tp-pp-evidence/`, and named final-package directories — raw evidence referenced by this list.
- Accepted source/report worktrees and their final verifiers, especially the gfx1030 native paths and vocabulary-sharding work.

Duplicate narrative cleanup completed on 2026-08-20. The QPN raw archive and stale FA/q4s8 worktrees were subsequently removed on request; only this concise index remains in the production repository. Removed temporary narrative files after their metrics were condensed above:

- `/Users/edwin/.tmp/beat-tp4-RESULTS.md`
- `/Users/edwin/.tmp/tp2-pp2-vs-tp4-q8-optimization.md`
- `/Users/edwin/.tmp/qwen38-tp4-performance.md`
- `/Users/edwin/.tmp/gfx1030-native-optimizations.md`
- `/Users/edwin/.tmp/qwen27-mmvq-autotune/RESULTS.md`
- `/Users/edwin/.tmp/qwen27-mmvq-autotune/EXTENDED-RESULTS.md`

Raw run directories, final-package manifests, source patches that are the only record of rejected correctness experiments, Ralph ledgers, and accepted worktrees were not deleted.

## Bottom line

The largest durable gains came from narrow, exact, model-aware paths: vocabulary-sharded output, Q4/Q6/MXFP4/NVFP4 dispatch improvements, native GDN/sibling fusion, exact Q8 cache reuse, and carefully gated gfx1030 arithmetic. The broad ideas that repeatedly failed were alternate tensor topologies, generic communication shortcuts, persistent alternate weight layouts, and arithmetic changes that relaxed raw-logit/recurrent exactness.
