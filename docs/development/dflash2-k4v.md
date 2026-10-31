# Bounded DFlash2 + K4V with adaptive KV streaming

The bounded seven-token implementation is complete on `feature/adaptive-kv-dflash2-k4v`. It combines the existing CPU K4V map with the admitted Qwen DFlash2 sidecar, shared phase arena and target verification. K4V remains opt-in: skipping neural predictions does not guarantee faster generation.

## Supported scope

- The pinned Qwen3.8-27B target (`qwen35` metadata) and matching [DFlash2 sidecar](../../benchmarks/data/dflash2-qwen27b-artifacts.json), one serial text request on one CUDA GPU.
- Independently configured DFlash proposal maximum F=1-7 and K4V continuation maximum M=1-7. Target verification and recurrent rollback use max(F,M); verification executes TG1-TG8, including intermediate widths. Neural prediction still respects its own trained block size.
- Full-model qualification covers Q8_0/Q4_0, Q8_0/Q8_0 and Q5_1/Q5_1. Adaptive drafting inherits target cache types; explicit draft-cache overrides are rejected.
- No simultaneous MTP, image/projector, parallel-slot, multi-GPU, Gemma+DFlash, DSpark or EAGLE admission. Native non-streaming modes retain their separate contracts.

The unfinished K15/TG16 extension is preserved on `wip/dflash2-k4v-k15-20261031` at `e59e929e9`. It is outside this release. Its source and partial evidence were preserved; they are not labelled qualified.

## Run it

Build this branch using the [normal fork build instructions](../../README.md#quickstart). Substitute model paths and fit the arena on your own GPU. No existing service is required; these tools start and stop their own server.

```bash
KV_SERVER=./build-v2/bin/llama-server
KV_MODEL=/path/to/Qwen3.8-27B-UD-IQ4_XS.gguf
KV_DFLASH=/path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

python3 benchmarks/find-max-arena.py \
  --server "$KV_SERVER" --model "$KV_MODEL" --model-draft "$KV_DFLASH" \
  --context-size 8192 --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-mode dflash-k4v --draft-length 7 \
  --ngram-n 12 --ngram-m 7 --ngram-min-hits 2 \
  --batch-size 256 --ubatch-size 256 --decode-tokens 256 \
  --probe-mode preflight --validate-selected --no-webui \
  --output benchmarks/results/my-dflash-k4v-fit
```

Preflight retains the requested context allocation and executes bounded text plus actual decoding. Mixed sizing also runs a separate 32-output copying/resource probe and requires both K4V proposals and real DFlash prediction forwards. That probe uses constrained target sampling only to prove the routes and is not a performance or quality measurement. `--validate-selected` executes the requested full text workload once at the selected arena. A failed selected validation is reported, not silently replaced by a smaller workload. The default `--probe-mode full` executes the full workload at every candidate.

`--initial-arena-mib` is an optional search guess. Read `arena_mib` from the resulting `result.json`; `next_mib_rejected` records the tested next MiB when available. Fit depends on weights, context, cache types, batch sizes, draft settings and other GPU allocations. Refit when they change.

```bash
KV_ARENA=$(python3 -c 'import json; print(json.load(open("benchmarks/results/my-dflash-k4v-fit/result.json"))["arena_mib"])')
"$KV_SERVER" --model "$KV_MODEL" --model-draft "$KV_DFLASH" \
  --ctx-size 8192 --parallel 1 --batch-size 256 --ubatch-size 256 \
  --n-gpu-layers 999 --spec-draft-ngl 999 --flash-attn on --fit off \
  --shared-device-memory-mib "$KV_ARENA" \
  --cache-type-k q8_0 --cache-type-v q4_0 --no-mmproj \
  --spec-type ngram-map-k4v,draft-dflash \
  --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-draft-p-min 0 \
  --spec-ngram-map-k4v-size-n 12 --spec-ngram-map-k4v-size-m 7 \
  --spec-ngram-map-k4v-min-hits 2 --offline --no-context-shift
```

N is the lookup-key length; M is the proposed continuation length. K4V's four values are alternative continuations, not simultaneous verification branches. The native default M=48 exceeds the bounded limit, so set M explicitly. H2 above is a conservative example; the H1 research measurements below do not establish a universal best setting. CLI order does not change K4V-before-DFlash priority. A miss or empty proposal falls through to DFlash.

## State, ownership and memory

K4V stores its index, continuation statistics and token-history references in RAM. The hash index is approximately 1 MiB, plus dynamic keys, statistics, history capacity and allocator overhead. Maps and acceptance feedback rebuild at each generation boundary, including equal-length prompt replacement. Restarts/restores may therefore change proposal counts relative to uninterrupted generation.

Every successful target verification still supplies DFlash features, even when K4V supplied the proposal. Acceptance trims both target and child provisional suffixes; source-specific acceptance feedback goes to the implementation that supplied the draft. Clear, cancellation, checkpoint/window recovery and reconstruction preserve the existing target safety guards.

Target/sidecar weights stay outside the arena. Target resident KV/ring, persistent bounded draft KV, phase workspace and staged rollback publication retain the existing owner. Recurrent snapshots remain host-spilled; the selected restore uses the existing publication copy backend. The release adds no GPU map, independent arena, GPU buffer or attention arithmetic.

Two qualified scheduling fixes are included:

1. `985c8cac4` registers flexible target-prefetch width only after successful DFlash construction. The owner restores prior append/read intents at teardown. Generation, physical layout and width checks still govern prime adoption.
2. `8793469b2` submits the selected recurrent restore after draft split inputs are uploaded and before draft graph execution. Target re-entry, state save/load, clear and destruction retain completion fences. Logical rollback remains single-use until a fresh target capture, even after DMA completes. The scheduler hook defaults to absent for other callers.

## Measured performance

The following are separate fixed-1-GiB, ordinary baseline/candidate experiments on natural NVIDIA Open-SWE prefixes. Each has one fresh pair per mode/prompt, alternating order, unchanged IQ4_XS target/Q4 sidecar/Q8K-Q4V/bub256/UVM-off and normal graphs. Each pair matches generated IDs, request/source counters and physical decode layout. They are not repeated statistical trials.

| Optimization | Prompt tokens | DFlash7 decode change | DFlash7 + K4V decode change |
| --- | ---: | ---: | ---: |
| Flexible target prefetch | 8,186 | +0.85% | +0.01% |
| Flexible target prefetch | 64,865 | +0.38% | +0.11% |
| Flexible target prefetch | 126,794 | +3.40% | +3.40% |
| Restore overlap, over the prefetch-fixed baseline | 8,186 | +5.35% | +5.13% |
| Restore overlap, over the prefetch-fixed baseline | 64,865 | +5.20% | +5.39% |
| Restore overlap, over the prefetch-fixed baseline | 126,794 | +0.86% | +0.30% |

Do not add or multiply gains from different cohorts. Prefill changes were within 0.11% and 0.42%, respectively. Restore-overlap MTP3 controls changed +1.05%, +2.48% and -0.09% at these prompts. The old introductory full-context chart was not rerun with these fixes.

Final-source warm Nsight captures match ordinary outputs/work and show restore-copy/draft-kernel overlap of 78.02% for DFlash7 and 51.14% for MTP3. These are actual CUDA launch correlations and clipped interval unions, not occupancy, bandwidth-saturation or instrumented throughput claims.

Adding K4V itself has a different, smaller effect. On 20 natural Open-SWE cases, H2 matched all DFlash outputs with median decode change -0.042%; H1 matched 18 with median +0.176%. Two H1 responses differed and were excluded individually from equal-work timing. Saved predictions often leave the same number of expensive target passes. K4V is not advertised as a general speedup or a task-quality improvement.

A separate six-context sample used independent maximum-fit arenas for MTP3, DFlash7 and mixed mode, with selected full requests and actual rejected +1 MiB neighbors at all 18 points. That sample uses the earlier prefetch-only runtime. It covers natural prompt bands near 8/64/128/160/200/256 Ki; it is not an equal-memory/equal-response comparison or a fresh restore-overlap sweep. The earlier 96-point sweep remains intentionally stopped.

Data: [prefetch CSV](../../benchmarks/data/dflash2-k4v-flexible-prefetch.csv), [restore CSV](../../benchmarks/data/dflash2-k4v-restore-overlap.csv), [maximum-arena sample](../../benchmarks/data/dflash2-k4v-swe-maxarena.csv), [NK7 evidence index](../../benchmarks/data/dflash2-k4v-nk7.json). Inputs are pinned historical prefixes, not synthetic repetition or live agent task-success evaluation; no replayed tool command is executed.

## Lookup cost

Historical accepted whole-call diagnostics, before the two scheduling fixes, measured the following. They include initial indexing/statistics work rather than timing one hash access. Full p50/p95/p99 counts and first-statistics/outcome classes are in the NK7 JSON.

| Context / fixture | Calls | All-call p50 / p95 / p99 (us) | Steady p50 (us) | Map reserved bytes |
| --- | ---: | ---: | ---: | ---: |
| prose-8192 | 89 | 0.586 / 0.872 / 161.787 | 0.584 | 1048576 |
| prose-65536 | 69 | 0.638 / 0.973 / 879.224 | 0.635 | 1048664 |
| copy-code-8192 | 32 | 1.618 / 10.920 / 33.531 | 1.010 | 1049984 |
| copy-code-65536 | 32 | 14.805 / 97.487 / 639.936 | 1.059 | 1049984 |
| ambiguous-code-65536 | 32 | 1.063 / 84.538 / 269.912 | 1.036 | 1051392 |

Small nearest-rank tail samples are not latency guarantees. The map column excludes history capacity, allocator overhead and host KV. Lookup, driver and process host intervals can nest; adding them would double-count work. These synthetic copy/ambiguity diagnostics characterize the mechanism, while the natural Open-SWE comparisons above establish the practical performance scope.

## Qualification and reproduction

NK0-NK6 cover CPU n-gram correctness, independent widths, every accepted prefix, exact quota boundaries, native/streamed values, lifecycle/recovery and three full-model cache pairs. The two optimization receipts preserve all output/source/layout checks. Restore qualification adds 21 accepted GPU gates, five strict zero-error summaries, nine coherent CPU ASAN/UBSAN suites, file-parser ASAN, 18 ordinary requests and two final-source profiles.

Strict checks keep default PDL, API-all reporting, error exit 99 and no filters; eager execution is limited to instrumented runs. Full F7 matched-streamed controls compare 3,195 records/~10.47 GB. Fitting native F3 controls cover the three cache pairs. The streamed MTP strict process checks all 1,725 records of the complete ordinary-native oracle with actual target/MTP ring activity. Two native-instrumented-prefill attempts did not finish and are not passes. Preserved native F7 capacity failures are not claimed as native passes, and long streamed measurements are not long-context native-equivalence proof.

The normal runtime was rebuilt consistently across all 16 dependent host translation units; CUDA libraries remain byte-identical. Commit/source mappings preserve the measured pre-commit identities rather than relabelling binaries after Git metadata changes. Raw local receipts remain under `benchmarks/results/dflash-prefetch-fix-20261029/` and `benchmarks/results/recurrent-restore-overlap-20261030/`; their hashes are indexed in the NK7 JSON.

For a portable ordinary reproduction, use `benchmarks/run_dflash_controls.py` with `--spec-mode dflash-k4v`, explicit `--ngram-n/--ngram-m/--ngram-min-hits`, the pinned target/draft/config files and a fresh output directory. Start with several representative prompts. The optional `--cpu-profile --trace-library` adds a separate exact-output/counter-matched host diagnostic; its whole-call lookup p50/p95/p99, first-use/statistics classes and reserved-map bytes are not ordinary speed evidence. The finder and control harness start only their own servers and do not assume a production service exists.

The [roadmap](../../DFLASH2_K4V_ADAPTIVE_KV_ROADMAP.md) retains the stage history, red/green evidence and scoped completion checklist.
