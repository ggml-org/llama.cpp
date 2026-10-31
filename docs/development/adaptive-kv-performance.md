# Adaptive KV Streaming performance report

This report groups the recorded benchmarks by model, then separates implementation A/B studies and reproduction commands. The [README](../../README.md#performance-and-reproduction) keeps the short overview; [KV-cache support](../../README.md#kv-cache-formats) is a capability list, not a list of benchmarked context lengths.

## Reading the results

The reference device is an RTX 5070 Ti with 16,303 MiB of VRAM. Compare only the configurations and runtime cohorts stated for each experiment. Maximum-fit sweeps, fixed-arena A/B studies, stock-UVM controls and image-phase measurements answer different questions; they are not interchangeable samples of one curve.

Most sweep points are single retained measurements, not repeated statistical estimates. Independent maximum-fit arenas are not equal-memory comparisons between draft modes. Operator or fitting native controls establish numerical coverage separately; completion of a long streamed workload is not long-context native equivalence. Weight/KV quantization and draft acceptance can also change outputs or speed.

## Qwen3.8-27B measurements

### DFlash2 + K4V scheduling studies

The completed [bounded K7 support report](dflash2-k4v.md#measured-performance) separates two matched fixed-arena optimization studies from the older context curves. Flexible target prefetch improved the selected long DFlash7/mixed sample by 3.40%; restore overlap, over that fixed baseline, improved the selected 8K/64K samples by about 5.1-5.4%. These are separate single-pair cohorts, not additive gains or a rerun of the introductory sweep. Whole-call lookup studies and a separate six-context maximum-arena Open-SWE sample are indexed in [NK7 data](../../benchmarks/data/dflash2-k4v-nk7.json). K4V itself has no demonstrated broad natural-workload gain and remains opt-in.

### Stock UVM without KV streaming

The introduction adds a measured stock baseline using the same IQ4_XS weights, Q8_0 K / Q4_0 V and article corpus as the four-mode V2 sweep. Stock is pristine upstream revision `10a60cf303566e10d6a7a2774c17d2085503d87b`, the upstream base used by this V2 implementation. UVM is enabled for stock; there is no adaptive KV pool, phase arena, MTP, DFlash or host-rollback override. This is not the older Q3-weight measurement.

All 16 stock points from 8 through 128 Ki completed with 256 output tokens, one slot, b/ub=256, eight CPU threads and prompt length equal to context capacity minus 271. All stock output token IDs match the corresponding V2 target-only runs. The baseline ends at its last measured point; no stock performance is extrapolated to 256 Ki.

| At 128 Ki capacity | Stock + UVM | V2 streamed target-only |
| --- | ---: | ---: |
| Prefill | 837.1 tok/s | 934.6 tok/s |
| Decode | 2.48 tok/s | 26.15 tok/s |

The measured decode ratio at that point is about 10.6x; it is not a general speedup claim. At short contexts, stock decode is roughly 2-3% faster in this sweep. UVM migration can make performance uneven under pressure, so the plotted points are kept as measured rather than smoothed. [Stock-UVM measurements and configuration (CSV)](../../media/stock-uvm-iq4-throughput.csv).

### Q8/Q4 four-mode context scaling

The completed four-mode sweep measures target-only, MTP=3, DFlash2=3 and DFlash2=7 at every 8 Ki capacity from 8 Ki through 256 Ki: 32 points per mode, 128 total. It uses IQ4_XS weights, Q8/Q4 KV, the checked-in article corpus, one slot, b/ub=256, 256 generated tokens and 15 positions of headroom. Each mode/context independently uses its maximum-fit arena, without UVM or a safety reserve.

![Four-mode full-context comparison: decode, prefill, effective KV grant and draft acceptance](../../media/adaptive-kv-four-mode-comparison.png)

[All 128 plot points (CSV)](../../media/adaptive-kv-four-mode-comparison.csv). This is the earlier completed Q8/Q4 sweep, recorded before cache-format generalization. The later five-capacity Q8/Q4 regression checks in the matrix below retain identical arenas and output IDs, with throughput differences below 0.7%. The figures use their own recorded measurements; datasets are not mixed to fill missing modes or context points.

### Full-context comparison: KV formats and drafting modes

The recorded Qwen 45-point matrix uses the same Unsloth IQ4_XS target on an RTX 5070 Ti, context capacities 8/64/128/192/256 Ki, b/ub=256, one slot, 256 generated tokens and the [checked-in Wikipedia corpus](../../benchmarks/data/online-articles-262144-words.txt). Prompt length is capacity minus 271 positions. Each format, mode and context independently uses its maximum-fit arena, with UVM disabled and no safety reserve. This is separate from Gemma's Q8/Q4 60-point sweep and the newer fixed-budget prefill A/B cohorts.

![Decode/prefill, effective KV grant and draft acceptance](../../media/kv-quant-comparison.png)

[All 45 plot points (CSV)](../../media/kv-quant-comparison.csv)

| KV format at 256 Ki capacity | Target-only decode | MTP=3 decode | DFlash=7 decode |
| --- | ---: | ---: | ---: |
| Q8_0 / Q4_0 | 9.60 tok/s | 22.27 tok/s | 21.80 tok/s |
| Q8_0 / Q8_0 | 6.66 tok/s | 14.53 tok/s | 18.04 tok/s |
| Q5_1 / Q5_1 | 10.56 tok/s | 23.27 tok/s | 27.71 tok/s |

These are single-run measurements for this model, prompt and machine. Acceptance and memory pressure both affect speculative speed; DFlash is not always faster than MTP. Different K/V formats can change outputs, and this graph does not evaluate model quality.

All 15 Q8/Q4 comparisons with the pre-generalization build choose identical arenas and retain all 256 output token IDs. Decode differences are -0.67% to +0.57%; prefill differences are -0.11% to +0.20%. No material regression is observed; these small differences are not claimed as speed gains.

### Draft-length comparison at a fixed budget

This earlier experiment holds the parent at 2,240 MiB and compares embedded MTP lengths 0-3 with Q8/Q4. It is historical data, not another run of the maximum-fit matrix above.

![Fixed-budget MTP draft-length prefill and decode comparison](../../media/adaptive-kv-stream-v2-throughput-combined.png)

[Detailed plot: pool size and acceptance](../../media/adaptive-kv-stream-v2-mtp-sweep.png). Fixed-arena and independently maximum-fit curves answer different questions: the latter includes differences in available resident KV space. Neither equal arena size nor equal KV grant means equal total VRAM across different draft models.

## Gemma 4 measurements

### Gemma 4 through 120 Ki

The final 60-point sweep covers every 8 Ki step from 8 to 120 Ki at target-only and MTP lengths 1-3. It uses the Q3_K_M target, matching Q8_0 assistant when enabled, Q8/Q4 KV, b/ub=256, UVM off and 256 outputs after a fresh capacity-minus-271-token prompt. Every point has its own full-workload maximum-fit arena and an actual rejected 1 MiB-larger neighbor.

![Gemma 4 all-mode decode, prefill, physical arena and draft acceptance](../../media/gemma4-120ki-qualification.png)

[All 60 points (CSV)](../../media/gemma4-120ki.csv) · [Profile, hashes and qualification scope](../../media/gemma4-120ki.json)

| Capacity | Target-only decode | MTP1 | MTP2 | MTP3 |
| --- | ---: | ---: | ---: | ---: |
| 8 Ki | 39.61 | 67.68 | 81.50 | 91.16 |
| 64 Ki | 25.79 | 34.27 | 45.69 | 56.04 |
| 120 Ki | 10.33 | 16.97 | 22.70 | 28.26 |

Rates are tokens/s for this corpus and machine, with one measurement per mode/capacity. Independent maximum-fit arenas are 1,502-1,510 MiB target-only and 1,024-1,032 MiB with the assistant; its weights reduce the space left for the arena. This is not an equal-memory comparison or a statistical speedup claim. All 15,360 output IDs and counters are checked for completeness; native equivalence is established separately at fitting contexts and by exact operator/model controls. A successful 120 Ki workload is not a full-256-Ki qualification. The earlier stopped 128 Ki cohort is preserved separately, not mixed into this graph.

### MTP3 through 256 Ki

The separate Gemma MTP3 follow-up completed all 32 points from 8 through 256 Ki, using `find-max-arena.py` independently at every point and retaining each rejected+1 MiB neighbor. Its rates are published in the [introductory chart data](../../media/gemma4-256ki-throughput.csv), with [cohort provenance](../../media/gemma4-256ki-throughput.json); its local comparison with the historical120 Ki MTP3 series is audited in the [Gemma ledger](../../GEMMA4_ADAPTIVE_KV_ROADMAP.md). The historical all-mode120 Ki data and the image Q3_K_S cohort remain separate; this is not all-mode256 Ki or full-context vision qualification.

![Measured Gemma target-only/MTP1/MTP2 through 120 Ki and post-D512 MTP3 through 256 Ki](../../media/gemma4-256ki-throughput.png)

This overview preserves the earlier target-only/MTP1/MTP2 measurements through 120 Ki and uses the separate post-D512 MTP3 series through 256 Ki. It does not extrapolate missing modes, join two MTP3 cohorts into one line, or attribute every difference to drafting. Both throughput y-axes start at zero.

## Optimization comparisons

### Common prefill improvements

Two separate repeated A/B studies use fixed 1 GiB arenas, Q8/Q4, b/ub=256, UVM off and 256 outputs. Each has 52 fresh requests, three paired trials at 8/64 Ki for target-only/MTP3, plus single 120 Ki target-only controls. They compare successive adaptive implementations, **not stock UVM**; do not multiply different cohorts into an unmeasured combined speedup.

| Repeated 64 Ki prefill improvement | Gemma target-only / MTP3 | Qwen target-only / MTP3 |
| --- | ---: | ---: |
| Bounded conversion reuse vs pre-conversion adaptive | +32.44% / +32.12% | +5.14% / +5.23% |
| Two-wave pipeline vs conversion-only | +3.64% / +3.65% | +1.25% / +1.82% |

Paired prompts, output IDs and draft counts match; decode median differences stay within 0.17% for conversion and 0.15% for the pipeline. At 8 Ki, conversion gains are smaller and the incremental pipeline change is negligible. [Conversion CSV](../../benchmarks/data/prefill-conversion-v1.csv) / [profile](../../benchmarks/data/prefill-conversion-v1.json), [pipeline CSV](../../benchmarks/data/prefill-pipeline-v1.csv) / [profile](../../benchmarks/data/prefill-pipeline-v1.json).

Matched Nsight windows confirm the mechanism: at 96 Ki, H2D time overlapping observed GPU kernels increases from 3.2% to 86.6% for Gemma and 2.9% to 62.1% for Qwen, with reader/converter GPU interval unions roughly stable. These are clipped time-interval intersections, not SM occupancy, bandwidth or hardware counters. Instrumented throughput is not the speedup measurement; these original captures did not include Nsight Compute counters.

### D512 attention feature pipeline

The SM120 D512 resumable 8-by-8 attention path now double-buffers K/V feature chunks inside the GPU. It preserves the native MMA, softmax, reduction and checkpoint order. Eligibility follows geometry and capabilities, not the model name; Gemma is the full-model reference and Qwen D256 is the regression control.

The following fresh A/B screen compares the shipping conversion-plus-ring implementation with the D512 change. Both use a fixed 1 GiB arena, Q8/Q4 KV, b/ub=256, UVM off, uncached `capacity-271` prompts and 256 outputs. There is one pair per point, not a repeated statistical trial matrix.

| Gemma capacity / mode | Baseline prefill tokens/s | D512 pipeline prefill tokens/s | Prefill change | Decode change |
| --- | ---: | ---: | ---: | ---: |
| 8 Ki, target-only | 1324.05 | 1342.56 | +1.40% | +0.34% |
| 8 Ki, MTP3 | 1304.80 | 1322.29 | +1.34% | +0.14% |
| 64 Ki, target-only | 834.46 | 895.73 | +7.34% | +0.07% |
| 64 Ki, MTP3 | 826.07 | 887.00 | +7.38% | -0.16% |

All paired prompt/output IDs, draft counts and decode-pool layouts match. Qwen target-only controls at 8/64 Ki change by less than 0.1% in prefill and 0.4% in decode. [Measurements (CSV)](../../benchmarks/data/d512-feature-pipeline-v1.csv) / [profile and qualification (JSON)](../../benchmarks/data/d512-feature-pipeline-v1.json). Do not combine these fixed-arena rates with the independently fitted 120 Ki sweep as if they were one cohort.

The extra buffer is **16 KiB of on-chip shared memory**, not a device allocation or a larger arena minimum. Total user shared memory is 100,136 bytes against the measured 101,376-byte limit; exact-limit and one-byte-short admission are tested. The existing quantized prefill preparation still requires its separately documented 8/16 MiB arena stage. Dense-strided, encoded-direct, D256, other tiles and older architectures retain their previous paths.

Qualification includes all 64 K/V pairs, native-bit and frozen-checkpoint oracles, 120 focused mask/tail/wrapped-wave profiles, 120 native-matched server records, ASan/UBSan and 17 zero-error CUDA memcheck/racecheck/synccheck gates. Only the wide sanitizer fixture uses the user-approved 769 MiB arena; ordinary model execution remains 1 GiB. Two representative affected CUDA units also pass functional compilation for SM61/70/75/86; that is not physical older-GPU runtime qualification. Further model admission is independent of generic D512 kernel support.

## Reproducing text sweeps

Use a separate output directory for every run. No existing server or container is required: these commands start their own test servers. `--no-manage-production` disables the script's optional container-management integration. Make sufficient GPU memory available before running. The fixed-span script accepts a model path and configurable cache types:

```sh
python3 benchmarks/run-fixed-span-sweep.py \
  --server build-v2/bin/llama-server --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --contexts 8192,65536,131072,196608,262144 --prompt-headroom 15 \
  --cache-type-k q8_0 --cache-type-v q4_0 --mtp-lengths 0,3 \
  --decode-tokens 256 --arena-mib 2240 --auto-max-arena \
  --no-manage-production --output benchmarks/results/text-q8-q4
```

For the completed Gemma grid, use the separate assistant and explicitly stop at 120 Ki:

```sh
python3 benchmarks/run-fixed-span-sweep.py \
  --server build-v2/bin/llama-server --model /path/to/gemma-4-31B-it-Q3_K_M.gguf \
  --model-draft /path/to/mtp-gemma-4-31B-it-Q8_0.gguf \
  --min-context 8192 --max-context 122880 --context-step 8192 \
  --mtp-lengths 0,1,2,3 --cache-type-k q8_0 --cache-type-v q4_0 \
  --batch-size 256 --ubatch-size 256 --decode-tokens 256 --prompt-headroom 15 \
  --auto-max-arena --request-timeout 3600 --no-manage-production \
  --output benchmarks/results/gemma-final-120ki
```

The external assistant is not loaded for length 0. Omit `--auto-max-arena` to keep a supplied arena fixed. The general script's default endpoint is still 256 Ki, so **keep Gemma's explicit 120 Ki endpoint** for this qualification. Keep `--no-manage-production` for standalone use; only the test processes launched by these commands are managed.

The DFlash control harness verifies the pinned target, sidecar and publisher config before running. It is deliberately an artifact-specific reproduction tool, unlike the general arena finder:

```sh
python3 benchmarks/run_dflash_controls.py \
  --server build-v2/bin/llama-server \
  --target /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --draft-config /path/to/DFlash2/config.json \
  --contexts 8192,65536,131072,196608,262144 --depths 3,7 \
  --prompt-tokens 262144 --prompt-headroom 15 --decode-tokens 256 \
  --target-k q8_0 --target-v q4_0 --arena-mib 1500 --auto-max-arena \
  --output benchmarks/results/dflash-q8-q4

python3 benchmarks/plot_dflash_controls.py \
  --series target-and-MTP benchmarks/results/text-q8-q4 \
  --series DFlash2 benchmarks/results/dflash-q8-q4 \
  --output benchmarks/results/speculation-comparison.png
```

For the dense four-mode sweep, pass all 32 capacities to both commands: in Bash, use `--contexts "$(seq -s, 8192 8192 262144)"`, retaining MTP lengths `0,3` and DFlash depths `3,7`. The plotting command combines both result directories without mixing in an earlier-build baseline.

Repeat with the other cache pairs and label each series accordingly. The control harness also supports `--spec-mode none --depths 0` and `--spec-mode mtp --depths 3`. It starts and cleans up its own test servers without requiring or managing another service. Its `--lifecycle` checks are separate from throughput measurement. The corpus is Wikipedia text under CC BY-SA 4.0 with article attribution, separate from the code license.

Correctness qualification compares matching formats and execution modes against native attention, including logits, hidden rows, KV bytes and recovery state. TG1 and multi-row stock kernels can themselves differ numerically; cross-mode output equality is not the numerical oracle. See [KV-format qualification](../../KV_QUANTIZATION_ROADMAP.md) and [the upstream attention port](../../UPSTREAM_ATTENTION_PORT_ROADMAP.md).

The Gemma command above reproduces the earlier all-mode 120 Ki cohort. To schedule only MTP3 through the native 256 Ki capacity, use `--mtp-lengths 3 --max-context 262144` with the matching external assistant. Use a new output directory and record the actual source, binaries, inputs and fitted arena at every point; a new run is a new cohort, not a replacement for the published records.

Full sweeps can take hours. They are distinct from the [quick arena-sizing examples](../../README.md#find-a-usable-arena-size), which use short runtime probes and optionally one final full-workload validation. Neither tool needs an existing inference service.

## Image measurements

Qwen and Gemma share the vision launch/request mechanism, but their projector artifacts, positions, cache lifetimes and memory budgets differ. Image and text-only measurements must remain separately labelled. See the [Gemma vision report](../multimodal/gemma4-vision.md#phase-measurements) and [Qwen vision measurement instructions](../../tools/server/tests/README.md#vision-memory-and-handoff-measurements) for the image-specific records and reproduction scope.
