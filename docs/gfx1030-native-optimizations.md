# Automatic gfx1030 native optimizations

This branch contains HIP paths tuned and validated on AMD RDNA2/gfx1030 (four Radeon Pro V620 GPUs). On the tested V620 launch, `HSA_OVERRIDE_GFX_VERSION=10.3.0` automatically selects the validated RDNA2 profile and its structural feature paths. Set the global kill switch to return to stock behavior:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export GGML_HIP_RDNA2_AUTO=0
```

`GGML_HIP_GFX1030_NATIVE=0|1` remains an explicit native-profile override. Per-feature variables also remain explicit overrides: when unset they inherit the HSA umbrella default, and `0` disables that feature. Unsupported models, formats, shapes, layouts, and topologies retain their stock fallback.

## Environment variables

| Variable | Default | Effect |
|---|---:|---|
| `GGML_HIP_GFX1030_NATIVE` | unset: inherit HSA umbrella; explicit `0|1` override | Selects or disables the validated gfx1030 kernel specializations: Q4_0 DOT8 MMVQ, exact-shape Muse Q8_0 eight-warp MMVQ, bounded six-row Q4_K/Q6_K routed MMVQ, native tiled-FlashAttention arithmetic/reductions, and chunked GDN prefill loads. |
| `GGML_HIP_GFX1030_Q8_1_FUSION` | unset: inherit HSA umbrella; explicit `0|1` override | Fuses routed SwiGLU evaluation into Q8_1 activation staging for eligible prompt-processing `MUL_MAT_ID` down projections. |
| `GGML_HIP_GFX1030_GDN_SIBLING_FUSION` | unset: inherit HSA umbrella; explicit `0|1` override | Creates and uses fused Qwen3.5/Qwen3.6 DeltaNet sibling projection weights for their structural loader/graph gates. |
| `GGML_HIP_GFX1030_Q8_CACHE` | unset: inherit HSA umbrella; explicit `0|1` override | Enables graph-owned reuse of exact standard Q8_1 TG activations and the eligible dual RMSNorm F32/Q8_1 producer. Q4_0 `sum_hi`, packed layouts, MMQ, and routed operations remain outside this cache contract. |
| `GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY` | unset / `0` | Reports eligible standard-MMVQ Q8_1 sources, safe reuses, cache hits, and storage. Telemetry does not allocate or reuse entries by itself. |
| `GGML_HIP_GFX1030_TARGET_BACKEND_SAMPLING` | unset: inherit native auto policy; explicit `0|off|false` disables | Automatically keeps eligible greedy native-MTP/DFlash target selection on the backend instead of materializing and scanning the full vocabulary on the CPU. |
| `GGML_TP_SHARDED_OUTPUT` | unset / `auto`: automatic hidden-axis/full-logit policy; explicit `0` disables; explicit `1` selects vocabulary-axis output unless target backend sampling is enabled at model load | Enables TP output-head sharding. For explicit `1` without server-wide `--backend-sampling`, the primary Qwen35/Qwen35MoE LM head is vocabulary-sharded, removing the full-logit output AllReduce for CPU-sampled inference. With `--backend-sampling` enabled at model load, the loader safely retains hidden-axis/full-logit output so target backend sampling works; vocabulary-axis sharding and target backend sampling are not combined yet. Sidecar-local draft sampling is unaffected. |
| `GGML_HIP_GFX1030_STACKED_TARGET_BACKEND_SAMPLING` | unset: disabled; explicit `1` enables the experimental path | Allows the automatic greedy target sampler for the validated neural profile cascaded with `ngram-mod`. Requires separate GPU qualification; unset is the safe fallback. |
| `GGML_HIP_GFX1030_STOCHASTIC_TARGET_BACKEND_SAMPLING` | unset: disabled; explicit `1` enables the experimental path | Allows the automatic target sampler for compact stateless stochastic chains (`top-k`/`top-p`/`min-p`/temperature/distribution). Requires separate GPU distribution and E2E qualification; active stateful/unsupported samplers fall back. |
| `GGML_HIP_GFX1030_DFLASH_MMVQ_ROWS2` | unset: enabled; explicit `0|off|false` disables | Uses two output rows per block for gfx1030 width-six Q4_0/Q4_K/Q6_K MMVQ. This is the validated default for the measured V620 DFlash2 path; the global RDNA2 auto-off switch still disables it. |
| `GGML_HIP_GFX1030_P2P_ALLREDUCE` | unset / `auto`: expanded automatic policy | Selects the RDNA2 host-snapshot policy. The default `auto-expanded` policy enables exact ordinary consumer-fused boundaries when their structural TP4 gate passes; `auto-basic` restores the former control policy, and `0`, `off`, or `false` disables this host-snapshot feature while leaving other native paths enabled. Unsupported TP counts/topologies fall back to RCCL. |
| `GGML_HIP_GFX1030_MMVQ_W8_ROWS4` | unset: inherit native auto policy; explicit `0` retains rows2 | Automatically selects the certified Q4_0 DFlash2 M8 rows/block=4 MMVQ path; unsupported shapes retain rows2/stock fallback. |

The selectors are read once during backend or model initialization. Set them before starting `llama-cli`, `llama-server`, `llama-bench`, or a test binary. Explicit `0` values are useful for A/B and fallback verification; an unset validated feature follows the automatic HSA profile when `HSA_OVERRIDE_GFX_VERSION=10.3.0` is active. The stacked and stochastic target-sampling extensions are deliberately opt-in until their matched V620 GPU correctness/performance matrices are complete.

Example with every accepted path enabled:

```bash
GGML_HIP_GFX1030_NATIVE=1 \
GGML_HIP_GFX1030_Q8_1_FUSION=1 \
GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1 \
GGML_HIP_GFX1030_Q8_CACHE=1 \
build/bin/llama-bench \
  -m model.gguf -ngl 999 -sm layer -ts 1/1/1/1 -fa on \
  -p 512 -n 128 -b 512 -ub 256 -r 5
```

To return to stock behavior, unset the optimization variables:

```bash
unset GGML_HIP_GFX1030_NATIVE
unset GGML_HIP_GFX1030_Q8_1_FUSION
unset GGML_HIP_GFX1030_GDN_SIBLING_FUSION
unset GGML_HIP_GFX1030_Q8_CACHE
unset GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY
export GGML_HIP_GFX1030_DFLASH_MMVQ_ROWS2=0
```

Multi-GPU ROCm state save/restore stability has a separate opt-in, `GGML_HIP_SAFE_STATE_IO=1`. It does not require the gfx1030 master switch and does not change inference kernels. See [the multi-GPU ROCm state-I/O workaround](rocm-multi-gpu-state-io.md).

## Automatically selected native paths

### Q4_0 DOT8 MMVQ

For Q4_0 decode with one destination column, the native MMVQ specialization preserves the ordinary Q4_0 weights and Q8_1 activation bytes. A two-byte `sum_hi` sidecar per Q8_1 block supplies the exact high-nibble correction needed by gfx1030 `UDOT8`/`SDOT8`; no model conversion or persistent layout change is required. Other quantization types remain on their normal vector-dot implementations.

The path is exact and automatically eligible under the tested HSA profile, but the measured Qwen end-to-end result was neutral to approximately 0.7% slower. It is retained as a validated native arithmetic path rather than advertised as a speedup. See [the Q4_0 DOT8 experiment](rdna2-v620-q4-0-dot8-experiment.md).

### Muse Q8_0 eight-warp MMVQ

Muse-Glimmer-30B tensor splitting exposes its promoted attention K/V weights to each backend as single-token, non-routed Q8_0 MMVQ with `K=6656` and `N=128`. Native mode launches an eight-wave kernel for that exact standard-Q8_1 shape. All other dimensions, quant types, routed operations, packed Q8_1 layouts, and multi-column batches retain the normal architecture policy; in particular, the measured `6656x2048` Q/gate sibling remains at one wave.

Across four paired synthetic passes, eight waves reduced production-graph latency for `6656x128` by 17.4%; `6656x2048` regressed by 6.2%, which is why the selector is exact rather than a general RDNA2 warp table. On four V620s with Muse Q4S8 tensor split, TG128 improved from a paired mean of `42.057` to `42.992 tok/s` (**+2.22%**) in A/B/B/A testing. Q8_0 outputs agreed with the one-wave path with zero tolerance failures, maximum absolute difference `2.44e-4`, and RMSE `4.78e-5`; different reduction order means byte identity is not claimed. Q4_0 and non-selected Q8_0 controls remained byte-identical.

### Bounded six-row routed MMVQ

On RDNA2, stock Q4_K and Q6_K `MUL_MAT_ID` dispatch changes from MMVQ to MMQ above five routed token rows. Native mode extends MMVQ through six rows when all of the following hold:

- the expert weights are Q4_K or Q6_K;
- the destination has at most six token rows;
- the routed IDs select at most four experts per token.

The top-k bound is intentionally conservative. Across 48 Q4_K/Q6_K cases with top-k 2 or 4, K from 256 to 8192, N from 256 to 4096, 8 to 256 experts, and both uniform and concentrated routing, six-row MMVQ reduced operation latency by 15.9% to 61.2%. Top-k 6 regressed in one tested Q6_K case, while concentrated top-k 8 routing regressed in 12 of 16 grid cases by as much as 54.5%; those cases therefore retain stock dispatch.

The exact Qwen3.6 35B four-GPU layer-split configuration currently carries an advisory tensor flag that permits its validated top-k 8 MTP path to use six-row MMVQ. The flag is generic: another model loader may set it on validated Q4_K/Q6_K routed weights after equivalent testing. It is inert when the native profile is disabled, so an explicit `GGML_HIP_GFX1030_NATIVE=0` or global `GGML_HIP_RDNA2_AUTO=0` keeps execution stock. Without the hint, every model automatically receives the validated top-k 1--4 path; higher top-k routing remains on stock dispatch until separately validated.

MMQ and MMVQ accumulate floating-point products in different orders and are not generally byte-identical. The validation sweep measured NMSE from `4.23e-10` to `9.15e-9`, compared with the backend `MUL_MAT_ID` allowance of `5e-4`; MMVQ graph and non-graph outputs were byte-identical. This path does not alter quantized weights or Q8_1 activation encoding.

### Tiled FlashAttention

The native tiled-F16 specialization uses gfx1030 `fdot2` accumulation and native wave-32 sum/max reductions. Host dispatch selects a separate compile-time kernel specialization, so the inner loops do not contain a runtime branch. Older ROCm compilers that do not expose the wave-reduction builtins compile the exact shuffle fallback instead. Vector and MMA FlashAttention variants are unchanged.

Both stock and native runs passed `2920/2920` `FLASH_ATTN_EXT` backend tests. Four-GPU PP4096 measurements remained within run-to-run variance, so no end-to-end FlashAttention gain is claimed. The guarded benchmark and verification workflow is documented in [the native FA harness](gfx1030-native-fa-harness.md).

### Chunked Gated DeltaNet prefill

For GDN calls with more than one token, the native specialization has lane 0 load scalar `beta` and per-column value inputs and broadcast them across the wave. In the non-KDA form it also loads and broadcasts the scalar gate; the KDA form retains its per-row gate loads. Decode (`n_tokens == 1`) keeps the stock specialization.

Direct GDN measurements improved by about 7.9% at 256 tokens and 17.7% at 512 tokens. The GDN backend suite passed all 36 cases across all five tested backends. Full-model PP4096 measurements were sensitive to process order and GPU temperature, so only the direct-kernel improvement is claimed.

### Vocabulary-sharded tensor-parallel output

The normal Qwen35 tensor-parallel output path splits the primary LM head along its hidden dimension and AllReduces partial full-vocabulary logits. This preserves a complete logit vector on every GPU and supports the target backend sampler. For workloads that sample target logits on the CPU, set `GGML_TP_SHARDED_OUTPUT=1` to enable output sharding and split only the primary output head across vocabulary rows, concatenating those shards during host readback instead. Native-MTP auxiliary heads retain the existing full-logit policy when distinct; the Qwen3.8-27B artifact currently reuses the primary full-vocabulary head, so sidecar-local sampling remains provider-local.

The explicit `GGML_TP_SHARDED_OUTPUT=1` mode is deliberately separate from the automatic `unset`/`auto` policy: without server-wide backend sampling it selects vocabulary-axis output for CPU-target sampling. If `--backend-sampling` is enabled at model load, the loader selects the hidden-axis/full-logit primary policy instead and preserves target backend sampling; this is a safe compatibility fallback, not distributed backend sampling over vocabulary shards. Request-level backend sampling cannot change a model that was already loaded in vocabulary-sharded mode and therefore still falls back to CPU there. Use `GGML_TP_SHARDED_OUTPUT=0` to disable output-head sharding.

On the Qwen3.8-27B AutoRound Q4_0 qualification workload, matched fixed-work decode improved by 4.59% on TP2 and 5.74% on TP4. Greedy target-only server output was exact and improved 3.84%/5.84%; greedy MTP-sidecar output and acceptance were exact and improved 4.00%/5.39%. Native MTP retained identical outputs/acceptance and improved 1.68%/1.64% after its required CPU draft-sampling fallback.

### Greedy speculative target backend sampling

The server automatically enables target-side backend sampling for validated single neural native-MTP and DFlash workloads when the gfx1030 profile is active. Eligibility is structural: tensor-parallel mode over four ROCm devices; MTP `n_max=4` or DFlash `n_max=5|7`; vocabulary size at least 65,536; temperature-zero sampling with neutral filtering/penalty controls; no grammar, reasoning-budget sampler, requested probabilities, logit bias, or model-supplied suppress-token list. An explicit request-level `backend_sampling` value takes precedence unless the model was loaded in vocabulary-sharded mode with `GGML_TP_SHARDED_OUTPUT=1` and without server-wide `--backend-sampling`; that layout cannot expose one complete target logit vector on a single backend and therefore forces CPU fallback. A server-wide `--backend-sampling` request selects the hidden-axis/full-logit compatibility layout at model load. Unsupported backend chains fall back to CPU sampling, and `GGML_HIP_GFX1030_TARGET_BACKEND_SAMPLING=0` or `GGML_HIP_RDNA2_AUTO=0` disables automatic selection. The current automatic path does not include stacked `ngram-mod` unless `GGML_HIP_GFX1030_STACKED_TARGET_BACKEND_SAMPLING=1` is set.

For deterministic one-candidate distributions, the backend uses a compact `temperature(0) -> greedy` chain. The greedy backend maps a reduced candidate index back to its vocabulary token ID; a focused backend-sampler test covers this composition. This avoids both the full 248,320-entry CPU sampler pass and the mutable random-input tensor used by the general distribution sampler.

On the fixed Qwen3.8-27B Q4_0 TP4 MTP workload, a fresh production-tree 5-request-per-leg A/B/B/A improved the pooled warm median from `65.7854` to `75.9478 tok/s` (**+15.45%**) with identical content/token hashes and draft/accepted counts. Median cycle time fell from `45.345` to `39.356 ms`; warm request E2E throughput improved from `59.37` to `66.69 tok/s`. A 1,024-token comparison was exact and measured `73.4756 -> 85.7679 tok/s`; a 20-request varied-prompt stress test, prompt-cache replay, graph-disabled execution, and two concurrent slots were exact.

On the separately measured DFlash2 `n_max=7` workload, the dedicated greedy chain improved pooled warm throughput from `62.3902` to `69.7006 tok/s` (**+11.72%**) and cycle time from `62.880` to `56.285 ms`. A 1,024-token comparison improved `69.8188 -> 79.0672 tok/s`; five-prompt and 20-request persistence runs, prompt-cache replay, fallback cases, and two concurrent slots were exact. The path is independent of target weight quantization, but Q4_0 is the production format validated end to end for the single-neural path.

For the normal DFlash2 `n_max=5` configuration, a fresh isolated five-request reproduction improved the warm median from `64.68` to `73.17 tok/s` (**+13.1%**) with identical content/token hashes, draft counts, and accepted counts. Temperature/top-k/top-p, grammar, requested-probability, reasoning, and combined n-gram cases retained the CPU fallback or their existing sampler semantics; prompt cache, parallel slots, cancellation/recovery, repeated requests, Q8 KV, and long outputs passed the server-safety matrix.

#### Experimental stacked and stochastic extensions

The isolated `perf/specdecode-universal-backend-sampling` worktree contains a policy-only stacked extension and a stochastic target-chain extension. `ngram-mod + MTP/DFlash` remains a first-success cascade, not a concatenated proposal. The stacked greedy path is enabled only with `GGML_HIP_GFX1030_STACKED_TARGET_BACKEND_SAMPLING=1`; the stochastic path additionally requires `GGML_HIP_GFX1030_STOCHASTIC_TARGET_BACKEND_SAMPLING=1`, only admits compact stateless chains with active `top-k <= 256`, and is automatically limited to MTP. The validated DFlash target-backend path remains greedy-only.

The stochastic MTP path reproduced a fresh positive result on the longer Qwen3.8-27B A/B/B/A workload (`+4.2729%` median generation throughput with exact token hashes and `204/75` draft/accepted counts). The corresponding realistic four-prompt DFlash `ngram=0` qualification was negative: target-backend sampling was slower on every prompt and aggregate backend eval time was `+3.617095%`; backend-on B1/B2 replay was exact, but control/backend trajectories and draft counts differed. Two distribution-only DFlash prototypes were rejected, one for an output-buffer contract crash and one for fixed-seed replay divergence. DFlash stochastic automatic sampling is therefore not retained; the result is documented here rather than exposed through a permanent opt-in switch. DFlash n-gram stacking remains a separate correctness-only path without a performance claim.

The stateless automatic chain remains attached to its server slot between compatible requests. This avoids rebuilding the target scheduler's prompt, token-generation, and prompt reservation graphs on every request; incompatible or explicitly configured sampler requests still detach and use the normal path. On the fixed DFlash2 workload this reduced prompt latency by about 81 ms and improved useful client-wall throughput by 1.38%.

For the same structurally certified path, a request with `cache_prompt: false` retains the checkpoint-driven prompt chunk boundaries required for exact recurrent/DFlash behavior but does not serialize a checkpoint that it will not reuse. It first discards older prompt checkpoints so a later cache-enabled request cannot restore stale recurrent state. Cache-enabled and fallback requests retain normal checkpoint creation. Incremental A/B/B/A reduced prompt latency from `424.264` to `169.662 ms` and improved useful client-wall throughput by **6.50%**, with identical output hashes, draft/accepted counts, and generation throughput. Together with persistent sampler attachment, comparison against the original merged implementation improved client-wall throughput from `62.4933` to `67.7398 tok/s` (**+8.40%**). If a later request switches from uncached to cached operation, it reprocesses once to create a checkpoint; subsequent cached requests resume normal reuse.

### Default-on DFlash width-six rows/block=2 MMVQ

The gfx1030 native profile automatically packs two output rows into each block for width-six Q4_0, Q4_K, and Q6_K MMVQ. Set `GGML_HIP_GFX1030_DFLASH_MMVQ_ROWS2=0` (or `off`/`false`) to restore the existing dispatcher. Non-gfx1030 execution, unsupported quantization or width, and global `GGML_HIP_RDNA2_AUTO=0` also retain the existing dispatcher. End-to-end evidence is validated on four V620s with TP4; TP2 is not claimed until separately benchmarked.

With backend target sampling already active, controlled Q4 DFlash2 `n_max=5` testing improved `72.717 -> 77.783 tok/s` (**+6.97%**). Mean speculative-cycle time fell `51.872 -> 48.554 ms`: target verification saved `2.317 ms/cycle` and draft evaluation saved `1.064 ms/cycle`. Synthetic width-six Q4_0/Q4_K/Q6_K raw-F32 checks were exact; 1,024-token Q4, Q8, reasoning, grammar, requested probabilities, prompt-cache, parallel, cancellation/recovery, and repeated-request gates passed.

This candidate does not by itself reach sustained 2x. A warm five-workload matrix averaged `83.85 tok/s` versus `44.74 tok/s` AR (**1.87x**), with per-case speedups from `1.12x` to `2.34x`. Rows/block=3/4 was slower, an eight-wave Q4_0 variant failed raw-F32 exactness, `p_split` changes were neutral, and forced RCCL settings regressed. A corrected exact width-six host-snapshot reduction gained only `0.74%`; the precompiled TP4-only route is retained behind its installed-RCCL startup exactness self-test. Fusing width-six gate/up/GLU regressed `4.63%`. The terminal width-six nwarps=2 screen changed raw F32 results for Q4_0/Q4_K/Q6_K and regressed target-dominant Q4_0 by 24.1%; no further local width/warp schedule is retained.

### Certified Q4_0 M8 rows/block=4 MMVQ

For native gfx1030 TP4 DFlash2 width-eight target verification, the dispatcher automatically uses rows/block=4 for the certified Q4_0 standard-Q8_1, non-routed shapes already covered by the rows2 whitelist. The kernel keeps width eight and the exact per-row K/lane/reduction order while halving the row-block grid. Unsupported shapes, IDs, packed layouts, non-Q4_0 types, non-native devices, and `GGML_HIP_GFX1030_MMVQ_W8_ROWS4=0` all fall back to the retained rows2/stock paths; `GGML_HIP_RDNA2_AUTO=0` disables it globally.

All eight certified Q4_0 shapes were direct byte-exact. Production-equivalent DFlash2 safety checks covered 1,024-token output, five varied prompts, prompt-cache transitions, graph on/off, semantic fallback requests, and two concurrent slots. Same-work ABBA improved `69.7819 -> 70.8576 tok/s` (`+1.5414%`); this narrow path is retained as a safe incremental optimization, not as a >=5% material gain. Resource census is `128 VGPR / 128 SGPR`, one wave, no LDS/scratch.

## Secondary fusions

### Graph-scoped standard Q8_1 reuse

With the native profile enabled and `GGML_HIP_GFX1030_Q8_CACHE` not explicitly disabled, eligible TG matrix multiplications can share an exact standard Q8_1 activation instead of independently staging the same F32 source. An explicit `GGML_HIP_GFX1030_Q8_CACHE=1` is still a useful self-documenting enable. The initial contract is deliberately narrow:

- a single-token, non-routed `MUL_MAT` using Q8_0 weights and MMVQ;
- standard `block_q8_1` activation layout only;
- the same source tensor/data, padded K, byte size, device graph, and stream;
- no packed 64/128/256 Q8_1 layouts, Q4_0 `sum_hi`, MMQ, or `MUL_MAT_ID`.

Storage belongs to the existing per-execution CUDA/HIP graph object and is freed when that graph is evicted. Readiness resets for every execution: the first consumer normally refreshes the entry, and later consumers reuse it only after an intervening-output overlap scan confirms that the F32 source remains live. This source-version and lifetime rule also applies when CUDA graph capture is disabled; it does not use an untyped tensor sidecar or a fixed node TTL.

After one execution has proved a reusable group, an eligible already-fused contiguous single-row RMSNorm+MUL can write both the normal F32 output and exact standard Q8_1 bytes into the planned entry. If no matching safe entry exists, the ordinary fused RMSNorm and MMVQ staging paths run unchanged. The dual producer preserves the materialized F32 arithmetic boundary before Q8_1 scale, sum, and rounding operations.

On the validated four-V620 Qwen3.6 35B graph, the ten full-attention `attn_norm` sources each feed three Q8_0 projections. The cache uses ten 2304-byte entries distributed across the four device graphs. Cache-only reuse removed 20 staging launches per token; the dual producer removed the remaining ten. In a matched rocprof run, standard `quantize_q8_1<false>` dispatches fell from 935 to 815.

Temporary verification compared every cached Q8_1 byte, scale, and sum with a fresh stock quantization and every dual-producer F32 byte with the stock fused RMSNorm+MUL result. A deterministic 64-token completion and every reported top-10 probability also matched. Four-process ABBA TG128 measurements improved from 86.2102 to 86.7479 tok/s (**+0.62%**). Packed layouts, prompt-processing MMQ, and routed operations retain stock behavior.

Set `GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY=1` to print per-device source/hit summaries during graph warmup. Telemetry alone only reports opportunities; it does not allocate or reuse cache entries.

### TP4 consumer-fused host-snapshot boundaries

With the automatic `GGML_HIP_GFX1030_P2P_ALLREDUCE=auto-expanded` policy, structurally eligible ordinary TP4 boundaries named `linear_attn_out-*`, `ffn_out-*`, and `attn_output-*` with contiguous F32 shape `[5120,1,1,1]` may use the existing exact consumer-fused host-snapshot kernel. It performs the validated mapped-host reduction and the dependent residual add/RMSNorm/mul in one kernel while preserving the F32 materialization and graph-prefix fallback contract.

The gate is structural and model-independent: it requires the existing four-rank RDNA2 topology/self-test, exact boundary shape/name, contiguous tensors, and an unshared `RESHAPE -> ADD -> RMS_NORM -> MUL` graph prefix. Any miss uses the ordinary host reduction or RCCL fallback. MTP width five (`ne[1]=5`) and external DFlash graphs do not activate this ordinary-only path.

The isolated integrated candidate was byte/content exact across deterministic, prompt-cache, grammar, long/stateful, Flash Attention, graph-reuse, and fallback validation. A clean production-control versus integrated-candidate ABBA measured `53.4202 -> 53.8177 tok/s` (**+0.744%** mean; **+0.742%** median). This is a small measured gain, not a general communication redesign; the existing `auto` control policy remains available.

### Routed SwiGLU to Q8_1 staging

With the native profile enabled and `GGML_HIP_GFX1030_Q8_1_FUSION` not explicitly disabled, graph fusion can replace:

```text
F32 gate + F32 up -> SwiGLU F32 tensor -> Q8_1 staging -> routed down projection
```

with register-level SwiGLU evaluation inside the Q8_1 staging kernel. Eligibility is deliberately narrow:

- prompt processing only; batches eligible for MMVQ/decode are rejected immediately;
- routed `GGML_OP_MUL_MAT_ID` down projection using the MMQ path;
- F32 gate/up inputs with exact matching shapes and supported alignment;
- SwiGLU, quantized down weights, and the ordinary non-deduplicated routed layout.

TG retains normal MMVQ dispatch. Shared dense experts and broadcast/deduplicated MoE layouts retain the stock graph.

Unsafe-math can otherwise reassociate arithmetic after removing the materialized F32 tensor. The fused kernel therefore keeps an opaque register-level compiler boundary after SwiGLU. Verification compared about 330 MB across 280 Qwen dispatches with zero Q8_1 byte differences. The targeted GLU plus staging sequence fell from about 41.5 microseconds to 11.6 microseconds; alternating PP512 runs measured a smaller end-to-end improvement of roughly 0.3% (with earlier runs near 1%).

### DeltaNet sibling projections

With the native profile enabled and `GGML_HIP_GFX1030_GDN_SIBLING_FUSION` not explicitly disabled, model loading creates two persistent row-concatenated weights for recurrent Qwen35MoE 35B layers:

```text
Q8_0 [wqkv | z]       : [2048, 8192] + [2048, 4096] -> [2048, 12288]
F32  [beta | alpha]   : [2048,   32] + [2048,   32] -> [2048,    64]
```

Packed rows are copied byte-for-byte; no dequantization or requantization occurs. The graph performs two matrix multiplications instead of four and exposes the original logical tensors through correctly-strided views. Non-contiguous inputs to CUDA unary operations are materialized before use.

The loader enables this only for the Qwen35MoE 35B architecture in layer-split mode, matching ROCm buffer types, expected Q8_0/F32 types, contiguous row layouts, and models without per-weight or input scales. If an active LoRA adapter is present, graph construction conservatively falls back to all four original `build_lora_mm` paths.

The original weights remain resident to support fallback, so the fused weights add **780 MiB** total (about 195 MiB per GPU with an even four-way layer split). Observed model initialization increased by roughly 190 ms. An exact two-pointer TG prototype consumed the original Q8_0/F32 sibling weights directly but was 0.46% slower than the packed path in four-process ABBA testing; it would also forfeit the packed PP gain. The prototype was therefore rejected and the known-correct packed implementation remains.

Exact full-byte callback hashes matched for 181 canonical tensors in both PP and TG: 120 projection outputs, 30 convolution inputs, 30 recurrent final outputs, and final logits. A deterministic 32-token completion also matched byte-for-byte.

Four-V620 ABBA benchmarks with seven repetitions measured:

| Test | Sibling fusion off | Sibling fusion on | Change |
|---|---:|---:|---:|
| PP512 | 3052.22 tok/s | 3094.22 tok/s | **+1.38%** |
| TG128 | 82.57 tok/s | 86.19 tok/s | **+4.39%** |

Both arms used `GGML_HIP_GFX1030_NATIVE=1`, `GGML_HIP_GFX1030_Q8_1_FUSION=1`, `-sm layer -ts 1/1/1/1`, FlashAttention, and `-ub 256`; only the sibling-fusion switch differed.

## Validation commands

Build for gfx1030 using the normal HIP configuration, then run stock and native arms separately. Representative checks are:

```bash
# Gated DeltaNet
build/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0
GGML_HIP_GFX1030_NATIVE=1 \
  build/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0

# FlashAttention
build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0
GGML_HIP_GFX1030_NATIVE=1 \
  build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0

# Six-row selector contract
build/bin/test-mmvq-batch6-config

# Synthetic MMVQ and bounded generic routed MMID
GGML_HIP_GFX1030_NATIVE=1 build/bin/test-mmvq-rdna2
GGML_HIP_GFX1030_NATIVE=1 build/bin/test-mmid-rdna2 \
  --type q4_k --k 2048 --n 512 --batch 6 \
  --experts 64 --top-k 4 --routing hot

# Exercise the native-only validated high-top-k advisory override
GGML_HIP_GFX1030_NATIVE=1 build/bin/test-mmid-rdna2 \
  --type q4_k --k 2048 --n 512 --batch 6 \
  --experts 256 --top-k 8 --mmvq-batch6-hint

# Report and exercise exact graph-scoped standard-Q8_1 TG reuse
GGML_HIP_GFX1030_NATIVE=1 \
GGML_HIP_GFX1030_Q8_CACHE=1 \
GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY=1 \
build/bin/llama-bench -m model.gguf -ngl 999 -fa on -p 0 -n 128 -r 5
```

Use the guarded scripts when collecting reproducible artifacts:

- `scripts/benchmark-gfx1030-mmvq.py`
- `scripts/verify-gfx1030-mmvq-run.py`
- `scripts/benchmark-gfx1030-native-fa.py`
- `scripts/verify-gfx1030-native-fa-run.py`

## Related model preparation

The native paths do not require a special GGUF. The separate Qwen Q4S8 quantization, calibration, quality, and benchmark report is maintained in [edwinbrowwn/gguf-q4s8](https://github.com/edwinbrowwn/gguf-q4s8). It is independent of the runtime environment variables above.