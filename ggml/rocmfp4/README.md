# ROCmFP4

ROCmFP4 is the experimental `Q4_0_ROCMFP4` GGUF tensor type for Strix Halo.
The current AMD-tuned variant stores 32 weights per block as packed
E2M1-derived 4-bit values plus two unsigned E4M3 scale bytes, one per
16-weight half, for 18 bytes per block. The codebook stores half-scale signed
integer levels up to `10`, representing `5.0` raw-scale units after the scale
factor is applied. This keeps outlier pull lower than the original wider test
range while preserving a fast integer-dot backend shape.

This directory owns the format-specific implementation. The rest of ggml only
registers and dispatches the type.

Current status:
- The format runs on CPU, Vulkan, and ROCm/HIP in this custom tree.
- `Q4_0_ROCMFP4` is the pure 4.50 BPW dual-scale path.
- `Q4_0_ROCMFP4_LEAN` keeps ROCmFP4 for dense tensors but protects token
  embeddings with `Q5_K`. On the Qwen3-4B Strix test pass this closed most of
  the Q4_0 coherence gap while staying around normal Q4 size.
- `Q4_0_ROCMFP4_COHERENT` protects token embeddings with `Q6_K` and is the
  quality-first ROCmFP4 preset.
- `Q4_0_ROCMFP4_FAST` is the 4.25 BPW single-scale speed path. It is the
  smallest and fastest decode variant, but the pure version gives up too much
  PPL to be the default quality target.
- `Q4_0_ROCMFP4_FAST_COHERENT` combines the fast 4.25 BPW transformer layout
  with `Q6_K` token embeddings. On the Qwen3-4B Strix test pass it is the
  current balanced AMD target: smaller than `Q4_0`, faster than `Q4_0` on
  decode on both Vulkan and ROCm, and close to `Q4_0` PPL on the short
  WikiText-2 check.
- `Q4_0_ROCMFP4_STRIX` is the current quality-biased Strix Halo preset. It
  keeps most transformer tensors on `Q4_0_ROCMFP4_FAST`, protects token
  embeddings with `Q6_K`, and uses the dual-scale `Q4_0_ROCMFP4` layout for
  attention-K and attention-V tensors. On Qwen3-4B it improved the short
  WikiText-2 PPL to `13.8865` at `4.49 BPW` while still beating the
  same-flags stock `Q4_0` decode baselines on both Vulkan and ROCm.
- `Q4_0_ROCMFP4_STRIX_LEAN` is the compact Strix Halo preset. It keeps the
  STRIX all-layer dual-scale attention-K/V protection, uses the FAST
  single-scale transformer layout for the dense tensors, and protects token
  embeddings/output with `Q5_K` instead of `Q6_K`. On the Qwen3-4B validation
  pass it landed at `4.38 BPW`, improved short WikiText-2 PPL versus
  `FAST_COHERENT`, and kept Vulkan decode in the `81 tok/s` band.
- A smaller first/last-layer-only K/V protection recipe was tested but not
  promoted. It reached `4.48 BPW`, `80.13` Vulkan decode, and `69.85` ROCm
  decode, but PPL regressed to `14.0167`, so the all-layer K/V STRIX preset
  remains the quality target.
- The ROCm/HIP MMQ path for `Q4_0_ROCMFP4_FAST` uses one scale per 32-weight
  block, matching the actual FAST layout instead of duplicating the scale into
  two half-block slots.
- The ROCm/HIP vector-dot and MMQ loaders use a ROCmFP4-owned Codebook10
  expander backed by AMD `amdgcn_perm` constants. This avoids the generic
  table-load helper on the hot ROCm path.
- The ROCm/HIP hot paths use a ROCmFP4-owned unaligned 32-bit quant-byte load
  for packed nibble bytes. ROCmFP4 blocks are 17 or 18 bytes wide, so the older
  byte-safe assembly path was conservative but expensive on Strix Halo HIP.
  Direct unaligned dword loads improved ROCm `MUL_MAT` and FlashAttention
  microbenchmarks while the Qwen3.6 27B MTP guard held `33.4 tok/s` short and
  `27.7 tok/s` sustained. It is enabled by default and can be disabled with
  `-DGGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD=0` for isolation.
- The ROCm/HIP FAST MMVQ/MMQ path now uses the same ROCmFP4-owned unaligned
  quant-byte load instead of the generic byte-assembly helper. The full
  promoted gate measured FAST ROCm `MUL_MAT` at `45.17`, `58.38`, `90.54`,
  and `157.83` us for `n=1/2/4/8`, and the Qwen3.6 27B MTP guard improved to
  `33.6 tok/s` short and `28.0 tok/s` sustained. The ROCm runtime guard now
  tightens the FAST ceilings to protect this band.
- Qwen3.6 35B A3B MTP ROCmFP4 STRIX_LEAN was checked separately at native
  `262144` context on ROCm0. In the reasoning-off discovery sweep, `n-max 1`
  was best sustained at `72.2 tok/s`, while `n-max 5` was best burst at
  `107.3 tok/s` but slower sustained at `64.9 tok/s`. With reasoning on, a
  follow-up sweep found `n-max 2` best sustained at `92.6 tok/s` short and
  `80.6 tok/s` sustained; `n-max 3` was close at `104.3` short and `80.1`
  sustained, while `n-max 5` stayed burst-only at `98.6` short and `73.3`
  sustained. KV-cache isolation then found that q8 main KV with q4 draft KV
  improved the reasoning-on `n-max 2` profile to `93.7 tok/s` short and
  `85.6 tok/s` sustained; full q8 main/draft KV reached `93.2` / `85.2`,
  draft-only q8 stayed at `80.5` sustained, and K-only / V-only q8 main KV
  regressed to `71.8` / `73.3` sustained. Rechecking draft depth under q8 main
  KV moved the best sustained profile to `n-max 3` at `104.3 tok/s` short and
  `89.3 tok/s` sustained, while `n-max 4` became the best burst-only profile
  at `111.2 tok/s` short and `78.7 tok/s` sustained. The Pi sustained profile
  therefore uses `n-max 3` with q8 main KV and q4 draft KV, while a separate
  `n-max 4` burst profile is available for short-response experimentation.
  The older `n-max 5` burst alias remains available for comparison.
  Batch and CPU thread follow-ups did not beat the promoted runtime shape:
  `-b 1024 -ub 512`, `-b 2048 -ub 512`, `-b 512 -ub 256`,
  `-t 24 -tb 32`, and `-t 12 -tb 32` all measured in the `88.7`-`89.1 tok/s`
  sustained band, so `-b 512 -ub 512 -t 16 -tb 32` stays promoted.
  Current KV isolation rechecked the promoted shape at `104.3 tok/s` short and
  `90.1 tok/s` sustained with q8 main KV, q4 draft KV, `n-max 3`, and
  `p-min 0.25`; full q8 draft KV tied at `104.5` / `90.0`, while draft-only
  q8 reached only `82.2` sustained. K-only and V-only q8 main KV regressed to
  `70.5` and `74.6` sustained, so both accepted K and V need q8 and the draft
  KV can stay q4. The updated default 35B guard then passed at `103.3 tok/s`
  short and `90.1 tok/s` sustained.
  Sampler-chain trims and backend sampling did not beat the promoted sustained
  profile: `top_k;top_p;temperature` reached `104.2` / `87.5`, the milder
  `penalties;top_k;top_p;min_p;temperature` chain reached `104.1` / `88.9`,
  and `--backend-sampling` reached `104.3` / `89.2`, so the default sampler
  path stays promoted.
  A single-sequence MTP `process()` fast path for `-np 1` was also prototyped,
  built, and rejected because the guard dropped to `104.1 tok/s` short and
  `88.5 tok/s` sustained; the change was removed.
  MoE `rows_per_block` compile-time variants were checked against the same 35B
  guard and also rejected: `rows_per_block=4` measured `103.8` short /
  `89.1 tok/s` sustained, `rows_per_block=3` measured `86.8 tok/s` sustained,
  and `rows_per_block=1` measured `103.6` / `88.7 tok/s`, none beating the
  promoted `104.3` / `89.3` band.
  The Pi server profile was also started and stopped successfully with
  `n_ctx = 262144`, `draft-mtp` initialized, built-in tools enabled, and
  `thinking = 1`; ROCm reported no KFD PIDs after shutdown. The promoted
  profile is now covered by the focused
  `scripts/check-rocmfp4-qwen35-a3b-mtp-regression.sh` guard, which defaults
  to `n-max 3`, q8 main KV, q4 draft KV, and `p-min 0.25` with `100.0` short
  and `85.0 tok/s` sustained floors.
  Follow-up `p-min 0.25/0.50/0.75`, `p-split 0.05/0.20`, and `n-min 1`
  checks on the q4-KV baseline all tied the same `80.5`-`80.8 tok/s` sustained
  band, while `p-min 0.25` on the q8-main/q4-draft path tied at `85.4 tok/s`
  sustained, so the
  conservative `n-min 0`, `p-min 0.0`, `p-split 0.10` defaults remained
  promoted until the MTP internal sampler was changed from `top_k=1` to
  `top_k=10`, making the `p-min` cutoff operate on a meaningful candidate
  distribution while the draft loop still selects the top sorted candidate.
  A pre-change p-min sweep on the actual promoted `n-max 3`, q8-main/q4-draft
  profile measured `104.2` / `89.1` at `p-min 0.25`, `104.1` / `89.0` at
  `0.50`, `104.0` / `89.3` at `0.75`, and `104.2` / `89.1` at `0.90`.
  Follow-up p-split checks measured `103.7` / `88.9` at `0.05` and
  `103.8` / `89.3` at `0.20`; n-min checks measured `103.8` / `88.9` for
  `1` and `104.2` / `89.2` for `2`. None beat the promoted
  sustained-plus-short profile. Combined `p-min`/`p-split` follow-ups also
  tied rather than beat the default: `p-min 0.25` with `p-split 0.30`
  repeated at `104.6` / `89.5`, `p-split 0.40` and `0.50` measured
  `104.6` / `89.5` and `104.4` / `89.5`, and `p-split 0.70` / `0.90`
  measured `104.8` / `89.5` and `104.7` / `89.5`; the same-session default
  tied at `104.7` / `89.5`. With `top_k=10`, the 35B A3B profile now promotes
  `--spec-draft-p-min 0.25`: default `p-min 0.0` measured `104.3` / `89.3`,
  while `p-min 0.25` measured `103.9` / `90.0` after a first sustained run at
  `89.8 tok/s`. The dense 27B profile should stay at `p-min 0.0`; the same
  `p-min 0.25` filter regressed it to `24.6 tok/s` sustained. The full
  all-regression harness can include the 35B guard with
  `INCLUDE_QWEN35_A3B_GUARD=1`.
  The MTP draft sampler now uses a probability-only top-10 helper for this
  path. The draft loop only consumes the sorted top candidate and its
  probability, so skipping the unused final RNG sampler selection preserved the
  output path while improving the 35B A3B guard to `104.6 tok/s` short and
  `90.2 tok/s` sustained. The dense 27B guard remained stable at `33.9` /
  `28.1 tok/s`.
  A fixed insertion top-10 replacement for `std::partial_sort` was tested and
  rejected after the 35B A3B short guard dropped to `73.0 tok/s`, below the
  `100.0 tok/s` floor. A `std::nth_element` plus top-k slice sort variant was
  also rejected after the same guard dropped to `71.5 tok/s`.
  Narrowing the MTP top-k probability accumulator from `double` to `float` was
  also rejected: dense 27B still passed at `34.0` / `28.1 tok/s`, but the 35B
  A3B short guard repeated below floor at `84.1` then `93.4 tok/s`; restoring
  the `double` accumulator recovered the short guard to `104.4 tok/s`.
  A `std::partial_sort_copy` top-k buffer variant built and passed the same
  guard at `104.4 tok/s` short and `90.2 tok/s` sustained, but a same-session
  promoted-build comparison measured `104.1` / `90.2`, so it was not promoted:
  it added sampler buffer/clone complexity without a sustained decode gain.
  Jackrong Qwopus3.6 27B v2 MTP BF16 was converted to STRIX_LEAN ROCmFP4 at
  `4.34 BPW`. At native `262144` context with reasoning on, q4 main/draft KV,
  and `n-max 4`, ROCm0 initially measured `34.9` / `29.6 tok/s`; increasing
  only the batch to `-b 1024 -ub 512` moved sustained decode to `29.9 tok/s`.
  Follow-up batch shapes `1280/512`, `1536/512`, and `1536/768` measured
  `29.8`, `29.9`, and `29.8 tok/s` sustained respectively, so the smaller
  `1024/512` profile remains promoted. Light acceptance filters `p-min 0.05`
  and `0.10` also tied at `29.9 tok/s` sustained without beating the default.
  Lower draft-depth checks with the same promoted batch rejected `n-max 1`,
  `2`, and `3`, which measured only `19.9`, `26.6`, and `27.3 tok/s`
  sustained. Qwopus therefore stays on `n-max 4`, unlike the 35B A3B profile
  where `n-max 3` plus q8 main KV is best.
  `--backend-sampling` tied sustained decode at `29.9 tok/s` but lowered prompt
  throughput, so it is not promoted. Thread split checks at target/draft
  `12/32` and `24/32` also tied `29.9 tok/s` sustained; the simpler default
  `16/32` thread shape remains the recommended Qwopus profile.
  KV isolation confirmed this is not like the 35B A3B q8-main profile:
  draft-only q8 KV measured `35.1` / `29.8 tok/s`, and full q8 main plus q8
  draft KV measured `36.6` / `26.0 tok/s`. The full-q8 path improves burst
  only and regresses sustained decode. Split accepted-KV checks also regressed:
  q8 K only measured `35.7` / `22.3 tok/s`, and q8 V only measured `34.2` /
  `24.7 tok/s`. Qwopus therefore keeps q4 main and q4 draft KV.
  Vulkan0 measured `40.0` / `27.7 tok/s`, with Vulkan `n-max 3` and `n-max 5`
  slower sustained; Vulkan `-b 1024 -ub 512` repeated the same `27.7 tok/s`
  sustained band. q8 main KV regressed this model on both ROCm0 and Vulkan0.
  A normal-path shortcut that only normalized `data[0].p` and filled the rest
  of the top-k probabilities only for debug logging was rejected after the
  35B A3B short guard fell to `92.7 tok/s`, below the `100.0 tok/s` floor.
  Disabling internal sampler timing with `sparams.no_perf = true` was also
  tested and rejected after the same short guard dropped to `96.2 tok/s`.
  Skipping the per-draft `common_sampler_reset()` in the MTP path was rejected
  after the 35B A3B short guard dropped to `68.3 tok/s`; that reset remains
  required to preserve the expected sampler/logit state for this helper.
  A small MTP host-path cleanup now delays `llama_get_embeddings_pre_norm_ith()`
  until the draft loop has confirmed that another draft token will be queued.
  This avoids unused embedding-row pointer fetches on p-min rejects and final
  `n-max` draft tokens. It passed the dense 27B guard at `34.0` / `28.1 tok/s`,
  the 35B A3B guard at `104.4` / `90.1 tok/s`, and the Qwopus best ROCm0
  profile at `35.0` / `29.8 tok/s`. The default all-regression gate also
  passed after this cleanup and ended with no KFD PIDs running. The 2026-05-25
  serial pass measured Qwen3.6 27B MTP at `33.9` / `27.9 tok/s`, ROCm runtime
  FAST `45.66` / `57.81` / `88.27` / `155.05` us and dual-scale `49.16` /
  `51.58` / `83.34` / `151.42` us for `n=1/2/4/8`, ROCm FlashAttention
  `70.86` / `66.51` us for 64d dual-scale / FAST and `189.45` / `172.73` us
  for Qwen-style 128d dual-scale / FAST, and ROCm CPY source-to-dual
  `1106.89` / `1008.56` / `1006.60` us with source-to-FAST `1050.49` /
  `958.98` / `950.50` us for F32/F16/BF16.
  A second attempt to skip the final `n-max` `common_sampler_accept()` call was
  rejected: the 35B A3B short check still reached `104.5 tok/s`, but sustained
  decode fell to `81.1 tok/s`, below the `85.0 tok/s` floor. Reverting only
  that sampler-accept change recovered the guard to `104.3` / `90.0 tok/s`.
  Retesting the internal MTP sampler candidate count on the 35B A3B
  reasoning-on profile rejected both directions around the promoted `top_k=10`
  setting: `top_k=5` fell to `77.3 tok/s` sustained and `top_k=20` fell to
  `69.6 tok/s` sustained.
- Reasoning-off checks on the final 35B q8-main/q4-draft profile are a
  separate lower-throughput mode. `n-max 1/2/3/4` measured `77.7` / `73.9`,
  `90.3` / `75.5`, `100.3` / `71.9`, and `85.7` / `66.1` short/sustained
  tok/s respectively. If reasoning is disabled, `n-max 2` is currently the
  best sustained profile in this bracket; the promoted fastest profile remains
  reasoning-on `n-max 3`.
- Draft-thread-only checks on the promoted 35B profile also tied below the
  promoted band. Keeping target threads at `16/32`, draft `8/16`, `16/16`,
  and `24/32` all measured around `104.1`-`104.2 tok/s` short and
  `89.1 tok/s` sustained, so the default matching draft thread counts remain.
- ROCm/HIP single-token MMVQ uses a full-block vector-dot ratio for the
  dual-scale layout (`VDR_ROCMFP4_Q8_1_MMVQ=4`) while keeping the FAST layout
  on the previous half-block ratio (`VDR_ROCMFP4_FAST_Q8_1_MMVQ=2`). This
  lets dual-scale ROCmFP4 consume one full 32-value block per vector-dot call
  without slowing the FAST dense-tensor path used by STRIX_LEAN. The focused
  ROCm0 `MUL_MAT` guard improved dual-scale from the prior `78.81` us/run
  serial pass to `54.89` us/run, and the Qwen3.6 27B MTP sustained guard
  improved from `26.2` to `27.8 tok/s`. A broader version that also moved
  FAST to `vdr=4` was rejected because sustained Qwen MTP dropped to
  `24.2 tok/s`. A later FAST-only retest after the packed-byte load improved
  the focused FAST ROCm guard to `41.37`, `49.29`, `80.91`, and `139.58`
  us for `n=1/2/4/8`, but was still rejected because sustained Qwen MTP
  dropped to `24.7 tok/s`. The remaining narrower FAST setting,
  `GGML_ROCMFP4_FAST_Q8_1_MMVQ_VDR=1`, was also rejected after it failed the
  focused FAST ROCm `n=1` guard at `60.18` us/run. The knob now rejects
  invalid FAST MMVQ values at compile time; only `1`, `2`, and `4` are valid.
- A direct ROCmFP4 `vec_dot_q_cuda_dispatch<type>` wrapper was tested in the
  MMVQ kernels to bypass the generic constexpr function pointer call. It built
  and passed the focused ROCm guard, measuring FAST `45.16` / `57.52` /
  `89.44` / `156.49` us and dual-scale `50.83` / `51.11` / `84.51` /
  `143.27` us for `n=1/2/4/8`, but Qwen3.6 27B MTP only reached `33.7 tok/s`
  short and `27.9 tok/s` sustained. Because it did not beat the promoted
  sustained band, the code change was removed.
- ROCm/HIP batched MMQ keeps the upstream-style `vdr=8` default for both
  ROCmFP4 layouts, with ROCmFP4-owned compile-time test knobs
  `GGML_ROCMFP4_Q8_1_MMQ_VDR` and `GGML_ROCMFP4_FAST_Q8_1_MMQ_VDR`.
  FAST-only `vdr=4`, FAST-only `vdr=16`, and dual-scale `vdr=16` were tested
  against the Qwen3.6 27B STRIX_LEAN ROCmFP4 bench and did not improve decode;
  the tested runs all stayed at `13.56 tok/s` generation, while dual-scale
  `vdr=16` slightly reduced prompt throughput. The default remains `8`.
- ROCm/HIP RDNA3.5 small-batch MMVQ now uses a ROCmFP4-specific two-warp
  launch geometry for the ROCmFP4 layouts through `n=2`. Strix Halo previously
  inherited the older RDNA2 one-warp table for this path. The promoted
  `GGML_ROCMFP4_RDNA35_NWARPS=2` and
  `GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS=2` defaults keep the single-token
  microbench win and improve the guarded ROCm0 `n=2` shape: the latest serial
  pass measured FAST/dual `66.56` / `58.40` us/run for `n=2`, versus the prior
  `68.85` / `60.98` us/run band before extending the two-warp route. The
  Qwen3.6 27B MTP guard held `33.5 tok/s` short and `27.7 tok/s` sustained, so
  this is promoted as a backend micro-optimization with no sustained decode
  regression. `GGML_ROCMFP4_RDNA35_NWARPS=4` was rejected because dual-scale
  regressed to `57.87` us/run. The remaining 8-warp candidate was also rejected
  because FAST `n=1` regressed to `59.33` us/run and failed the focused ROCm
  runtime guard before multi-column checks. Extending the promoted two-warp
  launch from `n=1` to `n=1..4` improved some focused multi-column ROCm
  microbench rows (`dual n=2` reached `57.14` us/run), but Qwen3.6 27B MTP
  sustained decode fell to `23.6 tok/s`, so the promoted upper bound stops at
  `n=2`. A midpoint `GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS=3` build was also
  checked on the Qwen3.6 35B A3B reasoning-on 262k profile; it reached
  `87.5 tok/s` sustained versus `89.6 tok/s` for the same-session promoted
  build, so `n=2` remains the default.
- ROCm/HIP RDNA3.5 wide-column rows-per-block now has ROCmFP4-owned compile-time
  test knobs: `GGML_ROCMFP4_RDNA35_RPB_WIDE`,
  `GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL`, and
  `GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST`. The defaults stay at `1`. A full
  `RPB_WIDE=2` test improved FAST `n=8` `MUL_MAT` from `167.90` to
  `131.72` us/run, but dual-scale collapsed from `148.11` to `1382.72`
  us/run. A FAST-only `RPB_WIDE_FAST=2` build kept dual-scale safe
  (`145.75` us/run) and improved FAST to `135.31` us/run, but Qwen3.6
  27B MTP did not improve (`33.3 tok/s` short and `27.6 tok/s` sustained
  at `n-max 4`; `n-max 5` remained burst-only at `45.5` short and `24.9`
  sustained). Keep the knob off by default until a real decode guard benefits.
- ROCm/HIP RDNA3.5 `MUL_MAT_ID` routing now has a ROCmFP4-only compile-time
  guard knob, `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH`, for testing whether MTP/MoE
  batches should leave MMVQ earlier. The default keeps the accepted generic
  RDNA3 behavior (`MMVQ_MAX_BATCH_SIZE`). A Strix test with the knob set to `1`
  tied the promoted Qwen3.6 27B MTP path at `33.4 tok/s` short and `27.7 tok/s`
  sustained. Follow-up threshold `2`, `3`, and `4` builds also tied at
  `33.6` / `33.8` / `33.7 tok/s` short and `27.7` / `27.9` / `27.9 tok/s`
  sustained. The same `3` threshold was then checked on the MoE-heavy
  Qwen3.6 35B A3B MTP ROCmFP4 path and regressed to `95.8 tok/s` short and
  `74.0 tok/s` sustained versus the promoted `104.4` / `89.3` band. Reusing
  the older threshold `4` exploratory build on the same 35B guard measured
  `104.2` / `89.2`, a near tie but still below the sustained promoted band.
  No lower routing threshold is promoted.
- Target/draft "dual-stream" MTP overlap was inspected and is not a promoted
  optimization. The current common MTP implementation verifies with the target
  context, mirrors pre-norm embeddings into the MTP context, then drafts through
  serial `llama_decode(ctx_dft, ...)` calls. ggml exposes async graph execution
  and pipeline parallelism, but llama enables pipeline parallelism only for
  multi-device layer-split offload cases. On single-device Strix Halo ROCm0,
  target/draft overlap would require speculative scheduler changes and new
  correctness guards rather than a runtime flag. The long-context optimization
  focus remains KV-cache traffic, FlashAttention, memory bandwidth, and MTP
  acceptance, while dual-scale ROCmFP4 mainly protects coherence.
- The MTP host loop now avoids two small sources of scheduler overhead. It
  reuses a per-sequence drafting-state buffer instead of allocating a
  `std::vector<bool>` on every draft call, and it tracks verify-batch sequence
  bounds with one pass over token seq-ids instead of scanning every sequence for
  every token. It also avoids copying all target verify rows when no previous
  draft is pending, keeping only the carryover hidden row needed for the next
  MTP step. The MTP path also uses a single-sequence batch append helper
  instead of constructing a temporary sequence vector for each `common_batch_add`
  call. It stores only non-final verify rows for partial-accept rollback,
  copying the final row directly to `pending_h` for the next MTP step. It also
  skips the draft-model decode call entirely when no sequence is drafting.
  Debug candidate token formatting is skipped unless debug logging is enabled,
  and the debug verbosity state is hoisted once per draft call instead of
  rechecked inside the per-token loop. This is mainly a cleanup for
  multi-sequence and long-running MTP sessions; the latest single-sequence
  Qwen3.6 27B guard stayed in range at `33.7 tok/s` short and `27.9 tok/s`
  sustained.
- The simple draft path also reuses an object-owned `uint8_t` drafting-state
  buffer instead of allocating `std::vector<bool>` on every draft call. This
  keeps both draft implementations on the same host-side allocation pattern;
  the full serial gate stayed clean, with Qwen3.6 27B MTP at `33.8 tok/s`
  short and `27.9 tok/s` sustained.
- The all-in-one regression harness now accepts a candidate `BUILD_DIR` and
  derives separate `TEST_BACKEND_OPS_BIN` and `LLAMA_CLI_BIN` paths, so backend
  microbench guards and Qwen CLI guards can run from the same candidate build.
  DeepSeek remains opt-in only through `INCLUDE_DEEPSEEK_SMOKE=1`.
- ROCm/HIP vector FlashAttention now uses a ROCmFP4-only RDNA K/Q thread-group
  default of `1` instead of the generic quantized-K RDNA default of `2`. This
  affects only `Q4_0_ROCMFP4` and `Q4_0_ROCMFP4_FAST` K-cache cases; other
  quantized FA paths keep the existing backend default. On the guarded ROCm0
  `FLASH_ATTN_EXT` shape, dual-scale improved from `122.33` to `113.03`
  us/run, and FAST improved from `115.41` to `109.33` us/run. The Qwen3.6
  27B MTP guard remained in the promoted band at `33.5 tok/s` short and
  `27.6 tok/s` sustained. A wider `GGML_ROCMFP4_FATTN_KQ_NTHREADS=4` variant
  was rejected because it regressed the same FA guard to `136.79` us/run
  dual-scale and `124.37` us/run FAST. After the V-side default moved to `2`,
  retesting `KQ_NTHREADS=2,V_NTHREADS=2` and `KQ_NTHREADS=4,V_NTHREADS=2`
  still regressed the focused FA guard, so K/Q remains `1`.
- ROCm/HIP vector FlashAttention also uses a ROCmFP4-only V thread-group
  default of `2`, down from the generic `D/4` path used by other quantized V
  types. On the same guarded ROCm0 `FLASH_ATTN_EXT` shape, this moved
  dual-scale from `113.03` to `85.63` us/run and FAST from `109.33` to
  `80.74` us/run. The Qwen3.6 27B MTP guard stayed in range at `33.4 tok/s`
  short and `27.6 tok/s` sustained, so this is promoted as the new default.
  `GGML_ROCMFP4_FATTN_V_NTHREADS=4` also passed but was slower on the focused
  FA guard; `1` was rejected because gfx1151 HIP compilation exceeded the 64
  KiB local-memory limit for ROCmFP4 FA instances. The FA guard now also
  includes a Qwen3.6-style 128-head-dim, 8-KV-head, 12x-GQA ROCmFP4 shape
  (`hsk=128,hsv=128,nh=8,nr23=[12,1],kv=7680,nb=1`). The accepted default
  measured `246.91` us dual-scale and `219.23` us FAST on that shape.
  Retesting `V_NTHREADS=4` on this wider shape produced a tiny dual-scale
  improvement (`244.01` us) but regressed FAST (`224.21` us) and the existing
  64d guard, so it remains rejected. Retesting `KQ_NTHREADS=2,V_NTHREADS=2`
  regressed the Qwen-style shape to `268.53` us dual-scale and `226.32` us
  FAST, so K/Q remains `1`.
- ROCm/HIP vector FlashAttention now uses a ROCmFP4-owned single-half
  Codebook10 expander in K/Q and V decode paths. These FA call sites already
  know whether they need the low or high nibble stream, so they no longer pay
  to expand both streams and discard one. The focused ROCm0 FA guard improved
  from `86.13` / `81.23` us to `82.24` / `78.15` us for the 64d dual-scale /
  FAST shapes, and from `247.06` / `221.19` us to `237.12` / `206.50` us for
  the Qwen-style 128d shapes. The Qwen3.6 27B MTP guard held at `33.5 tok/s`
  short and `27.7 tok/s` sustained in the full serial gate, so this is
  promoted.
- ROCm/HIP vector FlashAttention also specializes the ROCmFP4 K/Q path when
  `GGML_ROCMFP4_FATTN_KQ_NTHREADS=1`. Each thread owns the full head dot in
  this promoted setting, so the kernel now expands both low/high packed
  Codebook10 streams once per ROCmFP4 block and accumulates both half-blocks
  together instead of loading the same packed bytes twice. The full serial gate
  measured 64d FlashAttention at `81.62` / `78.13` us for dual-scale / FAST,
  and the wider Qwen-style 128d guard improved from `237.12` / `206.50` us to
  `228.58` / `199.32` us. Qwen3.6 27B MTP held `33.2 tok/s` short and
  `27.7 tok/s` sustained, so this is promoted as a real Qwen-relevant FA
  micro-optimization.
  A post-specialization retest of `GGML_ROCMFP4_FATTN_V_NTHREADS=4` was
  rejected. It passed the tightened FA guard and nudged the Qwen-style
  dual-scale microbench to `228.33` us, but regressed 64d dual-scale to
  `86.30` us and Qwen-style FAST to `201.81` us. Qwen3.6 27B MTP stayed at
  `27.7 tok/s` sustained, so the promoted V-side default remains `2`.
- ROCm/HIP vector FlashAttention now uses a ROCmFP4-only V dequant rows-per-
  thread default of `8`. The ROCmFP4 V helper supports 8-value chunks, and the
  post-unaligned-load retest produced a real FA win: 64d dual-scale / FAST
  `68.82` / `66.32` us and Qwen-style 128d dual-scale / FAST `201.13` /
  `172.22` us. Qwen3.6 27B MTP held `33.5 tok/s` short and `27.7 tok/s`
  sustained, so this is now promoted. It can be isolated with
  `-DGGML_ROCMFP4_FATTN_V_ROWS_PER_THREAD=4` to return to the prior setting.
  Retesting `GGML_ROCMFP4_FATTN_V_NTHREADS=4` on top of the 8-row default
  improved only the Qwen-style dual-scale FA microbench (`187.72` us) while
  regressing 64d dual-scale / FAST to `77.35` / `77.25` us and Qwen-style FAST
  to `182.56` us. Qwen3.6 27B MTP tied at `33.5` / `27.7 tok/s`, so the
  shipped V thread-group default remains `2`.
  Retesting `GGML_ROCMFP4_FATTN_V_NTHREADS=8` on top of the 8-row default
  regressed every focused FA row, including Qwen-style 128d dual-scale / FAST
  at `208.94` / `200.45` us, so it was rejected without a longer MTP run.
  Retesting `GGML_ROCMFP4_FATTN_KQ_NTHREADS=2` on top of the 8-row default
  also regressed the focused FA guard: 64d dual-scale / FAST measured
  `80.68` / `78.27` us and Qwen-style 128d measured `245.84` / `207.04` us.
  Because the Qwen-style dual-scale row failed the guard, it was rejected
  without a longer MTP run and K/Q remains `1`.
- ROCm/HIP vector FlashAttention now has a narrow dual-scale 128d V-thread
  specialization: `GGML_ROCMFP4_FATTN_V_NTHREADS_D128_DUAL=4`. This keeps the
  promoted 64d default and all FAST paths on `V_NTHREADS=2`, but uses the
  previously promising 4-thread V grouping only for the Qwen-style dual-scale
  128d shape. The full serial gate measured FA at `70.37` / `66.31` us for
  64d dual-scale / FAST and `188.75` / `171.89` us for Qwen-style 128d
  dual-scale / FAST, while Qwen3.6 27B MTP held `33.7 tok/s` short and
  `27.9 tok/s` sustained with no KFD PIDs left running.
  Retesting `GGML_ROCMFP4_FATTN_V_NTHREADS_D128_DUAL=8` was rejected. It
  slightly improved the 64d dual-scale row (`69.40` us versus a same-session
  `70.92` us promoted build), but regressed the Qwen-style 128d dual-scale row
  from `194.40` to `211.10` us/run. Because long-context Qwen is the relevant
  guard, the promoted D128 dual-scale V grouping remains `4`.
  `GGML_ROCMFP4_FATTN_V_ROWS_PER_THREAD=16` was rejected during compilation:
  the ROCmFP4 V dequantizer intentionally supports only `2`, `4`, and `8`
  rows per thread, and the fixed-copy helper does not support the resulting
  32-byte move. The source now rejects unsupported values with a direct
  ROCmFP4 compile-time error.
  Retesting `GGML_ROCMFP4_FATTN_V_ROWS_PER_THREAD=2` on the promoted
  D128-specialized build was rejected by the focused ROCm FlashAttention
  guard. It regressed 64d dual-scale / FAST to `94.31` / `88.77` us and
  Qwen-style 128d dual-scale / FAST to `270.00` / `223.88` us, so the
  promoted rows-per-thread default remains `8`.
- Vulkan ROCmFP4 scale decode now uses a shared UE4M3 lookup table with
  ROCmFP4's half-scale semantics. This moved the focused Vulkan dual-scale
  `MUL_MAT` guard from `82.86`, `120.77`, and `181.28` us/run to `65.05`,
  `83.07`, and `122.70` us/run for `n=1/2/4`. On Qwen3.6 27B MTP at 262k
  context, the same change moved Vulkan sustained output from the older
  `20.4 tok/s` at `--spec-draft-n-max 4` to `25.0 tok/s`; `n-max 3` now
  reaches `25.3 tok/s` sustained. ROCm0 remains the promoted backend because
  it still holds `33.5 tok/s` short and `27.7 tok/s` sustained.
  A post-LUT Vulkan runtime sweep found a better fallback profile with q8 KV
  and q8 draft KV: `--spec-draft-n-max 4` reached `34.8 tok/s` short and
  `27.0 tok/s` sustained. q8 with `n-max 5` improved only the short burst
  (`47.8 tok/s`) and regressed sustained output to `23.0 tok/s`; f16 KV
  regressed sustained output to `22.5 tok/s`.
  Follow-up isolation showed the Vulkan sustained gain comes from the main KV
  cache rather than the draft KV cache: q4 main KV with q8 draft KV reached
  only `34.6 tok/s` short and `25.0 tok/s` sustained, while q8 main KV with
  q4 draft KV reached `34.7 tok/s` short and `26.9 tok/s` sustained. For a
  Vulkan fallback, q8 main KV plus q4 draft KV is therefore the leaner near-tie
  profile. Splitting q8 across only one accepted KV side was also rejected:
  q8 main K with q4 main V reached `34.8 tok/s` short and `25.4 tok/s`
  sustained, while q4 main K with q8 main V reached `34.6 tok/s` short and
  `23.7 tok/s` sustained. The Vulkan fallback needs both accepted K and V at
  q8. Adding mild acceptance filtering to the lean fallback
  (`--spec-draft-p-min 0.25`) tied full q8/q8 at `27.0 tok/s` sustained while
  keeping draft KV at q4; adding `--spec-draft-n-min 1`, stricter `p-min 0.75`,
  and `p-split` checks stayed at `26.9 tok/s`. Retrying `n-max 5` with q8 main
  KV, q4 draft KV, and
  `p-min 0.25` still produced only a burst win (`47.8 tok/s` short) while
  sustained output stayed at `23.0 tok/s`, so `n-max 4` remains the Vulkan
  fallback setting. ROCm0 q4/q4 remains the overall promoted backend/profile.
  Replacing the shared Codebook10 table with inline integer decode was tested
  and rejected because it regressed the focused Vulkan FAST `n=1` guard to
  `99.43` us/run.
- ROCm/HIP fallback dequant, copy, get-rows, GPU-side quantization scoring,
  and standalone dequant helpers use ROCmFP4-owned HIP helpers for finite
  scales and Codebook10 nibbles instead of relying on generic FP8 handling.
  This keeps non-MMQ conversion paths aligned with the custom AMD format.
- A branchless HIP scalar Codebook10 nibble decoder was tested and rejected.
  It passed the focused ROCm CPY, FlashAttention, and Qwen MTP guards, but
  did not improve end-to-end decode. Qwen3.6 27B MTP measured `33.4 tok/s`
  short and `27.7 tok/s` sustained, and the focused FlashAttention guard
  slowed to `86.38` / `81.13` us for dual-scale / FAST, so the original
  scalar decoder remains the default.
- CPU-side Codebook10 table decode in the quantizer/dequantizer was tested
  and rejected. The full table variant failed the CPU quant guard with
  dequantization at `49.04` cycles/32 for dual-scale and `84.23` cycles/32 for
  FAST. The MSE-loop-only variant passed but slowed normal quantization to
  `4183.33` / `4018.68` cycles/32 versus the restored arithmetic baseline at
  `4034.33` / `3738.85` cycles/32, so the arithmetic CPU decoder remains the
  default.
- UE4M3 scale decode in the ROCm/HIP software path uses a ROCmFP4-owned finite
  scale decoder. It avoids `ldexpf`, builds normal FP32 values directly from
  exponent/mantissa bits, and skips the generic FP8 NaN handling because
  ROCmFP4 row validation already rejects non-finite scale bytes.
- A constant-memory HIP lookup table for the finite UE4M3 scale values was
  tested and rejected. The latest isolated `GGML_ROCMFP4_USE_SCALE_LUT=1`
  pass failed the focused ROCm runtime guard because FAST `n=1` regressed to
  `69.48` us/run; prior checks also regressed ROCm CPY source-to-quant paths
  and FlashAttention. The arithmetic finite-scale decoder remains the default.
- ROCm/HIP dequant conversion kernels use the same ROCmFP4 finite scale
  decoder, keeping tensor conversion aligned with the hot MMQ/MMVQ backend
  path instead of falling back to the generic FP8 helper.
- ROCm/HIP backend CPY now advertises and executes quantized
  `Q4_0_ROCMFP4 -> F32` and `Q4_0_ROCMFP4_FAST -> F32` conversion paths.
  This keeps diagnostic graph ops and fallback tensor conversion inside the
  custom AMD decoder instead of being rejected by backend capability checks.
  Contiguous q-to-f32 copies now use a ROCmFP4-specific packed-byte kernel:
  one HIP thread reads one packed byte and writes the matching low/high
  half-block output values. This gives coalesced output writes for normal
  contiguous graph copies while the existing stride-aware block kernel remains
  the fallback for views and non-contiguous tensors.
- ROCm/HIP backend CPY also supports `F16 -> Q4_0_ROCMFP4`,
  `F16 -> Q4_0_ROCMFP4_FAST`, `BF16 -> Q4_0_ROCMFP4`, and
  `BF16 -> Q4_0_ROCMFP4_FAST`. The kernels convert each 32-value half/bfloat
  block to local FP32 and then run the same exhaustive ROCmFP4 scale search,
  so runtime graph copies keep the coherence-first quantizer instead of
  falling back to unsupported behavior.
- ROCm/HIP ROCmFP4 source-to-quant and quant-to-F32 CPY wrappers now use
  normal multi-thread HIP launch geometry instead of launching one active
  thread per quant block. This keeps F32/F16/BF16 runtime quantization and
  ROCmFP4 dequantization on the same exact conversion math while removing a
  launch-shape bottleneck in helper/fallback graph paths.
- ROCm/HIP ROCmFP4 quant-to-F32 CPY uses ROCmFP4-specific block dequant
  helpers that decode each block scale once before unpacking all 32 values.
  This avoids repeatedly decoding identical scale bytes through the generic
  two-value dequant helper while preserving the same output values. For the
  guarded contiguous `8192x512x2` ROCm0 CPY shape, the packed-byte path moved
  dual-scale `Q4_0_ROCMFP4 -> F32` from the old `740` us/run band to
  `184.48` us/run and FAST `Q4_0_ROCMFP4_FAST -> F32` to `169.99` us/run.
  A split launch geometry is now used: source-to-ROCmFP4 quantization uses
  128-thread workgroups, while quant-to-F32 keeps the accepted 64-thread
  packed-byte launch. A whole-path 128-thread launch was roughly tied overall
  but slightly regressed quant-to-F32; isolating 128 threads to source
  quantization preserves the dequant copy win while shaving the source paths.
  A 256-thread launch regressed F16 source-to-quant paths and remains rejected.
  After the FAST direct-value scoring win, FAST-only 256-thread and 64-thread
  source launch splits were rechecked and rejected as well. The 256-thread
  split regressed FAST F16 source-to-quant to `1047.24` us/run, and the
  64-thread split measured FAST F32/F16/BF16 `1055.32`, `955.35`, and
  `954.78` us/run, so the shared 128-thread source launch remains promoted for
  both ROCmFP4 layouts.
- ROCm/HIP contiguous quant-to-F32 CPY shared-scale staging was tested and
  rejected. Decoding one block scale into shared memory and synchronizing each
  64-thread launch reduced duplicate scale decode work, but the synchronization
  overhead regressed the guarded `8192x512x2` shape: dual-scale
  `Q4_0_ROCMFP4 -> F32` moved from `181.84` to `188.63` us/run, and FAST
  `Q4_0_ROCMFP4_FAST -> F32` moved from `170.65` to `180.03` us/run. The
  direct packed-byte kernel remains promoted.
- ROCm/HIP backend CPY supports same-type packed-block copies for
  `Q4_0_ROCMFP4 -> Q4_0_ROCMFP4` and
  `Q4_0_ROCMFP4_FAST -> Q4_0_ROCMFP4_FAST`, including block-aligned views.
  The kernel copies the packed 18-byte or 17-byte ROCmFP4 blocks directly, so
  graph/view copies preserve exact bytes and avoid dequantize/requantize
  fallback behavior. The launcher uses normal multi-thread HIP workgroups
  rather than one-thread launches, so large packed-view copies scale with the
  number of ROCmFP4 blocks.
- ROCm/HIP runtime quantization now finds the finite UE4M3 candidate nearest
  `max_abs / 10` with a monotonic binary search, matching the CPU/Vulkan
  reference tie behavior while avoiding the older 126-step linear nearest-scale
  scan in the HIP copy utility. It also uses the same conservative lower-scale
  pruning as the CPU quantizer, so runtime F32/F16/BF16-to-ROCmFP4 copies skip
  smaller scales once their unavoidable clipped max-value error cannot beat the
  current best exact scale. The HIP max scan uses the same plain absolute
  compare as the CPU quantizer, preserving NaN/Inf handling while avoiding
  `fmaxf` in the runtime quantization hot loop.
- ROCm/HIP runtime quantization specializes the exact scale search for the
  only block shapes used by ROCmFP4 CPY: `0..15`, `16..31`, and `0..31`.
  This keeps the same candidate order and tie behavior while letting HIP
  compile the dual-scale and FAST source-to-quant paths with fixed offsets.
- ROCm/HIP dual-scale source-to-quant scoring uses a direct Codebook10 value
  helper for 16-value half blocks. Final packed quantization still uses the
  normal code-index helper, but scale scoring no longer has to map the selected
  nibble back through the generic decode helper on the dual-scale path.
- ROCm/HIP FAST source-to-quant scoring now uses the same direct Codebook10
  value helper during scale search. This preserves the exact nearest-value
  thresholds and final packed output, but avoids index-then-decode work in the
  32-value FAST scoring path. On the guarded `8192x512x2` ROCm0 CPY shape,
  FAST source quantization improved from F32/F16/BF16 `1218.69`, `1138.32`,
  and `1138.78` us/run to a repeat guard pass of `1047.21`, `950.93`, and
  `951.00` us/run.
- ROCm/HIP `F32 -> ROCmFP4` runtime CPY stages each 32-value block into local
  FP32 before running the exhaustive scale search, matching the existing
  F16/BF16 source paths. This preserves exact F32 input values but avoids
  rereading the global source pointer throughout scale scoring. On the guarded
  `8192x512x2` ROCm0 CPY shape, dual-scale F32 source quantization improved
  from roughly `9916` to `1117` us/run, and FAST improved from roughly
  `10671` to `1231` us/run.
- ROCm/HIP contiguous-only source-to-quant CPY kernels for
  `F32 -> ROCmFP4` were tested and rejected. They avoided the generic
  multidimensional offset math, but on the guarded `8192x512x2` shape they
  did not beat the existing multi-thread generic CPY path and slightly
  regressed `F32 -> Q4_0_ROCMFP4_FAST`, so source-to-quant ROCmFP4 CPY keeps
  the same guarded implementation for view and contiguous tensors.
- ROCm/HIP backend GET_ROWS supports both ROCmFP4 layouts. This gives pure
  ROCmFP4 tensors the same direct row-gather coverage as stock small-block
  quants on ROCm and keeps embedding-row access on the custom finite-scale
  decoder.
- ROCm/HIP `MUL_MAT` support now covers `F16` activation tensors for both
  ROCmFP4 layouts. The backend stages half activations to contiguous FP32 on
  the GPU, including non-contiguous/views, then feeds the existing Q8
  activation quantizer and ROCmFP4 MMVQ/MMQ kernels. This keeps the forward
  path on the AMD backend instead of rejecting the op and falling through to a
  slower dequantized matrix path. The generic matmul runtime guard explicitly
  allows this ROCmFP4 x F16 forward-inference case, so the support probe and
  execution wrapper agree for batched activations.
- The standalone HIP dequant skeleton covers both the dual-scale and FAST
  single-scale layouts, so future fused ROCm kernels can target the current
  balanced FAST artifact without reintroducing the older scale path.
- The standalone HIP dequant launch now maps one thread to one packed
  ROCmFP4 byte across a 256-thread global grid. This avoids the older
  one-16-thread-block-per-32-values launch shape and gives future fused or
  diagnostic dequant paths normal GPU occupancy scaling.
- Vulkan ROCmFP4 shaders also decode UE4M3 scales directly to the half-scale
  value used by the codebook, matching CPU/HIP and avoiding repeated `* 0.5`
  fixups at dequant and matmul call sites.
- Vulkan ROCmFP4 shaders keep a shared `kvalues_rocmfp4` Codebook10 table.
  Arithmetic/direct Codebook10 decode variants compiled and preserved
  coherence, but measured slower on Strix Halo Vulkan, so the table path
  remains the active backend implementation.
- Vulkan `dequantize4()` for ROCmFP4 and ROCmFP4_FAST decodes each block's
  UE4M3 scale once per 4-value vector instead of calling the 2-value helper
  twice. This improves tested dual-scale ROCmFP4 small-batch matvec shapes
  while keeping the promoted Vulkan runtime guard clean.
- Vulkan `Q4_0_ROCMFP4_FAST` matvec/MMQ kernels have a single-scale dot
  specialization. They combine the two half-block dot sums and apply the one
  FAST scale once, instead of taking the dual-scale path used by
  `Q4_0_ROCMFP4`.
- Vulkan `Q4_0_ROCMFP4_FAST` MMQ stores its block scale as a scalar in
  shared/register cache instead of duplicating it into a `vec2`. The focused
  Strix Halo Vulkan `MUL_MAT` microbench for
  `m=4096,n=1,k=14336,type_a=q4_0_rocmfp4_fast` improved from `62.18` to
  `61.45` us/run.
- A Vulkan packed16 view for ROCmFP4 quant bytes was tested and rejected. It
  compiled, but the focused FAST MMQ guard regressed to `83.03` us/run from the
  accepted `61`-`62` us/run range, so ROCmFP4 Vulkan keeps the byte-view load
  path.
- Vulkan backend CPY/SET_ROWS now has generated `F32 -> Q4_0_ROCMFP4`,
  `F32 -> Q4_0_ROCMFP4_FAST`, `Q4_0_ROCMFP4 -> F32`,
  `Q4_0_ROCMFP4_FAST -> F32`, and indexed SET_ROWS shaders. The SET_ROWS path
  uses the same exhaustive finite UE4M3 scale search as the CPU reference, so
  quantized K/V cache writes favor coherence over a cheap max-abs shortcut.
- Vulkan `F32 -> ROCmFP4` runtime quantization now uses the same exact ordered
  UE4M3 scale search as CPU/HIP: find the scale nearest `max_abs / 10`, expand
  outward, and stop a candidate once its partial error cannot beat the current
  best scale. The candidate set is unchanged, so this avoids a slower linear
  scan without falling back to a lower-quality shortcut.
- Vulkan source-to-ROCmFP4 runtime quantization now also prunes lower scale
  candidates once clipping the block max alone cannot beat the current best
  error. This mirrors the CPU/HIP scale search bound and keeps the candidate
  set exact. The 2026-05-25 Vulkan CPY guard passed after this shader change:
  F32/F16/BF16-to-dual measured `9525.39`, `2350.54`, and `2418.09` us/run,
  dual-to-F32 `516.65` us/run, F32/F16/BF16-to-FAST `10111.85`, `2923.67`,
  and `2949.42` us/run, and FAST-to-F32 `509.65` us/run. The full promoted
  gate passed after rebuild with Qwen3.6 27B MTP at `33.9` / `28.0 tok/s` and
  no KFD PIDs left running.
- Vulkan backend CPY also supports `F16 -> Q4_0_ROCMFP4`,
  `F16 -> Q4_0_ROCMFP4_FAST`, `BF16 -> Q4_0_ROCMFP4`, and
  `BF16 -> Q4_0_ROCMFP4_FAST`. The runtime quantization shader can load
  `float`, `float16_t`, and BF16 source bits and then runs the same exact
  ordered finite UE4M3 scale search. Backend tests keep a bounded NMSE
  tolerance only for the half/bfloat runtime quantization cases because those
  paths are inherently lossy around source-precision tie points; the
  `F32 -> ROCmFP4` checks remain strict.
- Vulkan same-type CPY supports packed-block copies for
  `Q4_0_ROCMFP4 -> Q4_0_ROCMFP4` and
  `Q4_0_ROCMFP4_FAST -> Q4_0_ROCMFP4_FAST`, including block-aligned
  non-contiguous/permuted/view copies. The non-contiguous path uses a
  byte-addressed block shader and preserves exact 18-byte or 17-byte
  ROCmFP4 blocks. Contiguous same-type dual-scale and FAST copies now both use
  the direct byte-copy path, avoiding the generic halfword copy route for
  ROCmFP4's custom 18-byte and 17-byte block layouts.
- Vulkan CPY is now covered by a dedicated regression guard so copy-path
  changes cannot silently fall back or regress outside the ROCm-only CPY gate.
  The guard runs same-type and source/dequant CPY correctness before measuring
  the large copy performance shape, and now streams the performance phase
  through `tee` so long Vulkan runs do not look idle to the command runner.
  The ceilings were tightened after the lower-scale pruning gain, so the old
  slow source-to-quant path no longer passes this guard.
  On Strix Halo RADV Vulkan, the large guarded shape currently measures
  F32/F16/BF16-to-dual at `9525.39`, `2350.54`, and `2418.09` us/run,
  dual-to-F32 at `516.65` us/run, F32/F16/BF16-to-FAST at `10111.85`,
  `2923.67`, and `2949.42` us/run, and FAST-to-F32 at `509.65` us/run.
- Vulkan scalar FlashAttention can now decode ROCmFP4 and ROCmFP4_FAST K/V
  cache blocks. ROCmFP4 K/V is forced to the scalar FA path because the current
  custom decode is not a coopmat/native matrix-core FP4 path.
- Vulkan scalar FlashAttention can use the integer-dot MMQ K path for both
  ROCmFP4 K-cache layouts. The FAST layout expands each 4-bit Codebook10 value
  into packed signed int8 lanes and uses its single UE4M3 scale as the K block
  multiplier. The dual-scale layout also uses packed signed int8 lanes, but
  splits the dot accumulation by half-block so each 16-value half uses its own
  UE4M3 scale. This keeps the quality-biased STRIX K-cache path fast without
  applying one scale to a two-scale block.
- Build and runtime verification generated the Vulkan SPIR-V entries for
  ROCmFP4 copy/SET_ROWS shaders, linked `libggml-vulkan.so`, and passed
  Vulkan ROCmFP4 CPY plus MUL_MAT smoke tests on Strix Halo.
- Row validation rejects invalid scale bytes outside finite unsigned UE4M3
  (`0x00` through `0x7e`) so corrupted custom GGUF tensors fail early.
- Quantization keeps the exhaustive 126-scale UE4M3 search for both normal and
  imatrix paths. Candidate-window scale search was tested and improved GGUF
  creation speed, but it regressed the Qwen3-4B short WikiText-2 PPL on the
  compact FAST path, so it was rejected for coherence.
- The exhaustive scale search now visits the UE4M3 candidate nearest the
  block's `max_abs / 10` first using a monotonic binary search, expands
  outward, and exits a candidate scale once its partial error cannot beat the
  current best scale. This remains exact because every finite scale is still
  evaluated; on the Qwen3-4B FAST_COHERENT artifact it produced a
  byte-identical GGUF while cutting FAST quantization cost sharply.
- CPU scale decode now uses a 127-entry finite UE4M3 half-scale table for the
  exhaustive scale search and row dequantization. The table preserves the same
  exact FP32 values as the former bit-construction decoder, but avoids
  rebuilding them for every scale candidate.
- CPU scale search now has separate unweighted and imatrix-weighted MSE
  scoring paths. The normal path avoids per-value weight branches, while the
  imatrix path precomputes the row-energy calibration weight once per
  16/32-value block instead of recomputing the same `sqrtf` term for every
  candidate scale.
- CPU normal quantization scans each block once for finite values and uses a
  finite-only nearest-Codebook10 helper in the exhaustive MSE loop when the
  block is clean. Non-finite blocks still use the guarded helper. This
  promoted conversion-speed path measured focused dual-scale / FAST normal
  quantization at `3844.38` / `3582.57` cycles per 32 values in the latest
  guard.
- CPU imatrix-weighted scale-MSE scoring now uses the same finite-only
  nearest-Codebook10 helper when the source block is finite. This preserves the
  guarded path for non-finite input while improving FAST imatrix GGUF creation:
  same-session pre-candidate FAST imatrix was `5258.07` cycles / 32 values, and
  two guarded candidate passes measured `4448.73` and `4447.32`. Dual imatrix
  stayed in the noisy guarded band, so the claim is limited to FAST imatrix.
- The weighted and unweighted exhaustive scale loops are split as well, so
  the normal quantizer does not branch on imatrix state for every finite
  UE4M3 candidate. This keeps the exact candidate order and tie behavior while
  improving CPU GGUF creation speed for both ROCmFP4 layouts.
- The CPU quantizers fill every packed nibble byte in each ROCmFP4 block, so
  they no longer clear `qs` before packing. This is an exact store cleanup:
  correctness stayed clean, `q4_0_rocmfp4` quantize improved to `5244.46`
  cycles / 32 values, `q4_0_rocmfp4_fast` improved to `4668.28` cycles / 32
  values.
- CPU scale selection scans block maxima with a plain absolute-value compare
  instead of `fmaxf`. NaNs still do not raise the maximum, infinities are still
  rejected before scale search, and finite output is unchanged. This moved
  `q4_0_rocmfp4` quantize to `5228.64` cycles / 32 values and FAST to
  `4516.45` cycles / 32 values.
- CPU dequantization and quantization-error scoring use the ROCmFP4 arithmetic
  Codebook10 decoder, which avoids a table fetch per unpacked nibble. CPU
  fallback vec-dot keeps the 16-entry codebook table because measured Strix
  Halo fallback dot speed was better with the table there. This hybrid keeps
  the dequant win without regressing CPU vec-dot.
- CPU fallback vec-dot now reuses each packed ROCmFP4 byte for both low and
  high nibbles before table decode. This preserves exact output, moved focused
  dual-scale vec-dot from `31.86` to the `29.77`-`29.82` cycles / 32 values
  band, and kept FAST in the `27.04`-`27.06` cycles / 32 values band.
- Two additional CPU decode shortcuts were rechecked on 2026-05-24 and
  rejected. Replacing arithmetic Codebook10 decode with table decode in
  row dequantization and scale-MSE scoring slowed CPU dequantization from the
  guarded `33`-cycle band to `51`-`84` cycles / 32 values. Returning a direct
  decoded value during MSE scoring instead of index-then-decode also passed
  correctness but regressed normal quantization in the measured guard, so the
  original arithmetic decoder remains the CPU path. A narrower full-block
  weighted-MSE-only retry was also rejected because it regressed the focused
  FAST imatrix timing to `5007.73` cycles / 32 values.
- A direct finite-scale table helper for CPU quantizer scale-search candidates
  was also rejected. It helped dual-scale normal quantization, but repeat
  guard runs regressed FAST normal quantization to `4043.60` and
  `4289.66` cycles / 32 values. The guarded scale helper remains in place
  because the compact STRIX_LEAN preset depends heavily on FAST tensors.
- NaN-only and branchless finite-scan variants were also rejected for default.
  The NaN-only scan was noisy and the same-session `isfinite()` scan measured
  better dual-scale normal quantization (`3735.41` vs `3776.53` cycles / 32
  values), while the branchless boolean-and scan regressed FAST normal
  quantization to `3773.44` cycles / 32 values.
- Per-value Codebook10 quantization uses exact nearest-neighbor thresholds
  instead of a 16-entry scan. The hot quantizer path uses one reciprocal scale
  per candidate/block and multiplies each value by that reciprocal instead of
  dividing per value. On the Qwen3-4B FAST_COHERENT check this kept PPL tied
  with the accepted artifact while cutting GGUF creation time further.
- CPU exhaustive scale search now prunes lower-scale candidates after the
  clipped max-value error alone cannot beat the current best error. The
  imatrix path uses the positive calibration weight of the max element for the
  same conservative bound. This does not change accepted values because every
  skipped smaller scale has a larger unavoidable clip error. The weighted path
  also combines imatrix weight preparation, max-absolute scan, and max-value
  pruning-weight selection into one pre-scan, while both exhaustive loops stop
  once no valid lower or upper scale candidates remain. Latest 40-iteration
  quant guard run: `q4_0_rocmfp4` normal `4049.28` cycles / 32 values, FAST
  normal `3721.48`, imatrix dual-scale `5251.76`, imatrix FAST `4898.75`.
- Adding `GGML_RESTRICT` to the local CPU scale-scoring helper pointers was
  tested and rejected. It compiled and passed the quant guard, but did not
  improve normal quantization and made the imatrix guard noisier/slower in
  the measured pass, so the existing helper signatures were kept.
- Regression guards:
  - `scripts/check-rocmfp4-deepseek-regression.sh` is an optional compatibility
    smoke guard for a second ROCmFP4-converted model family. It is not part of
    the promoted-gain gate because no reproducible DeepSeek speedup has been
    established yet. The default all-regression script skips it unless
    `INCLUDE_DEEPSEEK_SMOKE=1` is set.
  - `scripts/check-rocmfp4-qwen-mtp-regression.sh` verifies the promoted
    Qwen3.6 27B MTP ROCmFP4 262k-context `draft-mtp` path. The stable guard
    uses ROCm0 and `--spec-draft-n-max 4`. On ROCm0, `n-max 4` improved the
    short guard prompt from `27.7` to `33.6 tok/s` versus `n-max 3`, and the
    longer forced-output prompt from `25.6` to `26.3 tok/s`; the dual-scale
    MMVQ VDR tune then moved the longer prompt to `27.8 tok/s`. `n-max 5`
    and `n-max 6` improved only the bursty short prompt after the MMVQ tune
    but regressed sustained output to `24.8` and `24.0 tok/s`, respectively.
    Follow-up `n-max 5` checks with `--spec-draft-p-min 0.75`,
    `--spec-draft-p-min 0.90`, and `--spec-draft-n-min 1` still held the
    short prompt at `45.0`-`45.1 tok/s` but kept sustained output at only
    `24.7 tok/s`, so `n-max 4` remains promoted.
    The guard now checks both short and sustained prompts so high-acceptance
    best-case gains do not hide sustained-output regressions.
    Additional ROCm0 checks rejected larger batch sizes, smaller ubatch,
    q8/f16 KV cache, `--swa-full`, `--no-host`, `--no-op-offload`,
    `--no-repack`, polling changes, lower and higher CPU thread counts,
    greedy sampling, backend sampling, p-split changes, sampler chain trimming,
    and combined MTP+ngram speculation because none improved the sustained
    prompt. After the MMVQ VDR tune, `-b 1024 -ub 512`, `-b 2048 -ub 512`,
    and `-b 512 -ub 256` still tied the promoted `27.8 tok/s` sustained
    result, while `-b 1024 -ub 1024` and
    `--poll-batch 1 --spec-draft-poll-batch 1` stayed at `27.7 tok/s`.
    q8 and f16 KV cache dropped sustained decode to `25.4` and `26.0 tok/s`.
    The MTP draft loop now honors `--spec-draft-p-min`. With that cutoff active,
    `n-max 4, p-min 0.75` tied the promoted path at `33.5 tok/s` short and
    `27.6 tok/s` sustained, while `n-max 5` stayed burst-only at `45.0`-`45.1`
    tok/s short and `24.6`-`24.7 tok/s` sustained for `p-min 0.75` and `0.90`.
    Follow-up ROCm0 KV isolation confirmed this is not a draft-cache issue:
    q8 main KV with q4 draft KV measured `33.5 tok/s` short and `25.3 tok/s`
    sustained, q8 main K only measured `31.5` and `21.4 tok/s`, and q8 main V
    only measured `31.1` and `21.9 tok/s`. The promoted ROCm0 path therefore
    keeps both accepted and draft KV at q4.
    `--spec-draft-p-split 0.05` and `0.20` both tied the `27.7 tok/s`
    sustained guard, `-t 24 -tb 32 --spec-draft-threads 24
    --spec-draft-threads-batch 32` tied, and trimming the sampler chain to
    `top_k;top_p;temperature` regressed sustained decode to `26.6 tok/s`.
    Draft-only q8 KV (`SPEC_DRAFT_TYPE_K=q8_0`,
    `SPEC_DRAFT_TYPE_V=q8_0`) with main KV left at q4 tied the promoted path
    at `33.6 tok/s` short and `27.7 tok/s` sustained, so the leaner q4 draft
    KV default remains promoted.
    The heavier Qwen3.6 27B MTP `STRIX_MTP_Q6` ROCmFP4 profile was also
    checked with the same ROCm0 guard. It reached `30.1 tok/s` on the short
    prompt but only `21.3 tok/s` sustained, below the guard floor, so the
    Q6/quality-biased profile is not promoted for throughput.
    The guard script supports `SPEC_DRAFT_N_MAX`, `SPEC_DRAFT_N_MIN`,
    `SPEC_DRAFT_P_MIN`, `SPEC_DRAFT_P_SPLIT`, `THREADS`, `THREADS_BATCH`,
    `SPEC_DRAFT_THREADS`, `SPEC_DRAFT_THREADS_BATCH`, `SAMPLERS`,
    `BATCH_SIZE`, `UBATCH_SIZE`, `CACHE_TYPE_K`, `CACHE_TYPE_V`,
    `SPEC_DRAFT_TYPE_K`, `SPEC_DRAFT_TYPE_V`, `EXTRA_ARGS`, and
    `SPEC_EXTRA_ARGS` environment overrides for
    controlled sweeps while defaulting to the promoted `n-max 4`, `n-min 0`,
    `p-min 0.0`, `p-split 0.10`, q4 KV, `-t 16`, `-tb 32`, `-b 512`, and
    `-ub 512` settings.
  - `scripts/check-rocmfp4-quant-regression.sh` runs quant correctness plus
    CPU quantizer, dequantizer, and vec-dot cycle ceilings for both ROCmFP4
    block layouts. It checks normal and imatrix quantization, and now also
    protects CPU fallback dequant/vec-dot paths so decode-helper experiments
    cannot pass while slowing non-GPU fallbacks. Recent pass after keeping the
    `isfinite()` finite-block scoring promotion and adding the weighted/imatrix
    finite scorer: dual-scale quant `3844.38`, FAST quant `3582.57`, dual
    dequant `33.59`, FAST dequant `33.13`, dual vec-dot `29.96`, FAST vec-dot
    `27.03`, dual imatrix `5587.43`, and FAST imatrix `4447.32` cycles / 32
    values.
  - `scripts/check-rocmfp4-vulkan-runtime-regression.sh` measures focused
    Vulkan `MUL_MAT` runtime for ROCmFP4 FAST and dual-scale layouts, catching
    shader regressions that can be hidden by end-to-end decode noise. The
    guard covers `n=1`, `n=2`, `n=4`, and `n=8` for both layouts. Recent
    tightened serial pass measured FAST `55.82`, `71.59`, `105.10`, and
    `163.41` us/run for `n=1/2/4/8`, and dual-scale `64.87`, `83.24`,
    `118.56`, and `194.27` us/run for `n=1/2/4/8`. The previous serial pass
    after adding the shared ROCmFP4 UE4M3
    scale LUT was FAST `53.66`, `71.67`, `105.14` us/run and dual-scale
    `65.05`, `83.07`, `122.70` us/run for the same shapes.
  - `scripts/check-rocmfp4-vulkan-cpy-regression.sh` measures Vulkan0 CPY for
    `F32/F16/BF16 -> Q4_0_ROCMFP4`,
    `F32/F16/BF16 -> Q4_0_ROCMFP4_FAST`, and the matching quant-to-F32
    dequant copy paths on the same large shape used by the ROCm CPY guard.
    It is included in the all-regression harness to catch accidental Vulkan
    copy fallback or shader-routing regressions.
  - `scripts/check-rocmfp4-rocm-runtime-regression.sh` measures the same
    focused `MUL_MAT` shapes on ROCm0. The guard covers `n=1`, `n=2`, `n=4`,
    and `n=8` for both FAST and dual-scale ROCmFP4 so MTP-style multi-column
    regressions are caught instead of only protecting single-token matvec.
    Recent tightened serial pass after extending the RDNA3.5 two-warp route
    through `n=2` measured FAST `51.66`, `66.56`, `101.27`, and `168.84`
    us/run for `n=1/2/4/8`, and dual-scale `53.66`, `58.40`, `87.32`, and
    `148.26` us/run for `n=1/2/4/8`. The `n=8` guard was added after a
    rejected wide-rows candidate regressed dual-scale `n=8` to `1382.72`
    us/run.
  - `scripts/check-rocmfp4-rocm-cpy-regression.sh` measures ROCm0 CPY for
    `F32/F16/BF16 -> Q4_0_ROCMFP4`,
    `F32/F16/BF16 -> Q4_0_ROCMFP4_FAST`, and the matching quant-to-F32
    dequant copy paths on a large `8192x512x2` shape. Recent serial-gate pass
    after direct FAST value scoring: F32/F16/BF16-to-dual `1111.73`,
    `1008.69`, `1006.93` us/run, dual-to-F32 `182.25` us/run,
    F32/F16/BF16-to-FAST `1047.21`, `950.93`, `951.00` us/run, and
    FAST-to-F32 `170.36` us/run.
  - `scripts/check-rocmfp4-all-regression.sh` runs the promoted-gain gate:
    quant, Vulkan runtime, Vulkan CPY, ROCm runtime, ROCm FlashAttention,
    ROCm CPY, and Qwen MTP guards serially, then checks ROCm KFD PIDs. Runtime
    microbenchmarks should use this serial path rather than being run in
    parallel with other GPU/UMA workloads. DeepSeek is not part of this
    promoted-gain gate because no reproducible DeepSeek speedup has been
    established; run the separate compatibility smoke only when explicitly
    needed with `INCLUDE_DEEPSEEK_SMOKE=1`. The K/Q block-pair FlashAttention
    serial pass held Qwen3.6 27B MTP at `33.2 tok/s` short and `27.7 tok/s`
    sustained, with no KFD PIDs left running. The same pass measured ROCm
    runtime `MUL_MAT` at FAST `54.16` / `66.13` / `101.73` / `178.36` us and
    dual-scale `51.65` / `57.98` / `88.33` / `148.48` us for `n=1/2/4/8`,
    and ROCm FlashAttention at `81.62` / `78.13` us for 64d dual-scale / FAST
    and `228.58` / `199.32` us for Qwen-style 128d dual-scale / FAST.
  - The MTP `accept()` path now skips re-copying `pending_h` when acceptance
    lands on the final verify row already staged by `process()`. The focused
    Qwen MTP guard held `33.8 tok/s` short and `27.9 tok/s` sustained after
    this host-side cleanup.
  - The MTP `process()` path now stores only the non-final verify rows needed
    for partial-accept rollback, while copying the final target hidden row
    directly to `pending_h` for the next MTP step. The focused Qwen MTP guard
    held `33.9 tok/s` short and `27.9 tok/s` sustained, and the full serial
    gate held `33.7 tok/s` short and `27.9 tok/s` sustained after this cleanup.
  - The simple draft and MTP draft paths now return before
    `llama_decode(ctx_dft, batch)` when no sequence is actively drafting. The
    focused Qwen MTP guard held `33.8 tok/s` short and `28.0 tok/s` sustained,
    and the full serial gate held `33.7 tok/s` short and `27.9 tok/s`
    sustained after this idle/no-draft cleanup.
  - The simple draft path now uses a reusable `uint8_t` drafting-state buffer
    instead of allocating `std::vector<bool>` per draft call. The focused Qwen
    MTP guard held `33.7 tok/s` short and `28.0 tok/s` sustained, and the full
    serial gate held `33.8 tok/s` short and `27.9 tok/s` sustained.
  - The MTP draft loop now skips its debug-candidate loop entirely unless
    debug logging is enabled, hoists the debug verbosity state once per draft
    call, and uses direct vector indexing in the hot per-token path. The
    focused Qwen MTP guard held `33.8 tok/s` short and `27.9 tok/s` sustained
    after this cleanup, so it is kept as a low-risk host-side simplification
    rather than claimed as a decode-speed gain.
  - The simple and MTP draft paths now use a shared direct one-sequence batch
    append helper instead of `common_batch_add(..., { seq_id }, ...)` in the
    hot speculative path. This avoids a temporary sequence-id container without
    changing multi-sequence batch behavior elsewhere. The focused Qwen MTP
    guard held `33.9 tok/s` short and `27.9 tok/s` sustained, and the full
    serial gate held `33.8 tok/s` short and `27.9 tok/s` sustained.
  - The MTP verify-row buffer now reserves for the configured draft depth and
    only grows when needed instead of shrinking/resizing on every verification
    pass. The focused Qwen MTP guard held `33.8 tok/s` short and `27.9 tok/s`
    sustained, and the full serial gate held `33.9 tok/s` short and
    `27.9 tok/s` sustained with no KFD PIDs left running.
  - The MTP `process()` path now copies retained verification hidden rows from
    the target embedding buffer in one contiguous `memcpy()` instead of one
    `llama_get_embeddings_pre_norm_ith()` plus `memcpy()` per row. Rollback
    behavior is unchanged. The focused Qwen3.6 27B MTP guard held
    `33.7 tok/s` short and `27.9 tok/s` sustained, and the 35B A3B guard held
    `104.1 tok/s` short and `89.2 tok/s` sustained. The same build later
    passed the full serial all-regression gate with `INCLUDE_QWEN35_A3B_GUARD=1`,
    including Qwen3.6 27B MTP at `33.8` / `27.9 tok/s` and Qwen3.6 35B A3B MTP
    at `104.1` / `89.3 tok/s`.
  - A matching contiguous-pointer cleanup inside the MTP `draft()` loop was
  tested and rejected. It replaced per-row
  `llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch)` calls with one
  `llama_get_embeddings_pre_norm(ctx_dft)` pointer per draft decode
  iteration. The 27B MTP guard held at `33.8` / `27.9 tok/s`, but the 35B A3B
  repeat measured `104.3` / `88.7` and `104.3` / `89.2 tok/s`, below the
  promoted `89.3 tok/s` sustained band, so the code change was removed.
  - A single-sequence MTP `draft()` fast path was tested and rejected on
    2026-05-25. It removed the active-sequence bookkeeping loop for the common
    `n_seq == 1` case and passed the dense 27B guard at `33.7` / `28.0 tok/s`,
    but the 35B A3B sustained guard collapsed to `25.7 tok/s` despite a
    passing `103.1 tok/s` short check. Reverting that path restored the 35B A3B
    guard to `104.3` / `90.3 tok/s`, so the shared multi-sequence draft loop
    remains the promoted implementation.
  - A dual-scale-only finite-pack CPU quantizer shortcut was tested and
    rejected on 2026-05-24. It passed correctness, but even after isolating
    the shared scale chooser it regressed the protected FAST quant path
    (`q4_0_rocmfp4_fast` normal quant rose into the `3882`-`4022` cycles / 32
    values band versus the clean `3623` cycles / 32 values baseline). The
    final packing loops therefore remain on the guarded nearest-code helper.
- The Strix ROCmFP4 build script now builds the quant regression test
  binaries (`test-quantize-fns`, `test-quantize-perf`, and
  `test-backend-ops`) alongside `llama-cli`, `llama-quantize`, and
  `llama-bench`, so quant and runtime guards work from a clean build
  directory.
- `test-quantize-perf --imatrix` benchmarks the ROCmFP4 quality/coherence
  path through `ggml_quantize_chunk(..., imatrix)` with synthetic importance
  weights. The quant regression guard includes this mode to reject imatrix-side
  CPU changes that would not show up in the normal non-imatrix timing.

High-upside Strix-specific work:
- The next significant speed step is likely a fused ROCmFP4 decode-and-dot
  ROCm matvec path. The current ROCmFP4 wins come from compact blocks and
  avoiding generic decode/copy overhead; a fused path would keep Codebook10
  values and UE4M3 scales in registers/shared memory through the dot product
  instead of materializing wider intermediates.
- A ROCmFP4-aware long-context attention path is the second major target. The
  long-context limit is mostly KV traffic, FlashAttention shape efficiency, and
  memory bandwidth. The current dual-scale layout protects coherence, but it is
  not itself a large speed lever unless the attention kernels consume it
  directly and efficiently.
- MTP target/draft overlap is a possible scheduler project, not a flag. The
  current common MTP flow is serial on one Strix Halo ROCm device. Real overlap
  would need scheduler changes plus acceptance/correctness guards, and it
  should only be promoted if it beats the sustained Qwen 262k guard.
- Coherence should stay protected by tensor-aware ROCmFP4 profiles: dual-scale
  for sensitive tensors and FAST only where guarded model tests show no
  sustained decode or quality regression. Pure speed-only profiles are not
  promoted in this tree without the serial regression gate.

Hardware note:
- This is a special AMD-targeted ggml/llama.cpp quantization and backend
  path. It includes custom Vulkan and ROCm/HIP handling for the new GGUF
  types, but it is not yet a native rocWMMA FP4 tensor-core implementation.
  Current speed gains come from the compact block layout and backend decode
  paths; deeper rocWMMA/cooperative-matrix work is future optimization work.
- NVIDIA CUDA is disabled in the Strix-FP4 build (`-DGGML_CUDA=OFF`). Some
  upstream llama.cpp HIP backend sources still live under
  `ggml/src/ggml-cuda` and are compiled by HIP for AMD, but the ROCmFP4-owned
  helper code and user-facing build/run path are ROCm/HIP/Vulkan targeted.
  This tree also accepts `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` as the AMD-named
  alias for the upstream unified-memory switch.
- The bundled rocWMMA 7.1.0 headers expose gfx12 WMMA paths for FP8/BF8 and
  integer 8-bit inputs, but no native FP4 input type or FP4 WMMA/MFMA builtin
  is visible locally. A true matrix-core ROCmFP4 path therefore needs a
  measured unpack/convert strategy first, such as ROCmFP4 Codebook10 to int8
  WMMA or FP8 WMMA tiles, before claiming native FP4 tensor-core execution.
- rocWMMA FlashAttention is intentionally opt-in via
  `GGML_HIP_ROCWMMA_FATTN=ON scripts/build-strix-rocmfp4-mtp.sh`.
  It currently compiles with the local rocWMMA headers in
  `/path/to/third_party/rocWMMA`, but the Strix Halo benchmark
  regressed sustained Qwen MTP decode (`23.3 tok/s` vs the promoted
  `26.2 tok/s` default HIP FlashAttention path), so the default build keeps it
  disabled. A follow-up on the 35B A3B ROCmFP4 MTP guard measured only
  `99.7` / `76.1 tok/s` versus the promoted `104.4` / `89.3` band, so this
  remains true for the MoE-heavy profile as well.
- TurboQuant and TriAttention are not present as runtime flags in this isolated
  tree. Integrating them would require a source-level merge into `strix-fp4`;
  do not claim support or promote them unless they beat the serial ROCmFP4
  regression gate.
