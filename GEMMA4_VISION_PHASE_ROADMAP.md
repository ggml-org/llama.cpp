# Gemma 4 vision phase arena roadmap

## Objective and status

Add image understanding to the qualified Gemma 4 31B adaptive target and, subsequently, its matching shared-KV MTP assistant. Reuse Qwen's single-parent phase arena, projector storage, host embedding handoff and serial execution machinery. Preserve Gemma's authoritative local SWA cache while lending only reclaimable regions to vision encoding.

Status: GV0-GV9 complete for the scoped Gemma 4 31B image profile on CUDA SM120, 2026-10-24. This is the user-approved Q3_K_S target, matching local Q8_0 projector and matching external Q8_0 assistant at draft lengths 1-3, plus target-only. The historical Q3_K_M text qualification and post-D512 MTP3 sweep remain separate and unchanged. No attention arithmetic, CUDA device library, second arena, new allocator, CPU/UVM fallback or additional decoder KV history was introduced by this feature.

The [published qualification receipt](benchmarks/data/gemma4-vision-qualification.json), [phase measurements](benchmarks/data/gemma4-vision-measurements.json), [README vision Quickstart](README.md#vision-launch-examples) and [detailed vision report/API guide](docs/multimodal/gemma4-vision.md) are authoritative. Source and binary build origins remain explicit: committing documentation does not rebuild or relabel the measured binaries. Failed attempts and accepted partial results remain immutable under `benchmarks/results/gemma4-vision-20261023/`.

Implementation commits are `c24069f2d` for protected arena lending and `c211bf0bd` for Gemma image/MTP integration and its tests. Both preserve the qualified source contents; the measured binaries retain their original build provenance.

### Qualified scope and limits

- Q3_K_S target GPU weights of 12583.79 MiB and the optional Q8_0 assistant's 475.81 MiB remain outside the parent and on GPU. The local Q8 projector has 807772384 file bytes, SHA256 `ce5b38c7b9f22468acd0156c278cf8d70622d9b1669f34e23ea01f448d45448e`; its GPU weight grant is 807838208 bytes. It retains required F16/F32 tensors. Q4_0 is a characterized smaller candidate, not the selected deployment artifact.
- All five requested image budgets 70/140/280/560/1120 are exercised. Canonical square rows are 64/121/256/529/1089; aspect ratios change actual counts. Set both batch and ubatch to 256/256/512/768/1280 respectively. Normal parents are 2048 MiB through budget 560 and 2143 MiB at 1120. These are workload-specific profiles, not universal fit guarantees.
- Complete per-scenario native API controls cover modes 0-3 through budget 560, plus target-only 1120. Native 1120 MTP1-3 passes text control, then fails a 671878528-byte encoder-workspace allocation. Its arena profile fits and has separate exact native component/logit/hidden oracles plus save/restore/restart self-consistency. It is not full native API equivalence at 1120 MTP.
- The largest two-image 1120 workload has an exact 2246487936-byte phase minimum. Independently fitted maxima are 2812 MiB target-only and 2332 MiB MTP1/2/3, each with an actual rejected +1 MiB neighbor. Protected SWA varies from 406.25 to 731.25 MiB with the admitted microbatch.
- The 4 Ki matched profile includes 1024-output image continuations at all modes. The long profile is capacity 131072/prefix 98304/budget 70/all modes: 68 completions, 4352 IDs, 16 state actions and eight initial/restarted ring proofs. Saved states exceed 4 GiB. Long evidence is lifecycle/self-consistency, not native equivalence or full 256 Ki image prefill.
- Strict default encoder-forward/public target-only/MTP3 and full-vector memory gates use the explicitly approved 1536 MiB sanitizer-only parent, API-all reporting/error-exit 99 and final zero-error summaries. Normal 2048/2143 MiB profiles are unchanged. Maximum-budget component/decoder strict checks are separate; no full joint 1120 MTP sanitizer pass is claimed.
- Q8/Q4 is the all-budget reference. Fresh basic image API controls at budget 70 also cover Q8/Q8 and Q5_1/Q5_1 across all modes. The 64 reader/operator pairs are not a 64-pair whole-model image performance matrix.
- The current synthetic fixture/F16 reference comparison has no additional label errors for Q8 or Q4 at budgets 70/280, but is not broad OCR/chart/detail-sensitive task-quality certification. Encoder native-versus-arena outputs are exact within each artifact; different quantizations need not agree bitwise.
- Only one serial request on one CUDA SM120 device is runtime-qualified. Other Gemma variants, Gemma DFlash, audio/video, parallel/multi-GPU and older physical GPU vision execution are not qualified. HOST/ASan/common-backend tests are not complete accelerator ports.

### Completed stage ledger

| Stage | Accepted evidence and scope |
| --- | --- |
| GV0 Quantized artifacts/reference | Independent Q8/Q4 quantization from pinned F16, tensor inventories/fallbacks, reference floor-aligned Gemma4V geometry, frozen native outputs, 15 aspect controls and 108 fixture-quality API requests. Qwen/Gemma4UV preprocessing stays unchanged. |
| GV1 Feasibility | Exact protected-plus-vision quotes, diagnostic accounting red/fix, all-mode two-image 1120 fits and +1 MiB rejection logs, explicit native encoder-workspace capacity failures. |
| GV2 Protected lending | Existing persistent-region plans retain SWA, reject unknown/held loans and unsafe serial readers, and explicitly drain/rebind. Exact/short/overflow/no-mutation and reconstruction checks pass. |
| GV3 Projector/handoff | GPU encoder outputs match native bits, with owned host outputs, partial-upload cancellation, queued compute failure, cleanup/retry and largest-budget protected-state preservation. |
| GV4 Image prefill | Native local non-causal and global causal mask policies agree with compact bounds; all-budget raw embeddings and exact target/assistant full vectors include the largest 1089-row inputs. Stock attention order is unchanged. |
| GV5 Target API lifecycle | Canonical budget 70 boundary run: 224 completions, 3584 IDs, 72 exact-cause rejections and 32 state actions. All-budget basic controls include adjacent/separated images, image-only/first/last, portrait/narrow and cancel recovery. |
| GV6 Shared-cache MTP | Modes 1-3 inherit target KV and use the existing registered read-only child. Image-conditioned nineteen full logit/eighteen hidden vectors match native; target widths 1-4 and source preservation/rebind checks pass. No assistant KV catch-up. |
| GV7 Long/failure/memory | Native-paired 4 Ki long decode: 80 records / 81920 IDs; all-mode 98K-prefix ring/state/restart: 68 records; 1120 MTP lifecycle: 22 records per mode; explicit normal and strict component/full-vector/public memory gates. Corpus-cap and fixture mistakes are preserved, not runtime passes. |
| GV8 End-to-end/regression | Exact-source/binary lineage, portable HOST/ASan 55-test inventories, all 64 D512 pairs with 43778 normal/strict assertions, Qwen all-pair/MTP/DFlash/vision and Gemma acceptance/API regressions. The text-only eight-pair screen uses a predeclared 3% material-drop review threshold, not repeated statistical claims. First-image phase time/memory and actual image-aware maximum-fit evidence are published separately. |
| GV9 Documentation/handoff | Actual artifact/finder examples, one-parent API/lifetime guide, supported/no-fit matrix, measurement CSV/JSON/plots, local command/link/Mermaid checks and unchanged upstream README bytes. Original production is healthy on 1234; no owned qualification children remain. |

### Evidence lineage and deliberate limitations

The final continuation adopts completed results only with unchanged tracked source, runtime, test binaries, inputs and command profiles. The first long-ring attempt failed before inference because 262144 corpus characters supplied only 56454 tokens; removing that fixture cap provides 351790 tokens. A vocab-only proof retains the exact 1333-token prefix and all 146 adopted request hashes. The corrected long run is separate and complete.

The first broad run failed one existing Qwen fixture assertion: a 1 GiB parent with 128 prefix repetitions was resident, but the invocation omitted `--resident-control`. All three native-logit comparisons already had zero error. No CXX/runtime/math/assertion changed. The corrected declared resident case passes 231 assertions, and a separate existing 6000-prefix/16 MiB-requested legacy-pool case passes 194 assertions with actual streaming (effective decode grant 82.08 MiB, 27 active / 11 resident / 26 ring pages). Do not treat its requested 16 MiB as a whole-parent or total-device-memory bound.

Native image capacity failures are not implementation passes, staged native vectors are not simultaneous residency, and functional output without a sanitizer footer is not a memory pass. The image arena trades simultaneous residency for projector reload and phase-handoff latency; it is not a universal image-speedup claim. Encoder wall time includes host output, and CPU preprocessing/image-text-prefill portions are not independently isolated. Device samples can miss short peaks. The arena does not bound all host allocations or driver/executable memory.

The original acceptance plan and chronological evidence below are retained for audit. Their future tense and older "pending"/"not qualified" statements describe development history, not unfinished GV stages; this completed ledger and the published manifest take precedence. G1-G6 text stages are not reopened, and the evaluated/deferred G5c async prototype remains removed.

## Archived development status (superseded)

Status: implementation in progress, 2026-10-24. The user selected Q3_K_S main-model weights for this feature's validation to free memory, while deferring layer-by-layer vision-weight streaming. Current focus is reference-preprocessing/feasibility closure and GV7 memory hardening, followed by fresh canonical GV5/GV6 lifecycle and broad regressions. Earlier pre-resize ownership/encoder/image-mask and public lifecycle results remain immutable but do not qualify the changed preprocessing. Gemma image execution with the adaptive arena is not yet fully qualified. The preserved Q3_K_M text baseline is implementation commit `6612846ed`, including the generic D512 feature pipeline. Its separate MTP3 sweep completed all 32 maximum-fit points from 8 through 256 Ki, with 256 outputs per point. This does not extend the historical all-mode text sweep beyond 120 Ki. Preserve both text cohorts unchanged.

### Current evidence (focused gates, not overall completion)

**GV7/GV8 follow-up,2026-10-24:** user requested continuation through GV9. Fresh55 HOST and55 ASan/UBSan registrations pass, each with two expected CUDA-only skips. The private image-vector fixture now declares only its actual1/4 output rows through the existing common output-limit helper; this corrected an observed1280-output scratch over-reservation without increasing the arena, changing image geometry or removing any vector comparison. `gv4-all-budget-vectors-v1` accepted22 all-budget Q8/Q4 encoder component gates (both1120 strict summaries zero) before that fixture quota rejection. V2 explicitly retained those exact-runtime/encoder-binary components, re-exported the1089-row input identically, then passed full target and assistant1089-row vectors. Its failed observer-based fault injection was a fixture semantic error: stock false returns mean successful observation stop. The corrected queued exception/upload-cancel fixture passes78 assertions in `gv7-encoder-exception-retry-v4`; current projector HOST/ASan each pass127 checks.

`gv6-split-vector-memory-1536-v3` freshly passes all five gates and both strict zero-error summaries after the output-limit correction. All19 full logit and18 hidden vectors remain exact, and all three ordinary-native files SHA-match the frozen pre-correction oracle. Actual public-memoryV1 completed36 exact functional records/14 valid phases but lacked both sanitizer footers after whole-group shutdown; it remains FAILED memory qualification. A CPU-red-backed launcher now stops only the owned application, leaves its profiler alive and requires both its zero-error footer and exitcode0. Fresh actual encoder-forward/public memoryV2, larger canonical public profiles, long/window/ring/state stress, broad regressions and GV9 documentation remain required.

**Newest reference-resize cohort:** Gemma4V now uses floor-aligned resize-to-budget from the reviewed reference geometry; Qwen and Gemma4UV retain their old preprocessor. The archived reference is Transformers commit `19d179f2165fc02dbc7c45456d794a3f78d7e310`, source SHA256 `5d280d5448b1c219183a27e95b6aa7178b350275772d07881c91c65f13fa815e`. Square outputs are64/121/256/529/1089 for the five budgets; the reference max-target rule does not use a separate minimum-resolution floor. Behavioral red28/40 then expanded119 resize/bounds assertions pass, alongside the common1807-owner checks and9 HOST/ASan suites. No CUDA device library or attention arithmetic changed.

`gv0-reference-encoders-v3` passed7 new F16/Q8/Q4 controls, including Q8/Q4 native-exact phase outputs and strict CUDA zero errors, plus20 exact quotes/grouping controls. At560/1120 two images must split under the existing1024-token encoder batch cap; a fixture that incorrectly demanded combining them is preserved as failedv2. Per-fixture F16-relative embedding L2 is6.41% for Q8 and26.72% for Q4; these are descriptive differences, not task-accuracy scores. `gv0-fixture-quality-v1` then completed108 chat requests: all18 closed-label checks per precision at budgets70/280 match F16 and known synthetic labels, with native/arena content and usage exact. The user explicitly selected current fixtures and F16 reference; no real-world OCR/chart accuracy claim. Q8 remains the primary validation projector, Q4 a smaller evaluated candidate, never an automatic fallback.

`gv1-image-fit-1120-v2` completed actual canonical two-image workloads at all four modes: minimum2143 MiB (2246487936 bytes), physical maxima2812 MiB target-only and2332 MiB MTP1/2/3, each with tested rejected+1 MiB. Its audit found diagnostics double-counting protected SWA during vision; functional capacity evidence is retained, but resource accounting was not accepted. Exact1-failure/1027 TDD isolated that bug; current diagnostics exclude persistent bytes from transient borrowed bytes. Fresh `gv1-image-fit-1120-accounting-v3` revalidated all four maxima and upper rejections on the corrected runtime, with both image phases requiring valid diagnostics, transient grants equal to weights plus compute, and returned borrowed bytes zero. This is workload-specific capacity, not native equivalence.

**Strict-memory scope / approved test exception:** combined native+streamed image-vector memcheck failed two allocation API errors at the second2GiB context, not invalid access, and is not a pass. Fresh-process split controls then passed ordinary native/streamed and strict native (zero errors, input/all19 logit/all18 hidden oracle files SHA-identical), but strict streamed still could not allocate2GiB. On2026-10-24 the user explicitly approved1536 MiB for sanitizer-only vision tests. Normal2GiB and larger-budget physical profiles remain unchanged. The private image-assistant fixture now honors the existing explicit test-arena override; missing-helper compile red and1022 HOST assertions cover explicit1536 and unchanged2048 defaults. Fresh ordinary1536 comparison precedes strict1536 and uses the same full-vector native oracle. Do not suppress API errors, relabel either2GiB failure, or treat this frozen-image gate as actual encoder-forward memory qualification. Broad/long-session gates remain separate.

`gv6-split-vector-memory-1536-v2` passed all five fresh-process gates: ordinary native, ordinary2048 streamed, ordinary1536 fixture control, strict native and strict1536 streamed. Both strict summaries report zero errors; input/all19 full logit/all18 hidden-vector oracle files from strict native are SHA-identical to ordinary native, and every streamed vector matches bitwise. Verification widths1-4, source preservation and entire reusable-prefix overwrite remain checked. All nine runtime hashes are unchanged; HOST and ASan/UBSan each pass1022 assertions. Production restored healthy and command scopes drained. Next `gv7-public-memory-1536-v1` uses the existing exact per-scenario native server oracle for actual Q8 encoder-forward plus target/MTP at modes0/3, budget70, with1536 only under sanitizer. This is separate from frozen-image-vector memory evidence.

Current canonical public lifecycle `gv5-vision-lifecycle-70-reference-v2` passed128 native-paired completions (2048 IDs),32 state actions and valid phase accounting at all four modes. `gv7-input-recovery-v1` passed24 exact-cause media rejections and64 native-paired completions/recoveries, including malformed/oversized images, unsupported audio and blocked file URLs. Larger-budget, long-ring and whole-feature regression gates remain open.

Evidence root: `benchmarks/results/gemma4-vision-20261023/`. Failed fixture attempts and intentional behavioral reds remain beside their corrected runs; no failure is relabelled a pass. Current CXX feature changes are uncommitted and CUDA device-library contents remain identical to the qualified D512 baseline.

- `gv0-artifacts`: pinned Q3_K_S target, matching Q8 assistant and independently quantized local Q8_0/Q4_0 projectors. Q3_K_S GPU model allocation is 12583.79 MiB versus historical Q3_K_M 14038.87 MiB. Q8 projector file is 807772384 bytes, SHA256 `ce5b38c7b9f22468acd0156c278cf8d70622d9b1669f34e23ea01f448d45448e`; Q4 file is 599140576 bytes, SHA256 `424ea8bdd1743a169328fc2fb2603907efd40cf7515c837b4e2b0aa85236d1a9`. Sensitive/unsupported F32/F16 tensors remain high precision. No broad cross-precision quality claim yet.
- `gv0-q3ks-text-controls`: fresh Q3_K_S native-oracle export and exact streamed eight-logit-vector comparison, plus actual assistant verification widths1-4. Historical Q3_K_M frames were not reused.
- `gv0-encoder-controls`: per-projector Q8/Q4 native-exact phase outputs and strict CUDA memcheck zero errors, encoder component only. `gv3-joint-encoder-green` and `gv3-joint-encoder-memcheck`: actual GPU Q3_K_S target plus Q8 projector, 65 assertions each/strict zero errors, preserved full state and native-exact text continuation. Measured single-image vision weight/compute grants 807838208/39485952 bytes; protected SWA 425984000 bytes in a 2048 MiB parent. This is not an image-prefill claim by itself.
- `gv2-expanded-green`: protected lending/restoration, invalid/exact-fit grants and unchanged target state, 78 actual assertions. `gv4-image-mask-green`: 61 actual assertions, frozen native Q8 encoder rows -> image prefill and subsequent full target vectors exact. The compact full mask now shares the native SWA-only noncausal policy; local image SWA is bidirectional, global remains causal.
- `gv6-read-role-full-prefix-green`: 340 actual assertions including entire reclaimable-prefix overwrite, unchanged committed source, real shared Q8 assistant outputs and target verification widths1-4 native-exact. Public target-only and MTP3 startup/image smokes passed with matching output IDs and draft/cache accounting; the inherited assistant model is permitted but foreign model/depth/type overrides remain rejected.
- `gv5-vision-lifecycle-70-v1`: all128 completions (2048 recorded output IDs) and32 state actions passed with actual GPU Q3_K_S+Q8projector, Q8assistant at depths1-3, target-only depth0, Q8/Q4 KV, ctx4096/prefix1333/bub256, budget70, 16 outputs and arena2048 MiB. Each fresh/cached/changed/multiple/adjacent/image-last/cancel/restored/restarted scenario matched its own native token/draft/cache/expanded-extent oracle. All four native/arena state-file pairs were SHA-identical (1411182980 target-only bytes;1411247528 MTP bytes). Every logged vision phase returned borrowed bytes to zero and restored text admission. Host harness API reds became33 green tests. This is not larger-budget, long-ring, whole-feature sanitizer or quality qualification.
- `gv1-budget-matrix-v1`: requested min=max140/280/560 profiles each passed72 native-paired completions across all depths0-3, with microbatches256/512/768 and the same2048 MiB parent. Requested min=max1120 stopped at the first arena image after all9 target-only native cases passed; no MTP1120 qualification. Production was restored healthy. Preserve this rejection, not a numerical/OOM pass. `gv1-budget1120-measure` measured the exact forced-square native preprocessing output:1156 rows, Q8 weights807838208 bytes, device compute713215232 bytes, host32044352 bytes; with ub1280 protected SWA766771200 bytes, the aligned requirement is2287824640 bytes (2181.8396 MiB, whole-MiB minimum2182), versus parent2048 MiB. The measurement used decoder metadata only and a4GiB operator parent, not a joint-target inference claim.
- **New reference-semantics check, before promoting the1120 profile:** current generic native resizing with forced min=max1120 rounds this small square input up to1156 rows. The [reference Gemma processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/image_processing_gemma4.py) instead floors aligned dimensions under the selected maximum (1089 rows for a square at1120). Thus the measured2182 MiB requirement applies to the existing1156-row fixture, not automatically the canonical1120-upper-bound profile. Separate fixture controls and reference preprocessing parity before changing production resize semantics or claiming strict budget compliance. Existing native/arena equality evidence remains valid for its explicitly recorded same-processor scenarios. The user selected current fixtures and pinned F16 reference for quality assessment on2026-10-24.

The initial sequence is quantized-projector selection and native-oracle preparation, measured feasibility, protected arena lending, target-only image execution, then MTP and whole-session qualification. The user explicitly requires a Q8 or Q4 mmproj instead of the large F16 deployment artifact. Keep the high-precision projector as a reference and evaluate Q8_0 and a smaller Q4 candidate with the approved Q3_K_S target and unchanged matching Q8_0 assistant. Whole-projector vision residency is the initial implementation; bounded layer-by-layer weight streaming remains a separate later optimization. No new attention algorithm, second device arena, hidden GPU allocation, automatic fallback or SWA eviction policy is part of the default design. The user's implementation authorization does not authorize changing production configuration/model, pushing or silently changing another profile if this one fails.

## Scope and contracts

### Initial supported configuration

- The approved Gemma 4 31B Q3_K_S target and a verified matching quantized image mmproj. Pin `gemma-4-31B-it-Q3_K_S.gguf` from revision `c1ac76e99d5513b141e8adde7288b85c3f9c32ec`: 13210856384 file bytes, SHA256 `02cdd36af753d2a61832c3e71b621e6d6be65906bbee4adfcde9f62b42154d97`. The Q3_K_M baseline remains unchanged. File-size savings of about 1455.07 MiB are not a measured GPU saving. Evaluate Q8_0 first as the quality-oriented projector candidate and Q4_0 as a smaller candidate with 32-value blocks; inspect actual tensor types, protected high-precision tensors and shape fallbacks. A Q4_K_M recipe is not proof of four-bit contents. Preserve a pinned high-precision reference, but do not require the deployment profile to use it.
- One serial request and one CUDA device, Flash Attention, GPU KV offload, UVM disabled for physical-fit qualification, and a fixed declared parent during each measured request.
- Q8_0 K / Q4_0 V for initial full-model image qualification. Common storage and attention must preserve existing all-64-pair support; three full-server cache-pair controls remain separate from an all-64 image claim.
- Target-only image execution first. Add the matching Q8_0 assistant at draft lengths 1, 2 and 3 only after the target and memory gates pass. Target and assistant weights remain outside the arena and resident under the starting policy.
- Still images, including multiple images and interleaved text/image chunks. Qualify actual image-token and batching limits explicitly.

Audio, video/temporal frame merging, other Gemma sizes or unified/encoder-free variants, DFlash with Gemma, parallel requests, multi-GPU execution, new accelerator backends and further decode optimization are outside this workstream. Gemma 31B is an image/text model; other family members have different modality and encoder designs. [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4).

For each chosen weight artifact, keep numerical operation order, conversion, masks, cache semantics, native checkpoints and source identity unchanged between native and arena execution. Quantizing projector weights changes the model inputs numerically; assess its quality against the high-precision reference separately instead of demanding cross-precision bit identity or weakening same-artifact implementation checks. An implementation may schedule independent work differently, but may not substitute causal text chunking for bidirectional image processing. Do not weaken history-availability guards, tolerances, CUDA API reporting or failed-profile acceptance.

Generate new Q3_K_S native controls and compare Q3_K_S arena execution against those controls with the same projector and request. Do not reuse Q3_K_M logit frames, generated IDs or draft counts as a Q3_K_S numerical oracle. Preserve existing Q3_K_M and Qwen regressions separately; any measured Q3_K_S memory/performance result is a new explicitly labelled cohort.

### Reusable and model-specific responsibilities

| Responsibility | Reuse | Gemma-specific work |
| --- | --- | --- |
| Parent storage and transitions | Existing arena, checked regions, leases, workspace measurement and completion | Preserve fixed local-cache regions while lending only the remaining space |
| Vision encoder and projection | Existing `gemma4v` graph and dynamic preprocessing | Pin the matching artifact and validate its actual shapes, rounding, normalization, positions, pooling and projection |
| Embedding handoff | Existing owned host embedding outputs and ordered session chunks | Ordinary decoder positions and Gemma image markers, not Qwen decoder M-RoPE |
| Decoder attention | Existing native SWA and streamed D512 full attention | Bidirectional image SWA with causal global attention; exact compact-mask parity |
| MTP | Existing Gemma shared-source reader, serial child, verification and stock driver | Suspend/rebind the read-only child and derive fresh post-image carry; no Qwen auxiliary image KV catch-up |
| Server and diagnostics | Existing request parsing, phase diagnostics, finder and vision harness | Capability-based admission of the complete Gemma path, not blanket removal of Qwen guards |

Gemma's encoder has its own 2D position handling; that is distinct from decoder M-RoPE. The existing [encoder graph](tools/mtmd/models/gemma4v.cpp) and [embedding helper](tools/mtmd/mtmd-helper.cpp) already express these different responsibilities. Do not copy Qwen's learned merger, decoder position construction or optional deep-stack feature handling into Gemma.

## Memory lifetime design

### One parent with a protected persistent region

The existing bounded local cache is a fixed suffix in the parent. Preserve that location and its content unless an explicit existing transition requires safe rebinding. Vision grants must come from the disjoint reclaimable region; they cannot require a globally empty lease table when legitimate persistent leases remain.

```text
Outside the device arena:
  target weights + optional assistant weights + driver/native allocations

One device parent:
  [ reusable phase region ...................................... ]
  [ fixed protected Gemma local/SWA K/V suffix                   ]

Reusable region, at different times:
  vision:          encoder/projector weights + encoder workspace
  image/text fill: target graph + attention/conversion/writer + KV pool
  decode/draft:    serial graph/attention/writer grants + larger KV pool

Host storage retained across transitions:
  authoritative full-attention KV, cache metadata and image embeddings
```

The diagram expresses lifetimes, not a new allocation API. A GPU parent remains allocated while its internal leases change. The encoder has no decoder KV cache: its transient attention K/Q/V tensors belong to its measured graph workspace.

| Resource | During vision encoding | Return to image/text prefill |
| --- | --- | --- |
| Local/SWA K/V and required rollback headroom | Protected device storage; no modification by encoder | Same valid bytes, positions and target-owned content identity |
| Full-attention host K/V history | Retained and authoritative after pending publication completes | Existing lazy upload supplies resident pages/ring |
| Full-attention GPU residency/ring | Reclaimable; zero text-pool grant is permitted during vision | New checked grant, mirror invalidation/reload as required |
| Target/draft graph and attention/checkpoint scratch | Drained and released; no active execution pins | Fresh measured bindings before text or draft execution |
| Text quantized-conversion stage | Released; no text-stage reservation needed by the encoder | Restore its existing geometry-derived requirement |
| Text KV writer/publication temporaries | Finish publication before releasing disposable grants | Reacquire before the next append |
| Projector/encoder weights | Reloaded into a checked temporary arena grant | Drain users, retire captures and unload before text reuse |
| Vision graph workspace | Live only during encoding; includes encoder attention and projection intermediates | Released after completion and output retention |
| Image embeddings | Transient encoder output copied to owned host storage | Retained until all relevant chunks are consumed |
| Host session state | Retain committed frontier, positions, generations, image identity and recovery metadata | Publish only valid new state; refresh bindings and speculative carry |

Any additional persistent region discovered in the selected configuration must be included in the protected-resource inventory. Do not treat a stable publication lease as disposable merely because its previous operation finished. Prove its release/reconstruction contract or preserve its bytes and reservation. Gemma does not need a separate assistant KV allocation.

### Admission and accounting

For the default keep-SWA-resident design, use aligned region placement rather than a naive sum:

```text
parent_required >= maximum over enabled phases of
    aligned_layout(protected_resources, that_phase_working_set)

vision_working_set = projector_weight_grant + measured_encoder_workspace
                    + any device output that remains live at the boundary

vision text-KV / text-attention / text-writer / text-graph grants = 0
```

Host output storage avoids retaining a device embedding buffer after encoding. During encoding, the graph's output allocation is still part of its measured workspace and must not be counted twice. Host preprocessing, pinned history, embedding outputs and scheduler scratch need their own host accounting and bounds.

The existing Q8/Q4, ubatch256 local-cache quote is 425984000 bytes, or 406.25 MiB. A 1024 MiB parent therefore has at most 617.75 MiB available for other regions before alignment. This is an example, not a fixed vision budget: image microbatch width can enlarge the persistent SWA allocation and text graph requirements.

The pinned Unsloth revision `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` lists `mmproj-F16.gguf` at 1198957024 bytes (1143.41 MiB), SHA256 `6edcca228213c28d3567a35d22f849eea52d8360875093851959adf5d2f270eb`. That is the reference file size, not a measured GPU allocation or selected deployment precision. The F16 projector is not the deployment target after the user's Q8/Q4 decision. [Pinned publisher metadata](https://huggingface.co/api/models/unsloth/gemma-4-31B-it-GGUF/revision/c1ac76e99d5513b141e8adde7288b85c3f9c32ec?blobs=true).

Published candidates found on 2026-10-22 provide useful size references, but neither has been downloaded, paired with our exact target revision or runtime/quality-qualified here:

| Candidate | Pinned repository revision and file | File bytes / MiB | Caveat |
| --- | --- | --- | --- |
| Q8_0 | `prithivMLmods/gemma-4-31B-it-F32-GGUF` at `d1966676904f1391cf0d16d0a45c1b8ef254aac2`, `GGUF/gemma-4-31B-it.mmproj-q8_0.gguf` | 809541568 / 772.04 | Metadata SHA256 `a52482cc25c90530175cfc23761857c80df55c17e2834581a7a7d6afe259c73c`; not yet a verified pairing |
| Q4_K_M recipe | `Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF` at `95996d565bd68232f4ae3cf7f3a9dc23a3a6c694`, `gemma-4-31B-it-mmproj-q4km.gguf` | 659537504 / 628.98 | Metadata SHA256 `4a03ccaeaaa49cde65a97addac0b2ccd07df4617858aac1472048589ab672033`; publisher reports Q5_0/Q8_0 fallbacks, not Q4 tensors |

Sources: [Q8 artifact metadata](https://huggingface.co/api/models/prithivMLmods/gemma-4-31B-it-F32-GGUF/revision/d1966676904f1391cf0d16d0a45c1b8ef254aac2?blobs=true), [Q4-recipe artifact metadata](https://huggingface.co/api/models/Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF/revision/95996d565bd68232f4ae3cf7f3a9dc23a3a6c694?blobs=true), [publisher tensor inventory](https://huggingface.co/Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF). The current quantizer also maps incompatible Q4_K row widths to Q5_0; see [shape fallback](src/llama-quant.cpp). Q4_0 uses smaller blocks, but excluded or sensitive tensors still need an actual inventory and supported backend operators; do not force every tensor into Q4.

Using file sizes only as rough weight-storage proxies, those candidates plus the current 406.25 MiB SWA reservation total 1178.29 MiB for Q8 and 1035.23 MiB for the Q4 recipe, both before encoder scratch and any wider-image SWA requirement. Neither establishes fit in the old 1024 MiB MTP parent. Derive reproducible candidates from the pinned reference where possible, then measure actual GPU allocations and quality; do not promise half/quarter size from the filename. Projector quantization does not reduce encoder activation or attention workspace automatically.

Target-only and MTP modes must be fitted independently with the assistant present when applicable. The previous maximum-fit text arena is not a universal minimum, a portable recommendation or proof of image fit. An allocatable parent is useful only if every required phase minimum fits inside it.

## Phase transitions and failure behavior

Use the existing serial coordinator; do not create a second model-specific state machine in the server.

1. Validate the requested images, effective preprocessing budget, context extent and all required phase sizes before changing target state. Describe projector weights without uploading into live text storage.
2. Establish a committed target boundary. Complete pending publication, copies, prefetch readers and target/assistant work. Reject new submissions while the parent transitions.
3. Retire text/draft captures and reclaim their transient grants and full-attention device mirror. Keep protected local-cache regions and authoritative host history intact.
4. Bind disjoint vision weight/compute grants, upload the projector and encode the actual image batch. The target and assistant cannot execute concurrently on aliased space.
5. Retain complete host embeddings. Drain encoder work and copies before detaching workspace, unloading weights or returning leases.
6. Restore the text layout, refresh capture/buffer bindings and revalidate target/assistant source views. Full-attention mirror contents may reload lazily; protected SWA contents must already be valid.
7. Prefill image embeddings and intervening text in prompt order. On completion, derive the post-image target hidden/carry state, then enable decode or drafting.

Encoding future images together is allowed only through the existing compatibility/batching contract. Images can be encoded early, but their decoder embeddings must still be consumed in prompt order; adjacent images must not accidentally gain cross-image bidirectional visibility.

| Failure point | Required behavior |
| --- | --- |
| Shape/quota/artifact rejection before suspension | No target-state mutation, no projector upload, no output publication |
| Drain or graph retirement fails | Keep ownership/pins; do not lend memory that may still be in use |
| Weight reload/encoder/copy failure after suspension | Drain or retain unresolved ownership; resume only through a proven safe reverse transition |
| Text resume/rebind fails | Fail closed; do not execute on stale pointers or report successful recovery |
| Cancellation after encoding, before image prefill | Discard unused host embeddings safely and preserve the committed conversation |
| Cancellation/failure during image prefill or speculative verification | Use validated checkpoint/rollback or explicit request/context reset; never imply the pre-image state survived without proof |

Physical arena generations and graph-binding revisions may change during a transition. Unchanged cache bytes must not acquire a false content generation, and unchanged token counts must not hide changed content. After physical rebinding, refresh source leases using actual target-owned identity; never force counters or keep a stale raw pointer to bypass validation.

## Implementation sequence

Each named substage is a review and qualification unit, not a requirement for an empty commit. Keep implementation gates closed until their dependencies pass. No stage is complete from a build, plausible output or process exit alone.

| Stage | Dependency | Deliverable and exit condition |
| --- | --- | --- |
| GV0 Quantized artifacts and reference semantics | None | Pinned Q8/Q4 candidate inventory and quality gate, native preprocessing/embedding/mask references and explicit image-budget behavior |
| GV1 Physical feasibility and phase quotes | GV0 | Measured target-only/MTP capacity matrix, protected bytes and exact aligned phase minima; resolve any no-fit decision |
| GV2 Protected lending and restoration | GV1 | Common parent lends around fixed SWA, with exact-fit/failure/lifetime tests and unchanged Qwen behavior |
| GV3 Projector execution and host handoff | GV2 | Native-exact encoder outputs through borrowed grants, safe unload and retained embeddings |
| GV4 Target image-prefill correctness | GV0, GV2, GV3 | Correct positions, bidirectional SWA/causal global masks and atomic image batching; exact target vectors/KV |
| GV5 Target-only server lifecycle | GV4 | Serial image/text API, cache, save/restore, cancellation and recovery against per-scenario native controls |
| GV6 Shared-cache MTP | GV5 and an approved fitting MTP profile | Image-conditioned drafting/verification at lengths 1-3 without assistant KV catch-up |
| GV7 Long-context and resource hardening | GV5, GV6 | Window/ring/phase/state stress, allocation and malformed-input failure coverage, strict sanitizer gates |
| GV8 End-to-end qualification and performance | GV7 | Fresh common/model regressions, measured memory/time, image-aware maximum-fit finder evidence |
| GV9 Documentation and handoff | GV8 | Supported-profile matrix, actual examples, limitations, audited artifacts and healthy production |

### GV0 Quantized artifacts and reference semantics

**GV0a Pin inputs.** Record repository revision, filename, SHA256, file size, tensor-type inventory and backend weight-allocation quote for the matching 31B mmproj. Reuse the existing target/assistant manifest without silently changing their hashes. Pin image bytes, preprocessing configuration, chat template, corpus and output request. Do not mistake an audio/unified/other-size projector for the 31B vision tower; matching embedding width alone is insufficient evidence of pairing.

Add the Q3_K_S validation artifact to a separate manifest instead of overwriting the historical Q3_K_M target entry. Verify architecture, hidden/input dimensions, layers, sliding-window pattern and assistant compatibility from actual metadata before GPU work.

**GV0b Establish the oracle.** Capture native preprocessed pixels, dimensions, positional inputs, patch count, pooled token count and projected embedding bits. Test square, portrait, landscape and narrow/tall images, then incompatible and mixed-resolution batches. Preserve Gemma normalization, 2D encoder positions, pooling, RMS normalization, clamping and projection through the existing graph.

**GV0c Define decoder semantics.** Record image marker tokens, sequential decoder positions, effective per-image token count and native attention visibility for text/image/text and adjacent images. Gemma exposes budgets 70/140/280/560/1120, but actual output count depends on preprocessing dimensions. Track requested budget, effective limit and actual count separately; a server-clamped budget is not qualification of the requested one. [Google vision documentation](https://ai.google.dev/gemma/docs/capabilities/vision).

**GV0d Select a quantized projector.** Prefer local Q8_0 and Q4_0 candidates derived independently from the pinned high-precision artifact using the existing quantizer, not Q4 generated by requantizing Q8. Record the quantizer binary/source, command, input/output SHA256, metadata compatibility, tensor names/shapes/types and byte totals by type, including every shape fallback and excluded tensor. Preserve positional, convolution, normalization and other tensors required by their operators; do not override exclusions merely to hit a file-size target. Published candidates may be used only after equivalent provenance and pairing checks. The generic quantizer supports multimodal artifacts, but this does not by itself prove every Gemma tensor/backend combination; see [quantizer guidance](tools/quantize/README.md).

Evaluate image/OCR/chart/detail-sensitive tasks at the same image preprocessing and token budget against the high-precision reference. Declare quality acceptance criteria before measurement, include target-conditioned logits/output behavior where useful, and do not report embedding error alone as task quality. Separately require exact native-versus-arena embeddings and downstream execution for each retained quantized artifact. Select the artifact only after both quality and GV1 physical fit are known; Q8 is a quality-oriented candidate, not an automatic choice if it cannot fit.

Freeze a native binary or framed numerical outputs before modifying shared execution paths. Verify full bit identity, including signed zero and frame/profile checks; a recompiled native reference alone can hide a common regression. If simultaneous native target/projector residency fails, retain the capacity failure and explicitly qualify a fresh-process encoder-output export/import control. Such a control proves numerical stages, not simultaneous memory fit. Do not silently enable UVM, move the projector to CPU or reduce the requested image to manufacture a native pass.

Exit: immutable references and a declared native image/mask contract for the exact 31B artifact. Small Gemma variants' causal behavior and PLE requirements must not be generalized to 31B.

### GV1 Physical feasibility and phase quotes

**GV1a Measure before upload.** Extend existing metadata-based projector and graph measurement only where necessary. Quote exact weight bytes, device/host encoder workspace, protected SWA and other fixed regions, text image-prefill working sets, target verification and assistant scratch. Reuse backend allocation-size/alignment queries; file size and parameter estimates are not allocation plans.

**GV1b Build the feasibility matrix.** Cover target-only and MTP1-3, all five image budgets, representative aspect ratios, one/two images and supported encoder batch groupings. Include populated-prefix and fresh-context cases. A 256-row warmup graph is not an upper bound for 1120 image tokens or the pre-pooling patch graph.

Measure Q3_K_S target GPU allocation directly, then fit whole-projector vision grants. Do not substitute file-size delta for available VRAM or start weight-streaming implementation to bypass a failed fit. If the approved target plus retained Q8/Q4 projector still cannot fit a required image budget, retain exact measurements and request a scope/storage decision.

**GV1c Resolve no-fit profiles.** Compare each retained quantized candidate's exact phase minimum with the independently maximum-fit physical parent under the real GPU target/assistant weights. Record required/available/additional bytes, the failing phase and whether failure is arena quota or an external allocation. If no quality-accepted Q8/Q4 candidate fits, stop for a decision. Further alternatives include approved target-weight encoding, bounded projector-weight staging, CPU execution or explicit SWA/assistant eviction; none is an implicit fallback or part of the default plan. Quantized projector outputs must be compared with native execution of that same artifact, with quality assessed separately from bitwise implementation correctness.

Exact-fit and one-byte-short tests must fail before tensor writes, resource publication or target mutation. Check overflow, alignment, shapes and unsupported backend groups. The previous approval of the text preparation stage's higher minimum does not approve any new vision profile or storage policy.

Exit: a fitting target-only profile and a clear fit/decision status for MTP and larger budgets. Publish no universal 16 GB, image-count or token-budget guarantee.

### GV2 Protected lending and restoration

**GV2a Add the missing common ownership contract.** Extend the existing suspended-parent lending path to preserve known fixed regions and lend only proven-disjoint free regions. Keep the original Qwen no-protected-region behavior. The current zero-total-lease assumption cannot simply be removed: distinguish retained protected-region leases from unknown, overlapping or still-executing readers.

**GV2b Adapt suspension/completion.** The existing context entry point expects Qwen recurrent completion, while Gemma needs native local-cache preservation and a full-cache host-publication boundary. Supply the correct model resources through the common interfaces rather than inventing fake recurrent state. Target-only Gemma and Gemma's read-only serial MTP child must both have a validated suspension policy.

**GV2c Restore bindings and graphs.** Preserve parent identity and protected bytes; retire stale native captures, obtain fresh transient grants, rebuild target graphs and refresh local/full source bindings before resuming. Clear or revalidate ring retention/prefetch coverage against the actual mirror epoch; identical addresses are not evidence that old residency remains valid after vision overwrites it.

TDD starts with fake consumers and real arena views: protected suffix canaries, fragmented/aligned free regions, exact-fit/one-byte-short, live external leases, wrong parent/type, pending-copy/compute readers, retained assistant pins, drain exceptions, failed restore, destruction order and repeated vision/text transitions. The expected red must isolate the missing lending/preservation behavior, not a numerical or fixture error. Assert no second device parent and no text graph submission while suspended.

Exit: common ownership tests pass normally and under ASan/UBSan; applicable CUDA views/completion tests preserve all bytes and report clean errors. No public Gemma vision admission yet.

### GV3 Projector execution and host handoff

**GV3a Bind existing Gemma encoder execution.** Use the measured weight and compute grants with the existing projector metadata/storage lifecycle. Validate the entire batch plan before upload; reject stale dimensions or mismatched precision instead of allocating private scratch. Include partial weight-upload failure, graph-capture lifetime, release/reload and encoder completion.

**GV3b Reuse owned host embeddings.** Preserve the existing host-output boundary. Host views must survive projector unload, scheduler/batch destruction and later consumption of other images from the same encoded batch. Publish only complete outputs and keep encoder allocation alive until its final device-to-host operation completes.

Compare preprocessed inputs and all projected embedding bits with GV0 at each image budget and representative shape. Test two-image batching, incompatible shapes split into valid encoder batches, output ordering, cancelled batches, short/invalid grants and exact retry. Measurement must not read or overwrite live text storage while describing hypothetical tensor addresses.

Exit: native-exact image embeddings with bounded arena storage, no lingering projector grant and safe host ownership. This is encoder qualification, not decoder/image-language correctness.

### GV4 Target image-prefill correctness

**GV4a Effective attention policy.** Keep bidirectional visibility in Gemma's image SWA layers and causal visibility in its full-attention layers. Audit the compact-mask producer as well as the native dense-mask path: passing a global `causal_attn=false` directly into the full-cache compact-mask generator would bypass the native per-cache policy. Test mask bounds against native masks before running real graphs. Restore causal text behavior on success, failure and cancellation.

**GV4b Preserve atomic image visibility.** The current non-causal helper requires an image block to fit its admitted microbatch. Start with an explicitly sized batch/ubatch that preserves that behavior, including boundary/marker accounting. Re-quote SWA headroom, full-attention writer/attention/graph scratch and preparation requirements for the actual width. Test 256-boundary cases and the selected 280/560/1120-token budgets; do not silently clamp, split an image into causal chunks or claim a larger budget from a smaller effective image. A future chunked bidirectional implementation would need a separate numerical design.

**GV4c Validate physical and semantic state.** Keep physical append indices separate from model positions, update only actual target cache ownership, and preserve image/token identity in checkpoints. Use native target logits, hidden rows, full-cache bytes/metadata and visible local-cache bytes after image prefill and subsequent text. Cover image-first, text-first, text/image/text, adjacent images, SWA wrap, narrow cached suffixes and context boundaries.

The native-mask baseline and fitting model oracle must pass before opening any server guard. Preserve the D512 feature pipeline's native arithmetic and the native local SWA path; a visibility-policy fix does not authorize a new attention reduction.

Exit: target-only image-prefill and later generation match their exact native scenario, with no mutation from rejected inputs and safe graph/phase restoration.

### GV5 Target-only server lifecycle

Extend existing server and vision harnesses with explicit Gemma capabilities. Do not encode Gemma as a fake M-RoPE model. Startup admission checks must cover artifact pairing, ordinary decoder positions, the qualified image mask/batch policy, protected parent lending, GPU offload and actual encoder capabilities. Keep unsupported media/speculation rejected by exact cause.

Test completion and chat image requests, text-only requests with a configured deferred projector, changed-image cache invalidation, identical-image prefix reuse, follow-up text, a later image after long decode, compatible/separate media batches, malformed payloads, repeated low/high/low image budgets and recovery after a rejected request. Cache keys must cover image bytes and effective preprocessing/projector settings, not just a placeholder token or nominal budget.

Bound compressed input bytes, decoded pixels, patch grids, image count, embedding element counts and combined host/device byte arithmetic before allocation. Include overflow, truncated/corrupt image data and decompression-expansion cases. Preserve existing server URL/file-access policies; image support is not permission to fetch arbitrary resources or expose local files.

Compare each arena scenario with its matching native scenario. Do not assume native fresh and cached/replayed requests are bit-identical across different batching layouts. Capture actual cache-hit/recomputed token counts rather than inferring a hit from the final frontier. Verify complete save/restore files and restore into a new server process, with image-conditioned continuation and old-prefix/window availability guards intact.

Add an early request admission pass before consuming reusable prompt state where practicable. Failures after partial prefill need an explicit rollback/reset contract; tests must state which committed state is restored and cannot treat silent cache loss as preservation.

Exit: a documented target-only image profile with correct serial API behavior and lifecycle recovery. Only this scope may be admitted publicly at this point; MTP-image admission stays closed.

### GV6 Shared-cache MTP

**GV6a Register the existing assistant role.** Replace embedded-MTP-only vision assumptions with capability checks for Gemma's registered read-only child. Keep shared physical parent ownership, independent target verification width, effective recurrent snapshot count zero and matching inherited K/V types. Preserve the existing rejection of explicit adaptive draft-cache overrides and foreign draft models.

**GV6b Image-conditioned drafting.** No assistant vision tower or auxiliary image KV history is needed. Target prefill populates the image-conditioned caches; the assistant reads those caches and uses the correct target hidden/embedding carry and fixed drafting position. Refresh semantic source bindings from the post-image target frontier and invalidate stale pre-image carry. The architecture is described in the [Gemma assistant documentation](https://huggingface.co/docs/transformers/model_doc/gemma4_assistant).

**GV6c Verification and recovery.** Exercise lengths 1/2/3 and target query widths 1-4, acceptance from zero through each maximum, confidence cutoff, correction rows, cancellation/retry, new-image requests between rounds, cache restore/restart and source-generation changes. At fixed scenarios compare complete assistant logits/hidden vectors, target verification outputs, accepted output IDs, draft/accept counts, source bytes and local-window state with matching native controls.

Assert that the assistant never executes during vision, appends image KV or retains a read pin that allows the encoder to overwrite its source. A failed transition must keep admission closed until ownership is resolved. Model/assistant joint-memory feasibility is required even if a cache-only or fresh-process numerical fixture passed.

Exit: real GPU target, projector and matching assistant complete the admitted image scenarios under the one-parent design. Label the actually tested artifact, image budgets, microbatch and arena; no Gemma DFlash or audio claim.

### GV7 Long-context and resource hardening

Run long text -> image -> decode -> another image at all admitted draft modes. Require measured streaming/ring activity rather than assuming a long context implies it. Populate enough history to force a full-attention source beyond device capacity, cross local-window/rollback boundaries, and verify unchanged historical state after each encoding phase. Include many serial requests and low/high/low batch shapes to detect retained allocations and stale captures.

Inject failure at every ownership boundary: preflight, last-reader drain, capture retirement, weight upload, encoder compute, output copy, lease return, text resume, image prefill and assistant rebind. Include repeated cleanup exceptions, teardown while host embedding views remain, short quotas, stale bindings, wrong generations and context exhaustion. Distinguish recoverable pre-mutation rejection from a context requiring reconstruction. No orphan GPU child may survive a harness timeout.

Use fresh normal and ASan/UBSan component suites, strict CUDA memcheck with API-all reporting and a nonzero error exit, and targeted racecheck/synccheck if new queue ordering requires them. Eager sanitizer execution is a separate explicitly recorded profile; normal graph-enabled tests remain mandatory. Instrumentation OOM/timeouts are not zero-error passes and do not justify silently shrinking model, image budget, arena or comparisons.

Long profiles without a fitting native control establish lifecycle/budget/self-consistency only. Preserve native capacity failures; do not relabel exported reference fixtures or finite outputs as full long-context native equivalence.

Exit: no unresolved ownership, numerical, state or tool-qualification failure in the declared profile matrix.

### GV8 End-to-end qualification and performance

Run fresh common regressions for affected paths: portable HOST/ASan tests, context/owner/workspace transitions, all-64 D256/D512 span/tile/progressive/prepared controls as applicable, Qwen text/embedded-MTP/vision, Qwen DFlash, Gemma text MTP1-3 and candidate/acceptance/recovery. Preserve text-only fast paths and the qualified conversion, host half-ring and D512 feature pipeline. No measured change to those mechanisms is implied by adding vision admission.

Use the existing image-aware `benchmarks/find-max-arena.py` for actual image workloads, independently for each mode/context/image budget and batch grouping. A `--mmproj` text-only fit proves only configured-projector text execution, not image encoding. Retain the successful full request, measured phase minima, effective image dimensions/tokens and a tested rejected +1 MiB neighbor when claiming a maximum; no arbitrary reserve or fixed text-arena substitution.

Measure image preprocessing, projector reload, encoder execution, output handoff, text resume, image-embedding prefill, subsequent text prefill, time to first generated token and useful decode throughput separately. Record actual phase GPU/host peaks, idle baseline, protected bytes, reclaimable grants and first KV refill. Device sampling complements, but does not replace, checked allocator accounting; external driver/native resources remain outside the parent.

Compare native/eager and phase-arena execution only at fitting, matched configurations, and label every changed condition. Image encoder throughput alone is not a full-request speedup. The functionality target is correct image inference within a budget that need not hold all phases simultaneously; projector reload can add latency. Declare regression thresholds before timing, require no material text-only regression and report the image latency/memory tradeoff rather than promising that the phase-arena path is always faster. Use the user's single retained full-model measurement preference; do not add a repeated trial matrix without a decision. Nsight is optional for attribution after ordinary results are valid, not a source of uninstrumented speedup numbers.

Exit: complete scenario matrix, current source/binary/input provenance, audited strict summaries, measured fit/latency and unchanged existing consumer behavior. A no-fit image budget is a documented scope limit, not a skipped passing case.

### GV9 Documentation and handoff

Publish only verified projector/model combinations, precision, image budgets/counts, batch/ubatch, target/MTP modes, device/backend and context limits. Include an actual successful finder result and commands using inherited draft K/V settings. Explain which bytes stay outside the parent, why SWA remains protected, how temporary encoder attention differs from decoder KV, and why text-fit results cannot be reused as image-fit guarantees.

Add per-phase memory and latency tables/plots, native comparison scope, artifact hashes and explicit unsupported/no-fit cases. Link this plan from the Gemma text ledger without renumbering or reopening completed G1-G6 stages. A completed stage updates its evidence and status; historical failure records stay unchanged. If code changes are later committed, use separately authorized, narrowly scoped completed-stage commits and do not push.

Overall completion requires all in-scope stage gates, readable/reproducible documentation, no active test/profiler descendants and the original production service restored healthy on port 1234. A metadata change, generated plot or successful server startup alone does not complete vision qualification.

## Required coverage matrix

| Axis | Required coverage |
| --- | --- |
| Projector precision | Pinned high-precision reference, Q8_0 and smaller Q4 candidates, real tensor inventory/fallbacks, per-artifact native exactness and separate quality assessment |
| Target and speculation | Target-only first, then matching shared-cache MTP1/2/3; unsupported draft modes reject |
| Image budgets and dimensions | Requested/effective/actual counts at 70/140/280/560/1120, square/portrait/landscape/extreme aspect ratios, boundary sizes |
| Image arrangement | Image-only prompt, text/image/text, adjacent and separated images, compatible batched and incompatible split encoding |
| Session history | Fresh, cached identical image, changed image/preprocessing, follow-up text/image, long decode, low/high/low, context end |
| KV and masks | Local SWA vs full-cache visibility, dense/compact parity, wrapping/headroom, all existing cache-type capabilities preserved |
| Ownership | Same parent, disjoint protected region, no active text/draft execution during encoding, retained host outputs, safe new bindings |
| Failures | Exact/short quota, malformed metadata, stale identity/epoch, queued compute/copy failure, repeated cleanup failure, cancellation and teardown |
| State and APIs | Completion/chat inputs, real cache hit/recompute counts, save/restore/restart, incomplete state-file rejection |
| Capacity | Independent target-only/MTP full-image fit, actual upper-neighbor rejection, image-aware phase minima, no-fit cases retained |
| Numerical evidence | Frozen native encoder embeddings, native masks, target/assistant logits/hidden rows and cache state; matching scenario/mode/layout |
| Platform | CUDA SM120 runtime reference; other backends/hardware only according to separately obtained evidence |

Budget axes are required test targets, not promises that all will fit. If feasibility excludes one, retain its exact rejection and obtain an explicit scope decision before declaring the workstream complete. Native-vs-arena equality checks compare matching artifacts and execution scenarios, not arbitrary cross-batch or cross-precision outputs.

## Existing implementation and test map

| Area | Existing sources to extend narrowly |
| --- | --- |
| Phase owner and protected regions | [context memory](src/llama-context-memory.cpp), [owner interface](src/llama-context-memory.h), common memory plans/transitions/bounded KV |
| Target suspend/resume and resources | [context](src/llama-context.cpp), [Gemma model adapter](src/models/gemma4.cpp), [native cache masks](src/llama-kv-cache.cpp), iSWA cache/source views |
| Encoder, measurement and storage | [Gemma encoder](tools/mtmd/models/gemma4v.cpp), [CLIP loader](tools/mtmd/clip.cpp), [workspace](tools/mtmd/mtmd-workspace.h), [projector storage](tools/mtmd/mtmd-projector-storage.h) |
| Image handoff and ordered execution | [MTMD session](tools/mtmd/mtmd-session.cpp), [embedding ownership](tools/mtmd/mtmd-embeddings.h), [decode helper](tools/mtmd/mtmd-helper.cpp) |
| Server admission and dispatch | [configuration/request gates](tools/server/server-common.cpp), [server context](tools/server/server-context.cpp), [speculative driver](common/speculative.cpp) |
| Component tests | `test-context-memory`, `test-memory-phase-owner`, `test-memory-transition`, `test-mtmd-workspace`, `test-mtmd-projector-storage`, `test-mtmd-session`, `test-mtmd-embeddings`, `test-server-vision` |
| Real-model controls | [vision harness](tools/server/tests/test_adaptive_vision.py), [Gemma harness](tools/server/tests/test_adaptive_gemma.py), [phase measurements](tools/server/tests/measure_adaptive_vision.py), existing context/model tests |
| Physical fit | [finder](benchmarks/find-max-arena.py), existing finder/vision/Gemma host-contract tests |

The Qwen vision harness currently assumes Qwen-specific launch/budget behavior and contains a fresh-vs-uncached comparison. Extend it with explicit model capabilities and per-scenario native controls; do not reuse those assumptions unchanged for Gemma. Prefer existing component suites over a parallel test framework and avoid new public APIs unless the common contract cannot express the needed behavior.

## Evidence and operating rules

Start every implementation substage with a failing test for the missing behavior, then run its focused normal and sanitizer gates before broader consumers. Preserve the red, corrected fixtures, original failure logs and accepted partial results; a fixture error, OOM or tool timeout does not count as the intended red.

Every GPU gate records source commit and diff, core/test binary hashes, resolved libraries, model/projector/assistant/image/corpus/template hashes, commands, environment changes, effective geometry and results. Archive a qualified runtime before changing it; preserve build origin separately from later documentation/commit metadata. Reuse evidence only with an explicit matching-source/binary/profile manifest and an explained test-only lineage, not because a previous process exited successfully.

Keep source and binaries frozen while a GPU owner is active. Use owned per-command cgroups/process groups, bounded timeouts and cleanup self-tests; drain all owned descendants before restoring production. Retire completed watchers and test scopes before another GPU job, but do not kill scopes hosting the restored production service. No duplicate jobs, agents, pushes, production configuration/model changes, clean compile-time benchmarking or code-generation packaging refactors.

Existing source contracts are the implementation baseline; external documentation explains model semantics but does not prove this fork's integration or physical fit. References: [Gemma text qualification](GEMMA4_ADAPTIVE_KV_ROADMAP.md), [common consumer and Qwen vision plan](DEVICE_MEMORY_CONSUMERS_ROADMAP.md#milestone-7-separate-vision-encoder-integration), [kernel optimization scope](QWEN_GEMMA_KERNEL_OPTIMIZATION_ROADMAP.md), [MTMD development guide](tools/mtmd/README-dev.md).
