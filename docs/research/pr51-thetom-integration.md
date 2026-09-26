# PR 51: TheTom integration and validation

This integrates the compatible delta from TheTom's default branch pinned at
`80007e71526b2566bda62a8fc98f68c4d231139c` into the fork, preserving the fork's
later upstream work, supported backends, TurboQuant ABI, and SYCL implementation.
It does not import CUDA, HIP, Metal, OpenCL, RPC, or other excluded backends.

The original integration was validated at `0bc989ac4`. The review corrections
and their additional validation are recorded below; quality probes were not
rerun for these corrections.

## Source and linear history

- Destination: `sync/thetom-on-master-fresh`, PR 51 into
  `Raudbjorn/ggml-llama.cpp:master`.
- Fork master at audit: `54d02625c723d2482abe20e9856113e8a6f1afc5`.
- Pre-integration PR tree: `fdfa74483c61ffafab2650b7f6e4ad232f7fa84c`.
- TheTom source: `80007e71526b2566bda62a8fc98f68c4d231139c`.
- Content reconciliation base: `15586e2d7165570fb3aa7c26e0d442e289ef69de`.

Fork upstream syncs were squashed. The June ancestry merge base is therefore
not the correct content base: merging from it replays upstream content that the
fork already contains. The integration uses the common upstream content base
and is submitted as a linear content integration onto fork master. Rebase or
squash merging is supported. TheTom and the superseded sync branches are source
provenance, not extra merge parents; their redundant upstream commits and
damaged intermediate trees are not imported into master history.

The three historical remote branches are accounted for without rewriting them:

| Branch | Pinned tip | Disposition |
| --- | --- | --- |
| `sync/thetom-turboquant-3` | `54d02625c723d2482abe20e9856113e8a6f1afc5` | Already an ancestor of the destination |
| `sync/thetom-turboquant-2` | `21680f772306205d1b580689fe73c2b093439b7a` | History excluded; conflicted core and mass deletion superseded; its 12 added files match the pinned TheTom payload |
| `merge/sycl-turboquant` | `987015aa257a32e46a659958bc9ed90d66dd5f06` | History excluded; its SYCL TurboQuant disablement is superseded by the working fork implementation |

The [file disposition ledger](pr51-thetom-integration-files.tsv) covers all 311
paths changed between the content base and TheTom: 38 source contents present,
95 reconciled, 43 fork contents retained, and 135 excluded/superseded paths
absent. These are content comparisons, not a claim that every upstream feature
is supported on every backend. Governance files retain fork policy; `AGENTS.md`
only updates the factual type table for the new CR types.

## Reconciliation decisions

- Preserve serialized TurboQuant/TQ IDs 43-47 and the fork's 68-byte Turbo4
  layout with `rnorm`. Append Q8_CR/Q5_CR/Q6_CR at 48/49/50; COUNT becomes 51.
- Move the Turbo3 Lloyd-Max table together in CPU, SYCL, and all five Vulkan
  table copies. A Python check detects table drift. Remove the rotation
  initializer's 64 KiB temporary stack array. Preserve WHT signs and InnerQ.
- Retain the fork's SYCL kernels and pruning. The incoming MMVQ launch formulas
  are equivalent at the fork's fixed subgroup size 16. Ingredient-emitting GDN
  operations fall back to CPU on SYCL/OpenVINO; Vulkan implements that mode.
- Integrate CR CPU codecs/dots, Vulkan TQ paths and expert caching, GDN replay,
  adaptive/chained MTP, Qwen4exp MTP, Laguna DFlash, benchmark/fitting controls,
  and the compatible server and conversion changes.
- Preserve the newer fork's Qwen4exp PLE/QSA/cache-cell handling, DSpark/DFlash2,
  Eagle3, PocketTTS/audio conversion, and InnerQ callbacks. Do not restore
  duplicate legacy graph builders or the alternate SYCL port.
- Wire and build `turboanchorkv` and the GDN ingredient test against the real
  cache accessors and current operation API. Fix documentation rather than
  presenting TheTom's Metal/CUDA measurements as this fork's results.
- Checkpoint sidecars include target, draft, and speculative state. Version 2
  binds them to the main saved state and bounds reads before allocation; stale,
  truncated, oversized, and allocation-failing sidecars fall back to the prompt
  tip. The fingerprint detects mismatches; it is not authentication.
- Device state restore copies across differing source/destination contiguous
  runs and handles empty tensors and quantized block lengths. Keep real logit
  buffer bounds checks around chained extraction. Flush graph-checker logs
  before exit so its shell regression check receives the complete results.
- Keep extra draft-model memory in every MoE fitting candidate. Vulkan caching
  uses synchronous fills on one device and has no fused SwiGLU cache dispatch;
  its documented limits differ from TheTom's CUDA provider.

## Review corrections and continuing fork divergences

- Restore the upstream `ExaoneMoeForCausalLM` conversion registry alias alongside
  `ExaoneMoEForCausalLM`; a real-import regression test checks both names.
- Remove `scripts/ppl_test.sh`: its foreign paths, CUDA selection, and OSCAR2
  settings do not describe a runnable fork benchmark.
- Restore llama-bench CSV/JSON examples, labeled as historical upstream output
  so their older schema and hardware are not presented as current measurements.
  Multi-value arguments remain documented; newly added scalar options are listed.
- New CR `general.file_type` values are 512/513/514 in C and Python, below the
  `GUESSED=1024` flag. These are fork-selected extension values, not an official
  upstream reservation. Tensor IDs remain 48/49/50; existing serialized IDs are
  unchanged. Python tests serialize/read all three metadata values and check
  agreement with the public C header. Audit both namespaces on future syncs.
- Vulkan FA now uses `fa_types.glsl` instead of incorrect hardcoded TurboQuant
  IDs 42/43/44. Vulkan Turbo4 centroids/midpoints now agree with the fork CPU
  table. These fix existing fork kernels as well as integrating source changes.
  The Turbo3 sign ballot selects the correct 32-bit word for wave64 lanes;
  SET_ROWS source indexing uses source extents and guards broadcast overruns.
  The imported FA fixture now honors the fork's 68-byte Turbo4 block and leaves
  reserved `rnorm` zero. A770 tests do not validate wave64 hardware execution.
- `ggml_gated_delta_net` adds `emit_mode`; `llama_memory_recurrent` adds the
  GDN replay request to its constructor. Upstream callers require reconciliation
  on future syncs. SYCL/OpenVINO ingredient emission falls back to CPU.
- Tensor-split and fit-target parsers use `>` rather than `>=` against the
  maximum device count, allowing exactly the array capacity.
- `--cache-ram -1` now selects half of free host RAM, with an 8192 MiB fallback
  if free memory cannot be determined, rather than upstream's unlimited cache.
  This user-visible policy and the API changes above are accepted divergences.

## Review validation

On the review-corrected source, CPU and SYCL full builds pass. All four required
SYCL tests pass; the safe oracle again reports 52 PASS, zero GATE-FAIL/XPASS.
The complete A770 Vulkan turbo FA selection passes 2594/2594 cases:

```bash
cmake --build /home/svnbjrn/build-pr51-vulkan --target test-backend-ops -j8
timeout 300 /home/svnbjrn/build-pr51-vulkan/bin/test-backend-ops \
  -b Vulkan1 -o FLASH_ATTN_EXT -p turbo
PYTHONPATH=. python3 tests/test-conversion-registry.py
PYTHONPATH=gguf-py python3 -m pytest gguf-py/tests -q
```

The Python checks pass (one conversion regression, 11 GGUF tests and seven
subtests). The Vulkan binary includes the fixture correction; the later CR
file-type metadata change does not affect these tensor-only FA cases. Both
visible GPUs report subgroup size 32; no wave64 runtime claim is made.
The CPU CTest selection again passes 67 tests with one expected no-provider
skip (68 total), including model architecture and state save/load tests.
Pruning and required-target checks pass. Review logs are in
`/home/svnbjrn/.cache/pr51-review-fixes/`.

## Original integration validation on this host

Builds are outside mergerfs under `/home/svnbjrn/build-pr51-*`.
CPU uses OpenBLAS; SYCL uses oneAPI icx/icpx 2026.0 with F16 enabled and empty
compiler launchers. Runtime selects `ONEAPI_DEVICE_SELECTOR=level_zero:0`,
persistent caching, and disabled SYCL graphs on the Arc A770. Vulkan operation
checks pass on both the AMD Raphael iGPU (RADV, Vulkan0) and Intel Arc A770
(Mesa, Vulkan1). The MoE cache test passes with default visibility and with
`GGML_VK_VISIBLE_DEVICES=1` selecting the A770.

| Check | Result |
| --- | --- |
| CPU + OpenBLAS full build | Pass |
| SYCL full build | Pass |
| Vulkan backend build | Pass |
| OpenVINO backend build | Pass |
| CPU CTest selection | 67 pass, one expected no-provider skip; 68 total |
| Standalone model state save/restore | 111 pass; one draft-only DFlash fixture skipped |
| Quantized state save/restore | Q8_0 and Turbo3 llama fixtures pass, including device scatter |
| SYCL correctness oracle | 52 PASS, zero GATE-FAIL/XPASS |
| Required SYCL correctness/fusion/FA-policy/status tests | Four pass |
| Vulkan GATED_DELTA_NET backend ops | 40/40 pass, including ingredient mode |
| Vulkan TQ3/TQ4 MUL_MAT backend ops | 305/305 pass |
| Vulkan MoE cache | Applicable hit, invalidation, fallback, batching and session tests pass; provider-specific cases explicitly skipped |
| GGUF Python | Eight tests and seven subtests pass |
| Conversion/Python syntax | Compileall pass |
| Slot save/restore | Five tests pass, including three malformed/stale sidecar cases |
| UI | Source build and embedding pass with HF fallback disabled |
| Pruning and required-target scripts | Pass for CPU/SYCL profiles |

The CPU selection excludes expensive external-tokenizer, exhaustive backend-op,
quantization-performance, model-load-cancel, and eval-callback runs. The default
SYCL oracle does not enable opt-in sections 5/7/8. OpenVINO is build-validated,
not inference-validated. Its installed package has stale CMake import paths;
validation used a scratch copy of that metadata pointing at the installed
libraries and system TBB. The installation itself was not modified.

The installed shader compiler accepts the OCP float extension but its optimizer
rejects capability 4229. The capability probe now compiles an actual operation
with optimization, checks the exit status, and clears the shader-generator
cache option on failure. Fallback shaders build successfully.

## Model quality: measured limits, not a blanket pass

Model: local `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`. Corpus: Wikitext-2 raw
test, context/batch/ubatch 512, first eight chunks, flash attention on, four
CPU threads, automatic fitting off. No throughput claim is made; these were
accuracy probes without exclusive desktop GPU tenancy.

| Execution | KV | PPL |
| --- | --- | ---: |
| Pre-integration CPU, tree `fdfa74483` | Turbo3/Turbo3 | 8.4752 |
| Integrated CPU | Turbo3/Turbo3 | 8.3698 |
| Pre-integration SYCL | Turbo3/Turbo3 | 8.4805 |
| Pre-integration SYCL | Q8_0/Q8_0 | 7.9861 |
| Integrated SYCL | Q8_0/Q8_0 | 7.9861 |
| Integrated SYCL | Turbo3/Turbo3 | 8.393943 |

CPU Turbo3 PPL is 1.24% lower and SYCL Turbo3 PPL is 1.02% lower in this
probe; the Q8 SYCL control is unchanged. The SYCL KLD probe reports mean
KL divergence 0.055825 +/- 0.002678. Its saved, compressed Q8 baseline gives
PPL 7.981711 and ratio 1.051647; the direct Q8 PPL above gives a 5.11% gap.
The existing quality script's 5% PPL threshold is therefore **not passed** for
this model/configuration. No threshold was relaxed.

An additional context-2048 Q8 SYCL control produced NaNs, both with 512-token
and 2048-token batches. A separate SYCL build of the pre-integration tree
reproduces the NaNs with context/batch/ubatch 2048 and two chunks: this failure
predates the integration and is tracked in [the open defect note](sycl-q8-context-2048-nan.md).
Its corresponding KLD run is invalid and is not
counted as quality evidence. The synthetic oracle's green status does not
establish that every model/context configuration is usable.

No real-model MTP, Laguna DFlash, Qwen4exp, or OpenVINO quality/performance result
is claimed. Imported architecture/conversion support, graph tests, replay tests,
and successful builds are narrower evidence.

Full local logs and commands: `/home/svnbjrn/.cache/pr51-integration/`.
PR 51 remains open for the owner to review and merge using rebase or squash.
