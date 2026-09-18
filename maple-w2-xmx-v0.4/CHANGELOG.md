# v0.4 Vulkan architecture port v1 (lab extension)

- Preserve v0.4 static expert grouping and existing T1/DPAS/quant/SwiGLU math.
- Add explicit legacy/direct/grouped/calibrated-auto dispatch selection.
- Add opt-in expert token descriptors and T2/T4 register B-load reuse, with independent gate/down sizes. DPAS remains RepeatCount=1.
- Expose the original A8 quantizer as an asynchronous producer; add optional grouping/quant fork-join and fresh prequantized input.
- Add producer masks and overlap-aware interval-union profiling.
- Add RNE contract auditing, stage-isolated same-q/scales comparison, raw dumps, freeze-input diagnostics and same-binary AB/BA workload suite.
- Add hash-checked, backup-preserving module overlay and unified diff.
- Local validation is CPU/reference/scalar-emulation/host-mock only. Intel SYCL compilation and GPU results are not claimed.

---

# v0.4 — one optimization: stable per-expert job grouping

- GPU count/prefix/stable-scatter builds a fresh job permutation every invocation.
- Gate/up and down share that permutation; activation/output slots and route reduction order remain unchanged.
- Retain G32/H32, s2tile8, gluquant, async, DPAS repeat=1 and existing arithmetic.
- Two-arm same-build comparator; alternate AB/BA; include grouping cost and per-stage profiling.
- Full GPU A/B on all outputs/quant planes; sampled CPU gold (up to three tokens) and exact full mapping validation.
- Q validation limit broadened to 2048 equally for both arms, because local Q-sweep source was not attached.
- Q1..2048 suite, no group/split/queue policy sweep. No auto cutover.
- Source/executable fingerprints reject stale -NoBuild binaries.
- No multi-token GEMM, no Vulkan/llama integration, no A750 performance claims before local execution.

---

# Changes

## v0.3 — MoE stage chaining after user's v0.2 A750 run

- Preserve standard TQ2 block scales and the user-tested A16/A8 DPAS packing/math.
- Add A8 prequantized consumer entry (same-call producer dependencies required).
- Add clipped SwiGLU and fused SwiGLU→hidden-A8 producer, groups 32/128/256.
- Add one-layer gate/up→epilogue→down→weighted-sum API with persistent scratch.
- Explicit event dependencies; no explicit intermediate host wait in the default path.
- Add a diagnostic variant with three host waits; not a Vulkan interop simulation.
- Add shared-queue A16 / A8-separate / A8-gluquant chain comparison, independently
  configurable input/hidden groups, split-K, in-order/out-of-order, Q1–8.
- Add stage-local arithmetic references and separate full-chain F32 error reporting.
- Add submit-return/tail-wait/GPU-gap timing; no cold-JIT claim for post-validation timings.
- Cache validated A750/A770 device identities; diagnostic repeated-query switch retained.
- Use user's working Windows exception flags and compiler-runtime builtins library discovery.
- One build/test wrapper, independent v0.3 directory, legacy matvec regression retained.
- User-supplied v0.2 results preserved in evidence; new v0.3 GPU compilation/execution unverified.

Previous descriptions: README.v0.1.ko.md and README.v0.2.ko.md. Previous validation files
are historical and do not certify newly added kernels.
