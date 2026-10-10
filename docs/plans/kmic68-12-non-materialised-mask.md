# P12 - Prefix-length causal-mask representation

**Kind:** core/backend feature
**Depends on:** P08

## Context

Cached causal attention normally builds an additive mask whose type is F16 when
flash attention is enabled and F32 otherwise. Its general shape includes KV,
query, and stream dimensions. For a single sequence whose cache cells form one
leading position-ordered prefix, each query needs only the count of visible KV
cells; an `int32_t` prefix length can reproduce the same zero/negative-infinity
mask values inside a supported FA kernel.

P12 is an explicit opt-in. `--causal-prefix-mask` enables it,
`--no-causal-prefix-mask` and the default disable it, and
`LLAMA_ARG_CAUSAL_PREFIX_MASK` is the boolean environment alias. Automatic
eligibility is intentionally narrow: one sequence, one stream, contiguous cache
cells with no holes whose selected K/V tensor row order is position-ordered and
forms the exact leading visible prefix for every query, strictly increasing
contiguous query positions, ordinary
causal full attention, no ALiBi, no M-RoPE mask ordering, and a CPU or SYCL
VEC/TILE FA route that implements prefix lengths. Every other case retains the
materialized mask.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`p100-docs/FINDINGS.md`, "Do not materialise the causal mask".

## EARS requirements

- **R12.1 (State-driven):** WHILE causal prefix masks are enabled and a cached decode satisfies the compact eligibility predicate, the cached-attention graph input shall represent visibility with one prefix length per query.
- **R12.2 (Event-driven):** WHEN a prefix-length graph input reaches a supported FA kernel, the kernel shall derive each query/key mask value from that query's visible-prefix length.
- **R12.3 (Ubiquitous):** The prefix-length path shall produce bit-identical attention tensors and token IDs to the materialized F16 FA mask path for eligible inputs.
- **R12.4 (State-driven):** WHILE a batch is multi-sequence, multi-stream, non-contiguous, query-position-unordered, cache-position-disordered or wrapped, cross-attention, or sliding-window attention, the mask builder shall allocate and populate the existing full mask tensor.
- **R12.5 (Event-driven):** WHEN graph reservation uses the prefix-length representation, `llama_context::graph_reserve` shall size the compute buffer from the compact input shape.
- **R12.6 (State-driven):** WHILE causal prefix masks are disabled, flash attention is disabled, or the selected backend/kernel lacks prefix support, the graph builder shall use the existing materialized mask.
- **R12.7 (Event-driven):** WHEN a context transitions between prefix and materialized eligibility, the context shall complete representation-specific re-reservation before binding graph inputs.
- **R12.8 (Event-driven):** WHEN the cached-attention input prepares an eligible query, `llama_kv_cache` shall set its prefix length to the exact number of leading cache cells visible to that query.
- **R12.9 (State-driven):** WHILE ALiBi, M-RoPE ordering, cache holes, non-monotonic cached-cell positions in exact K/V order, cache wraparound, or non-leading sequence membership changes additive mask values, the cached-attention graph input shall use the existing materialized mask.
- **R12.10 (Unwanted behaviour):** IF the causal-prefix-mask option receives an invalid boolean value, THEN the common argument parser shall fail with the option name and rejected value.
- **R12.11 (Unwanted behaviour):** IF the actual I32 prefix tensor is not strictly smaller than the applicable F16 FA mask tensor, THEN the cached-attention graph input shall use the materialized mask.
- **R12.12 (Event-driven):** WHEN compact eligibility is evaluated, the route preflight shall positively identify CPU FA or SYCL VEC/TILE on the intended backend before selecting prefix representation.
- **R12.13 (Unwanted behaviour):** IF representation-specific scheduler reservation fails, THEN the context shall abort the request before cache mutation or graph input binding.
- **R12.14 (Ubiquitous):** The FA operator shall keep `src[3]` as the tagged mask-or-prefix payload and `src[4]` as sinks for both representations.

## Approach

1. Add `causal_prefix_mask=false` to common/context parameters with positive/
   negative CLI aliases and `LLAMA_ARG_CAUSAL_PREFIX_MASK`; propagate through
   target and P08-capped draft contexts without changing the default.
2. Add an eligibility query to `llama_kv_cache_context`. Prove one sequence/
   stream, contiguous hole-free leading cache membership, and that selected K/V
   tensor row order is position-ordered with no wrap or disorder and forms the
   exact leading visible prefix for every query. Also require position-ordered
   queries, causal full attention, no ALiBi/M-RoPE/SWA/cross/recurrent/special
   mask, and a supported route. Build candidate tensor descriptors and require
   `ggml_nbytes(prefix_i32) < ggml_nbytes(materialized_f16_mask_shape)`;
   equality or larger size is fallback.
3. Add `build_attn_inp_kq_prefix`, I32
   `llm_graph_input_attn_kv::self_kq_prefix`, and
   `llama_kv_cache::set_input_kq_prefix`. Keep existing mask builders and population unchanged for fallback and all excluded call sites.
4. Extend the existing `GGML_OP_FLASH_ATTN_EXT` contract without moving source
   slots: `src[0]=Q`, `src[1]=K`, `src[2]=V`, `src[3]=mask-or-prefix`,
   and `src[4]=sinks`. Add an op-param enum `MATERIALIZED`/
   `PREFIX_LENGTHS`. Materialized FA requires src3 F16 with the existing mask
   shape; prefix requires src3 I32 with `ne[0]=n_queries` and other dimensions
   1. Sinks semantics remain unchanged. The representation tag, not source-slot
   position alone, determines interpretation.
5. Update `ggml.c` assignment/validation/op params, graph duplication,
   cloning, hashing/serialization if applicable,
   `ggml-backend-meta.cpp::handle_flash_attn_ext` split metadata, generic
   backend source heuristics, CPU/SYCL support+dispatch, and OpenVINO/other
   backend mask classification. Invalid tag/type/shape combinations are rejected;
   an I32 src3 can never be classified as an additive mask by fallback code.
6. Before building a compact graph, create a shape-correct prototype and run a
   positive route preflight on the intended backend. CPU must select its prefix
   implementation. SYCL must call the same best-kernel selector as dispatch and
   return VEC or TILE; XMX, sparse, MKL, oneDNN, OpenVINO, or another backend
   returns ineligible. Do not rely on scheduler migration or a later
   `supports_op` failure; construct the full mask instead.
7. In `llm_graph_context::build_attn`, select the prefix operator only after
   preflight. CPU FA and SYCL VEC/TILE load one visible count per query and use
   exact zero/negative-infinity values. No temporary mask is materialized.
8. Make graph parameters and reuse include representation enum, source shape,
   backend, and selected route. A mismatch rejects reuse before input binding.
9. Implement one-active-reservation protocol for target and draft contexts. Keep
   byte estimates per `(representation,effective_n_ubatch,route)`, but only one
   scheduler buffer live. Before the first live graph or any representation
   transition: synchronize; discard/reset the old scheduler buffers; build the
   actual new-representation reservation graph; call reserve; verify returned
   bytes; then set active representation and bind inputs. Failure aborts before
   cache mutation. Returning to prefix re-reserves compactly rather than keeping
   a full-size high-water buffer.
10. Use the P08 effective draft width in reservation keys. Test initial prefix,
   prefix-to-full-to-prefix, inherited and cap-64 draft widths, allocation
   failure, and route changes. Record raw reserved/actual bytes for each step.
11. Add operator, eligibility, source-layout, backend-preflight, transition,
   deep token, and fallback tests. Keep the materialized F16 FA full-mask oracle
   callable in one build; validate F32 only through the existing non-FA fallback.

## Critical files & anchors

- `common/common.h` and `common/arg.cpp` - opt-in parameter, CLI, and environment parsing.
- `src/llama-kv-cache.h:98-141` - `slot_info` and `is_contiguous`.
- `src/llama-kv-cache.cpp:2168-2317` - cached causal mask construction.
- `src/llama-kv-cache.cpp:2358-2395` - cached mask input population.
- `src/llama-graph.cpp:30-65,477-508,2751-2775,2985-3020` - representation-aware inputs, reuse, operator construction, and ordinary cached MHA.
- `src/llama-graph.cpp:1065-1099,3117,3501-3628,3823-3832` - mandatory materialized-mask call sites.
- `src/llama-context.cpp:1840-1896,3123-3182` - pre-bind reuse decision and representation-specific reserve protocol.
- `ggml/src/ggml.c:5592-5665` - fixed FA source slots and representation op params.
- `ggml/src/ggml-backend-meta.cpp:1016` and `ggml/src/ggml-backend.cpp:1186-1195` - split metadata and backend placement.
- `ggml/src/ggml-cpu` - CPU prefix FA implementation/support.
- `ggml/src/ggml-sycl/fattn.cpp:871-1017` and `ggml/src/ggml-sycl/ggml-sycl.cpp:7017-8123` - route selection, support, and dispatch preflight.
- `ggml/src/ggml-sycl/fattn-vec.hpp:155,407-409` and `ggml/src/ggml-sycl/fattn-tile.hpp` - supported consumers.
- `tests/test-backend-ops.cpp`, `tests/test-sycl-turbo-correctness.cpp`, and `tests/test-qwen4exp-mtp.cpp` - operator, route, transition, and token tests.
- `docs/research/qwen4exp-mtp-correctness-2026-10-04.md:412-418` - PR #90 note updated only after proof.
- `scripts/perf/verify-causal-prefix-mask.py` - planned deep/fallback/reservation comparison.

## Verification

Run the CPU and SYCL FA operator suites and existing Turbo FA gate:

```bash
timeout 180 ./build-sycl/bin/test-backend-ops -b CPU -o FLASH_ATTN_EXT
timeout 180 ./build-sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
LLAMA_TEST_TURBO_FA=1 timeout 180 ./build-sycl/bin/test-sycl-turbo-correctness
```

Then stop the service, verify A770 sole tenancy, name build/model/driver, and run
materialized/prefix modes plus representation transitions at P08 cap 64:

```bash
timeout 2400 python3 scripts/perf/verify-causal-prefix-mask.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --depths 4096,16384 --draft-ubatch 64 --seed 123 --modes materialized,prefix --transitions prefix,full,prefix --fallbacks multi-sequence,non-contiguous,out-of-order-cache,wrapped-cache,cross-attention,sliding-window,alibi,mrope,non-fa,xmx,mkl,onednn,openvino,unsupported-kernel --report /tmp/p12-prefix-mask.json
```

Expected evidence:

- CPU and SYCL VEC/TILE outputs are byte-identical between the I32 prefix and
  materialized F16 FA representations;
- every compact op has tag PREFIX_LENGTHS, shape-correct I32 prefix in src3,
  unchanged sinks in src4, and remains on the intended backend;
- every unsupported FA-route preflight builds the full F16 mask before graph
  construction, while the non-FA case retains its existing F32 non-FA mask; no
  compact op migrates to CPU or executes unmasked;
- strict byte comparison covers prefix smaller/equal/larger boundaries;
- seeded target tokens are identical at depths 4096 and 16384;
- prefix graphs allocate no additive mask and reserve fewer raw bytes;
- prefix-to-full-to-prefix tests re-reserve before binding, release the prior
  active buffer, use correct bytes at inherited/cap-64 widths, and abort safely
  on injected reserve failure before cache mutation;
- every named FA fallback passes its existing F16 additive mask in src3 with tag
  MATERIALIZED and no compact tensor; the non-FA fallback remains outside the FA
  operator and retains its existing F32 mask;
- graph transitions reject reuse and rebuild with the correct representation;
- only after all checks pass is the PR #90 residual note updated;
- the post-run two-driver fault gate passes and the service is restarted.

Run the exact README regression floor before and after the feature.

## Assumptions & contingencies

- Prefix length counts cells in exact selected K/V tensor row order, not absolute
  position; eligibility proves that this row order is position-ordered and forms
  the exact leading visible prefix for every query, rejecting wrapped or
  disordered cache layouts.
- The compact path is disabled by default; a default change needs the complete
  fallback/backend fleet.
- Unsupported routes materialize the existing mask before graph construction;
  they never reconstruct it from a compact tensor or rely on scheduler migration.
- Representation transitions may reallocate and synchronize; safety and exact
  reservation size take precedence over transition latency.
- Bit identity requires exact zero and negative infinity; non-binary masks stay
  excluded.
- P12 consumes P08's effective draft width in every reservation key/test.
