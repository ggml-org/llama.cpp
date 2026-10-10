# Plan set: adopt Kmic-68/llama.cpp findings

Source of findings: `Kmic-68/llama.cpp`, branch `p100-optimizations` at tip
`ae35056eba07a52dc85c91b035170013d5e46220`, compared at merge base
`f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`. The twelve plans select findings
and excerpts from that pinned tip; the merge base is only the comparison point.
This source pin applies to every plan's provenance below.
CUDA/Pascal-SASS-only work (`fattn-gemm`, `gemm-fold`,
`fattn-q4p`, `mmvq`, and `gdn-chunked`) remains out of scope.

Each included finding belongs to exactly one plan. Thresholds and defaults from
the P100 source are evidence, not A770 defaults; each A770 value must be derived
and recorded by the plan that consumes it.

## EARS contract

Requirements use the five forms from Mavin et al., *Easy Approach to
Requirements Syntax (EARS)*:

- Ubiquitous: `The <system> shall <response>.`
- Event-driven: `WHEN <trigger>, the <system> shall <response>.`
- Unwanted behaviour: `IF <trigger>, THEN the <system> shall <response>.`
- State-driven: `WHILE <state>, the <system> shall <response>.`
- Optional feature: `WHERE <feature included>, the <system> shall <response>.`

Angle brackets above are template notation only. A final requirement shall use
a stable `RNN.M` identifier, exactly one of the five structures, one concrete
repository component, and one testable response. Final requirement text shall
not contain template placeholders. Implementation rationale, scope notes,
measurement gates, and rollback policy belong outside the requirement sentence.

## Plan inventory

| Plan | Title | Kind | Hard dependency | Evidence input |
| --- | --- | --- | --- | --- |
| P01 | [Bench token count at depth](kmic68-01-bench-token-count.md) | measurement methodology | none | none |
| P02 | [Minimum free memory over a run](kmic68-02-min-free-memory.md) | measurement methodology | none | none |
| P03 | [Memory-path timing proxy](kmic68-03-memory-only-floor.md) | experimental methodology | none | none |
| P04 | [Speculative cycle log and phase profile](kmic68-04-cycle-log.md) | instrumentation | none | none |
| P05 | [Guarded CPU sampler top-k prefilter](kmic68-05-topk-prefilter.md) | common-layer feature | none | none |
| P06 | [Distribution-recording speculative acceptance](kmic68-06-speculative-sampling.md) | common-layer feature | P04 | P04 cycle dataset |
| P07 | [Block verification](kmic68-07-block-verify.md) | common-layer feature | P06 | P06 distributions |
| P08 | [Draft-context ubatch cap](kmic68-08-draft-ubatch-cap.md) | common-layer feature | none | none |
| P09 | [Fixed-width padded verification](kmic68-09-padded-verify.md) | common-layer feature | none | none |
| P10 | [Cumulative-probability draft-width clamp](kmic68-10-pcum-draft-width.md) | common-layer feature | P04 | P01 re-baseline |
| P11 | [Per-tile fp32 fold for F16 FA output](kmic68-11-fp32-vkq-fold.md) | SYCL backend feature | P03 | P03 proxy report |
| P12 | [Prefix-length causal-mask representation](kmic68-12-non-materialised-mask.md) | core/backend feature | P08 | P08 effective draft width |

P01, P02, P03, P04, P05, P08, and P09 are independently executable. P01 is an
evidence dependency for P10 rather than an implementation dependency.

## Dependency graph

Hard implementation dependencies:

```text
P04 -> P06 -> P07
  +-----> P10
P03 -> P11
P08 -> P12
```

Evidence dependency:

```text
P01 -- re-baselined long-depth results --> P10
```

P02 supplies shared measurement discipline but is not a hard dependency. P05
and P09 are independent; P09 is deliberately not coupled to P10. The graph is
acyclic.

## Recommended execution order

1. Run P01, P02, P03, P04, P05, P08, and P09 independently or in parallel.
2. After P04, run P06. After both P04 and the P01 re-baseline, run P10. After
   P03, run P11. After P08, run P12.
3. After P06, run P07.

The order controls prerequisites, not ownership: every plan remains an
independently executable document once its named dependencies are complete.

## Shared verification and A770 regression floor

- Before and after a plan that changes `common/` speculative or sampling
  behavior, run these exact gates plus the plan's focused test:
  `timeout 240 ./build-sycl/bin/test-qwen4exp-mtp`,
  `timeout 240 ./build-sycl/bin/test-qwen4exp-mtp --q8-kv`, and
  `timeout 240 python3 -m unittest scripts.test_bench_spec`.
- Before and after P11 or P12 backend work, run
  `timeout 180 ./build-sycl/bin/test-sycl-turbo-correctness`. Run the Turbo FA
  path only with the exact opt-in
  `LLAMA_TEST_TURBO_FA=1 timeout 180 ./build-sycl/bin/test-sycl-turbo-correctness`
  and its documented hang precautions. Use
  `ctest --test-dir build-sycl -L sycl --timeout 180 -V` when the full SYCL
  label is applicable.
- P05, P06, and P07 require seeded output-distribution parity against their
  disabled or predecessor path, not only a passing build.
- Every timing run follows the mandatory GPU discipline in `AGENTS.md`: stop
  the service, verify sole tenancy on `/dev/dri/renderD128`, wrap the workload
  in `timeout`, restart the service afterward, and apply the two-driver fault
  gate. A clean `dmesg` is not proof of health on `xe`.
- Every recorded A770 result names the kernel driver, complete command, build,
  model, repetition count, and observed output. A plan may not copy a P100
  threshold as an A770 default.
