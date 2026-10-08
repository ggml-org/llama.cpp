# Research corpus

Dated evidence behind the fork's decisions, sorted by topic. Each note states its date, host, and
what it measured; a later note or live command output overrides an earlier one. Raw artifacts
(`.json`, `.jsonl`, `.log`, `.csv`) keep the paths they recorded when they were written, which
predate this layout.

## sycl/ - SYCL backend on the Arc A770

Baselines and A/B runs:

- [standard-sycl-baseline-2026-07-11.md](sycl/standard-sycl-baseline-2026-07-11.md) - f16/q8_0 KV baseline
- [standard-sycl-upstream-ab-2026-07-11.md](sycl/standard-sycl-upstream-ab-2026-07-11.md) - fork vs upstream A/B
- [2026-07-09-fork-vs-upstream-overview.md](sycl/2026-07-09-fork-vs-upstream-overview.md) - first fork vs upstream overview
- [2026-07-09-a770-benchmark-results-incremental.md](sycl/2026-07-09-a770-benchmark-results-incremental.md) and
  [2026-07-09-a770-skipped-cases-ledger.md](sycl/2026-07-09-a770-skipped-cases-ledger.md), with raw data in
  [a770-fork-unique-2026-07-09/](sycl/a770-fork-unique-2026-07-09/) and [coherence-2026-07-09/](sycl/coherence-2026-07-09/)

Performance campaigns:

- [sycl-a770-p5-performance-campaign-2026-07-19.md](sycl/sycl-a770-p5-performance-campaign-2026-07-19.md) - P5 campaign, source of most measured dead ends
- [sycl-a770-round2-decode-candidates-2026-07-25.md](sycl/sycl-a770-round2-decode-candidates-2026-07-25.md) and
  [sycl-a770-round2-report-2026-07-26.md](sycl/sycl-a770-round2-report-2026-07-26.md) - decode round 2
- [round2-decode-probes-2026-08-13.md](sycl/round2-decode-probes-2026-08-13.md) - round 2 follow-up probes
- [fa-occupancy-prereg-2026-07-25.md](sycl/fa-occupancy-prereg-2026-07-25.md) - FA occupancy predictions, with
  [fa-occupancy-campaign-2026-07-25/](sycl/fa-occupancy-campaign-2026-07-25/),
  [fa-occupancy-campaign-2026-08-13-llama31-coverage/](sycl/fa-occupancy-campaign-2026-08-13-llama31-coverage/)
  and the two `fa-occupancy-*-ab-2026-07-25.csv` tables
- [upstream-fa-occupancy-issue-draft-2026-08-13.md](sycl/upstream-fa-occupancy-issue-draft-2026-08-13.md) - unfiled upstream issue draft
- [ffn-fusion-campaign-2026-07-26/](sycl/ffn-fusion-campaign-2026-07-26/) and [ffn-fusion-2026-07-26/](sycl/ffn-fusion-2026-07-26/) - FFN fusion A/B
- [sycl-fa-large-grf-2026-09-30.md](sycl/sycl-fa-large-grf-2026-09-30.md) - FA register spills and the 256-GRF knob
- [sycl-moe-expert-paging-research-2026-09-27.md](sycl/sycl-moe-expert-paging-research-2026-09-27.md) - MoE expert paging
- [sycl-prefetch-second-queue-2026-09-28.md](sycl/sycl-prefetch-second-queue-2026-09-28.md) - second queue for expert prefetch, probe in
  [sycl-prefetch-second-queue-probe.cpp](sycl/sycl-prefetch-second-queue-probe.cpp)
- [ornith-a770-perf-research-2026-09-27.md](sycl/ornith-a770-perf-research-2026-09-27.md) - where decode time goes on a 35B MoE
- [ngxson-branch-evaluation-2026-09-27.md](sycl/ngxson-branch-evaluation-2026-09-27.md) - evaluation of an upstream contributor branch

Bugs and harness checks:

- [sycl-q8-context-2048-nan.md](sycl/sycl-q8-context-2048-nan.md) - q8_0 KV NaN at n_kv >= 1024 (resolved 2026-09-05)
- [p6.6b-graph-profile-crash-fix-2026-08-15.md](sycl/p6.6b-graph-profile-crash-fix-2026-08-15.md) - SYCL graph crash under the FA profiler
- [p6.8-harness-smoke-2026-08-13/](sycl/p6.8-harness-smoke-2026-08-13/) - benchmark harness smoke run

## turbo/ - TurboQuant codec, quality and capacity

- [turbo-fa-research-artifact.md](turbo/turbo-fa-research-artifact.md) - correct turbo flash attention on SYCL (the VEC data contract)
- [turbo3-quality-gate-llama31-8b-2026-09.md](turbo/turbo3-quality-gate-llama31-8b-2026-09.md) - turbo3 PPL gate on Llama-3.1-8B
- [p4.5a-turbo4-capacity-reopen-2026-08-13.md](turbo/p4.5a-turbo4-capacity-reopen-2026-08-13.md) - turbo4 capacity validation, raw data in
  [p4.5a-turbo4-capacity-reopen-2026-08-13/](turbo/p4.5a-turbo4-capacity-reopen-2026-08-13/)
- [p4.6-canonical-advanced-ideas-2026-07-11.md](turbo/p4.6-canonical-advanced-ideas-2026-07-11.md) - audit of advanced KV ideas
- [p6.6a-turbo-k-predicate-fix-2026-08-13/](turbo/p6.6a-turbo-k-predicate-fix-2026-08-13/) - turbo K predicate fix logs
- [pr51-thetom-integration.md](turbo/pr51-thetom-integration.md) - integration of the TheTom TurboQuant fork, file ledger in
  [pr51-thetom-integration-files.tsv](turbo/pr51-thetom-integration-files.tsv)

## speculative/ - speculative decoding and MTP

- [qwen4exp-mtp-correctness-2026-10-04.md](speculative/qwen4exp-mtp-correctness-2026-10-04.md) - Qwen4Exp MTP head correctness
- [sycl-a770-spec-checkpoint-on-device-ab-2026-10-04.md](speculative/sycl-a770-spec-checkpoint-on-device-ab-2026-10-04.md) - on-device checkpoint A/B
- [spec-f16-exactness-2026-07-26/](speculative/spec-f16-exactness-2026-07-26/) - f16 KV output exactness, no drafting vs ngram-mod and ngram-map-k4v drafting

## software-stack/ - driver, compiler and runtime

- [sycl-build-runtime-pins.md](software-stack/sycl-build-runtime-pins.md) - canonical toolchain and runtime pins
- [xe-kmd-bcs-copy-engine-2026-09-30.md](software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md) - blitter copy hangs on the xe KMD, with the
  compute-runtime patch in [patches/](software-stack/patches/)
- [claude-research-1.md](software-stack/claude-research-1.md), [google-research-1.md](software-stack/google-research-1.md),
  [google-research-2.md](software-stack/google-research-2.md), [openai-research-1.md](software-stack/openai-research-1.md) -
  model-generated surveys of the Mesa / compute-runtime stack, kept as unverified reference
