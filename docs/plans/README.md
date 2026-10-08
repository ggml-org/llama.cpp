# Plan set: adopt Kmic-68/llama.cpp findings

Source of findings: `Kmic-68/llama.cpp`, branch `p100-optimizations` at tip
`ae35056eba07a52dc85c91b035170013d5e46220`, read against this fork at merge base
`f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`. The tip is the revision the findings,
thresholds and code excerpts were taken from and is what an audit needs; the
merge base is only for diffing. Its CUDA kernel work (`fattn-gemm`,
`gemm-fold`, `fattn-q4p`, `mmvq`, `gdn-chunked`) is Pascal SASS and is out of
scope; these plans cover the `common/` layer and the two findings that bear on
our SYCL build directly.

Kmic-68's `p100-docs/FINDINGS.md` records eleven ways its own measurements misled
it. Every default in these plans is therefore measured on the A770 rather than
copied from its thresholds.

Line numbers throughout this set refer to commit `001d906a6`. They drift as the
code moves; the plans would need re-verification when implemented, not just when
written.

Requirements use EARS (Easy Approach to Requirements Syntax): the *Ubiquitous*
form `The <system> shall <response>`, *event-driven* `WHEN <trigger>, the
<system> shall <response>`, *unwanted behaviour* `IF <trigger>, THEN the <system>
shall <response>`, *state-driven* `WHILE <state>, the <system> shall <response>`,
and *optional feature* `WHERE <feature included>, the <system> shall <response>`.

## The plans

| Plan | Title | Kind | Depends on |
| --- | --- | --- | --- |
| P01 | [Bench token count at depth](kmic68-01-bench-token-count.md) | methodology | none |
| P02 | [Minimum free memory over a run](kmic68-02-min-free-memory.md) | methodology | none |
| P03 | [Memory-only floor ceiling analysis](kmic68-03-memory-only-floor.md) | methodology | none |
| P04 | [Speculative cycle log and phase profile](kmic68-04-cycle-log.md) | instrumentation | none |
| P05 | [Top-k prefilter in the CPU sampler](kmic68-05-topk-prefilter.md) | port | none |
| P06 | [Distribution-recording acceptance](kmic68-06-speculative-sampling.md) | port | P04 |
| P07 | [Block verification](kmic68-07-block-verify.md) | port | P06 |
| P08 | [Draft-context ubatch cap](kmic68-08-draft-ubatch-cap.md) | port | none |
| P09 | [Padded fixed-width verify](kmic68-09-padded-verify.md) | decoupling | none |
| P10 | [Cumulative-probability draft width](kmic68-10-pcum-draft-width.md) | port | P01, P04 |
| P11 | [Per-tile fp32 fold for the F16 FA output](kmic68-11-fp32-vkq-fold.md) | port | P03 |
| P12 | [Non-materialised causal mask](kmic68-12-non-materialised-mask.md) | port | P08 |

## Dependency graph

```
P01 -------------------------------> (re-baselines the numbers P10 is judged against)
P02, P03 ---> (measurement discipline for every plan below)
P04 ---> P06 ---> P07
P01 ---> P10
P05 (independent)
P08 ---> P12
P09 (independent; P10 is deliberately NOT coupled to it)
P03 ---> P11
```

## Recommended order

P01 first: it costs no code and may invalidate recorded figures. Then P04,
because every measurement-driven plan needs its dataset. Then P05, the smallest
real code win. Then P06 and P07. P02, P03, P08 and P09 run in parallel. P10, P11
and P12 last.

## Regression floor

`test-qwen4exp-mtp` (75 assertions CPU on the final source, 50 with
`--q8-kv`; 76 and 51 on Arc A770) and the whole of `scripts/test_bench_spec.py`
(26 tests at `001d906a6`) pass
before and after every plan that touches `common/` or the FA kernels. The counts
are from `docs/research/qwen4exp-mtp-correctness-2026-10-04.md:351-352`; a lower
count than that is a regression. Plans that change output tokens (P05, P06,
P07) additionally require seeded output-distribution parity against their own
pre-change path.

Every A770 measurement follows the GPU discipline block in `AGENTS.md`, names the
kernel driver, and runs inside `timeout`. A clean dmesg is not evidence of health
under xe.
