# Turbo3 KV quality gate on Llama-3.1-8B-Instruct Q4_K_M (2026-09-05)

[`scripts/turbo-quality-gate.sh`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/435f47bb80b88b60887b44b3d8c06acca45cab3f/scripts/turbo-quality-gate.sh) stage 1 compares turbo3/turbo3 against q8_0/q8_0
perplexity at context 512 and fails when turbo3 exceeds 105% of q8_0. On this
model it fails by 0.11 percentage points. This note records the gate run, the
KV-policy sweep that was measured against it, and the decision: the threshold
and the defaults stay as they are; the README claim is corrected to the
measured number.

## Setup

- Host: `vinbonesjr`, Intel Arc A770, oneAPI 2026.0, `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
- Source: fork master `435f47bb8` plus the MKL FA quants-first fix (PR #52); the fix
  does not touch turbo3 or any context-512 path, so it is neutral here.
- Build: `~/build-master-435f47bb8` (JIT, `GGML_SYCL_F16=ON`, launchers empty).
- Model: `/mnt/mrgr/models/llama31-8b-q4km/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
  (32 layers, GQA 4:1, head dim 128). Corpus: Wikitext-2 raw test.
- Stage 1 protocol: `llama-perplexity -c 512 --chunks 8 -fa on -ngl 99`, run with
  `-ctk turbo3 -ctv turbo3` and separately with `-ctk q8_0 -ctv q8_0`.
- Stage 2 protocol: `llama-perplexity -c 4096 --chunks 4 -fa on -ngl 99`.
- GPU shared with a desktop IDE process during the runs; PPL is unaffected by
  contention, the stage 2 timing ratio is indicative only.

## Gate result summary

Human-readable summary of the recorded results, not verbatim script output:

```
PASS | 0.1 correctness (LLAMA_TEST_TURBO_FA=0)
FAIL | 1 perplexity (turbo3 vs q8_0, -fa on): turbo3 PPL 8.3939 > limit 8.385405 (q8_0 7.9861)
PASS | 2 context-scaling ratio: turbo3 335.46 t/s, q8_0 318.75 t/s, ratio 1.052 (required > 0.95)
```

Before PR 51 the same probe gave turbo3 8.4805 (+6.19%); the Turbo3 centroid
re-derivation brought it to 8.3939 (+5.11%). Still over the 5% line.

## KV policy sweep (stage 1 protocol)

Bytes per cached element are analytic from the block layouts (f16 2 B, q8_0
34 B/32 = 1.0625 B, turbo3 50 B/128 = 0.390625 B), averaged over K and V and
over the 32 layers; the compression column is relative to f16/f16.

| K / V | Policy | PPL | vs q8_0 | vs f16 | B/elem | vs f16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f16 / f16 | | 7.9844 | -0.02% | 0 | 2.000 | 1.00x |
| q8_0 / q8_0 | | 7.9861 | 0 | +0.02% | 1.063 | 1.88x |
| turbo3 / turbo3 | default | 8.3939 | **+5.11%** | +5.13% | 0.391 | 5.12x |
| turbo3 / turbo3 | `TURBO_LAYER_ADAPTIVE=1` (first 4 + last 4 layers K,V = q8_0) | 8.2428 | +3.21% | +3.24% | 0.559 | 3.58x |
| turbo3 / turbo3 | `TURBO_LAYER_ADAPTIVE=2` (last 8 layers K,V = q8_0) | 8.3339 | +4.36% | +4.38% | 0.559 | 3.58x |
| turbo3 / turbo3 | `TURBO_LAYER_ADAPTIVE=5` (V: first 2 + last 2 turbo4, rest turbo2) | 8.6085 | +7.79% | +7.82% | n/a | |
| turbo3 / turbo3 | `TURBO_LAYER_ADAPTIVE=6` (V: last 8 turbo4, rest turbo2) | 8.7709 | +9.83% | +9.85% | n/a | |
| turbo3 / turbo3 | `TURBO_LAYER_ADAPTIVE=7` (V: first 2 + last 2 q8_0, rest turbo2) | 8.6415 | +8.21% | +8.23% | n/a | |
| q8_0 / turbo3 | asymmetric K | 8.0292 | +0.54% | +0.56% | 0.727 | 2.75x |
| turbo3 / q8_0 | asymmetric V | 8.3444 | +4.49% | +4.51% | 0.727 | 2.75x |

Modes 5, 6 and 7 are turbo2-V policies: they replace the requested turbo3 V by
turbo2 on the non-boundary layers, so they lower quality for a turbo3 request.
They are listed because they were part of the sweep, not because they were
candidates.

## Reading

- The K side carries most of the loss on this model: K=q8_0 with V=turbo3 is
  within 0.6% of q8_0 at 2.75x; V=q8_0 with K=turbo3 is still +4.5%.
- Layer-adaptive modes 1 and 2 pass the 5% gate at 3.58x compression by moving
  8 of 32 layers to q8_0. Mode 1 (boundary layers) is 1.15 points better than
  mode 2 (last layers) for the same byte budget.
- Symmetric turbo3 at 5.12x does not pass the 5% gate on Llama-3.1-8B at ctx
  512. The auto-asymmetric downgrade does not fire here (GQA 4:1 < 6, not Qwen).

## Decision

Owner decision 2026-09-05: keep the 5% threshold in [`scripts/turbo-quality-gate.sh`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/435f47bb80b88b60887b44b3d8c06acca45cab3f/scripts/turbo-quality-gate.sh)
and keep the turbo3 defaults; record the measured gap instead. README.md now
states the measured figure for this fork and model in place of the imported
"<1.5% PPL loss" claim. Mode 1 (or asymmetric K) is the candidate if the owner
later wants turbo3 to pass the gate on this model; that is a separate decision
because it costs compression (3.58x or 2.75x instead of 5.12x).

Not claimed: one model, one corpus, one context length; no long-context PPL;
no KLD; no throughput claim (stage 2 ran under GPU contention); compression is
analytic block-size arithmetic, not measured allocation.

Logs: session scratchpad `gate-run.log`, `gate-*.log`; gate stage logs under
`/var/tmp/svnbjrn/tmp.MicZ3d0Ads/` at the time of the run.
