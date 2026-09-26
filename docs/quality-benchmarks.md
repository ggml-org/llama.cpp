# TurboQuant quality validation

A synthetic CPU/backend oracle checks implementation agreement. It does not
establish model quality. Measure perplexity and KL divergence against an f16
KV baseline using the same model, corpus, context, token count, and build.
Record the source commit, command, hardware, environment, and complete logs.

For the Arc A770 SYCL path, use `ONEAPI_DEVICE_SELECTOR=level_zero:0` and the
GPU tenancy procedure in `AGENTS.md`. The synthetic gate needs no model:

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 300 \
    "$HOME/build-sycl/bin/test-sycl-turbo-correctness"
```

For model quality, `scripts/turbo-quality-gate.sh` provides the existing PPL
and context-scaling workflow. Read its options and select local model and corpus paths.
Use `llama-perplexity --kl-divergence-base` to save baseline logits, then
`--kl-divergence` with that file to compare the candidate. Compare f16, q8_0,
and the intended asymmetric or symmetric turbo configuration.
Do not infer throughput from the compressed byte count. Timing runs require
exclusive GPU access and repeated measurements.

## Imported historical evidence

TheTom's original [quality notebook at the pinned integration source](https://github.com/TheTom/llama-cpp-turboquant/blob/80007e71526b2566bda62a8fc98f68c4d231139c/docs/quality-benchmarks.md)
reports an early Metal experiment on an Apple M5 Max with Qwen 3.5 35B-A3B
Q8_0, Wikitext-2, context 512, and eight chunks:

| KV type | Reported PPL |
| --- | ---: |
| f16 | 6.121 |
| q8_0 | 6.111 |
| q4_0 | 6.142 |
| turbo3 | 165.6 |

That notebook attributes the failure to missing rotations through hybrid
memory and contains proposed investigations. These are historical reports
from TheTom, not measurements of this fork, the current centroid table, or
SYCL. Its speed claims, block-size descriptions, and local paths do not
apply here. Current validation evidence belongs in the report for the
specific tested revision.
