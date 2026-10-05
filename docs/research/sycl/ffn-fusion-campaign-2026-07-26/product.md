# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

> [!CAUTION]
> INVALID FOR PROMOTION: the raw samples report build commit `b5ef0a84b`,
> which contains the known-bad interleaved activation quantizer. The fixed
> kernel starts at `02f848c83`. The table is retained as historical data only
> and does not validate the fixed kernel.

- bin-dir: /mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork-fa-occupancy/build/bin
- baseline label: baseline
- candidate label: candidate
- baseline env: {'GGML_SYCL_FFN_FUSION': '0'}
- candidate env: {'GGML_SYCL_FFN_FUSION': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: False
- invalid diagnostics: ['sample build_commit b5ef0a84b does not match repository commit 02f848c83a8aee4bfe0ce956b64cc4b9058e2bf2']
- candidate env log assertions: GGML_SYCL_FFN_FUSION=1 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | f16/f16 | pp512 | Y | 538.32 | 538.25 | 4.21 | +/- 5.23 | 539.32 | 539.53 | 4.31 | +/- 5.35 | +0.15 | +0.24 | 0.83 | +/- 1.03 | 0 | 0.000 | 0.000 | 5 |
| 0 | f16/f16 | tg128 | Y | 24.23 | 24.23 | 0.04 | +/- 0.05 | 26.02 | 26.04 | 0.05 | +/- 0.07 | +7.56 | +7.49 | 0.32 | +/- 0.39 | 0 | 0.000 | 0.000 | 5 |
| 0 | q8_0/q8_0 | pp512 | Y | 536.14 | 535.41 | 2.34 | +/- 2.91 | 528.26 | 529.96 | 3.38 | +/- 4.20 | -1.48 | -1.01 | 0.96 | +/- 1.20 | 0 | 0.000 | 0.000 | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 24.56 | 24.56 | 0.03 | +/- 0.03 | 26.43 | 26.43 | 0.07 | +/- 0.08 | +7.60 | +7.61 | 0.37 | +/- 0.46 | 0 | 0.000 | 0.000 | 5 |
| 4096 | f16/f16 | pp512 | Y | 279.87 | 279.84 | 0.32 | +/- 0.40 | 279.45 | 279.58 | 0.43 | +/- 0.53 | -0.06 | -0.09 | 0.11 | +/- 0.13 | 2147483648 | 46.382 | 49.524 | 5 |
| 4096 | f16/f16 | tg128 | Y | 21.60 | 21.59 | 0.02 | +/- 0.02 | 23.06 | 23.06 | 0.04 | +/- 0.05 | +6.78 | +6.77 | 0.27 | +/- 0.33 | 2147483648 | 46.382 | 49.524 | 5 |
| 4096 | q8_0/q8_0 | pp512 | Y | 279.06 | 276.65 | 5.52 | +/- 6.85 | 278.80 | 278.81 | 0.44 | +/- 0.54 | -0.02 | +0.81 | 1.92 | +/- 2.39 | 1140850688 | 24.116 | 25.692 | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 21.14 | 21.05 | 0.24 | +/- 0.30 | 22.52 | 22.50 | 0.09 | +/- 0.11 | +6.56 | +6.94 | 0.86 | +/- 1.07 | 1140850688 | 24.116 | 25.692 | 5 |
| 16384 | f16/f16 | pp512 | Y | 97.23 | 97.28 | 0.30 | +/- 0.38 | 96.98 | 95.75 | 3.16 | +/- 3.92 | -0.05 | -1.57 | 3.35 | +/- 4.16 | 8589934592 | 145.260 | 151.173 | 5 |
| 16384 | f16/f16 | tg128 | Y | 16.91 | 15.54 | 3.04 | +/- 3.77 | 17.60 | 16.73 | 1.72 | +/- 2.14 | +4.66 | +9.93 | 14.78 | +/- 18.35 | 8589934592 | 145.260 | 151.173 | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 97.54 | 97.56 | 0.07 | +/- 0.08 | 97.69 | 97.64 | 0.14 | +/- 0.17 | +0.04 | +0.08 | 0.13 | +/- 0.16 | 4563402752 | 75.617 | 79.336 | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 16.57 | 16.57 | 0.02 | +/- 0.02 | 17.39 | 17.39 | 0.01 | +/- 0.02 | +4.93 | +4.95 | 0.10 | +/- 0.13 | 4563402752 | 75.617 | 79.336 | 5 |
