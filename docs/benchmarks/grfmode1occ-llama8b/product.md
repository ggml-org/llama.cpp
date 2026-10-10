# Product campaign: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-fa-grf256/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 996.82 | 998.44 | 3.89 | +/- 9.66 | 1100.02 | 1098.28 | 3.28 | +/- 8.16 | +9.93 | +10.00 | 0.35 | +/- 0.88 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.88 | 46.87 | 0.02 | +/- 0.04 | 46.87 | 46.88 | 0.03 | +/- 0.08 | +0.05 | +0.02 | 0.07 | +/- 0.17 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 211.62 | 211.07 | 1.20 | +/- 2.98 | 211.17 | 211.19 | 0.20 | +/- 0.50 | -0.10 | +0.06 | 0.58 | +/- 1.45 | 2281701376 | 83.037 | 83.023 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 36.39 | 36.40 | 0.02 | +/- 0.05 | 36.39 | 36.37 | 0.04 | +/- 0.09 | -0.07 | -0.08 | 0.09 | +/- 0.23 | 2281701376 | 83.037 | 83.023 | 3 |
