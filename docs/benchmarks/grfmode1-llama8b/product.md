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
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 1008.64 | 1005.60 | 7.69 | +/- 19.11 | 1109.47 | 1107.80 | 3.27 | +/- 8.12 | +10.04 | +10.17 | 1.07 | +/- 2.66 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.90 | 46.91 | 0.02 | +/- 0.05 | 46.89 | 46.90 | 0.01 | +/- 0.03 | -0.01 | -0.02 | 0.06 | +/- 0.14 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 210.40 | 210.89 | 0.85 | +/- 2.12 | 210.90 | 210.93 | 0.42 | +/- 1.04 | +0.24 | +0.02 | 0.58 | +/- 1.44 | 2281701376 | 83.099 | 83.077 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 36.42 | 36.42 | 0.01 | +/- 0.02 | 36.41 | 36.41 | 0.02 | +/- 0.06 | -0.03 | -0.04 | 0.05 | +/- 0.13 | 2281701376 | 83.099 | 83.077 | 3 |
