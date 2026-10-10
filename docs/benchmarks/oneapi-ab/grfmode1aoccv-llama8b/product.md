# Product campaign: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=1 after=1 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 913.20 | 926.38 | 35.77 | +/- 88.86 | 1083.92 | 1061.80 | 51.19 | +/- 127.17 | +12.11 | +14.71 | 6.54 | +/- 16.25 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.39 | 46.36 | 0.11 | +/- 0.27 | 46.51 | 46.64 | 0.32 | +/- 0.80 | +0.24 | +0.60 | 0.93 | +/- 2.30 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 209.81 | 209.75 | 0.61 | +/- 1.53 | 208.54 | 208.36 | 0.75 | +/- 1.87 | -0.75 | -0.66 | 0.25 | +/- 0.62 | 2281701376 | 82.242 | 82.047 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 36.04 | 36.12 | 0.13 | +/- 0.32 | 35.96 | 35.99 | 0.05 | +/- 0.13 | -0.23 | -0.36 | 0.45 | +/- 1.12 | 2281701376 | 82.242 | 82.047 | 3 |
