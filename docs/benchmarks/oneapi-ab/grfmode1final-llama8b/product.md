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
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 975.00 | 942.17 | 78.96 | +/- 196.17 | 1060.16 | 1051.66 | 20.69 | +/- 51.40 | +9.41 | +12.23 | 11.05 | +/- 27.45 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.78 | 46.64 | 0.38 | +/- 0.94 | 46.89 | 46.87 | 0.06 | +/- 0.14 | +0.30 | +0.52 | 0.89 | +/- 2.21 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 210.63 | 210.52 | 0.27 | +/- 0.67 | 208.99 | 209.16 | 0.69 | +/- 1.72 | -0.58 | -0.65 | 0.30 | +/- 0.76 | 2281701376 | 81.852 | 81.844 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 35.87 | 36.00 | 0.29 | +/- 0.71 | 35.87 | 35.97 | 0.21 | +/- 0.52 | -0.01 | -0.07 | 1.26 | +/- 3.13 | 2281701376 | 81.852 | 81.844 | 3 |
