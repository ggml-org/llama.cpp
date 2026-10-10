# Product campaign: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-grf-split/bin
- baseline label: grf-off
- candidate label: grf-mode1
- baseline env: {}
- candidate env: {'GGML_SYCL_FA_LARGE_GRF': '1'}
- candidate_enabled: True
- model shape: {'model_layers': 32, 'query_heads': 32, 'head_dim': 128}
- campaign valid: False
- invalid diagnostics: ['sample build_commit b23424e30 does not match repository commit 5b97abd57472e316c5c206516a97517af03a5850']
- candidate env log assertions: GGML_SYCL_FA_LARGE_GRF=1 (validated in 8/8 candidate samples; valid=True)
- dmesg fault hits before=1 after=1 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 984.68 | 975.38 | 22.71 | +/- 56.42 | 1077.26 | 1059.59 | 51.84 | +/- 128.79 | +9.40 | +8.59 | 2.82 | +/- 7.02 | 0 | 0.000 | 0.000 | 3 |
| 0 | q8_0/q8_0 | tg128 | Y | 46.62 | 46.70 | 0.21 | +/- 0.53 | 46.94 | 46.92 | 0.09 | +/- 0.23 | +0.68 | +0.46 | 0.66 | +/- 1.63 | 0 | 0.000 | 0.000 | 3 |
| 8192 | q8_0/q8_0 | pp512 | Y | 210.51 | 210.10 | 1.18 | +/- 2.92 | 208.15 | 208.75 | 1.36 | +/- 3.37 | -1.12 | -0.64 | 1.20 | +/- 2.99 | 2281701376 | 82.830 | 82.810 | 3 |
| 8192 | q8_0/q8_0 | tg128 | Y | 36.30 | 36.27 | 0.10 | +/- 0.25 | 36.29 | 36.25 | 0.10 | +/- 0.24 | -0.02 | -0.06 | 0.50 | +/- 1.25 | 2281701376 | 82.830 | 82.810 | 3 |
