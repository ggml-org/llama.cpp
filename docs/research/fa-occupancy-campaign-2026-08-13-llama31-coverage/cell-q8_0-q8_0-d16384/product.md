# Product campaign: Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-p63-80d52e708/bin
- baseline label: wg2
- candidate label: wg16
- baseline env: {'GGML_SYCL_MAX_WG_PER_CU': '2'}
- candidate env: {'GGML_SYCL_MAX_WG_PER_CU': '16'}
- candidate_enabled: True
- model shape: None
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: GGML_SYCL_MAX_WG_PER_CU=16 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | q8_0/q8_0 | pp512 | Y | 111.65 | 111.64 | 0.17 | +/- 0.21 | 111.60 | 111.63 | 0.06 | +/- 0.07 | +0.01 | -0.01 | 0.13 | +/- 0.16 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 11.17 | 11.19 | 0.05 | +/- 0.06 | 19.26 | 19.27 | 0.08 | +/- 0.09 | +72.92 | +72.22 | 1.31 | +/- 1.62 | n/a | n/a | n/a | 5 |
