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
| 4096 | q8_0/q8_0 | pp512 | Y | 346.98 | 346.50 | 2.28 | +/- 2.83 | 347.54 | 347.72 | 0.63 | +/- 0.78 | +0.13 | +0.36 | 0.68 | +/- 0.85 | n/a | n/a | n/a | 5 |
| 4096 | q8_0/q8_0 | tg128 | Y | 19.01 | 18.84 | 0.29 | +/- 0.37 | 23.00 | 22.92 | 0.21 | +/- 0.26 | +21.13 | +21.65 | 1.76 | +/- 2.18 | n/a | n/a | n/a | 5 |
