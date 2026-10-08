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
| 8192 | f16/f16 | pp512 | Y | 204.43 | 204.53 | 0.45 | +/- 0.56 | 204.78 | 204.64 | 0.40 | +/- 0.50 | -0.02 | +0.05 | 0.18 | +/- 0.23 | n/a | n/a | n/a | 5 |
| 8192 | f16/f16 | tg128 | Y | 11.27 | 11.27 | 0.01 | +/- 0.02 | 19.97 | 20.00 | 0.06 | +/- 0.07 | +77.43 | +77.49 | 0.36 | +/- 0.44 | n/a | n/a | n/a | 5 |
