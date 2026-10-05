# Product campaign: mistral-7b-instruct-v0.1.Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-p63-80d52e708/bin
- baseline label: imm1-batch-unset
- candidate label: imm1-batch64
- baseline env: {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '1'}
- candidate env: {'UR_L0_USE_IMMEDIATE_COMMANDLISTS': '1', 'UR_L0_BATCH_SIZE': '64'}
- candidate_enabled: True
- model shape: None
- campaign valid: True
- invalid diagnostics: none
- candidate env log assertions: UR_L0_BATCH_SIZE=64 (not validated from backend logs; key not emitted); UR_L0_USE_IMMEDIATE_COMMANDLISTS=1 (not validated from backend logs; key not emitted)
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | Y | 935.28 | 936.42 | 7.65 | +/- 9.50 | 918.58 | 921.29 | 17.92 | +/- 22.25 | -1.95 | -1.61 | 1.87 | +/- 2.32 | n/a | n/a | n/a | 5 |
| 0 | q8_0/q8_0 | tg128 | Y | 26.13 | 26.16 | 0.08 | +/- 0.09 | 26.13 | 26.18 | 0.11 | +/- 0.13 | +0.04 | +0.09 | 0.21 | +/- 0.26 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | pp512 | Y | 111.64 | 111.55 | 0.16 | +/- 0.20 | 111.65 | 111.65 | 0.09 | +/- 0.11 | -0.01 | +0.09 | 0.22 | +/- 0.27 | n/a | n/a | n/a | 5 |
| 16384 | q8_0/q8_0 | tg128 | Y | 19.85 | 19.81 | 0.10 | +/- 0.12 | 19.82 | 19.82 | 0.02 | +/- 0.03 | -0.13 | +0.02 | 0.52 | +/- 0.65 | n/a | n/a | n/a | 5 |
