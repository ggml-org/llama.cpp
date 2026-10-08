# Product campaign: Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

- bin-dir: /home/svnbjrn/build-p63-80d52e708/bin
- baseline label: stock
- candidate label: candidate
- baseline env: {}
- candidate env: {}
- candidate_enabled: False
- model shape: None
- campaign valid: False
- invalid diagnostics: ["cell 1 d=0 kv=['q8_0', 'q8_0']: pp512 baseline rep=1 (missing, non-finite, or non-positive avg_ts)", "cell 1 d=0 kv=['q8_0', 'q8_0']: pp512 baseline rep=2 (missing, non-finite, or non-positive avg_ts)", "cell 1 d=0 kv=['q8_0', 'q8_0']: tg128 baseline rep=1 (missing, non-finite, or non-positive avg_ts)", "cell 1 d=0 kv=['q8_0', 'q8_0']: tg128 baseline rep=2 (missing, non-finite, or non-positive avg_ts)", "cell 1 d=0 kv=['q8_0', 'q8_0']: pp512 baseline has 0 retained samples (expected 2)", "cell 1 d=0 kv=['q8_0', 'q8_0']: tg128 baseline has 0 retained samples (expected 2)", 'baseline samples do not report build_commit']
- candidate env log assertions: none
- dmesg fault hits before=0 after=0 new=0

| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |
|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | q8_0/q8_0 | pp512 | N | 0.00 | 0.00 | 0.00 | +/- 0.00 | 0.00 | 0.00 | 0.00 | +/- 0.00 | +0.00 | +0.00 | 0.00 | +/- 0.00 | n/a | n/a | n/a | 0 |
| 0 | q8_0/q8_0 | tg128 | N | 0.00 | 0.00 | 0.00 | +/- 0.00 | 0.00 | 0.00 | 0.00 | +/- 0.00 | +0.00 | +0.00 | 0.00 | +/- 0.00 | n/a | n/a | n/a | 0 |
