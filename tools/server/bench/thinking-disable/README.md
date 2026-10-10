# Thinking disable benchmarks

See [BASELINE.md](BASELINE.md) for gate A prompt-suffix expectations.

## Gate C (optional nightly)

```bash
# Compare on vs off; prefer Jinja off as reference arm
python tools/server/bench/thinking-disable/think_suppress_bench.py \
  --url http://127.0.0.1:8080 --label nojinja-off --out nojinja-off.json
```

Accept vs Jinja `--reasoning off`: `think_hit_rate` within +5pp; vs `--reasoning on`: drop >= 50%.
