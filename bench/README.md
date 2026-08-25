# Benchmarks

Structured, append-only performance data for this fork against pinned upstream, plus the tooling
that produced it. `results.jsonl` is the running document: one JSON object per measured cell, never
rewritten, so a later run adds samples and tightens the same table rather than replacing it.

## Files

| file | what |
|---|---|
| `run-depth-sweep.sh` | runs the fork-vs-upstream matrix and appends to `results.jsonl` |
| `summarize.py` | `results.jsonl` -> markdown or CSV, with deltas and significance |
| `results.jsonl` | the running document, append-only |
| `results-ub2048-partial.jsonl` | earlier partial run at ubatch 2048, kept for the depth curve |

## Running it

```sh
./bench/run-depth-sweep.sh                      # defaults below
ROUNDS=2 ./bench/run-depth-sweep.sh             # more samples, appends
DEPTHS=0,4096 REPS=1 ./bench/run-depth-sweep.sh # quick smoke
python3 bench/summarize.py                      # markdown table
python3 bench/summarize.py --format csv         # feed the charts
```

Defaults: depths 0/4096/16384/32768/65536/131072, pp2048 and tg64, ubatch 512, flash attention on,
3 internal repetitions, one palindrome round.

## Method, and why each part is there

**Upstream is pinned at the exact commit this fork merged** (`95b8e33e1`). Comparing against a
moving `master` would fold upstream's own progress into our delta.

**Arms alternate in palindrome order** (`fork mainline mainline fork`) inside each round, so linear
clock drift cancels between arms instead of being charged to whichever ran later. An earlier sweep
without this showed a baseline sliding 301 -> 273 t/s across six runs, which is larger than most of
the effects being measured.

**The first run of a set gets a boost clock**, worth 15 to 20 percent on this APU, so every process
opens with a throwaway warmup that is not recorded.

**Deviation is captured twice.** `stddev_ts` is llama-bench's spread across its internal
repetitions; the spread across rounds comes from multiple samples per cell. `samples_ts` keeps the
raw per-run values, so any statistic can be recomputed later without re-running.

**`summarize.py` marks anything under 2 sigma as noise** rather than reporting it as a result.

**Flash attention is forced on** to match `models.ini`, which sets `flash-attn = on`. Benchmarks
that leave it at `auto` are not measuring the deployed configuration.

**Ubatch is 512**, the llama.cpp default and what the servers run. This matters twice over: at
depth 65536, ub 2048 hangs the compute ring on gfx1151 (see `WORKLOG.local.md`), and even where it
completes, a larger ubatch is *slower* at depth - 61.3 t/s at ub 512 against 48.4 at ub 1024. The
depth-0 ordering is the opposite, which is how the wrong value got picked first. Ubatch is recorded
per row; deep and shallow rows are not comparable across different ubatch values.

**A failing cell does not abort the matrix.** Deep-context cells can fail for reasons that are not
a property of either build. Partial rows are kept and the sweep continues.

## Every row says which model and quant produced it

`results.jsonl` (depth sweep) carries llama-bench's own `model_filename`, `model_type` - which
includes the quant, e.g. `qwen35 27B Q4_K - Small` - `model_size` and `model_n_params`.

`results-spec.jsonl` carries a `models` object with both the target and the draft: filename,
architecture, size on disk, and the tensor-type histogram read from the GGUF rather than guessed
from the filename. It also records `spec_method` (`none`, `draft-mtp`, `draft-dflash`).

This matters more than it looks. Draft choice is a speed/acceptance tradeoff, not a ranking: the
0.96 GB FP4 sidecar beat the 1.92 GB z-lab Q8_0 on this hardware because the APU is bandwidth
bound. Rows that do not name the exact target/draft pairing cannot be compared, and an append-only
log accumulates pairings over time.

## Reading the numbers

Every generation figure needs its depth attached. These models declare 262144 context, and token
generation at depth is roughly a third of its depth-0 value - a bare "TG = 60" is not a claim. The
same applies to prefill, which falls faster: 282 -> 87 t/s between depth 0 and 32768 on
Qwen3.8-27B. That is why depth is an axis here rather than a footnote.

## Power profile and thermals

`power_dpm_level` and `power_dpm_state` are stamped on every row, because the profile moves
throughput by more than most of the effects measured here and can be changed mid-session.

Two figures are worth reporting separately, and conflating them would misdescribe the machine:

- **Sustained** - the profile this box runs on for normal work. Everything in the fork-vs-mainline
  and gate-ablation tables was collected here.
- **Peak** - `power_dpm_state = performance`, measured in a short burst. Observed at 79 C, 115 W and
  2900 MHz within minutes of starting, and this chassis cannot hold that all day. A number taken in
  the first minutes of a performance burst is a peak, not a rate anyone sustains.

Comparisons are safe *within* a profile. The fork-vs-mainline and gate figures are ratios with arms
interleaved inside one session, so they survive a profile change; absolute t/s does not.

## Hardware

Single Radeon 8060S (Strix Halo APU, gfx1151, RDNA 3.5), RADV / Mesa 26.0.8, 96 GB VRAM budget.
Run with nothing else on the GPU: co-resident models change the numbers, and `gpu_busy_percent`
reads 0 percent while they are resident, so check
`/sys/class/drm/card1/device/mem_info_vram_used` instead.
