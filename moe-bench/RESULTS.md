# MoE expert cache on a VRAM and PCIe constrained machine

Research log for the `moe-gpu-optimize` branch. Everything here is measured on one
machine with one model; read the Limitations section before trusting a number.

Research material, not an upstream submission. `src/` and `tools/` changes are
split per topic in the branch history; see "Upstream status" at the end.

Sections 1 to 11 are the original measurements, taken with the branch based on
upstream `78a8ae1`. The branch was later rebased onto `d854d98b`; section 12
re-runs them there and reports which numbers moved and why.

## 1. Setup

| | |
|---|---|
| GPU | NVIDIA RTX 4060 Ti, 8 GiB (8188 MiB), CUDA arch 89 |
| Link | PCIe 4.0 x8, confirmed current and max (`nvidia-smi --query-gpu=pcie.link.*`), about 14 GB/s usable |
| CPU / RAM | AMD Ryzen 7 9700X, 8 cores / 16 threads, 31.11 GiB |
| OS / toolchain | Windows 11, MSVC 19.29, CUDA 12 runtime |
| Baseline | upstream master `78a8ae1` built in `llama.cpp-src/build-base` |
| Branch | `moe-gpu-optimize`, based on `78a8ae1` here, rebased onto `d854d98b` in section 12 |
| Model | Qwen3.6-35B-A3B, `UD-Q4_K_XL`, 20.81 GiB |

Model shape that matters for the cache:

- 40 MoE layers, 256 experts each, 8 used per token
- 18.32 GiB of the 20.81 GiB is expert weights, so the experts, not the dense
  weights, are what cannot fit in 8 GiB of VRAM
- the mixed `UD` quantization splits the layers into **three cache groups**
  (36 layers at 1.812 MiB per expert, 1 at 2.195, 3 at 1.945), each with its own
  slot budget and its own LRU
- the experts of **one token across all layers are 586 MiB**. This is the scale
  that every result below is relative to, so cache sizes are also quoted as
  "slots per token" = cache slots / experts used by one token.

All runs use `-ngl 99 -ncmoe 40`: every expert stays in host memory, the dense
weights and the attention run on the GPU. That is the only configuration this
model fits in on 8 GiB.

## 2. What is already upstream

Checked before writing any code, so as not to redo existing work:

- **master already has a MoE expert cache.** `src/llama-moe-cache.cpp`,
  `--moe-cache-mib`. It keeps host-resident experts in a per-device SLRU; each
  layer has a slot map in host memory, and when the scheduler copies that slot
  map to the device the copy callback uploads the experts that are missing.
  Generation (ubatch <= 32) reads through the cache; bigger ubatches bypass it
  but may still read experts out of it.
- **In flight upstream:** #30848 small-cache protected-set ramp (open, by another
  contributor), #30966 run missed experts on CPU, #30594 profile prefilling and
  zero-upload inference, #29963 pipeline parallelism with host experts, #30205
  per-device cache sizes, #31139 / #31095 cache group splitting and allocation
  limits, #21067 tensor prefetch. #30112 (multi-GPU cache) is merged.
- **Closed on process, not merit:** #31121 (`-mcmoe` llama-bench axis) and
  #31122 (host expert page pinning) were closed by the automated checker for the
  PR template, for having two open PRs as a new contributor, and for an
  AI-written description. The code was never reviewed on technical grounds.

So the remaining unexplored lever was the **replacement policy at small cache
sizes**, which #30848 explicitly leaves alone ("performance may suffer").

## 3. Method

Timing on this machine is noisy (section 7), so the primary metrics are the two
deterministic ones: **cache hit rate** and **bytes uploaded over PCIe**. Both are
exactly reproducible for a fixed model, prompt and greedy decode.

Instrumentation added to the branch (all opt-in, default off):

- `LLAMA_MOE_CACHE_LOG=N` - log hit rate and upload volume every N slot-map copies
- `LLAMA_MOE_CACHE_TRACE=path` - CSV of the expert ids selected at each copy
- upload accounting for the large-batch path, which was previously unmeasured

Four traces were collected by greedy-decoding 256 tokens on four different
prompts (technical prose, C++ code, a history summary, a translation task);
`moe-bench/prompt*.txt`. Each is 10202 slot-map copies (40 layers x 255 tokens).

`moe-bench/moe_sim.py` is a faithful port of `moe_cache_slru`, including the
three-group budget split and the "never evict an expert this request needs" rule.
It was **validated bit-exact against llama.cpp**: at a 2048 MiB cache the
simulator reports hit rate 60.82%, 58867.3 MiB uploaded, 2044.1 MiB of VRAM;
llama.cpp reports 60.82%, 58867.26 MiB, 2044.11 MiB. That makes it safe to
compare 20+ policies at 10+ cache sizes without spending GPU time, and every
policy claim below was then re-checked on hardware.

## 4. Main finding: LRU falls off a cliff below one token

Expert selection during generation is a cyclic sweep: the model touches layer 0,
1, ... 39 and comes back to layer 0 for the next token. When the cache holds
fewer slots than one token needs, LRU evicts every expert shortly before the next
token asks for it again - the textbook LRU thrash.

Generation-only hit rate, all four traces, simulated and spot-checked on
hardware. The last column forces the frequency policy on at every size to show
why the switch has to exist; the branch default only uses it below 1.0.

| cache MiB | slots per token | upstream (SLRU + #30848 ramp) | **branch default** | PCIe traffic | frequency policy forced on |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.44 | **0.00%** | **15.77%** | **-15.7%** | 15.77% |
| 384 | 0.66 | **0.00%** | **23.90%** | **-23.8%** | 23.90% |
| 528 | 0.90 | **1.17%** | **29.98%** | **-29.1%** | 29.98% |
| 586 | 1.00 | 24.27% | **31.97%** | **-10.1%** | 31.97% |
| 645 | 1.10 | 35.50% | 35.50% (unchanged) | 0% | 33.84% |
| 1024 | 1.75 | 44.34% | 44.34% (unchanged) | 0% | 42.47% |
| 2048 | 3.49 | 61.65% | 61.65% (unchanged) | 0% | 58.71% |

Three things to read off this table:

1. Below 1.0 slots per token the existing cache returns **nothing at all** while
   still occupying its VRAM. A frequency-aware victim recovers 16 to 30 points of
   hit rate, which is 10 to 29% less traffic over PCIe.
2. Above 1.1 slots per token recency is **better** by 2 to 3 points, so the
   frequency policy must be switched off there - which the branch does.
3. The crossover sits at 1.0 slots per token, exactly the condition the upstream
   code already detects and warns about, so the switch adds no new tuning
   constant.

### Why the hot set exists

From `moe-bench/trace_stats.py`, averaged over layers:

| hottest k experts of a layer (of 256) | share of that layer's selections |
|---:|---:|
| 8 | 31 - 33% |
| 16 | 46 - 47% |
| 32 | 64 - 66% |
| 64 | 83 - 85% |

Routing is strongly skewed, so a small resident hot set pays off - but only if the
policy can find it. Recency cannot, under thrash; frequency can.

## 5. The change

`src/llama-moe-cache.cpp`, about 45 lines inside the existing `moe_cache_slru`:

- the victim becomes **the least used of 16 uniformly sampled slots** instead of
  the least recently used one, with the existing "cannot evict what this request
  needs" rule unchanged, and a fall back to the normal LRU walk if none of the
  sampled slots is evictable
- one `uint32_t` use counter per (layer, expert), 40 KiB for this model, halved
  every `8 * n_slots` fills so stale popularity is forgotten
- enabled **only** when `n_slots < experts of one token`, the case that already
  logs a warning. Above it, behaviour is byte-for-byte the old one.
- `LLAMA_MOE_CACHE_SAMPLE=N` overrides the sample count (`0` = upstream LRU),
  for A/B measurement

Sampling the whole cache matters. Sampling the LRU end instead recovers only 1.1
points at 256 MiB versus 14.2 for uniform sampling, because the LRU end is
precisely where the entries that are about to be reused sit.

### Verified on hardware

255 greedily decoded tokens on `moe-bench/prompt.txt`, same seed, measured with
the cache's own counters (`LLAMA_MOE_CACHE_SAMPLE=0` is the upstream policy,
default is the branch):

| cache MiB | upstream hit / uploaded | branch hit / uploaded | traffic | simulator said |
|---:|---:|---:|---:|---:|
| 384 | 0.00% / 145030.88 MiB | **20.94%** / 114725.94 MiB | **-20.9%** | 21.65% |
| 528 | 1.10% / 143324.84 MiB | **27.37%** / 105374.88 MiB | **-26.5%** | 27.87% |
| 586 | 21.73% / 113516.56 MiB | **29.50%** / 102296.00 MiB | **-9.9%** | 29.87% |

Every measured hit rate lands within 0.7 points of the simulator, the residual
being that the C++ uses a small LCG where the simulator uses Python's RNG, so the
two draw different slot samples. Output is unchanged (section 6).

### Throughput

`llama-bench -n 64 -p 0 -r 4`, mean of two order-balanced passes (upstream arm
first then branch arm, and the reverse), single process, page cache pre-warmed,
18 GiB of RAM free:

| cache MiB | slots per token | upstream tok/s | branch tok/s | change |
|---:|---:|---:|---:|---:|
| 384 | 0.66 | 7.35 | 8.60 | **+17.0%** |
| 528 | 0.90 | 8.54 | 11.35 | **+32.9%** |
| 586 | 1.00 | 11.34 | 13.28 | **+17.1%** |
| 645 | 1.10 | 13.63 | 13.68 | +0.4% |
| 1024 | 1.75 | 17.20 | 16.98 | -1.3% |
| 2048 | 3.49 | 24.16 | 24.25 | +0.4% |

The three rows at and above the crossover are the control: the branch is inert
there by construction, and they come out at -1.3% to +0.4%, which is both the
noise floor of this setup and the evidence that the order balancing removed the
page-cache bias that an unbalanced sweep had produced.

Below the crossover the gain tracks the traffic reduction, a little under it
because compute does not shrink with the bytes: 528 MiB saves 29.1% of traffic
for +32.9% throughput (a pure bandwidth model would predict +41%), and 384 MiB
gains less than its traffic saving because at that budget one group gets no cache
at all and falls back to the bypass path.

## 6. Correctness

The cache only moves expert weights into VRAM; the kernels and the values are the
same, so generation must be bit-identical. `moe-bench/check-correctness.sh`
greedily decodes the same prompt under every configuration and diffs the output.
All nine agree byte for byte, including the unmodified upstream build:

```
== diff against cache-off ==
  OK    cache1024-pinned      OK    cache528-auto
  OK    cache2048-auto        OK    cache528-lru
  OK    cache2048-lru         OK    upstream-2048
  OK    cache2048-s16         OK    upstream-off
```

That covers the new eviction policy on and off, above and below the crossover,
host page pinning, and upstream master with and without the cache. GGUF loading,
the quantization formats and the CUDA path are untouched.

## 7. Other results

### Host expert page pinning (#31122)

`cudaMemcpyAsync` from pageable memory is neither asynchronous nor full speed; it
goes through a driver staging buffer. Registering the mmap'd expert pages with
`cudaHostRegister` removes that staging copy.

Dividing the known per-token upload volume by the measured time per token gives
the effective rate of the pageable path:

| cache MiB | hit rate | MiB uploaded per token | tok/s | effective GB/s |
|---:|---:|---:|---:|---:|
| 384 | 0.00% | 586.2 | 7.35 | 4.21 |
| 528 | 1.17% | 579.3 | 8.54 | 4.83 |
| 586 | 24.27% | 443.9 | 11.34 | 4.92 |
| 645 | 35.50% | 378.1 | 13.63 | 5.03 |
| 1024 | 44.34% | 326.3 | 17.20 | 5.48 |
| 2048 | 61.65% | 224.8 | 24.16 | 5.30 |

So **4.2 - 5.5 GB/s**, against roughly 14 GB/s of usable PCIe 4.0 x8 bandwidth.
The rate is near-constant across a 2.6x range of transferred bytes, which says
generation here is essentially **bandwidth-bound on the host-to-device copy** -
that is why cutting bytes converts almost linearly into tokens per second. It
also says the pageable path leaves most of the link unused, which is what the
pinning change targets. (The figure is derived, not a bus measurement: compute
overlaps the copies, so it is a lower bound on the true rate. The 384 MiB row is
lowest because at that budget one group gets no cache and takes the bypass path.)

It is capped at a quarter of system RAM by default because pinned pages compete
with the OS; `LLAMA_MOE_PIN_MIB` overrides the cap and any range the driver
refuses simply stays pageable. Sweeping the cap at a 1024 MiB cache, where the
eviction change is inert, order-balanced over two passes:

| pin cap | actually pinned | share of the 18760 MiB | tok/s | vs pageable |
|---|---:|---:|---:|---:|
| off (pageable mmap) | 0 | 0% | 9.32 | - |
| 2048 MiB | 1954 MiB | 10% | 8.47 | **-9%** |
| 4096 MiB | 3986 MiB | 21% | 9.70 | +4% |
| 8192 MiB | 8162 MiB | 44% | 13.61 | **+46%** |
| 16384 MiB | 15266 MiB | 81% | 17.53 | **+88%** |

Read this carefully, because the headline number is partly an artifact of this
machine: pinning does remove the staging copy, but it also **locks those pages
resident**, so at 81% coverage it additionally eliminates the mmap page-cache
misses that this 31 GiB machine suffers against an 18.32 GiB working set. The
+88% is therefore the sum of two effects and would be smaller on a host with
enough RAM to keep the model cached anyway. The default RAM/4 cap lands near the
44% row. Low coverage is mildly **counterproductive** (-9% at 10%): it spends RAM
without covering enough of the working set to pay for itself.

Run-to-run spread here is wide (+-25 to 45%), but the 81% configuration measured
17.59 and 17.47 across the two passes, so the top of the trend is solid.

`--load-mode none` is the existing upstream alternative - it puts all experts in a
pinned host buffer, at the cost of pinning the full 18.3 GiB and reading the whole
file at startup. Pinning changes no output (section 6).

### Prompt processing is uncached by design

A 54-token prompt pushed **5696.85 MiB** over PCIe with zero cache reuse. Ubatches
above `max_batch = 32` bypass the slot map, and nothing populates the cache from
the prefill. This is deliberate upstream (large batches use most experts of a
layer and would evict the generation hot set), and it is what #30594 attacks.

## 8. Approaches tried and rejected

Recorded because the negative results were the expensive part.

- **Prefetch the previous token's experts.** Token-to-token overlap within a layer
  is only **31 - 37%** (and 19 - 23% at a lag of 8 tokens), well below what the hot
  set already delivers, so a previous-token prefetch would spend bandwidth to do
  worse than the cache it feeds. Dropped before implementing.
- **Frequency sampling from the LRU end.** 1.1% / 5.4% at 256 / 512 MiB, versus
  14.2% / 25.6% for uniform sampling. Wrong place to look.
- **Pure random eviction.** Cheapest possible fix for thrash and it does help
  (3.2% / 14.0% at 256 / 512 MiB) but it is far behind frequency sampling and it
  costs 8 to 10 points at large caches.
- **S3-FIFO.** Marginally the best at large caches (86.94% vs 86.22% at 6144 MiB)
  but erratic at the small sizes that matter here, and it needs a new data
  structure for a fraction of a point.
- **Full LFU everywhere.** Wins below the crossover but loses up to 5 points above
  it; an exact min-frequency structure is also more machinery than sampling.
- **No counter aging.** Loses 6 to 9 points over a 1024-token four-topic session,
  and at 0.99 slots per token it is actually *worse* than upstream. Aging is not
  optional.
- **Timing the policy with `llama-completion -v`.** The per-token debug logging is
  large enough to hide the effect; it showed 6.61 vs 6.53 tok/s where llama-bench
  on the same configurations shows a clear difference. Use `llama-bench`.

Two measurement traps worth writing down, because both produced confident
nonsense before being caught:

- **Two benchmark processes at once.** Each instance maps the whole 20.81 GiB
  model, so two of them exhaust 31 GiB of RAM, the run becomes disk bound, and
  every cache size returns the same ~6 tok/s - the cache appears to do nothing.
  Check for stray processes and free RAM before believing a flat sweep.
- **Arms run in sequence rather than interleaved.** The second arm inherits a
  warmer page cache. Run in sequence, the frequency policy appeared to gain 4 to
  12% even at cache sizes where its hit rate is *lower* by construction, which is
  how the bias was spotted. Sweeps must be order-balanced.

### Remaining headroom

Belady (offline optimal, same slot count) for reference:

| cache MiB | upstream | this branch | Belady |
|---:|---:|---:|---:|
| 256 | 0.00% | 15.77% | 31.86% |
| 384 | 0.00% | 23.90% | 43.35% |
| 528 | 1.17% | 29.98% | 51.53% |
| 586 | 24.27% | 31.97% | 53.93% |
| 1024 | 44.34% | 44.34% | 64.74% |
| 2048 | 61.65% | 61.65% | 77.30% |

A large gap remains everywhere, which is the space a profile-driven resident hot
set (#30594) is aiming at.

## 9. Limitations

- **One model, one GPU.** The policy result is validated on four traces of a
  single MoE model (256 experts, 8 used). Models with fewer experts per layer, or
  a flatter routing distribution, will show a smaller effect. Nothing here is
  tested on a second GPU or on multi-device splits.
- **Timing is sensitive to host memory pressure.** The 18.32 GiB host expert
  working set only fits in this machine's 31 GiB when little else is running.
  With a desktop session and other applications loaded, free RAM fell to under 4
  GiB, mmap page-cache misses added disk reads (the array does about 3.7 GB/s
  sequential) on top of the PCIe transfers, and `tg64` spread reached +-25% with
  the mean moving by 30% between sweeps. The numbers reported here were taken
  with a single process, the page cache pre-warmed and about 18 GiB free, which
  brings the spread down to +-5 to 8%; even then the absolute tok/s are not
  comparable against a differently loaded machine. This is why hit rate and
  uploaded bytes, which are exact, carry the claims, and throughput is reported
  as corroboration with its control rows shown.
- **The pinning gain is confounded with page residency** on this machine, as
  section 7 explains: locking the pages both speeds up the DMA and stops the
  mmap page-cache misses. Separating the two needs a host with enough RAM to keep
  the whole model cached without pinning, which this one does not have.
- **`--load-mode none` was not benchmarked.** With 18.3 GiB of experts and 31 GiB
  of RAM it is at the edge of what this machine can pin, so the comparison
  against capped in-place pinning is still open.
- **PCIe width was not varied.** The link is fixed at x8; the constrained-bandwidth
  axis was approximated by varying cache size, which changes bytes transferred
  rather than link speed.
- Run order matters for throughput because of the page cache; the sweeps in
  section 10 are order-balanced for that reason.

## 10. Reproducing

```sh
# build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench llama-completion

# throughput, telemetry and pinning sweeps
./moe-bench/run-bench.sh /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

# the MoE cache must not change what the model generates
./moe-bench/check-correctness.sh /path/to/model.gguf [baseline_build/bin]

# offline policy exploration from a trace, no GPU needed
LLAMA_MOE_CACHE_TRACE=trace.csv ./build/bin/llama-completion -m model.gguf \
    -ngl 99 -ncmoe 40 --moe-cache-mib 2048 -c 2048 -n 256 \
    -f moe-bench/prompt.txt --temp 0 --seed 1 -no-cnv -v
python moe-bench/moe_sim.py trace.csv \
    --layout moe-bench/layout-qwen35moe-35b-a3b-ud-q4kxl.json \
    --sizes 384,528,645,1024,2048 --policies slru-ramp,ramp-s16-age8,lfu,belady
python moe-bench/trace_stats.py trace.csv --layout moe-bench/layout-*.json
```

`layout-*.json` describes the cache groups of one model; regenerate it for
another model from the GGUF expert tensor types and shapes.

## 11. Upstream status

The branch deliberately keeps the topics separate:

| commit | upstream status |
|---|---|
| `-mcmoe` llama-bench axis | was #31121, closed on process grounds |
| host expert page pinning | was #31122, closed on process grounds |
| SLRU protected-set ramp | **someone else's open PR #30848**, applied locally only so it could be measured; must not be resubmitted. The `LLAMA_MOE_CACHE_SLRU_RAMP` toggle on top is local |
| telemetry and expert tracing | measurement scaffolding, not intended for upstream as is |
| sampled least-used eviction | new; would be a follow-up to #30848, and belongs in that discussion first |

Before any resubmission: one PR at a time, the PR template filled in, the
description and commit messages written by hand, and the AI usage disclosed. Per
`AGENTS.md` this repository forbids automated PR submission, so submitting is a
manual step.

## 12. Rebase verification on `d854d98b`

The branch was rebased from `78a8ae1` onto upstream master `d854d98b`, 21 commits
later. No conflicts: only `src/llama-context.cpp` is touched by both sides, and
`git range-diff` reports all 7 commits as unchanged. The build is clean, 563 of
563 targets, with no warnings in any file this branch touches.

Everything in this section was re-measured on the rebased branch. Sections 1 to
11 are left as the `78a8ae1` record.

### Upstream changed the routing, so the exact figures moved

The deterministic numbers no longer reproduce to the digit, and the cause is not
in this branch. With `--moe-cache-mib 0`, where the cache is never constructed
and none of this branch's code runs, the two bases generate different text from
the same greedy decode of the same prompt:

```
<     *   Scenario
>     *   Scenario: GPU reads model weights stored in system RAM.
```

So upstream's own commits perturbed the arithmetic enough to change the router's
top-8 at some positions. This is consistent with `c49edf096` (fused FFN gate/up
matmuls and GLU in MMQ) and `a55c16e4e` (fused MRoPE and set-rows) altering
reduction order and therefore rounding, but intermediate commits were not built,
so the attribution is not isolated. Prefill traffic moved the same way, 5696.85
to 5707.74 MiB.

An expert-selection change does not move the uploaded byte count at a 0% hit
rate: every selected expert is uploaded, 8 per layer per token, and expert size
depends on the layer rather than on which expert. That is why the 384 MiB
upstream row below is unchanged to the last digit.

### Hit rate and uploaded bytes

255 greedily decoded tokens on `moe-bench/prompt.txt`, from the cache's own
counters, against the section 5 table:

| cache MiB | upstream hit / uploaded | branch hit / uploaded | traffic | section 5 branch hit | delta |
|---:|---:|---:|---:|---:|---:|
| 384 | 0.00% / 145030.88 MiB | **21.32%** / 114174.28 MiB | **-21.3%** | 20.94% | +0.38 |
| 528 | 1.04% / 143422.10 MiB | **27.53%** / 105147.00 MiB | **-26.7%** | 27.37% | +0.16 |
| 586 | 20.93% / 114692.98 MiB | **29.64%** / 102097.78 MiB | **-11.0%** | 29.50% | +0.14 |

Every hit rate reproduces within 0.8 points and every conclusion of section 4
holds: upstream returns almost nothing below one token's experts, the sampled
victim recovers 8.7 to 26.5 points of hit rate, and traffic falls by 11 to 27%.

Above the crossover the two arms came out **byte-identical** - same hits, same
misses, same uploaded bytes to the hundredth of a MiB:

| cache MiB | slots per token | both arms | uploaded |
|---:|---:|---:|---:|
| 645 | 1.10 | 29.96% | 101592.12 MiB |
| 1024 | 1.75 | 41.36% | 87882.56 MiB |
| 2048 | 3.49 | 61.38% | 58033.62 MiB |

Section 4 claimed this inertness from the simulator; it is now measured on
hardware.

### Throughput

`llama-bench -n 64 -p 0 -r 4`, mean of two order-balanced passes, this time with
**16.1 GiB of host RAM free** instead of the roughly 18 GiB of section 5:

| cache MiB | upstream tok/s | branch tok/s | change | section 5 change |
|---:|---:|---:|---:|---:|
| 384 | 5.71 | 7.54 | +32.2% | +17.0% |
| 528 | 7.24 | 10.65 | +47.1% | +32.9% |
| 586 | 10.07 | 12.67 | +25.8% | +17.1% |
| 645 | 12.15 | 12.24 | +0.7% | +0.4% |
| 1024 | 15.02 | 15.57 | +3.7% | -1.3% |
| 2048 | 22.10 | 21.24 | -3.9% | +0.4% |

**The larger gains are not an improvement.** Absolute throughput is down 10 to
15% everywhere, and the host expert working set is 18.32 GiB against 16.1 GiB
free, so mmap page-cache misses penalise whichever arm reads more bytes - which
is the upstream arm. Host memory pressure inflates the ratio. The gain below the
crossover is real; its size is a property of how loaded the machine is.

The control rows put a number on that. The table above this one proves the two
arms are byte-identical at 645, 1024 and 2048 MiB, so those three rows measure
nothing but noise, and they span +3.7% to -3.9%. That is the noise floor of this
session, wider than the +-1.3% of section 5 at 18 GiB free. The sub-crossover
gains sit far outside it, but carry at least that uncertainty.

### Effective bandwidth of the pageable path

Section 7 derived this from the upstream arm's known upload volume and time per
token; recomputed from the measurements above, using measured rather than
theoretical bytes per token:

| cache MiB | MiB uploaded per token | tok/s | GB/s | section 7 GB/s |
|---:|---:|---:|---:|---:|
| 384 | 568.7 | 5.71 | 3.40 | 4.21 |
| 528 | 562.4 | 7.24 | 4.27 | 4.83 |
| 586 | 449.8 | 10.07 | 4.75 | 4.92 |
| 645 | 398.4 | 12.15 | 5.08 | 5.03 |
| 1024 | 344.6 | 15.02 | 5.43 | 5.48 |
| 2048 | 227.6 | 22.10 | 5.27 | 5.30 |

The three largest caches agree closely with section 7. The two smallest are
lower, which is the same host memory pressure: those are the configurations that
read the most bytes through mmap. The conclusion is unchanged - the pageable path
delivers a fraction of the roughly 14 GB/s the link can carry, and generation
stays bandwidth-bound on the host-to-device copy.

### Correctness

`moe-bench/check-correctness.sh` with no baseline build, so 6 configurations
against cache-disabled rather than the 8 of section 6:

```
== diff against cache-off ==
  OK    cache528-lru      OK    cache2048-lru     OK    cache1024-pinned
  OK    cache528-auto     OK    cache2048-auto    OK    cache2048-s16
```

All byte-identical: the policy on and off, above and below the crossover, and
host page pinning. The `upstream-off` and `upstream-2048` rows of section 6 need
a build of upstream master at this base, which was not made.

### Not re-run

- **The simulated tables of sections 4 and 8.** They average four traces
  collected on `78a8ae1`; the traces would have to be collected again on this
  base, and the simulator re-validated against it, before those numbers can be
  quoted for `d854d98b`.
- **The pinning sweep of section 7.** It was already the noisiest measurement
  here (+-25 to 45%) and explicitly confounded with page residency; with less
  free RAM than before, re-running it would be dominated by the confound rather
  than informative about the rebase.
- **Prompt processing.** Only the one figure above, prefill traffic, was
  re-checked.
