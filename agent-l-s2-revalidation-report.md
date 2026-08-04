# agent-l s2 LLAMA_KV_LAZY_CLEAR re-validation

Date: 2026-08-03
Branch: scratch/s2-revalidation/agent-l (5fa932bc0 + s2 cherry-pick)
Baseline: evolve-review/s2-revalidation/agent-l (df3e9c6cd)
Cherry-pick: 6f3a7495fad0c7fb3616a5bcdf34f9591eb93272 (s2: gen1 lazy KV clear)
Build: cmake -G Ninja -B build-agent-l -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
Binary: build-agent-l/bin/llama-bench
Machine: Apple M1, MTLGPUFamilyApple7, has unified memory, recommendedMaxWorkingSetSize 12713 MiB

## Setup

- Model: /Volumes/Julian T7/models/unsloth-nonaut/gemma-4-12b-it-Q5_K_M.gguf (7.82 GiB, 11.91 B params, Q5_K - Medium)
- llama-bench: -p 512 -n 32 -ngl 99
- n_ctx (effective for pp512/tg32): 512; n_ctx_train: 262144 -> KV buffer is sized for the full 262144-token context (this is the buffer the eager clear touches)
- Per setting: 5 runs of /usr/bin/time -l llama-bench
- Two env-var settings: LLAMA_KV_LAZY_CLEAR=0 (default, eager clear) and =1 (lazy clear)

## Raw peak RSS (bytes), /usr/bin/time -l "maximum resident set size"

LLAMA_KV_LAZY_CLEAR=0 (eager clear, default):
- run 1: 8748548096  (8.148 GiB)
- run 2: 8749547520  (8.149 GiB)
- run 3: 8777826304  (8.175 GiB)
- run 4: 8777908224  (8.175 GiB)
- run 5: 8778006528  (8.175 GiB)

LLAMA_KV_LAZY_CLEAR=1 (lazy clear, opt-in):
- run 1: 8601976832  (8.011 GiB)
- run 2: 8601780224  (8.011 GiB)
- run 3: 8602648576  (8.012 GiB)
- run 4: 8601550848  (8.011 GiB)
- run 5: 8601714688  (8.011 GiB)

## Medians (5-run median, resistant to outliers)

- RSS OFF: 8777826304 bytes (8371.19 MiB)
- RSS ON:  8601780224 bytes (8203.30 MiB)
- delta_rss = -176046080 bytes = -167.89 MiB (RSS drops when lazy clear enabled)
- pp t/s OFF: 64.30
- pp t/s ON:  66.06
- delta_pp  = +1.76 t/s (+2.74%)
- tg t/s OFF: 6.23
- tg t/s ON:  6.47
- delta_tg  = +0.24 t/s (+3.85%)

## Spread (stdev over 5 runs)

- RSS  OFF stdev = 15.08 MiB (min 8343.27 / max 8371.36)
- RSS  ON  stdev =  0.41 MiB (min 8203.08 / max 8204.12)
- pp t/s  OFF stdev = 4.85 (min 53.90 / max 66.11)   <- large, cold-cache outliers
- pp t/s  ON  stdev = 0.84 (min 64.27 / max 66.36)
- tg t/s  OFF stdev = 0.75 (min  4.72 / max  6.51)   <- large, cold-cache outliers
- tg t/s  ON  stdev = 0.03 (min  6.43 / max  6.50)

The OFF group was run first, so runs 1-2 were cold-cache (tg 4.72 and 5.82, both
well below the ~6.5 steady-state). The ON group ran after, all warm.

Apples-to-apples (OFF 3-5 warm vs ON 1-5 all warm):
- delta_rss = -167.97 MiB (same conclusion)
- delta_pp  = +1.55 t/s (+2.40%)
- delta_tg  = -0.03 t/s (-0.46%) (essentially zero)

## Verdict

REGRESSION REFUTED -> WAVE-3 NON-REPRO WAS STALE.

delta_rss = -167.89 MiB is more than 3x the wave-3 noise floor (50 MiB)
and is in the OPPOSITE direction from the wave-3 "+0.59 GB regression" claim.
The s2 commit's original claim of ~178 MiB win reproduces cleanly on
current main. t/s impact is neutral-to-slightly-positive; well within
the 20% threshold.

## Anomalies

- OFF runs 1-2 had cold-cache tg t/s (4.72, 5.82) far below steady-state
  (~6.5). The metal library reloads between runs in verbose mode and
  pp/tg can spike on the first 1-2 invocations. The median of 5 absorbs
  this; the warm-only subset (OFF 3-5) gives the same answer.
- ON runs were tighter (stdev 0.41 MiB RSS, 0.03 tg t/s) than OFF
  (stdev 15.08 MiB RSS, 0.75 tg t/s). The lazy clear path appears to be
  more deterministic - likely because no memset memset touches the full
  n_ctx_train-sized buffer.

## Recommendation

Promote s2 to main. Safety rails:
- Keep LLAMA_KV_LAZY_CLEAR opt-in via env var (already the case in the patch).
- Add a comment in the commit (or follow-up) that this is safe because
  KV cells guard all reads (flash/paged/naive only read cells the cell
  tracker has marked written).
- Optional: add a one-time logit_probe correctness check on first
  enable, as a startup sanity test (cells-guard claim should hold).
- Do NOT flip the default until at least one additional model + quant
  has been re-validated (see Optional sanity check below).

## Optional sanity check (not run in this validation, but easy follow-up)

gemma-4-12B-it-DFlash-Q5_K_M (525 MB) at
/Volumes/Julian T7/models/drafters/gemma-4-12B-it-DFlash-Q5_K_M.gguf
would confirm the s2 finding is model-independent. The DFlash variant
has a smaller drafter alongside the main model, so the KV buffer shape
differs slightly; a clean +/- result there strengthens the case.
