# A770 campaign, 2026-10-06 03:14 to 2026-10-07 11:23 (b12327.c4b9d6430, xe, oneAPI 2026.1)

State at the end: Ornith service active on b12327 with production flags unchanged, ccs_mode=1 (was 2 before the campaign), sysfs knobs
back at 600/2400, base profile, timeslice 1000. The campaign changed no unit, drop-in or package. Local commit 7d183b5 (PKGBUILD without
the DG2 patch, .SRCINFO, CLAUDE.md) is on feature/raudbjorn-fork-arc-tuning, unpushed.

Files here: ALL-TABLES.md (per-arm tables for phases 1-6), COMPLETE-MATRIX.md/.csv (phase 1 per-run with env), issues/ (two drafts, not filed),
index.tsv (per-cell idle, load, foreign CPU, timestamps), master.log, monitor/ (kernel log stream, FAULTS.txt, 5 s series), klog-*.txt.
Raw per section: <section>/*.md|.err|.freq|.fdinfo|.vmstat. Scripts: ../matrix-lib.sh, matrix2..6-run.sh, matrix2-spec.py.

"Clean" below means other processes used <=18% of host CPU (own benchmark CPU subtracted), logged per cell as foreignpct; this is a shared
desktop host (Plex analysis job, browser, other sessions), so the floor is 1-15% depending on the hour. Phase 6 reruns are clean in that sense.

## Findings

1. ccs_mode 1 > 2 > 4. Ornith production, paired in one session (4 rounds): mode 2 -12.5% pp / -13.2% tg / -17.5% tg@8k, mode 4 -35%.
   8B in VRAM: mode 2 -29% pp / -18% tg, mode 4 -62% / -52%. Two concurrent processes gain nothing from 2 or 4 (aggregate at mode 1:
   1525 pp / 59 tg vs 1634 / 62 single; threads -t 6 equals -t 12 on the 8B). Real text (llama-completion, 560-token prompt): 34.9-38.8 t/s at
   mode 1 vs 34.2-35.2 at mode 2 (about +7%); bench tg64 (49 t/s) uses random tokens and overstates Ornith decode by about a third.
   fdinfo: capacity matches the mode (2, 4); drm-cycles-ccs for one process is about 452M at mode 2 and 453M at mode 4 (read at ~25 s), i.e. the
   busy-cycle count does not scale with capacity, consistent with one engine in use; not conclusive, no mechanism proven.
2. MKL flash attention default flips sign with head dim (off vs on, pp512 at depth; one or two rounds):
   d=128: Llama-3.1-8B +32% @2k, +57% @8k, +58% @16k, +63% @32k (clean rerun: 63.5 vs 103.7 t/s); Qwen2.5-7B (7:1) +4%, -7% @4k, +30% @8k, +38% @16k.
   d=256: Ornith -10% @2k, -24% @8k, -34% @16k, -43% @32k (clean: 208 vs 118 t/s), -53% @64k; Qwen3.5-9B -15% to -31%. Depth 0 unchanged.
   Q8_0 Ornith (clean, 3 rounds): MKL off -21.7% pp@8k, same sign as Q4_K_M.
   Numerics (perplexity, ctx 8192): on minus off: Llama-8B -0.007, Qwen2.5-7B +0.07, Qwen3.5-9B +0.008, Ornith inside its 0.01 noise.
   Oracle [4c] covers only d=128 GQA 4:1 f16/q8_0 and fails by design with MKL_FA=0; no oracle for d=256, 7:1, or the fallback at n_kv>=1024.
   A head-dim gate for the default looks warranted: draft in issues/01 (not filed).
3. MAX_WG_PER_CU on Ornith Q4_K_M, clean, 3 rotated rounds, tg64 vs default 16: 24 gives +5.7% @16k, +7.9% @32k, +11.5% @64k; 32 gives +5.2%,
   +10.3%, +18.0%; 64 gives +6.9%, +3.3%, +6.9%; d=0 unchanged (-0.3..-0.5%). Spread between rounds 0.0-0.4 t/s (64 @32k: 0.9). Q8_0 Ornith: 32 gives
   +5.9% tg @8k. On the 8B (d=128, in VRAM) it does nothing (+/-1.7%, two clean rounds). 32 is the best single value for a ctx 131072 service;
   APPLIED 2026-10-07 as drop-in wg32.conf on the Ornith unit; output check: 8B perplexity bit-identical 12.7626 in 4 runs, Ornith long-context greedy within noise, real-source decode +5% @27.5k / +20% @61k; live chat request answered coherently.
4. Ornith env sweep (mode 1, 3 rounds, 8k columns): ILCL0 -23% tg, OPT0 -20%, FUSION0 -11%, PINNED0 -8%, ESIMD0 -6%, WG8 -7% tg@8k, MKL0 -24% pp@8k;
   GRAPH0, Q8_KV_QUANTS_FIRST=0, Q8_GQA_TILE=1, VECSTD, ASYNC0, DMMV1 within about +/-2%; GRF0 -2% pp. Unset immediate command lists behaves as =1.
   8B results do not transfer (GRAPH0 -6.7% there, nothing on Ornith).
5. Host: -t 12 near best; CCD0 (V-cache) -t 6 +2.4% tg; CCD1 -3..-6%; -t 24 -10.5%; -ub 256 -33% pp; -ub 2048 -4% at 8k. -fitc 131072 = 32768 in bench.
6. Frequency: pin 2400 = default; pin 2000 -7% tg; pin 1500 -19%. power_saving verified applied (sysfs shows it active), no effect.
7. Speculative decoding on Ornith (production flags, 10 varied local-file prompts, 3 rotated launches per config, foreign CPU <=1%, launch-to-launch
   spread of the baseline 0.4 t/s): none 45.4 (t0) / 45.3 (t0.6); ngram-mod -0.1% (t0) / -6.2% (t0.6); ngram-simple -3.3% / -2.8%.
   Per prompt the spread is large: ngram-mod +40% (code_comment), +18% (csv_to_md), -36% (summarize), -25% (repeat_code); ngram-simple +43% (repeat_code),
   -35% (csv_to_md), -26% (sh_to_py). No spec-type beats no speculation on this mix; production's ngram-mod is neutral at t0 and -6% at the unit's
   sampling. Earlier readings (-26% in phase 2, -1.5% in phase 4) came from runs I cannot vouch for and are superseded. Verify-batch cost explains
   why gains need high acceptance: Ornith 4 tokens cost 3.9x one decode step, 8 tokens 5.6x (dense 8B: 2.3x, 5.2x); no fusion/graph knob changes it.
   My earlier ngram-simple recommendation is withdrawn.
8. Qwen4Exp (PR #90, installed): MTP n=6 16.8, n=3 17.6, n=6 + --spec-chain 4 17.4, n=6 @fit-target 3072 17.0 t/s vs 13.3 with no speculation.
9. Server aborts "Failed to allocate physical memory" in ggml_sycl_pool_vmm::alloc (phys.emplace) / UR_RESULT_ERROR_OUT_OF_RESOURCES, at --fit-target 1024:
   - Ornith + --prefetch-experts-slots 4 (+ ngram-mod): SYCL0 free after load 104.7 MiB vs 951 MiB without prefetch, i.e. the slots take about 850 MiB
     outside the fit budget; the same config at --fit-target 3072 runs; prefetch alone and ngram-mod alone run. Prefetch gives no speedup (44.7 vs 45.5 t/s).
   - Qwen4Exp + ngram-mod/-simple aborted at the first request with 953 MiB free; they run at --fit-target 2048 and 3072 (16.3-19.0 t/s). Cause not shown.
   - Ornith llama-perplexity -b 2048 (8 of 8 aborted); the retry with -fitt 3072 -b 512 ran (Ornith ppl 6.807-6.822 both MKL routes).
   - A stray llama-server I leaked (killed script) reproduced the abort on a plain Ornith ngram-mod launch with 947 MiB free after load. So a competing
     holder of VRAM is an unexcluded explanation for the earlier aborts I did not check for strays; Per-process VRAM was not logged.
   Drafts: issues/02 (not filed).

## Problems in my own harness (all corrected before the numbers above)
ssh swallowing loop stdin; setvars.sh failing under set -u; a fault counter that read 0 for everything; Q8 path pointing at an empty directory; a gate that
counted the benchmark's own CPU as foreign load; a leaked server that held VRAM across a restart. Per-cell fault evidence for 03:14-06:00 does not exist
(no kernel log survived); from 08:14 the monitors saw only the GT resets from my ccs_mode writes and no devcoredump.

## Not done / not proven
- fallback-route oracle and a d=256 / 7:1 oracle (needs fork test changes); output check with the other changed defaults (WG32 check done).
- Why mode 1 beats 2 (fdinfo is suggestive only); why Qwen2.5-7B favors MKL at 4k but not at 8k.
- Not filed: issues/01, issues/02. Not changed: the unit, ccs_mode persistence across reboot, PKGBUILD pkgrel. Hindsight DB corruption is the user's.
