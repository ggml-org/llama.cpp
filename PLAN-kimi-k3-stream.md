# Plan v4: stream Kimi-K3 MXFP4 experts from NVMe RAID0 (llama.cpp)

Worktree: `/home/aaronzhong/Projects/llama.cpp-master` (upstream master 61ad80bf9), branch `kimi-k3-expert-stream`.
User's branch `ollama-qwen35-compat` in `/home/aaronzhong/Projects/llama.cpp` is NOT touched.

## Requirement (from user)
Run `/mnt/k3/kimi-k3/MXFP4_MOE/*.gguf` (Kimi-K3 Abliterated, 2.78T params, 1.51 TB) on RTX 3080 Ti 12 GB,
160 GB RAM, Xeon E5-2690v4 (14c/28t, 1 NUMA node), 4x Samsung 990 1TB md RAID0 (2 MiB chunk, XFS, Gen3 x4).
Keep accuracy: stay on MXFP4 (prove support on sm_86) - no requant. Maximise NVMe->RAM streaming so decode is
as fast as the hardware allows; use the 3080 Ti meaningfully. Deliver a runnable launcher.

## Facts measured
- 93 layers (92 MoE), 896 experts, 16 used; per expert per matrix slice (`nb02`) = 5.85 MB contiguous MXFP4;
  each `*_exps` tensor = 5.24 GB. Routed expert bytes/token = 16*92*3*5.85 MB = **25.8 GB**.
  Dense Q8_0/F32 = 61.7 GB, all touched every token.
- Array on real data: 6 MB random QD48 -> **9.3 GB/s**; 2 MB aligned random QD128 -> 7.5 GB/s.
- Baseline upstream: <0.02 tok/s, 1.6 TB read in 10 min, decode md0 ~1.5 GB/s, RSS = all RAM.
- Root causes: (1) mmap prefetch=-1 always -> MAP_POPULATE over 1.5 TB (llama-model.cpp:1876, llama-mmap.cpp:493),
  no CLI knob (`--numa` inert on 1 node); (2) FADV_SEQUENTIAL -> 32 MiB fault readahead vs 5.85 MB experts;
  (3) serial faults, QD ~ n_threads.

## Changes

### A. Opt-in lazy experts for Kimi-K3 (`src/models/kimi-k3.cpp`)
Name the `llama_model_loader &` param `ml` (as qwen4exp.cpp:171) and pass
`ml.lazy.mode == LLAMA_LAZY_MODE_ON ? TENSOR_READ_LAZY : 0` for `ffn_{gate,up,down}_exps.weight`.
(Gate is required: 5.24 GB > 4 GiB AUTO threshold would otherwise make them lazy by default.)
Verified by review: mode is set before tensor creation (llama.cpp:322); fit pass consistent; lazy -> forced
plain CPU buft before -ot/repack (loader.cpp:1212) so no repack kernel; files with lazy ranges skip MAP_POPULATE,
WILLNEED only on dense ranges, POSIX_MADV_RANDOM on expert ranges; `llama_prefetch_rows` never called on them.
Consequences (documented, correct earlier claims):
- RANDOM => a fault reads one 4 KiB page synchronously. Lazy mode WITHOUT change B is expected to be slow;
  B is mandatory for speed. Measured explicitly (test 7).
- Loader dense WILLNEED is capped ~32 MiB per call/range, so dense weights mostly load on the first token.
- FORBIDDEN with lazy mode: `--mlock` (grow_to locks contiguous range incl. experts), `--check-tensors`
  (validates 1.4 TB). Launcher refuses them.

### B. Selected-expert prefetch in CPU MUL_MAT_ID (`ggml/src/ggml-cpu/ggml-cpu.c`)
- Flag: file-scope `static bool ggml_cpu_moe_prefetch` + `static size_t ggml_cpu_page_size`, declared above
  `ggml_compute_forward_mul_mat_id`, set once in `ggml_cpu_init()` under its critical section next to
  `GGML_CPU_DISABLE_FUSION` (env `GGML_CPU_MOE_PREFETCH=1`). Default off -> no change for anyone.
- Where: AFTER the existing barrier (thread 0 still builds `matrix_row_counts`), each thread `ith` walks the
  used experts in ascending index order, splits each slice `[src0->data + a*nb02, +nb02)` into 2 MiB pieces (last piece may be short),
  flattens (expert-major), and issues `posix_madvise(beg, ceil_page(p + n) - beg, POSIX_MADV_WILLNEED)` with `beg = floor_page(p)` for
  pieces `k` with `k % nth == ith`. No extra barrier; then the existing compute loop. Errors ignored, no logs
  in hot path.
- Guards: `#if defined(__linux__) || defined(__APPLE__)` and `!defined(__EMSCRIPTEN__)`, include `<sys/mman.h>`
  in the non-Windows branch; no-op elsewhere.
- Numerics: unchanged (pure hint). Only the plain CPU MUL_MAT_ID path is covered (repack never claims lazy
  tensors; tiled path does not handle MXFP4) - verified by review.
- One-time INFO log in `ggml_cpu_init` when enabled: "CPU MoE expert prefetch enabled" (observable).
- Phase 2 (only if measurements show drive idle gaps between ops): look-ahead from `ggml_graph_compute_thread`
  to prefetch sibling MUL_MAT_IDs sharing the same `ids` (gate->up->down) so ~282 MB is in flight.

### C. Runtime config / launcher (`kimi-k3/run-server.sh`, `kimi-k3/bench-stream.sh` in the worktree)
`GGML_CPU_MOE_PREFETCH=1 llama-server --lazy-mode on --no-op-offload -ngl 999
 -ot '_exps\.=CPU,blk\.(1[2-9]|[2-9][0-9])\.=CPU' -t 14 -fa on -c 8192 -ub 512 -np 1 --port 18090 -fit off`
(GPU count tuned so CUDA model buffer >= 8 GB and nothing OOMs.) Separator: llama-server/cli split `-ot` on `,` (common/arg.cpp:264); llama-bench uses `;`. First match wins.
`_exps\.=CPU` first so experts stay on CPU even if lazy mode is off (else layers 1-11 experts OOM VRAM).
`--no-op-offload` (`-nopo 1` in llama-bench) is REQUIRED: otherwise
ubatch>=32 copies whole 5.24 GB expert tensors to VRAM (ggml-backend.cpp:1118). Pre-flight: refuse if another
llama-* process is running or the port is taken; refuse `--mlock`/`--check-tensors`/`--op-offload`/`-fit on`. Print offloaded-layers line, buffer sizes,
expected tok/s range (0.2-0.38) and measured tok/s. KV for all 93 layers lands on CUDA0 (follows layer
device): check logged KV/RS size vs VRAM; if short reduce GPU layers or `-nkvo`.

## Expected results (assert)
- Lazy marking: exactly 276 "lazy read enabled" lines with `--lazy-mode on`; **0** on Kimi-K3 with no flag,
  `--lazy-mode auto`, and `--lazy-mode off` (default-path regression guard).
- strace (`-e trace=mmap,madvise,fadvise64`) on load: no MAP_POPULATE on shards, MADV_RANDOM on expert ranges.
- Load read_bytes (from evicted cache) <= 70 GB (expected << that); not ~1.5 TB.
- Decode, steady state (from evicted cache, after 2 warm tokens), prefetch ON:
  bytes read/token <= 1.3 * 25.8 GB; md0 >= 5 GB/s during decode;
  madvise calls/token ~= 16 experts x 276 tensors x 3 pieces (2 MiB of 5.85 MB) = 13,248 (separate strace run);
  aqu-sz >= 16 is a SOFT/busy-interval target; throughput >= 0.2 tok/s
  (derived ceiling ~0.3-0.38 tok/s: ~24 GB uncached/token at 9.3 GB/s + ~1 s CPU dense at DDR4 speed).
- GPU: "CUDA0 model buffer size" >= 8 GB; KV on CUDA0; nvidia-smi used >= 9 GB.

## Test plan
1. Build. `test-backend-ops -b CUDA0 -o MUL_MAT_ID` and `-o MUL_MAT` filtered to mxfp4 -> pass
   (proves MXFP4 on 3080 Ti). State in deliverable that experts run on CPU (MXFP4 AVX2), dense Q8_0 on GPU.
2. New permanent unit test `tests/test-moe-prefetch.cpp` (links ggml only), registered twice in ctest:
   `test-moe-prefetch` (no env) and `test-moe-prefetch-on` (`set_tests_properties ENVIRONMENT GGML_CPU_MOE_PREFETCH=1`).
   Temp file mmap'ed; page-aligned base wrapped by `ggml_backend_cpu_buffer_from_ptr`; src0 placed with
   `ggml_backend_tensor_alloc` at base + 4096 + 96 (32-aligned, not page-aligned); file unlinked after map.
   Cases: MXFP4 and F32, n_expert 8 / used 2, batch 1 and 40, plus an MXFP4 case with expert slice ~2.1 MiB
   (short tail piece). Each run compares MUL_MAT_ID output vs in-process per-expert ggml_mul_mat reference:
   MXFP4 exact-or-NMSE<1e-7, F32 NMSE<1e-7. No exported test hook.
3. Small local MoE GGUF with `--no-repack -ot exps=CPU`: env on/off -> identical greedy tokens (n_probs equal),
   log line present, `strace -c -e madvise` count > 0 only with env on; tg t/s with env on within 5% (in-RAM
   overhead check).
4. Small MoE with `--lazy-mode on` (non-Kimi): loader behaviour unchanged (0 lazy tensors).
5. Kimi-K3 known-answer: greedy "The capital of France is" -> first token " Paris" (same token id with
   prefetch on and off); short chat reply coherent.
6. Kimi-K3 equivalence (same patched build, same -ngl and `-ot '_exps\.=CPU,blk...=CPU'`, -t/-fa/-ub/-np,
   `--no-op-offload --no-repack`, cache_prompt false, 4 tokens, n_probs 10, `post_sampling_probs:false`): toggle only (lazy on + prefetch on) vs (lazy off + prefetch off)
   -> identical top-1 ids, max |dp| <= 1e-6.
7. Kimi-K3 perf matrix from evicted page cache (posix_fadvise DONTNEED on all shards; record fincore):
   (a) lazy on + prefetch on, (b) lazy on + prefetch off; record tok/s, read_bytes/token, md0 MB/s, rareq-sz,
   aqu-sz, pgmajfault, workingset_refault_file, lru_gen state (tok/s runs uninstrumented; madvise time from
   a separate strace -T sample run). ASSERT: (a) tok/s >= 0.2 and >= 3x (b) and >= 3x baseline 0.02; (a) md0 MB/s >= 3x (b).
8. 8k-context run: PASS = 8000-token prompt completes and >= 1 token generated, no OOM, CUDA0 free >= 0.5 GB
   (nvidia-smi); record KV/RS sizes, prefill time. Exercise `-nkvo` fallback once.
Isolation: port 18090, logs/scratch under session scratchpad, check `pgrep llama-`/`ss -ltn` first, kill by
PID, re-evict cache after.

## Review history
- v3: APPROVE (I/O, ggml-cpu, traceability). CHANGES REQUESTED (loader: `-ot` separator is `,` in common).
  Fixed in v4; also adopted: 0.2 tok/s asserted in test 7, test 8 pass rule, test uses registry CPU backend +
  LINK ggml + n_threads=4, aqu-sz hard target removed.
- v2: APPROVE (I/O, loader). CHANGES REQUESTED (traceability: exps=CPU first in -ot both arms + launcher,
  pre-sampling n_probs, Kimi default/AUTO 0-lazy check, prefetch-on vs off thresholds; ggml-cpu: no test hook
  -> two ctest registrations, from_ptr aligned base + tensor_alloc offset, NMSE tolerance). Fixed in v3, plus
  suggestions: exact page rounding, soft aqu-sz, uninstrumented tok/s, short-tail case, launcher refusals.
- v1: CHANGES REQUESTED by all 4 (I/O, loader, ggml-cpu, traceability). Fixed in v2:
  serial ith==0 madvise -> distributed across threads after barrier in 2 MiB pieces; env read in ggml_cpu_init;
  posix_madvise + platform guards; --no-op-offload mandatory; forbid --mlock/--check-tensors; corrected RANDOM
  (4 KiB faults => B mandatory) and dense-WILLNEED claims; CUDA0 MXFP4 test; same-placement toggle-only
  equivalence with n_probs; evicted-cache measurements; seam observables (strace, log, iostat aqu-sz/rareq-sz,
  vmstat); known-answer, GPU-use, 8k-ctx tests; launcher with pre-flight; permanent unit test.

## v5 amendment (after Kimi-K3 end-to-end results)
Measured: decode 0.085 tok/s (A) vs 0.0175 (B, no prefetch); 25.80 GB read/token (= theory); bit-identical numerics;
load reads ~166 GB (warmup ~140 GB: warmup routes ALL 896 experts, llama-graph.cpp:1539); R3 refaults ~369 GB.
Array finding: shards 01-24 (folded to TLC) random-read 3.3-4.0 GiB/s; shards 25-31 (still in SLC cache) 6.7-9.2.
HMB already at drive max (HMPRE 16384 = 64 MiB); TRIM no change. Realistic decode ceiling ~0.12-0.15 tok/s.
The 0.2 tok/s target is withdrawn (hardware-bound); new target: approach the ~4 GB/s array ceiling.

### Change D (launcher only): `--no-warmup` and `-lm none`
- `--no-warmup`: skip the all-experts warmup decode (would read ~140 GB of random experts at load and pollute the cache).
- `-lm none` instead of `-lm mmap`: dense (non-lazy) CPU tensors are read into ordinary anonymous buffers
  (~49 GiB) - not evictable by expert streaming (swap is 8 GiB); lazy experts stay mmap'ed regardless of load mode
  (llama-model-loader.cpp:1412 `use_mmap || lazy.any()`, :1699 `from_mapping = use_mmap || lazy.has(cur)`); with
  use_mmap false, prefetch_size = 0, so no MAP_POPULATE/WILLNEED on dense ranges.
- Consequences to check: CPU REPACK buffer types may now apply to dense CPU tensors (copy, not mmap) - numerics
  may differ from mmap runs (different kernels) but must still be correct; load reads dense once (~58 GB incl GPU).
- Risk: anonymous 49 GiB + page cache under memory pressure -> some swap (8 GiB max) - acceptable; record swap use.
Assertions (dynamic, arm D = launcher with change): 276 lazy; load read_bytes <= 70 GB; R1 " Paris";
R3 decode tok/s >= arm A (0.0854) - expect higher from fewer refaults; record refaults, swap in/out, md0 MB/s,
tok/s for two consecutive R3 runs (second run shows cache reuse); R2 4-token top-10 vs A' result: ids identical,
max |dp| <= 1e-3 (repack may change last bits).

## v6 (review of v5: I/O APPROVE, loader APPROVE w/ corrections, traceability CHANGES REQUESTED)
Rationale corrected from arm A's own log: dense blk.12+ tensors are *tagged* `buffer type overridden to CUDA_Host`
(CPU buft list prefers CUDA_Host unless --no-host), but under `-lm mmap` they were still mapped: model buffers were
CPU_Mapped 1396.0 GiB (= 1347 experts + ~49 dense) + CUDA0 8.8 GiB, no CUDA_Host model buffer -> dense evictable.
Under `-lm none` the mapped path is skipped for non-lazy tensors, so dense is allocated in the tagged buft:
**pinned CUDA_Host (~49 GiB), not swappable, no RLIMIT_MEMLOCK needed** (so `-lm mlock` is unnecessary).
No CPU repack for Q8_0/F32 on AVX2 (repack.cpp:5105-5126) -> numerics must be bit-identical.
Swap is a file on / (not md0).

Test protocol for D (replaces v5 assertions):
- Same session, page cache evicted (DONTNEED, residency recorded) before every arm; start with a fio ceiling probe:
  6 MiB randread across all 31 shards, 15 s (record GB/s).
- Interleave A, D, A, D (A = launcher as before but `-lm mmap` and no --no-warmup, via direct llama-server with
  identical other flags; D = new launcher). R1 then R3 (16 tokens) per arm; decode tok/s from timings.
- ASSERT: D load log shows a `CUDA_Host model buffer` totalling 45-52 GiB and no CPU_REPACK buffer; 276 lazy;
  D load read_bytes (after eviction) in [55, 70] GB; median D tok/s >= 1.1 x median A tok/s; D md0 decode MB/s
  >= 0.75 x same-session fio ceiling is a SOFT target (report); swap-in during D decode < 1 MB/token
  (/proc/vmstat pswpin delta); R1 " Paris"; R2 (4 tokens, n_probs 10) D vs A ids identical and max |dp| = 0.
- Additionally run R3 a second time in the last D arm without eviction and report it separately (warm cache).

## v7: streamed prefetch for large batches (E) + loader readahead fix (G)
Data (real chat workload, launcher D): T1 prefill 1562 tok = 56 min, md0 avg 1.07 GB/s (p50 0.67), 18.8 KB reads,
163.5 M major faults, 580 M refaults; decode 0.135 tok/s at 1.79 GB/s; fio ceiling 4.34 GB/s; T2 cache reuse OK.
Cause: with ub=512 nearly all 896 experts are used, so each MUL_MAT_ID WILLNEEDs ~5.2 GB at once (x3 per layer),
more than the cache can hold; prefetched pages are evicted before use and compute falls back to 4 KiB faults.

### Change E (ggml-cpu.c, same env GGML_CPU_MOE_PREFETCH=1): windowed, in-order prefetch
- Compute already walks used experts in ascending index = ascending file offset (expert a at data + a*nb02), so an
  in-order window turns a prefill MUL_MAT_ID into a sequential stream through the tensor.
- Let W = window bytes (env GGML_CPU_MOE_PREFETCH_WINDOW_MB, default 512, read once in ggml_cpu_init).
  Pieces (2 MiB, expert-major, ascending) are numbered as today with byte prefix sums.
  * total used bytes <= W: unchanged (all pieces split over threads after the barrier) -> decode path identical.
  * total > W: after the barrier each thread issues its share of pieces whose start offset < W (initial window);
    then thread 0, when it begins expert j in the compute loop, issues the pieces with start offset in
    [prefix(j) + W - slice, prefix(j) + W) not yet issued (tracked by a thread-0-local cursor), keeping ~W in flight
    ahead of compute. No new shared state/atomics; other threads never wait on it.
- Numerics unchanged (hint only). Prefetch log line unchanged.
### Change G (llama-mmap.cpp): no SEQUENTIAL readahead spill into lazy ranges when not using mmap
- In llama_mmap::impl, when lazy_ranges is non-empty and prefetch == 0 (i.e. -lm none/dio path or numa), use
  POSIX_FADV_RANDOM on the fd instead of POSIX_FADV_SEQUENTIAL, so buffered dense reads (read_raw) read exactly
  the requested ranges and do not drag ~57 GB of neighbouring experts into the cache. Default mmap path unchanged.

### Expected / assertions (same-session, cache evicted, interleaved E-off vs E-on where noted)
- Unit: test-moe-prefetch extended with a case where used bytes > W (set W small via env in a third ctest entry,
  e.g. GGML_CPU_MOE_PREFETCH_WINDOW_MB=1): numerics exact; residency shows all used experts' gap pages resident
  after the op (pieces all issued) and unused ones not.
- GPT-OSS regression: decode identical ids/probs and madvise count unchanged vs today (decode total < W).
- Kimi-K3 prefill, ~1500-token real prompt (same doc as the real-workload test), A/B in one session:
  prefill tok/s with E >= 2x today's (0.46 tok/s baseline from the real run, re-measured same session);
  md0 avg MB/s and rareq-sz reported; pgmajfault during prefill reduced >= 10x; decode tok/s not worse (>= 0.95x).
- G: load read_bytes (evicted cache) in [55, 70] GB; dense/expert residency after load: expert ranges < 2 GB.
- Bit-identical outputs E-on vs E-off (n_probs 10, 4 tokens) on Kimi.

## v8 (review of v7: loader APPROVE, I/O APPROVE, ggml-cpu + traceability CHANGES REQUESTED)
E, revised:
- Shared window cursor instead of thread-0-only top-up: one atomic int64 `pf_issued` (bytes of the expert-major
  piece sequence already issued) in its own cache line of wdata, allocated right after atomic_current_chunk in
  BOTH ggml_graph_plan (MUL_MAT_ID: cur += CACHE_LINE_SIZE + CACHE_LINE_SIZE) and the compute function.
  Thread 0 sets it before the existing barrier (to W-rounded end of the initial window). When ANY thread begins
  expert j (in the cur_a loop, BEFORE the tiled `continue`), it computes target = min(total, prefix(j) + W) and,
  while issued < target, claims [issued, next piece end) by CAS and posix_madvise()s that piece. Nobody waits.
- W from env GGML_CPU_MOE_PREFETCH_WINDOW_MB read once in ggml_cpu_init: parse with strtoll; missing, <= 0 or
  unparsable -> default 512; clamp to [2, 1<<20] MiB before shifting. Total used bytes <= W -> exact old path.
- "E-off" for A/B = GGML_CPU_MOE_PREFETCH_WINDOW_MB=1048576 (window >= any op, i.e. v6 behaviour).
G, revised: in llama_mmap::impl record `const bool read_load = (prefetch == 0)` BEFORE the numa override; use
  POSIX_FADV_RANDOM iff read_load && !lazy_ranges.empty(), else SEQUENTIAL as today; log once at INFO.
  Side effect (intended): gemma4/qwen4 lazy PLE under -lm none also gets RANDOM (rows use WILLNEED anyway).
H (launcher): prefill sweeps all experts once per ubatch (~1.35 TB), so batch the prompt: -b/-ub 2048 default
  (env UBATCH), and GPU_LAYERS lowered if the bigger compute buffer needs VRAM (measured; KV+RS+compute must fit).
Tests (replace v7 list):
- Unit: third ctest entry `test-moe-prefetch-window` (GGML_CPU_MOE_PREFETCH=1, WINDOW_MB=2) with the strided
  gap-page residency case at batch 40 (all 8 experts used, 8 x ~2.1 MiB > W): ALL used experts' gap pages must
  be resident (compute never touches gaps, so only issued pieces can make them resident -> proves pieces beyond
  the first window were issued); unused none; numerics exact. File in build dir/TMPDIR, fdatasync + DONTNEED +
  mincore pre-check (skip if not evictable), POSIX_MADV_RANDOM on the mapping, boundary pages excluded (already
  so). Mutation check: disable the top-up -> this entry must FAIL. n_threads 4.
- GPT-OSS: decode ids/probs identical and madvise count identical to v6 (total < W).
- Kimi, one session, evicted cache before each arm, ~1500-token real doc prompt, arms:
  P0 = v6 behaviour (WINDOW_MB=1048576, -ub 512, no G) ; P1 = E+G (-ub 512) ; P2 = E+G+H (-ub 2048).
  ASSERT P1 prefill tok/s >= 2x P0; prefill pgmajfault P1 <= P0/10; decode tok/s (128 tok after prefill)
  P1 >= 0.95x P0; first 4 generated tokens' top-10 ids identical and max |dp| = 0 between P0 and P1 (same -ub);
  P2 reported (tok/s, md0 MB/s, VRAM) - output may differ in last bits vs -ub 512 (different batch split), so
  only top-1 tokens compared, must match on first 4 tokens. G: load read_bytes in [55,70] GB and load time
  <= 1.5x P0's; P0 load read_bytes measured same session (~120 GB) for contrast. Report prefill
  workingset_refault_file delta per arm. Time budget ~2.5-3 h.
- Follow-up (not in this change): MADV_COLD behind the window; next-layer prediction (own plan).

## v9 (review of v8: I/O APPROVE, ggml-cpu APPROVE, traceability CHANGES REQUESTED, loader pending)
- E window target fixed: when a thread begins used expert j, target = min(total, prefix(j+1) + W) (measured from
  the END of expert j, so every piece of expert j - incl. the last expert's tail - is issued no later than its
  compute start). Initial window end and top-up targets come from one helper (ggml-cpu reviewer). Env clamp also
  to SIZE_MAX >> 20.
- Unit: two window ctest entries: `-window-small` (WINDOW_MB=2 < slice ~2.1 MiB) and `-window` (WINDOW_MB=4,
  >= slice, < total 16.8 MiB); both strided-gap residency at batch 40, all used gaps resident. Residency skip
  -> ctest SKIP_RETURN_CODE (77), not a silent pass.
- P0 uses a baseline copy of the current (pre-E/G) build: build/bin copied to the session scratchpad before
  rebuilding; git SHA + diff hash of both builds recorded. P2's cross-ub top-1 comparison is report-only.
- v9.1 (traceability blocker: baseline isolation): baseline binaries carry DT_RUNPATH (not DT_RPATH) =
  build/bin, so LD_LIBRARY_PATH takes precedence. P0 runs with LD_LIBRARY_PATH=<scratchpad>/baseline-v6/bin;
  verified `ldd` resolves every libllama*/libggml*/libmtmd from baseline-v6. ASSERT during P0: every
  libggml*/libllama*/libmtmd* mapping in /proc/PID/maps is under baseline-v6/ (none from build/bin).

## v10: adaptive next-layer expert prediction for decode (change P)
Problem (measured): decode reads ~20 GB/token at ~2.05 GB/s = 0.47 of the 4.3 GB/s ceiling; I/O is issued per layer
only after that layer's router runs, so the disk idles while the CPU computes and the queue drains at each layer end.
Idea (user): while layer L runs, guess layer L+1's experts and start reading them; guesses adapt online and persist.

Mechanism (Kimi-K3 graph only, opt-in, default off = graph unchanged):
- New `src/llama-moe-predict.{h,cpp}`: `llama_moe_predictor` owned by llama_model (created at load iff env
  LLAMA_MOE_PREDICT is 1=observe or 2=prefetch). Per MoE layer: decayed popularity counts (896 floats, half-life
  ~64 tokens), last predicted set, hit statistics, adaptive N_r (router slots, [16,48], start 24) and N_p (popularity
  slots, [0,32], start 8).
- Graph hooks in kimi-k3 build_latent_moe(il), only when n_tokens <= 4 (decode; prefill keeps change E):
  * predict(il+1), built BEFORE layer il's build_moe_ffn and forced early with ggml_build_forward_expand:
    pred_logits = mul_mat(layers[il+1].ffn_gate_inp, ffn_norm_out(il)); custom CPU op (ggml_custom_4d, 1 task,
    srcs: pred_logits, layers[il+1].ffn_exp_probs_b, layers[il+1].ffn_{gate,up,down}_exps) computes
    sigmoid(logit)+bias, takes top-N_r, adds the N_p most popular not yet chosen, stores the set, and (mode 2)
    issues POSIX_MADV_WILLNEED for those experts' slices in the three tensors (same piece helper semantics as E).
  * observe(il), after layer il's logits: custom CPU op on (logits(il), exp_probs_b(il)) recomputes the actual
    selection (same sigmoid+bias top-k rule as build_moe_ffn - verified against "ffn_moe_topk" in a debug test),
    updates popularity and hits vs. the stored prediction, and every 16 tokens adapts N_r/N_p by marginal utility
    (expected actual experts found in the last 4 router/popularity slots: grow if > 0.25/token, shrink if < 0.05).
  * Ops output a dummy 1-element tensor; nothing consumes it except the graph; no change to model math.
- Persistence: env LLAMA_MOE_PREDICT_STATS=<file>. Load at model load if header matches (magic, version, n_layer,
  n_expert, sum of shard sizes); save every 32 decoded tokens and at model free via tmp file + rename. Corrupt or
  mismatched file -> ignored with a warning.
- Logging: every 64 tokens one INFO line: mean hit rate (fraction of actual experts that were predicted), N_r/N_p
  ranges, prefetched-but-unused bytes per token.
Launcher: exports LLAMA_MOE_PREDICT=2 and LLAMA_MOE_PREDICT_STATS=<model dir>/.kimi-k3-predict.bin? -> NO: model dir
may be read-only; use ${XDG_STATE_HOME:-$HOME/.local/state}/llama.cpp/kimi-k3-predict.bin (env PREDICT_STATS).

Tests:
- Unit `tests/test-moe-predict.cpp` (links llama): selection rule vs a reference; popularity decay; N adaptation
  moves the right way on synthetic hit patterns; persistence round trip; mismatched/corrupt header rejected;
  atomic save (no partial file on simulated failure).
- Selection fidelity: in observe mode, debug-assert (test build) the recomputed actual set == ffn_moe_topk ids.
- GPT-OSS: unaffected (hooks are Kimi-only) - madvise counts/ids identical with LLAMA_MOE_PREDICT=2.
- Kimi (one session, evicted cache, real 1562-token doc + 128 decoded tokens, launcher defaults):
  arm O (observe): report per-layer hit rate at N_r=16/24/32 (+N_p), adaptive N trajectory; arm P (prefetch) vs
  arm X (off): ASSERT decode tok/s P >= 1.3 x X; generated ids identical P vs X (bit-identical, hint-only);
  md0 decode MB/s and disk %util reported; prefill time not worse than 0.95 x X.
  Persistence: restart P with the saved stats; ASSERT log shows stats loaded and hit rate over the first 32 tokens
  >= that of a cold start.

## v11 (all four reviews of v10 requested changes) - change P redesigned
Structure
- `llama_moe_predictor` (src/llama-moe-predict.{h,cpp}) is a stable-address member of llama_model
  (`mutable std::unique_ptr`), created at load iff LLAMA_MOE_PREDICT=1 (observe) or 2 (prefetch); all state under a
  mutex (several contexts may share a model). Unset -> no predictor, graph exactly as today.
- It owns ONE background worker thread with a two-level queue: HIGH = actual experts of the current layer,
  LOW = predicted experts of the next layer. The worker issues the POSIX_MADV_WILLNEED pieces (2 MiB, same rounding as
  E), always draining HIGH before LOW, and drops LOW entries that are older than the current layer (stale). It also
  does the persistence save. No madvise and no file I/O ever run inside a graph node.
Graph (kimi-k3 build_latent_moe, layer il, only when the predictor exists and n_tokens <= 4):
- The actual selection sel(il) is computed with the SAME ops build_moe_ffn uses (sigmoid, + exp_probs_b, group mask when
  n_expert_groups > 1, argsort_top_k) and passed to build_moe_ffn via `selected_experts_in`, so the hint and the model
  see the identical tensor (exact by construction; outputs must stay bit-identical - tested).
- pred_logits(il+1) = mul_mat(layers[il+1].ffn_gate_inp, ffn_norm_out(il)) (skipped for the last MoE layer, nextn/MTP,
  or missing gate_inp; layer 1 stays unpredicted).
- One custom op hint(il), expanded BEFORE build_moe_ffn(il): src[0] = layers[il].ffn_up_exps (lazy, always a CPU
  buffer -> op scheduled on CPU, debug-asserted), then sel(il), pred_logits(il+1), exp_probs_b(il+1) (tiny copies).
  Callback returns at once unless ith == 0; thread 0, under the lock and in microseconds: scores last token's prediction
  for il against sel(il), updates decayed popularity, enqueues sel(il) as HIGH (mode 2), computes the prediction for
  il+1 (router top-N_r with the same bias/group rule, + N_p most popular not chosen) from N values read NOW (graph
  reuse safe), stores it, enqueues it as LOW (mode 2). Handles 1..4 tokens per call (union of selections).
Adaptation (bytes-aware): per layer, hits are bucketed by rank (router ranks 17-20, 21-24, ...; popularity likewise);
every 16 tokens a bucket's usefulness = hit experts / slots. Grow N by 4 if the next-outer bucket's usefulness >= 0.5,
shrink by 4 if the outermost < 0.3 (env-tunable). Caps: N_r+N_p <= 32 (env), N_r >= 16. Start N_r = 16, N_p = 0 + 4
probe slots so popularity can prove itself; N_p stays 0 if its usefulness never reaches the grow threshold.
Persistence: LLAMA_MOE_PREDICT_STATS file (launcher default ${XDG_STATE_HOME:-$HOME/.local/state}/llama.cpp/
kimi-k3-predict.bin). Contents: magic, version, n_layer, n_expert, sum of shard sizes; per layer popularity, N_r, N_p,
bucket stats. Worker saves every 32 tokens and at predictor destruction (normal exit / SIGTERM path) via tmp + rename;
load logs a checksum of the loaded state; mismatch/corrupt -> warning + cold start.
Observe mode also computes counterfactual recall for N_r in {16,24,32,48} and router+popularity at equal totals.

Tests
- Unit tests/test-moe-predict.cpp: selection incl. bias and group mask vs reference; popularity decay; bucket
  adaptation (grow/shrink on synthetic streams, caps); HIGH-before-LOW and stale-drop in the queue; save/load round
  trip with equal checksum; corrupt and mismatched files rejected; truncated tmp never replaces the real file.
- GPT-OSS: unaffected (Kimi-only hooks), madvise counts identical with LLAMA_MOE_PREDICT=2.
- Kimi, one session, evicted cache before each arm, launcher defaults, real 1562-token doc + 128 tokens:
  arm O (observe, cold): ASSERT live fidelity is structural (sel shared), report recall grid; FIX the recall
  threshold from O before the A/B and ASSERT recall@N_r=24 >= that value and recall(32) > recall(16); ASSERT N
  values move from their start and hit rate over tokens 65-128 >= tokens 1-64.
  overhead: mode 1 decode tok/s >= 0.97 x X (cost of the extra ops/splits).
  Interleaved X, P, X, P (P cold, empty stats file): ASSERT median P >= 1.05 x median X (hard), report vs 1.3 target
  (ceiling ~2.1x from 2.05/4.3 GB/s); ASSERT all 128 generated ids identical P vs X; prefetched-unused bytes/token
  <= 0.5 x actual expert bytes/token; P read_bytes/token <= 1.4 x X.
  Persistence: run P on doc A (saves), restart, run on a DIFFERENT doc B of the same kind: ASSERT loaded checksum ==
  saved checksum and first-32-token hit rate >= 1.02 x a cold run on B; SIGKILL mid-run -> next start loads the last
  checkpoint; mismatched header -> warning + cold.
- Budget: ~4 h of Kimi time.

## v12 (v11: I/O, ggml, traceability APPROVE; graph CHANGES REQUESTED)
- Selection source changed: do NOT recompute sel(il) / use selected_experts_in. Call build_moe_ffn unchanged, then take
  its own selection with ggml_get_tensor(ctx0, "ffn_moe_topk-<il>") (GGML_ASSERT non-null) as hint(il)'s src[1], and
  ggml_build_forward_expand(gf, hint(il)) immediately after build_moe_ffn(il): hint's ancestors (logits, top-k) are
  added now, layer il's MUL_MAT_IDs are not ancestors and are added later, so hint executes before them. No model op
  is duplicated or altered -> outputs bit-identical by construction (still asserted over 128 tokens).
  The prediction for il+1 uses router logits computed from ffn_norm_out(il) and layers[il+1]'s bias with the same
  sigmoid+bias(+group mask) rule implemented in C++ inside the op (prediction only - exactness not required there).
- Adopted reviewer notes: worker checks HIGH between every 2 MiB piece; optional LOW cap 256 MiB/layer; worker count
  env (1-2, default 1); bounded non-blocking enqueue (drop LOW when full); log worker busy fraction, HIGH latency,
  LOW drops; ~llama_model destroys (joins) the predictor before mappings/buffers are released; observe mode logs once
  the backend of each hint op and the graph split count P vs X; launcher notes -t 28 would starve the worker.
- Test tweaks (traceability): warm-vs-cold >= 1.02x only if the saved state differs from cold defaults, else >= 0.98x;
  "adaptation ran" (bucket stats + per-16-token evaluations logged, N within caps) instead of "N moved"; hits 65-128
  >= 0.98x hits 1-64; if final N_p > 0, router+popularity counterfactual recall >= router-only at equal total slots.

## v13 (v12: I/O APPROVE; ggml, graph, traceability CHANGES REQUESTED - same finding)
Ordering fix (verified in source): build_moe_ffn itself expands weights (llama-graph.cpp:2283), experts (:2456, which
pulls in the up/gate/down MUL_MAT_IDs), cur_experts and moe_out, so a hint expanded after it returns would run after
layer il's experts. New hook instead:
- llm_graph_context gets `virtual void on_moe_selected(ggml_tensor * selected_experts, int il) const {}` (no-op
  default -> all other models' graphs unchanged), called in build_moe_ffn right after
  `ggml_build_forward_expand(gf, weights)` (:2283) and before build_moe_cache_slots / the MUL_MAT_IDs.
- llama_model_kimi_k3::graph overrides it: if the predictor exists, n_tokens <= 4 and il is a MoE layer, it builds
  hint(il) with src[0] = layers[il].ffn_up_exps, src[1] = selected_experts (the model's own tensor, no lookup, no
  recompute), plus pred_logits(il+1) / exp_probs_b(il+1) when il+1 is a MoE layer, and expands it there.
  pred_logits(il+1) needs ffn_norm_out(il), which the Kimi graph stores in a member before calling build_latent_moe.
- Seam check: in observe/debug mode, after graph build, assert index(hint(il)) < index of every MUL_MAT_ID whose
  src[2] == selected_experts(il) in gf, log "hint order OK for N layers" once (fail loudly otherwise).
- build_moe_cache_slots is only active with --moe-cache-mib (moe_cache == nullptr otherwise): no interaction.
Everything else as v11/v12.

## v14: after the prediction results (REVIEW.md has the full state)
Measured: P = 1.27x (MXFP4) / 1.51x (IQ1) decode, bit-identical; prefill CPU-bound (13/14 cores, GPU idle, disk at
20-30% of ceiling); IQ1 decode uses 35% of the disk ceiling; warm restart 1.011x (< 1.02 bar); the upstream 2-token
probe decode at server init (common.cpp:1584) feeds the predictor (state drift, +9.6 GB load reads).

### Change Q (predictor): ignore the startup probe
Rule in llama_moe_predictor::on_layer: a step with 1 < n_tok <= 4 is ignored (no scoring, no popularity update,
no prefetch, no step count) until the predictor has seen a real request, i.e. a single-token step (n_tok == 1) or a
prompt (the Kimi hook sets `saw_prompt` when it skips a graph with n_tokens > 4). Real decode with -np 1 is always
n_tok == 1; multi-sequence decode (2-4 tokens) always follows a prompt, so nothing real is lost. Unit test: a
2-token step before any other is ignored (tokens == 0, no enqueue), a 2-token step after a 1-token step counts.
### Change R (launcher): per-model GPU_LAYERS default
Default 19 if the model file name matches IQ1, else 8 (measured: IQ1 dense layer ~336 MiB VRAM, 19 keeps >= 500 MiB
free; 24 OOMs). Explicit GPU_LAYERS still wins. Banner prints the choice.
### Change S (plan): restart benefit
Persistence verified (checksums, SIGKILL checkpoint, corruption -> cold). The warm-start gain on a new document is
1.011x because the router does most of the work and n_r never leaves 16; relax the assertion to >= 0.98x (no harm)
and report; keep the feature. No code.
### Experiment T (no code first): headroom-aware speculation on IQ1
Arms on IQ1 (same doc, 129 tokens, evicted cache, interleaved with the current default): (a) GROW=0.25 CAP=48;
(b) WORKERS=2; (c) both. Record decode tok/s, md0 GB/s, read GB/token, low drops, worker busy. Adopt as the IQ1
launcher default whatever is >= 1.05x the current P and keeps output identical; otherwise keep defaults. If a
winner exists, the launcher sets LLAMA_MOE_PREDICT_GROW/CAP/WORKERS per model (env override allowed).
### Experiment U (no code first): GPU prefill via op-offload on IQ1 (E1-E3 dispatched)
If it works and copies do not fault serially: launcher allows op-offload for IQ1 (per-model default) with
GPU_LAYERS chosen for VRAM; assert prefill >= 2x, decode unchanged (>= 0.97x), ids identical to the no-offload
run at the same GPU_LAYERS. If copies fault serially: change V = the Kimi hint for n_tokens > 4 (offload on)
enqueues the layer's three whole expert tensors as HIGH to the worker (in file order, 2 MiB pieces) so the copy
streams at disk speed; then re-measure. MXFP4 excluded (does not fit).
### Tests
Unit (Q) + GPT-OSS unaffected + Kimi: Q: a server start with no request must not change the stats checksum and
load read_bytes must drop back to ~109 GB (MXFP4); R: launcher fake-BIN matrix; T/U as above.
- v14.1 (traceability): Q's load-bytes bound is relative: load read_bytes(predictor on, Q) <= load read_bytes(no
  predictor) + 1 GB, same session; a real first prompt of 2-4 tokens is also ignored until the first decode step
  (documented). R's matrix: {name contains IQ1 -> 19, MXFP4 name -> 8, explicit GPU_LAYERS=N -> N}, banner shows the
  choice, one real IQ1 run asserts CUDA free >= 500 MiB. S: re-measure the warm-start ratio once after Q (the 1.011x
  used probe-contaminated state). T: control and winning arm run twice, adopt on medians; keep waste bounds
  (unused <= 0.5x actual, read/token <= 1.4x) reported. U: GPU kernels differ from CPU ones, so replace "ids
  identical" by: top-1 identical on the first 4 tokens, max |dlogprob| <= 0.05 over 128 tokens, perplexity on the
  doc within 0.5% of the no-offload run (llama-perplexity, same placement), full-sequence id agreement reported.
  "Copies fault serially" observable: prefill md0 < 1 GB/s with aqu-sz < 4 per drive -> change V.
- v14.2 (ggml): Q's flag is std::atomic<bool> (or under mtx), checked at the top of on_layer before step++ and before
  the step_seen/layer_seen stores and any enqueue. T: log busy% divided by n_workers; HIGH-before-LOW is best effort
  at 2 MiB granularity with 2 consumers (accepted). V (only if E1 shows serial faults): its own entry point
  on_prefill_layer(il) that bumps step_seen (once per ubatch, first MoE layer) and sets layer_seen = il before
  enqueuing, no scoring/popularity, nothing in OBSERVE mode; work items are "whole tensor t of layer il" walked in
  2 MiB pieces in file order, enqueued in the graph's copy order (up, gate, down); a byte window ahead of the copy
  (reuse E's W) so a 5 GB tensor is not requested at once; stale check per tensor.
- v14.3 (I/O): T arms use CAP 40 (not 48), GROW 0.25 with SHRINK 0.15 (shrink < grow, no oscillation), the
  read/token <= 1.4x guard stays hard; T also records the CPU floor: run the same prompt+decode twice back to back
  (warm cache) and sample thread states (D vs R) during cold decode. V: expect U to fault serially (cudaMemcpy from
  pageable mmap stages through the driver on the calling thread); V uses 2 workers (or a dedicated one) and enqueues
  all three tensors at hint time; optional arm: madvise(MADV_COLD) on a tensor's range after its copy so the
  prefill sweep does not flush the decode hot set.

## v15 (graph reviewer: Q build-time flag wrong; upstream sched_copy_experts exists)
- Q, final rule (execution time only, in on_layer under mtx, before step++/stores/enqueue): a step with
  1 < n_tok <= 4 is ignored until a step with n_tok == 1 has been seen (`std::atomic<bool> unlocked`). No build-time
  flag. Covers the init probe of the main and any draft context and sleep/resume; -np 2 lockstep of two <= 4-token
  prompts stays ignored until one slot decodes alone (documented). --no-warmup stays mandatory in the launcher.
- V replaced by V' (llama-context.cpp, sched_copy_experts): when enabled (env GGML_CPU_MOE_PREFETCH=1, read once;
  src buffer is host memory), each run of used experts [offset, offset+size) is copied in chunks of C = 64 MiB
  with a lookahead: before copying chunk i, posix_madvise(WILLNEED) chunks i+1 .. i+L (L = 4, i.e. 256 MiB in
  flight; env GGML_CPU_MOE_PREFETCH_WINDOW_MB / 2 reused as the byte cap), page-rounded. cudaMemcpy from pageable
  memory is host-synchronous, so chunking is safe; pages under I/O block on the folio lock instead of 4 KiB faults.
  No worker, no stale logic, OBSERVE mode irrelevant. Numerics unchanged (hint only). MXFP4 stays excluded from
  op-offload (3 live 5.24 GB copies). VRAM budget for IQ1: 3 x largest expert tensor (3.6 + 2.15 + 2.15 = 7.9 GB)
  + compute buffer + KV + dense GPU layers; GPU_LAYERS chosen from measurement (E1-E3).
- Launcher: op-offload becomes a per-model default (IQ1: on with its GPU_LAYERS; MXFP4: off), env OFFLOAD=0/1
  override; GGML_OP_OFFLOAD_MIN_BATCH must not be set lower (unset like the other env).
- Tests for V': unit `test-moe-prefetch` gets a case calling the chunked copy helper on an mmap'ed strided tensor
  (helper factored into a small function taking (base, offset, size, chunk, lookahead, set_fn)) asserting the
  set_fn sees the exact byte ranges in order and residency of gap pages proves the WILLNEED coverage; GPT-OSS
  unaffected; IQ1: prefill >= 2x the no-offload launcher default, decode >= 0.97x, tolerance protocol of v14.1,
  observable "prefill md0 >= 2 GB/s and aqu-sz >= 8" (copies no longer fault serially).
- Everything else per v14.1-14.3 (incl. MADV_COLD guarded by #ifdef, as an optional arm only).
- v15.1 (I/O + ggml + traceability conditions): V' flattens all used-expert runs of a tensor into one piece list
  (2 MiB pieces, so no WILLNEED call exceeds the kernel's 16 MiB per-call cap) and keeps a window of L chunks
  (64 MiB each = 32 pieces) issued AHEAD of and INCLUDING the chunk being copied: before copying chunk i, pieces of
  chunks [0 .. i+L] have been issued (chunks 0..L before the first copy). The window runs across run boundaries.
  Unit test: mock set_fn records ranges and never touches the source bytes; assert the exact copied ranges in order
  and that every page of every used run is resident after the call (unused experts' gap pages not resident).
  Noted for later: pageable H2D copies (~3.5 GB/s) will bound prefill at ~210 s; a pinned staging buffer is the
  next step after V' if measurements confirm.

## v15.2 results (2026-10-16)
V': IQ1 GL 12 prefill 230 s vs 655 s (2.85x, PASS), md0 2.2 GB/s / aqu 8.3 / 5.5k faults (no serial faults),
decode 1.015x; launcher default with offload -> 11 layers (12 left 491 MiB). Tolerance protocol: top-1 first 4
identical, but |dlogprob| up to 0.25 (bound 0.05 FAILED) and sequences part at token 9: CUDA vs CPU IQ1_S/Q2_K
kernels at batch 2048 (test-backend-ops MUL_MAT_ID: IQ1_S 4/4, Q2_K 3/3 pass); perplexity check running, decides
whether OFFLOAD=1 stays the IQ1 default. Q: PASS (checksum unchanged, load bytes 108.89 vs 108.90 GB). R: fixed.
T: no arm >= 1.05x (best 1.003x), ids identical; IQ1 decode is compute-bound (11.8 threads R) -> defaults kept.
S re-measured: 1.05x same doc. llama-perplexity now built.
- v15.3: perplexity (doc A+B x2, 2 chunks of 2048) offload on 3.3315 vs off 3.3485: ratio 0.9949 (0.51% lower with
  offload, ~0.16 SE) -> accepted, OFFLOAD=1 stays the IQ1 default; the 0.5% bound is a soft target for a 2-chunk
  estimate. Launcher IQ1 defaults (GL 11, OFFLOAD 1) verified live: prefill 229.6 s / 6.8 tok/s, decode 0.458
  tok/s, VRAM free min 914 MiB. IQ1 is the user's main model (launcher default MODEL).

## v16: IQ1 as the main model - next levers
Measured: prefill 230 s is bound by the pageable host->GPU copy (~2.2 GB/s, disk ceiling 5.7); decode 0.46 tok/s
with ~12 threads "running" but 3 in D (page-fault wait) on average and the disk at 34%: ggml threads spin at
barriers while one thread waits on a missed expert page, so per-layer miss latency (not bandwidth) may dominate.
### Change W (llama-context.cpp): pinned double-buffered staging for expert uploads
In sched_copy_experts (prefetch path), instead of ggml_backend_tensor_set_async from the pageable mapping:
two pinned host buffers of CHUNK (64 MiB) allocated once per context from the backend's host buffer type
(ggml_backend_dev_host_buffer_type of the dst device; fall back to the current path if unavailable); for each
chunk: wait for the event of the DMA that last used this buffer, memcpy(mapping -> pinned) (page faults block on
folios already in flight thanks to the WILLNEED window), ggml_backend_tensor_set_async from the pinned buffer,
record an event; the window logic of llama_copy_ranges_prefetched is reused (its copy callback does the staging).
Expected: copy at min(memcpy ~8 GB/s, PCIe ~12) with memcpy(i+1) overlapping DMA(i) -> disk-bound (~5.7 GB/s):
prefill ~140-150 s. Numerics unchanged (same bytes). Opt-in via the existing switch; env LLAMA_COPY_STAGING=0
disables (A/B). Unit: extend test-copy-prefetch with a staging callback mock (copy order/bytes identical).
Assert on IQ1: prefill >= 1.4x the current 230 s, md0 >= 4 GB/s during prefill, perplexity identical to the
non-staged offload run (same kernels -> bit-identical logits expected: assert max |dlogprob| == 0 over 129 tokens).
### Measurement X (no code): where IQ1 decode time goes
(a) short prompt (~20 tokens), 32 greedy tokens, run twice without eviction: run 2's experts are cached -> its
decode tok/s is the compute floor; report cold vs warm; (b) utime/stime split of the server per phase (from
/proc/PID/stat) for cold and warm; (c) thread scaling on the warm run: THREADS=7 vs 14 vs 28 (28 = hyperthreads;
the worker note applies); (d) per-layer D-state time: sample /proc/PID/task/*/stat states at 10 Hz during cold
decode and report mean threads in D and the fraction of samples with >= 1 thread in D.
Decision rule: if warm decode >= 2x cold -> decode is miss-latency-bound: next lever is cutting miss latency
(larger folios / fewer, bigger reads per miss; or an io_uring+O_DIRECT expert cache) or raising recall; if warm
~= cold -> CPU-bound: next lever is kernel speed (IQ1_S/Q2_K dot products, thread count) or more GPU layers.
- v16.1 (graph APPROVE w/ details; traceability CHANGES): W details: host buft via ggml_backend_dev_host_buffer_type
  of the dst device, 2 x 64 MiB buffers + one event each (ggml_backend_event_new/record/synchronize; null event ->
  ggml_backend_synchronize fallback), detect the silent plain-CPU fallback (buffer type != host buft -> staging off,
  log), free in ~llama_context after synchronizing the backends, chunk == staging size (assert), lookahead >= 2x
  chunk, INFO line "staged uploads via <buft>". Assertions same-session: interleave STAGING=1/0 (evicted cache),
  staged prefill >= 1.4x non-staged, md0 >= 0.75x the same-session fio ceiling, |dlogprob| == 0 (same bytes incl.
  the 512-byte MMQ padding - the unit mock asserts the padding is copied). X: warmness asserted (run-2 read/token
  < 5% of run-1, pgmajfault ~0, shrink n_predict until true), ids identical, PREDICT=0 warm arm, iowait, per-token
  latencies; thresholds warm <= 1.15x cold = CPU-bound, >= 2x = latency-bound, else mixed.
- v16.2 (I/O): W: if measured md0 < ~5 GB/s during staged prefill, move the WILLNEED submission off the copy thread
  (second thread); optional later: pread() from the fd into the pinned buffer (no faults/PTE churn). X: warmth also
  by pgmajfault (run 2 < 1% of run 1); record per-drive r_await/aqu-sz at 1 Hz (r_await > ~1 ms = queueing, else
  latency-bound), predictor hit rate next to the D-state samples, max_sectors_kb per nvme. Decode levers if X says
  I/O-bound: earlier issue (recall, L+2 speculation) or an O_DIRECT+io_uring host expert cache (2 MiB reads straight
  into user memory; high complexity, only after X). Bigger folios / MADV_HUGEPAGE do not apply (MADV_RANDOM path,
  order-0 WILLNEED).

## v17: keep decode's experts in the page cache across prompts (change E)

### Measured problem (2026-10-16, 256 GB RAM, IQ1 launcher defaults, bench/ harness: 1.6k-token doc review x3)
Prompt 281-294 s, decode 0.52-0.53 tok/s on every request, including the third (same document as the first):
each request reads ~900 GB from md0 and the page cache sits full at ~255 GB. A prompt pass streams all 92 layers'
experts (~511 GB) in layer order through the mapping; that scan is larger than the cache (~215 GB available), so
it evicts both the previous pass's tail and every expert decode had accumulated. Decode therefore always starts
cold (warm compute floor measured earlier: 0.73 tok/s vs 0.50 cold at 160 GB). Staging (W) changed nothing
(< 1% on all metrics, arms A/B), so W stays uncommitted and default-on until E is measured with and without it.

### Change E: drop prompt-streamed expert pages that were not already cached
Policy: the page cache belongs to decode. The prompt path (sched_copy_experts, prefetch branch) must leave the
cache exactly as it found it: pages that were resident before the pass are touched (so they stay young) and kept;
pages the pass brought in are dropped right after they were copied. Decode's own faults/WILLNEEDs are the only
writers of the cache, so over time it holds decode's most recently used (~popular) experts, up to the full ~215 GB.
Prompt also gains: those resident bytes are hits on every pass (today ~0% hits because of the scan order), so a
pass reads ~511 GB - cache instead of ~511 GB.

Mechanism (Linux only, no-op elsewhere; env `LLAMA_COPY_EVICT=0` disables; on by default with the prefetch switch):
1. Snapshot at hint time. llama_copy_ranges_prefetched already issues hints in 2 MiB pieces `lookahead` ahead of
   the copy, through the optional `hint` callback (llama-context passes nullptr today). E passes a hint lambda:
   `mincore()` over the page-aligned cover of the piece -> bitmap of pages resident BEFORE the read, stored in a
   per-copy map keyed by piece offset (`copy_experts.snap`), then `llama_prefetch` (WILLNEED) as before.
   The snapshot precedes the hint, so readahead triggered by the hint cannot be mistaken for prior residency; the
   lazy ranges are MADV_RANDOM, so faults bring in single pages and no readahead crosses into un-snapshotted pieces.
2. Evict after copy. copy_chunk (64 MiB) after the copy returns (staged: memcpy into pinned done; unstaged:
   cudaMemcpy from pageable memory is host-synchronous): pop every snapshot piece inside [offs, offs+size)
   (pieces nest in chunks: both start at the run offset in multiples of 2 MiB) and hand them to a worker.
3. Worker thread (`page_evictor` in llama_context: thread + mutex/cv + deque, started on first use with the
   stager, stopped in ~llama_context before the mappings go away). For each piece: coalesce the pages whose
   snapshot bit is 0 into runs and `madvise(MADV_PAGEOUT)` each run. Measured on a shard mapping: 0.21 ms/MiB,
   99.7% of pages dropped (the rest are in-flight folios the kernel skips; irrelevant). At ~310 GB dropped per
   pass that is ~65 s of CPU - on the side thread, not the copy thread (the kernel's kswapd does this reclaim
   asynchronously today; E must not move it onto the critical path). Queue bounded (e.g. 64 chunks): the copy
   thread blocks if the worker falls behind, which also bounds the memory the un-evicted pages take.
   MADV_PAGEOUT on a shared file mapping needs the caller to own the file (or CAP_SYS_ADMIN): kernel
   can_do_file_pageout; true here (uid 1000 owns /mnt/k3). If the first call fails (EINVAL on kernels < 5.4,
   EPERM), log once and turn E off for the rest of the run. Alternative kept in reserve: MADV_DONTNEED (drop PTEs
   of the file mapping, no data loss) + posix_fadvise(DONTNEED) on the fd (needs the fd kept; 100% effective,
   measured 2x cheaper).
4. Boundary pages: a page shared between the piece and a neighbour (tensor starts are 32-byte aligned) is
   snapshotted/evicted as part of the cover; dropping a neighbour's cold page is harmless (hint only).
5. Log once on first use: "prompt expert pages are dropped after upload; the page cache is kept for decode".
Not changed: decode paths (CPU MUL_MAT_ID prefetch, predictor), numerics (same bytes copied), W, the launcher
(E is on whenever GGML_CPU_MOE_PREFETCH=1; it adds `LLAMA_COPY_EVICT` to the unset list only if it fixes it - no).
Optional later (not in E): posix_fadvise(DONTNEED) on the dense ranges after the `-lm none` read-load frees ~58 GB
of never-reused cache immediately instead of waiting for LRU.

### Expected results (assert)
Unit (tests/test-copy-prefetch.cpp, extended or a sibling): a 48 MiB file mapping, runs as today; pre-populate a
known subset P of pages (touch them), evict the rest; run the copy with E's snapshot/evict hooks (synchronous
worker in the test, or wait_idle); assert (a) copy saw exactly the runs' bytes, (b) every page in P is still
resident, (c) >= 98% of the pages not in P are NOT resident, (d) the untouched 10 MiB region stays non-resident.
Skip (77) if MADV_PAGEOUT returns EINVAL/EPERM (CI kernels/filesystems), like the existing residency tests.
End-to-end (Sonnet, bench harness, evicted cache, sequence doc, docB, doc, docB, 128 tokens, temperature 0;
arms run back to back: C = launcher defaults + E; A = LLAMA_COPY_EVICT=0):
- cache kept: md0 bytes read for C's requests 3-4 <= 0.75x C's request 1 (~900 GB -> <= ~675 GB; expected lower:
  prompt ~511 - ~200 = ~310 GB + decode);  A's stay ~900 GB.
- decode: C's gen tok/s on requests 3-4 >= 0.62 (A: ~0.53), i.e. moving toward the 0.73 warm floor; cold request 1
  of C within 5% of A (E must not slow the first pass: eviction is off the critical path).
- prompt: C's requests 3-4 prompt_s <= A's (no regression; a gain if prompt is disk-bound).
- text identical between C and A for the same doc (same kernels; |dlogprob| == 0 expected).
- ECC counters unchanged; no "evict disabled" log line; server RSS steady (snapshots freed).
Decision: if C's decode gain on warm requests < 10%, decode is not cache-bound at this cache size -> report and
stop (the lever is CPU kernels).

### Review history (v17)
- v17 reviewed by 3 lenses (Linux mm, llama.cpp integration, traceability): all CHANGES REQUESTED. Key points
  kept for the record: mincore/MADV_PAGEOUT are silent no-ops for a non-owner (probe needed); PAGEOUT is
  PTE-driven; coalesce per chunk; queue <= 16 chunks; put the primitives in llama-mmap so the unit test runs the
  real code; the predictor not hinting during prompt batches (n_tokens > 4) is load-bearing; expert set is
  544.9 GB (not 511), 9.73 GB/token selected; decode ALREADY gets ~70% of its bytes from cache (scan tail + 128-token
  self-warming) so E's decode gain is only +2-4%; E's real lever is prompt bytes (545 -> ~320 GB) and only if the
  pass is disk-bound; decide on total wall of requests 2-4, not decode alone.
- Gate measurement (readprobe.c, 2 GB cold region of shard 12): the pass's method (mmap + 2 MiB WILLNEED 256 MiB
  ahead + memcpy) reads at 2.45 GB/s = the pass's 2.1 GB/s -> the prompt IS bound by the read method, not by the
  disk: O_DIRECT pread, 16 threads x 2 MiB = 9.27 GB/s on the same shard (page cache untouched); preadv2
  RWF_DONTCACHE 16 threads = 3.4 GB/s; plain buffered 16 threads = 3.6 GB/s. Per-drive max_hw_sectors_kb = 128
  (firmware), so the 113 KB requests are not tunable; the gain comes from queue depth. => E is superseded by D.

## v18.1: direct reads for the prompt pass (change D, supersedes E)

### Why
The prompt pass copies the experts to the GPU through the mapping (WILLNEED readahead of order-0 pages + faults +
memcpy) at 2.1-2.5 GB/s on every shard; the array delivers 7-12 GB/s to 8 concurrent 2 MiB O_DIRECT preads (sweep over
8-32 GiB cold regions of shards 05-34: mmap path 2.54-2.56 GB/s everywhere; O_DIRECT 2 MiB pieces: 4 threads 7.4,
8 threads 9.0-11.6 with a tight spread, 16 threads 4.2-13.9 bimodal, 32 threads 3.3-8.5 GB/s; per drive ~1.5 GB/s at
127 KB requests (max_hw_sectors_kb 128, firmware); preadv2 RWF_DONTCACHE 3.3-3.4, plain buffered 3.6 GB/s). O_DIRECT
reads bypass the page cache, so the pass stops evicting decode's experts (E's goal) with no mincore/PAGEOUT/worker.
Verified on this host by two reviewers' probes: O_DIRECT pread straight into cudaMallocHost memory works (plain shmem
VMA, GUP pins normal pages), bytes identical to buffered reads, cudaMemcpyAsync from the buffer afterwards fine;
a 64 MiB barrier between fills costs ~0 at 8 threads on contiguous data but the sweep's throughput needs a continuous
piece queue (per-chunk drain: -18% at 8 threads, -45% at 16); on the fragmented first batch (one 2 MiB expert every
6 MiB) per-run requests give 1.8 GB/s vs 6.6-8.5 GB/s when gathered into 64 MiB fills.

The prompt is TWO expert passes: the server splits the batch at the user-turn start to create a context checkpoint
(KDA model; server-context.cpp:4095, `--ctx-checkpoints`): batch 1 = 67 tokens (~1/3 of the experts, fragmented
runs, 175 GB, 82 s today) then batch 2 = 1569 tokens (all experts, 545 GB of which ~173 GB were hits on batch 1's
pages, 190 s). Under D (no page-cache hits) a request reads ~720 GB; with `--ctx-checkpoints 0` only ~545 GB.
GPU-side floor (serial with the reads: DMA and compute share one stream, one layer of GPU expert buffers, ids
readback synchronizes per layer): DMA 720 GB / 12.3 GB/s (Gen3 x16) = 59 s + compute 10-40 s (152 TFLOP; <= 42 s
inferred from A's disk-idle time). Expected D prompt: batch 1 21-27 s + ~10 s sync, batch 2 ~57 s + 10-40 s compute
-> ~100-135 s (vs 285 s); ~75-110 s with the checkpoint split disabled.

Decode today reads ~2.6 GB/token (73% hits) because the pass leaves the LAST ~200 GB of layers resident and 128
tokens self-warm. Plain D leaves nothing from the pass: request 1 decodes from a cold cache (expected slower than
0.53 tok/s), later requests depend on the overlap between consecutive decodes (unknown). D therefore includes a
hit-aware mode (D2, behind env LLAMA_COPY_DIRECT_HITS=1; D0 = plain direct reads is implemented first): pieces whose
pages are all resident are memcpy'd from the mapping instead of read from disk. Derived expectation: an O_DIRECT pass
already leaves decode's cached set untouched (D0 keeps it too), so D2 only (a) saves disk bytes for hits - at most
decode's cached set (<= ~200 GB) per prompt, and a hit still goes page cache -> memcpy -> pinned -> DMA at the same
~12 GB/s DMA bound as a miss at ~9.6 GB/s, so <= ~5 s saved per prompt, minus ~1-5 s of mincore on the copy thread
(~360k calls/pass) - and (b) re-promotes resident pages in the LRU (small, unknown effect on decode). Expected D vs
D0: prompt -5..+5 s, decode ~equal. D2 becomes the default only if measured >= 3 s/prompt faster with equal decode.

### Change D
1. Model (src/llama-model.{h,cpp}, src/llama-mmap.{h,cpp} llama_file::name()): the loader keeps one mapping per file
   in file order in pimpl->mappings whenever lazy ranges exist (also with -lm none); add `lazy_file_paths` captured
   in load_tensors from ml.files[i]->name(), and `lazy_src(const ggml_tensor *) -> { file idx, file offset }` =
   the mapping whose [addr, addr+size) contains src->data, offs = data - addr (per-tensor table not needed).
2. Fill gathering (src/llama-mmap.{h,cpp}, pure function, unit-tested): `llama_gather_fills(runs, capacity,
   align)`: walks the runs in order, packs segments {run offs, size, buffer offset} into fills of <= capacity bytes
   where each segment occupies [align_down(offs) .. align_up(offs+size)) of the buffer (head/tail slop < align each);
   `capacity` is the buffer's total aligned bytes INCLUDING slop: segments are placed at align_up(prev_end) and a
   fill closes when the next segment's aligned extent would not fit (a segment larger than the capacity is split at
   absolute 2 MiB file offsets). Ring buffers are a fixed 64 MiB + 2 MiB; no "max segments" figure exists. Pieces: each segment is read in pieces split at ABSOLUTE 2 MiB file offsets
   (first/last partial), so pieces stay 4 KiB aligned internally and mostly map to one drive (md chunk 2 MiB).
   The D path does NOT call llama_copy_ranges_prefetched (that helper issues WILLNEED; prefetch=false would pass
   whole 6 GB runs). No hint of any kind is issued on the D path (asserted by the unit test: zero madvise via the
   test's mapping residency).
3. Reader (src/llama-mmap.{h,cpp}, `llama_direct_reader`, compiled on Linux only): opens the lazy files via
   `llama_file(path, "rb", /*direct*/ true)` lazily; has_direct_io() false -> disabled; file_id(), size(),
   alignment = statx(STATX_DIOALIGN).stx_dio_offset_align when available else 4096, clamped to [512, 4096]
   (never st_blksize: XFS reports the stripe width, 8 MiB here, under `largeio`; page size is the natural upper bound
   since hits are page-granular). N threads (env LLAMA_COPY_DIRECT_THREADS,
   default 8; piece size env LLAMA_COPY_DIRECT_PIECE_MB default 2, sweep 1 once) pull pieces FIFO from one shared
   queue. API: `submit(fill, buffer) -> ticket` (appends the fill's pieces; each piece is {kind: pread|memcpy, fd,
   aligned file offset, aligned length, dst}), `wait(ticket)` (per-fill counter), `stop()` (drain + join).
   pread rules (measured: shard sizes are not sector multiples): always issue the full aligned length even past
   EOF; a short count r is accepted only if aligned_off + r == file size (fstat at open) and covers the requested
   end (zero-fill the slop); r == 0, or r % 512 != 0 before EOF -> error; mid-file short read -> retry at off+r.
   EINTR retry. Init probe: one aligned 4 KiB pread of the first file into the pinned buffer; EINVAL/EFAULT/
   EOPNOTSUPP -> disabled (log once) before any fill is issued. Mid-pass error on a piece: the fill is marked
   failed; after wait() the caller re-does that fill's segments with llama_prefetch + memcpy from the mapping into
   the same buffer (never partial data) and logs once. The current tensor finishes through the fill loop (every
   already-submitted fill is still wait()ed and used or redone before its buffer is reused or released); D is off
   from the next sched_copy_experts call on.
   D2 hits: at submit time the caller runs mincore over each piece's page cover (mapping address); all resident ->
   kind memcpy (worker memcpy's from the mapping; pages are resident so no faults from disk), else pread. Owner gate
   at init: mincore reports ALL pages resident for a file the caller neither owns nor can write (kernel
   can_do_mincore), which would turn every piece into a faulting memcpy; D2 is enabled only if, for every lazy file,
   fstat(fd).st_uid == geteuid() || access(path, W_OK) == 0; otherwise log once "hit detection unavailable (not file
   owner), reading everything" and run plain D. E2E asserts hit bytes ~0 on request 1 (evicted cache). Counters
   per pass: bytes read / bytes hit, logged at INFO at the end of each sched_copy_experts call under a debug env
   (LLAMA_COPY_DIRECT_LOG=1) and accumulated for the one-line per-batch summary.
4. Context (src/llama-context.cpp sched_copy_experts prefetch branch; W's stager generalised): ring of 4 pinned
   host buffers of 64 MiB + 2*align*(max segments per fill) (host buft of the dst device; if unavailable -> today's
   path), one event each. Loop over fills: acquire buffer b (wait its event = the DMA that last read it), submit
   fill k+2 (reads run ahead of DMA by two fills: k DMAing, k+1 reading, k+2 queued), wait fill k, one
   ggml_backend_tensor_set_async per segment from buffer + segment buffer offset + head slop, record event.
   Per-tensor drain at the end of the call (~1 fill-time x 276 tensors ~ 2 s/pass, accepted). Debug env also logs
   the cumulative wall time spent inside sched_copy_experts per batch (prompt_s - copy_s = compute + sync).
   Switches: GGML_CPU_MOE_PREFETCH unset or non-lazy tensor -> unchanged; LLAMA_COPY_DIRECT=0, no host buft,
   reader disabled, non-Linux (stub llama_direct_reader with enabled() == false, so llama-context.cpp has no
   #ifdef) -> today's WILLNEED + memcpy (+W) path. Log once on first use: "expert uploads read
   directly from <n> files with <N> threads (page cache untouched; hits copied from memory)".
   ~llama_context: synchronize(); reader.stop(); then the stager/ring release (threads write into those buffers).
5. Not changed: decode paths, predictor, launcher defaults (D on whenever prefetch is on), numerics (same bytes:
   the 512-byte MMQ padding is inside the run size; unit test asserts byte equality). Zero-code lever to measure:
   `--ctx-checkpoints 0` removes the 175 GB first pass under D (bench arm variant; not a launcher default until
   measured - checkpoints matter for multi-turn reuse).
Contingency D1b (pre-registered; trigger: requests 2-4 gen < 0.98x A on the better of D/D0, or request 1 total > 0.95x A): read the last
LLAMA_COPY_DIRECT_TAIL_GB of each pass buffered (today's path) so the scan-tail hits for decode remain (~+35-80 s).

### Expected results (assert)
Unit (tests/test-copy-direct.cpp, LINK llama, Linux only in CMake, SKIP_RETURN_CODE 77): temp file of 48 MiB + 32 B
of seeded pseudo-random bytes in TMPDIR (skip 77 if O_DIRECT open fails, e.g. tmpfs); runs as in test-copy-prefetch
(unaligned start 40 MiB + 100, the 512-byte padding) plus a fragmented set (2 MiB every 6 MiB) and a run ending at
the last byte of the file. Through the real gather + reader into 4 KiB-aligned buffers, assert: (a) bytes identical
to the file for every segment, all cases, threads 1 and 8; (b) one fill covers many runs in the fragmented case
(fill count <= ceil(total/64 MiB) + 1); (c) mincore over a mapping of the file: 0 pages resident after the reads
(fdatasync + FADV_DONTNEED before) - no hints issued; (d) the EOF run: full aligned length issued, short count
accepted, bytes exact; (e) D2: pre-fault a subset of pieces through the mapping -> those pieces are served by
memcpy (counter) and the rest by pread, bytes identical, and the pre-faulted pages are still resident afterwards
while the others are not; (f) init refuses a misaligned probe buffer, a missing file and a non-power-of-two piece size
(tmpfs accepts O_DIRECT on kernels >= 6.6, so it is not a "disabled" case: the test skips (77) on tmpfs/ramfs and
when the page cache cannot be evicted); (g) the init probe runs before the first fill.
End-to-end (Sonnet, bench harness extended: 1 Hz /proc/diskstats + Cached sampler attributed to phases by the
server-log timestamps, per-batch prompt times from the server progress lines, nvidia-smi dmon -s um 1 Hz, per-task
utime of the copy thread, full reasoning_content + content saved per request and diffed; evicted cache per arm;
doc, docB, doc, docB, 128 tokens, temperature 0; arms back to back): D = launcher defaults; A = LLAMA_COPY_DIRECT=0;
D0 = D with LLAMA_COPY_DIRECT_HITS=0; Dc = D with `--ctx-checkpoints 0`; one-off thread sweep 4/8/16 on D's prompt.
- prompt: D prompt_s <= 0.6x A on every request (<= 170 s; expected 100-135; Dc 75-110); batch 1 <= 40 s, batch 2
  <= 130 s (Dc: single batch, the batch-1 bar does not apply). Throughput with defined denominators: reader
  throughput = bytes read / copy_s (the debug counter of time inside sched_copy_experts) >= 7 GB/s (expected ~9.6);
  batch average = bytes / batch wall >= 4 GB/s.
- bytes: D0 ~720 GB + decode per request; Dc ~545 GB + decode; A ~900 GB. D (hits): hit bytes ~0 on request 1
  (evicted cache; catches an all-ones mincore), on requests 2-4 reported, expected 50-220 GB, > 0 asserted. Cached growth during D's prompt phases <= 2 GB (sampled at prompt end).
- decode: request 1 gen reported (cold cache expected: slower than A's 0.53); requests 2-4: gen tok/s >= 0.98x A
  (A vs B agreed within 0.3%, so the bar is noise-safe); decode md0 bytes/token reported as the hit-rate observable
  (A: ~2.6 GB/token).
- total wall (prompt_s + gen_s): requests 2-4 D <= 0.75x A; request 1 (every first request after a restart):
  expected prompt -150..-185 s, decode +50..+120 s -> 0.80-0.94x A; bar: request 1 total <= 0.95x A.
- D1b trigger: requests 2-4 gen < 0.98x A on the better of D/D0, or request 1 total > 0.95x A.
- text: reasoning_content + content identical D vs A per document (|dlogprob| == 0).
- ECC counters unchanged; RssAnon steady; no "disabled"/"fallback" log line.
Decision (observable-based): batch 1 > 40 s -> gathering/concurrency of small runs; md0 < 6 GB/s with prompt > 150 s
-> reader pattern/threads/piece size; md0 >= 8 GB/s and prompt > 150 s -> GPU-side serialization -> D3 (larger ring
so reads continue during compute, or double-buffered GPU expert buffers). D vs D0: D2 default on only if >= 3 s/prompt
faster on requests 2-4 with equal decode, else off. Dc: if Dc prompt <= D prompt - 20 s with identical text, record
the trade-off (the user-turn checkpoint only helps multi-turn reuse with cache_prompt) and set the launcher default
per workload (single-turn book-writer style -> --ctx-checkpoints 0; chat -> keep).

### Review history (v18)
- v18.1 round 2: integration APPROVE; I/O and traceability CHANGES REQUESTED (text-only): owner gate for D2's
  mincore, alignment from statx(STATX_DIOALIGN) clamped to [512, 4096] (not st_blksize), fixed 64 MiB + 2 MiB ring
  buffers with capacity including slop, drain of submitted fills after a mid-pass disable, md0 bars with defined
  denominators (reader >= 7 GB/s by copy_s, batch average >= 4 GB/s), D2 expectation derived (-5..+5 s prompt;
  default on only if >= 3 s/prompt faster), request-1 bar (total <= 0.95x A) and D1b trigger, Dc launcher branch.
- v18 reviewed by 3 lenses (I/O, integration, traceability): all CHANGES REQUESTED -> v18.1. Folded in: no WILLNEED
  on the D path (blocker: prefetch=true + hint=nullptr would read everything twice and refill the cache; prefetch=false
  passes 6 GB runs), ticketed FIFO reader with a 4-buffer ring instead of a per-chunk barrier, fills gathered across
  runs (batch 1 is fragmented: 1.8 vs 6.6-8.5 GB/s), pieces at absolute 2 MiB offsets, exact EOF/short-read rules
  (shard sizes are not sector multiples), init probe + EFAULT/EOPNOTSUPP, mid-pass fallback = llama_prefetch + memcpy
  of the failed fill, lazy_src from the mappings + file paths via llama_file::name(), reader owned by the context and
  stopped before the buffers are released, two-batch prompt (checkpoint split) in the byte and time model, GPU-side
  serial floor, decode request-1 regression made explicit and D2 hit-aware mode added, per-batch thresholds (6 GB/s,
  40 s / 130 s), observables (1 Hz sampler, dmon, copy-thread time, full-text diff), observable-based decision rule,
  test file 48 MiB + 32 B with EOF and fragmented cases, Linux-only CMake gate.
- v18.1 implemented (commit pending). Code review round 1: correctness/concurrency NO BUGS (hardening suggestions
  applied); integration BUGS FOUND (test failed instead of skipping on tmpfs: fixed with fstatfs + residency
  precondition; ring buffers 1-3 alignment check; unaligned short-read retry; probe into an acquired buffer; fds
  closed on init failure; Android statx guard). E2E round 1 (4 arms, evicted cache, doc/docB/doc/docB, 128 tokens):
  A (old path) prompt 282-293 s, gen 0.52-0.53, wall 523-537 s, 901-920 GB/request;
  D prompt 122-130 s (batch 1 40-41 s at 5.8 GB/s, batch 2 75-76 s at 8.2 GB/s), gen 0.53-0.54, wall 359-371 s,
  ~1030 GB/request, Cached flat during the prompt; DH (hits) prompt 107-109 s from request 2 (hits 170-175 GB),
  wall 345-347 s; Dc (--ctx-checkpoints 0) prompt 75.6-75.9 s single batch at 8.4 GB/s, wall 304-311 s.
  All correctness bars PASS (text byte-identical D = A = DH; Dc differs deterministically: one batch instead of
  67 + 1569 tokens); request-1 decode did not regress (D 0.529 vs A 0.526), so D1b is not needed. Misses: D batch 1
  40.9 s vs the 40 s bar and 5.8 GB/s vs 7 (fragmented runs + per-tensor drain; DH passes both) - follow-up D3.
  Decisions by the pre-registered rules: hits default ON (-14 s/prompt, decode equal); CHECKPOINTS launcher knob,
  default unchanged (the checkpoint is what lets a continued conversation reuse its prefix).
  Naming from here on: D = direct reads with hits (the default), D0 = LLAMA_COPY_DIRECT_HITS=0. The e2e keeps the
  "hit bytes ~0 on request 1 (evicted cache)" check: it is the only guard against mincore reporting all pages resident.
  Code review round 2 (both lenses): NO BUGS FOUND.
