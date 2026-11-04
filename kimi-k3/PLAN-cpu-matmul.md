# Plan: CPU matmul tuning for Kimi-K3 IQ1_S decode (2026-11-03)

## Summary of what landed

| change | files | effect (CPU-only decode profile, ms/token) |
|---|---|---|
| C1 iq1_s AVX2 vec_dot: SIMD index prepass, broadcast+blend grid build, maddubs(shuffle(ls), sign(q8, grid)) | ggml/src/ggml-cpu/arch/x86/quants.c | kernel 1.5x, experts to the roofline per core |
| C4 MUL_MAT / MUL_MAT_ID: each src1 row converted to q8_K by one thread (multi-writer row caused a 4x slower dot loop on this 2-socket box) | ggml/src/ggml-cpu/ggml-cpu.c | dense GEMV 47 -> 85-92 GB/s, ~-300 |
| C5 AVX2 quantize_row_q8_K (bit-identical, reference fallback for NaN/inf/denormal) | arch/x86/quants.c | ~-12 |
| B2-B4, B6 column split for ops with fewer rows than threads (get_rows memcpy, unary, silu, concat as two strided copies, binary ops, scale) | ggml/src/ggml-cpu/common.h, ops.cpp, unary-ops.cpp, binary-ops.cpp | ~-170 |
| C7 predictor skips WILLNEED for slices whose first and last page are resident (mincore) | src/llama-moe-predict.{h,cpp} | worker 78% -> 10% of a core, ~-47 |
| C8 predictor quiet mode: router matmul moved into a custom op that returns at once when the selected experts have all been resident for 2 tokens; no guesses while quiet; HIGH hints keep probing | src/llama-moe-predict.{h,cpp}, src/models/kimi-k3.cpp | predictor free when warm, ~-40 |

All changes keep the model output byte-identical (kernel: 2M random rows; ops: 3020 cases x 5 thread counts; decode text crc unchanged on both prompts). Rejected after measurement: hierarchical barrier (C3), per-node expert split (C6), graph-level state-gather skip (B1), reductions split (B5), thread cap (C2, premise was a bench artifact).
Results: CPU-only decode 1256 -> ~740-755 ms/token; live config (GPU build, launcher defaults, 2.6k-token doc + 128): 0.71-0.78 -> 1.12-1.19 tok/s before C7/C8, final number in T3 below. Prompt unchanged (PCIe-bound).
Not done / ideas left: multi-row iq1_s GEMV (kernel is at its lookup-bound ceiling, ~85 GB/s at 44 cores), fused MoE op, nice 19 for the predictor worker, SUM_ROWS/RMS_NORM split (changes summation order).

Owner of the design: Aaron (fork `kimi-k3-expert-stream`). Planner: Fable 5.1. Coder: Opus 5.5. Tester: Sonnet 5.5.

## Where the decode token goes (measured 2026-11-03)

Box: 2x E5-2699 v4 (44 cores, AVX2/FMA/BMI2, no AVX-512, 2.6 GHz all-core), 448 GB DDR4, 2 NUMA nodes.
Raw read bandwidth, 44 cores, parallel first touch: 87 GB/s (`scratchpad/probe/bw 44 16 1`).
Decode: 0.82 tok/s = 1.22 s/token, reading ~47 GB of weights per token (dense ~35 GB, experts ~11.6 GB).

One-token matmul throughput, NUMA row split, `matmul_bench --paths a --numa` (GB/s of weight bytes):

| op | type | 44 thr | 33 thr | 22 thr | status |
|---|---|---|---|---|---|
| ffn_gate_inp 7168x896 | f32 | 90 | 90 | 88 | at roofline |
| ffn_down_exps@0:16 (layer 10) | q2_K | 88 | 85 | 77 | at roofline |
| attn_output 12288x7168 | iq4_xs | 59 | 63 | 69 | bandwidth-bound, better with fewer threads |
| attn_q 7168x12288 | iq4_xs | 47-52 | - | 47-58 | same |
| ffn_routed_up 3584x7168 | iq4_xs | 20 | - | 27 | same, small op |
| ffn_gate_shexp 7168x6144 | q5_K | 47 | 53 | 58 | same |
| ffn_gate_exps@0:16 (layer 50) | iq1_s | 33 | 30 | 17 | COMPUTE-BOUND, scales with cores |
| ffn_down_exps@0:16 (layer 50) | iq1_s | 43 | 38 | 25 | same |

Expert tensors in the model: 266 of 276 are iq1_s (all gate/up, 82 of 92 down), 10 down tensors are q2_K.

Estimated per-token split: iq4_xs dense ~0.41 s, q5_K shexp ~0.18 s, iq1_s experts ~0.30 s, f32 router ~0.03 s,
everything else (SSM scan, norms, barriers, sampling, expert prefetch) ~0.30 s.
Ceiling with every byte at 87 GB/s: 0.54 s/token = 1.85 tok/s.

The iq1_s AVX2 kernel (`ggml/src/ggml-cpu/arch/x86/quants.c`, `ggml_vec_dot_iq1_s_q8_K`) does, per 32 weights: 2 pdep,
4 scalar loads from the 16 KB `iq1s_grid` table, one `_mm256_set_epi64x` (vpinsrq/vinserti128 chain, port 5), then
`mul_add_epi8` (sign, sign, maddubs) + `madd` by the block scale. Measured ~21 cycles per 32 weights = 0.75 GB/s per core.
The q2_K kernel on the same shape reaches the memory roofline. Prior art: ik_llama.cpp PR 212 got 1.05-1.30x on AVX2 TG
for IQ1_S (memory-bound there; here we are compute-bound, so the headroom is larger); its AVX-512 kernels do not apply.

### Correction after T1 (2026-11-03, later)

- matmul_bench multi-thread times carry a per-graph floor of ~0.45 ms at 44 threads (~0.22 ms at 4-8): OpenMP threads sleep between the bench's graph computes and are woken by futex for every rep. A plain MUL_MAT of a 3.6 MB slice shows the same floor as MUL_MAT_ID, so it is not the expert path. The real decode runs the whole token inside one parallel region and does not pay this per op. Consequences: the table above understates every op below ~50 MB; "iq4_xs/q5_K faster with 22 threads" may be only the cheaper wake-up; the kernel is judged by kbench (single core, L1/DRAM), and the end-to-end numbers only by a real decode profile.
- kbench, single core, n = 3584, TSC cycles (2.2 GHz) per 32 weights, L1 / DRAM: iq1_s 7.30 / 7.52 (1.88 GB/s), q2_K 2.70 / 3.34, iq4_xs 4.36 / 5.27, q5_K 4.74 / 5.88. With 22 copies on one node: iq1_s 1.36 GB/s per core (so 44 cores ~60 GB/s), iq4_xs/q5_K ~2.9 GB/s per core = one node's bandwidth. So the iq1_s GEMV is still the only compute-bound kernel, but its real cost is ~0.19 s/token (11.6 GB at ~60 GB/s), not 0.30; a 2x kernel gain is worth ~0.06-0.08 s/token until the roofline (87 GB/s) binds. The dense ops are probably at the roofline already (~0.40 s/token), which leaves ~0.6 s of the 1.22 s token unexplained: that is now the biggest bucket and gets profiled first (T1b).

## Goals, in order

1. iq1_s AVX2 GEMV at >= 2x fewer cycles per 32 weights, bit-identical results. Expected: experts 0.30 -> ~0.15 s/token (+12-15% decode).
2. Thread cap for bandwidth-bound dense GEMVs inside the NUMA row split (iq4_xs/q5_K run faster with 11 cores per node than 22). Expected ~0.1 s/token (+8%).
3. Then: profile the ~0.3 s of non-matmul time per token; multi-row iq1_s GEMV if experts are still compute-bound; iq4_xs/q5_K kernels if they stay under ~80 GB/s.

## Rules for everyone

- No git here (the worktree's parent repo is gone). Before editing a repo file the first time, copy the original to
  `scratchpad/orig/<same relative path>` so a diff is always possible (`diff -u scratchpad/orig/x repo/x`).
- The repo has large uncommitted work (NUMA row split in ggml-cpu.c, llama-model.cpp, server checkpoint flag, launcher). Do not revert or reformat it.
- Build in `build-mm`, never in `build` (the live server runs from `build/bin`, keep it untouched until the e2e step):
  `cmake -S . -B build-mm -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_NATIVE=ON -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_UI=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache && cmake --build build-mm -j 44 --target ggml-cpu ggml llama-server test-quantize-fns test-backend-ops`
  (`ggml-cpu`/`ggml` are enough for the kernel loop; the rest only for the e2e step.)
- The live server (`llama-server` on 127.0.0.1:18090, behind a cloudflared tunnel) must keep running. Do not kill it, do not start another llama-server on the model, do not run anything that needs more than ~20 GB of RAM. Measurements that use all 44 cores are fine when `curl -s localhost:18090/slots` shows `"is_processing":false`; check before each timed run and keep runs short.
- Code style: AGENTS.md. ASCII only in code and comments (no unicode arrows/dashes), few and short comments, match the surrounding code. Keep the change inside the existing function/file structure; no new subsystems.
- Correctness bar for the kernel: bit-identical float output to the current kernel for every valid bit pattern (`qs` any byte, `qh` any 16-bit value, any Q8_K row), for row lengths 256..8192, over at least 10^6 random rows. Same accumulation order as today keeps this trivially true.
- Report numbers, not adjectives: cycles per 32 weights single-thread (L1-resident and DRAM-resident), and GB/s at 22/33/44 threads from matmul_bench, before and after.

## Tasks

### T1 (tester) kernel harness - `scratchpad/kern/`
- `kcheck.cpp`: links `build-mm/bin/libggml*`. Contains a verbatim copy of the current AVX2 `ggml_vec_dot_iq1_s_q8_K` (and `mul_add_epi8`, `hsum_float_8`, the grid via `iq1s_grid` from `ggml/src/ggml-cpu/ggml-cpu-quants.h` or wherever it is declared) renamed `ref_vec_dot_iq1_s_q8_K`. Generates random `block_iq1_s` rows (raw random bytes for qs/qh, random fp16 d in a sane range) and random `block_q8_K` rows (random qs, random d, bsums computed correctly), calls the library kernel through `ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S)->vec_dot` and the reference, and requires `memcmp` equality of the float result. Row lengths 256, 512, 3584, 7168, 8192; >= 10^6 rows total; exit code nonzero on the first mismatch with the row dumped to a file.
- `kbench.cpp`: single-thread timing of the library kernel for a type (iq1_s, q2_K, iq4_xs, q5_K): one row of 3584 repeated (L1), 64 rows (L2), and a 256 MB working set (DRAM), reporting ns per row, cycles per 32 weights (use the TSC and print its rate from `/proc/cpuinfo` or a calibration loop), and GB/s of weight bytes. Pin to one core with `taskset`.
- `run.sh`: rebuilds `build-mm` targets `ggml-cpu ggml`, runs kcheck, kbench, then `matmul_bench` for the expert and dense ops at 44,33,22 threads (`kimi-k3/tools/build_bench.sh` links against `build/bin`; add `BUILD=${BUILD:-build}` to it and build with `BUILD=build-mm`). Prints one summary table.
- Also run `build-mm/bin/test-quantize-fns` and `build-mm/bin/test-backend-ops -b CPU -o MUL_MAT` / `-o MUL_MAT_ID` filtered to iq1_s if the tool allows it; note the baseline pass/fail before any kernel change.

### T1b (tester) per-op profile of a real decode token
- Temporary, env-guarded profiler in `ggml/src/ggml-cpu/ggml-cpu.c`: when `GGML_CPU_PROFILE=1`, thread 0 of the graph compute records the wall time of every node (between the barriers of `ggml_graph_compute_thread`), aggregates per (op, src0 type) and, for MUL_MAT / MUL_MAT_ID, the src0 bytes actually read (for MUL_MAT_ID: used experts x expert slice size), and prints a table at the end of each graph compute to stderr: op, type, count, total ms, bytes, GB/s. Keep it under ~60 lines, ASCII, in one place; it will be removed later.
- Run a second, CPU-only server from `build-mm` on port 18190 (copy `scratchpad/bench/serve.sh` to `serve-mm.sh`: `build-mm/bin/llama-server`, add `-dev none`, drop `-ngl/-ot`, keep `-t 44 --numa distribute --no-host --no-repack -lm dio --lazy-mode on -fa on -c 32768 -ctk q8_0 -ctv q8_0 -ub 512`, env `GGML_CPU_MOE_PREFETCH=1 LLAMA_MOE_PREDICT=2 GGML_CPU_PROFILE=1`, port 18190). It allocates ~35 GB of dense weights; that is allowed for this task. The live server on 18090 must stay up; never use its port or its GPU (`CUDA_VISIBLE_DEVICES=` empty for the second instance).
- Send the essay prompt (`scratchpad/bench/bench.py` has the request format; use `--n 64`) twice with greedy sampling; report the profile of the second run's decode graphs (per-token average): the top 15 rows by total ms, the sum of all MUL_MAT/MUL_MAT_ID vs everything else, and the wall time per token from the server log. Then repeat with `-t 22`. Kill the second instance afterwards and confirm `curl -s localhost:18090/slots` still answers.
- Expected: total ~1.2 s/token at 44 threads; tell us where the ~0.6 s outside the weight matmuls goes (barrier counts and per-op overhead, SSM ops, GATED_DELTA_NET, norms, the output head, expert prefetch syscalls, anything with a surprising GB/s).

### C1 (coder) iq1_s AVX2 GEMV
Edit only the `#if defined __AVX2__` branch of `ggml_vec_dot_iq1_s_q8_K` (plus a helper if needed). Ideas, to be measured not assumed:
- Cut the vector-build cost: `_mm_loadl_epi64` + `_mm_insert_epi64`/`_mm_unpacklo_epi64` pairs then one `_mm256_inserti128_si256`, or `_mm256_i32gather_epi64`/`i64gather` (gather on Broadwell is slow; measure once, do not guess).
- Replace `mul_add_epi8` (3 uops) with `_mm256_maddubs_epi16(ones, _mm256_sign_epi8(q8, q1))` (2 uops) since the grid values are +-1; keep exact integer results.
- Keep the scalar index extraction cheap: the non-BMI2 form (`qs[k] | ((qh >> s) & 0x700)`) may be cheaper than pdep + movzx chains; try both.
- Unroll over 2 blocks to give the OoO core independent chains; keep per-block float accumulation order unchanged (one fmadd per block into `accum`, scalar `accum1`).
- Anything that changes the float accumulation order must be justified by a measured gain and still pass kcheck (then with a 1-ulp bar agreed with the planner first).
Deliver: the diff, cycles per 32 weights before/after (L1 and DRAM), matmul_bench GB/s at 44/22 threads for `blk.50.ffn_gate_exps.weight@0:16` and `blk.50.ffn_down_exps.weight@0:16`.

C1 result (done): 1.44-1.53x fewer cycles, bit-identical over 2M rows; 1.93 GB/s per core with 22 cores loaded (DRAM), so 44 cores reach ~85 GB/s = the roofline. Ablations: the 4 grid lookups per 32 weights are ~1/3 of the time, the design is front-end bound at ~25 uops per 32 weights; 2x needs a different algorithm. Accepted; no more kernel work on iq1_s. Diff: scratchpad/coder/c1.diff.

Real decode profile (T1b, CPU-only instance, 44 threads, 1.49 s/token): MUL_MAT iq4_xs 522 ms (21.6 GB, 41 GB/s), MUL_MAT_ID iq1_s 347 ms (9.15 GB, 26 GB/s; 353 ms at 22 threads), MUL_MAT q5_K 181 ms (46 GB/s), GET_ROWS of the 110592-float recurrent state 118 ms (231 ops, single thread, 0.5 ms each), UNARY/ADD/MUL/RMS_NORM/SUM_ROWS ~5200 tiny ops ~110 ms (~20 us each = barrier cost), CONCAT 34, SCALE 23, CPY 22, GATED_DELTA_NET 11, output head 9.5. Thread 0 waits ~90 ms per token at barriers of the dense ops: the nodes are not equally fast (T1c investigates).

### T1c result (2026-11-03): where the non-matmul time goes, and the dense gap
- NUMA placement and pinning are correct (dense 16.9/17.9 GB on N0/N1, experts 111/115 GB; thread i on cpu i, one per core; all cores at 2600 MHz; 0 page faults in decode). Node skew is 1-2%. Barriers cost 2-4 us per op (1000 dependent ADDs in one graph: 4.1 us at 44 threads), so the ~17.6k nodes per token cost at most 35-70 ms. C3 is cancelled.
- Decode at 1.256 s/token (44 threads) and 1.253 s (22 threads): halving threads costs nothing.
- Single-threaded work, ~230 ms/token: GET_ROWS of the SSM state in build_rs (node_41, 128x128x96 f32 = 6.3 MB per KDA layer, 69 per token, 1.55 ms each = 107 ms; `ggml_compute_forward_get_rows_f32` ops.cpp:5401 splits by row, nr = 1, thread 0 copies with the scalar loop `ggml_vec_cpy_f32` vec.h:119); CONCAT of the conv state (node_18, [3,12288] + [1,12288] along dim 0, 207 per token, ~70 us each = 27-31 ms; `concat_f32` ops.cpp:2037 is a per-element loop with a branch, parallel only over ne2 = 1); UNARY on 1x12288 (1085 per token, 47 us each = 41-51 ms, one row = one thread); SUM_ROWS 16 ms; RMS_NORM 11-13 ms. Sources: src/models/kimi-k3.cpp:498 and :542 call build_rs (src/llama-graph.cpp:3783-3824, get_rows at :3806; a `get_state_rows` hook exists at llama-graph.h:1016), kimi_k3_conv1d at kimi-k3.cpp:~466 builds the concat.
- Dense GEMV gap: per-thread compute inside MUL_MAT iq4_xs is 434 ms of the 472 ms op wall, balanced across threads (max/mean 1.004), i.e. 1.13 GB/s per thread, while kbench streams the same kernel at 2.9 GB/s per core with 22 cores loaded; 21.6 GB at the 87 GB/s roofline would be 250 ms. matmul_bench agrees with the in-situ op time (47 MB iq4_xs op ~0.9 ms), so this is in the op, not in the bench. Not isolated yet (C4).
- iq1_s experts at 44 threads: 227 ms wall vs 147 ms per-thread compute; the 80 ms is per-op straggler wait (max/mean 1.08 on node 0, 305 us per op). This is with the old kernel; re-measure with C1.

### C4 (coder 1) why the dense GEMV runs at 1.13 GB/s per thread inside ggml
Reproduce without the server: one graph with ~100 MUL_MAT nodes on distinct 47 MB iq4_xs weight copies (rows mbind-split like the server: see scratchpad/probe/numa_check.cpp and ggml_numa_bind_rows), ne11 = 1, 44 threads NUMA-pinned, so there is no per-graph wake-up; report ms per op and GB/s. If it reaches ~80 GB/s, the server differs environmentally (page placement, THP: check /sys/kernel/mm/transparent_hugepage/enabled and AnonHugePages of the server vs the probe; src1 layout) and the fix is there. If it does not, bisect inside ggml_compute_forward_mul_mat (ggml-cpu.c ~1380-1600): the src1 quantization + barrier, the llamafile/tiled attempts, the NUMA row split shape (each thread gets one contiguous 0.6 MB range), chunk/block sizes in ggml_compute_forward_mul_mat_one_chunk (blck_0 = 16 rows, tmp buffer), and compare with a direct vec_dot loop over the same rows in the same program. Also test the row split with the HT siblings idle vs 88 threads on the probe only. Deliver the number for each step and the fix or the explanation. The temporary profiler in ggml-cpu.c (GGML_CPU_PROFILE=1) is the tester's; keep it, use it if helpful, do not revert it.

C4 result (done): the f32 -> q8_K conversion of src1 split one 8 KB row over ~28 cores on both sockets; afterwards every thread's dot loop ran ~4x slower for the whole op (coherence pathology, measured: 1 writer 0.156 ms, 28 block writers 0.34, 128 line writers 0.82). Fix: MUL_MAT / MUL_MAT_ID convert each src1 row on one thread (round-robin over rows; prompt batches stay parallel). Dense iq4_xs/q5_K 47 -> 85-92 GB/s at 44 threads, byte-identical (probe, numa_check, matmul_bench NMSE). Diff: scratchpad/c4/c4.diff. 88 threads no better. Harness note: with OMP_PROC_BIND=true OMP_PLACES=cores GOMP sleeps the waiting threads and wake-ups cost ms; use the default env (as the live server) or OMP_WAIT_POLICY=active. With C1 + C4, iq1_s expert ops reach 63 GB/s at 44 threads (39 at 22): still below the roofline because of per-op stragglers (C6).

### C6 (coder 1) MUL_MAT_ID work split within a node
Today each used expert is split by rows over the node's 22 threads: ~70 rows = ~49 KB per thread per expert, 16 experts x 3 ops per layer, and the op waits for the slowest of 44 short streams (80 ms/token of straggler wait in T1c, 63 GB/s vs 87). Change the NUMA branch of ggml_compute_forward_mul_mat_id so that, within node k, the (used expert, row) space of node k's halves is cut into nth_node contiguous chunks: each thread gets ~1117 rows in at most two experts, long sequential streams, same rows-to-node mapping (the pages are placed by node), no atomics. Keep the non-NUMA path as is. Measure with a probe like scratchpad/c4/mm.cpp but with 100 MUL_MAT_ID nodes of 16 used experts (gate shape 3584x3072x16 iq1_s and down 3072x3584x16), 44 threads: target >= 80 GB/s; outputs byte-identical (rows are independent). Then C5.

C6 result: REJECTED. Per-node contiguous split measured gate iq1_s 1.00x, down iq1_s 0.93x op time, q2_K neutral: with C1 the expert ops run at 71-72 GB/s = 85% of the kernel ceiling (44 x 1.92 GB/s), straggler wait is no longer the limit, and the change alters the windowed prefetch order for large batches. Reverted; diff kept at scratchpad/c6/c6.diff.

### C5 (coder 1) AVX2 quantize_row_q8_K, bit-identical
Result: done, 3.7x faster per row (7168: 9.5 -> 2.6 us), byte-identical over 1e6 rows / 38M blocks incl. zero, ties, 1e37, denormals, .5 products, inf, NaN (NaN/inf/denormal-max blocks fall back to the reference). ~10-15 ms/token. Diff: scratchpad/c5/c5.diff.
The x86 `quantize_row_q8_K` is the scalar reference (ggml/src/ggml-cpu/quants.c or arch/x86); the single-writer conversion now costs 15-26 us per GEMV x ~1270 dense GEMVs = ~25 ms/token. Write an AVX2 version producing identical bytes (max-abs is exact in any order; the per-element nearest_int and the bsums are elementwise/integer). Verify with a kcheck-style comparison over 1e6 random rows of length 7168 and 12288 including rows with max at 0, ties and large values.

### B (coder 2) single-threaded ops
B2. `ggml_compute_forward_get_rows_f32` (ops.cpp ~5401): when the row count is below nth, split the work along ne0 across threads, and copy contiguous rows with memcpy. Same for the other get_rows variants only if trivial.
B3. UNARY (SILU/SIGMOID/EXP/GELU... whatever the 1x12288 and 1x7168 nodes are; the profiler keys tell: run the CPU-only instance with GGML_CPU_PROFILE=1 or read ggml_compute_forward_unary in ops.cpp/unary-ops.cpp): split along ne0 when nr < nth, keeping per-element results identical.
B4. `ggml_compute_forward_concat_f32` (ops.cpp:2037): parallel over ne1 (or ne0) when ne2 == 1, and two strided copies instead of a per-element branch.
B5. SUM_ROWS / RMS_NORM single-row: only if B2-B4 land cleanly; a two-phase reduction is not worth it below ~30 ms.
B1. Skip the state gather in build_rs when the copy is the identity (n_seqs = 1, no pending seq copy): only if the identity is known at graph-build time from the memory context (mctx: s_copy(i) == head + i) AND graph reuse (llm_graph_params comparison in src/llama-graph.cpp / llama-context.cpp) still rebuilds when that changes; otherwise do not do it and say why. The cached state must not be aliased by an op that writes its input in place.
Bar: every change keeps results bit-identical (these are copies and per-element ops); the tester checks 1 thread vs 44 threads outputs on the real shapes. Diffs small, in place, ASCII, AGENTS.md style.

B result (done): B2/B3/B4 in (common.h helper get_thread_range_rc: rows over threads when nr >= nth, else 16-aligned column blocks of every row; get_rows_f32 memcpy; apply_unary_op + silu; concat_f32 over all rows as two strided copies). Byte-identical over 682 cases x 8 thread counts. 44 threads: SSM GET_ROWS 1489 -> 263 us, conv GET_ROWS 94 -> 21, CONCAT 122 -> 17, sigmoid 12288 65 -> 6.7, tanh 7168 181 -> 10. Estimated ~150 ms/token. B1 rejected: s_copy() has a side effect (resets rs_idx) and llm_graph_input_rs::can_reuse does not compare s_copy, so a reused graph could read a stale state; after B2 the gather is ~20 ms/token, not worth it. B5 rejected: changes summation order. Trap: ggml_get_n_tasks returns 1 for GET_ROWS/SIGMOID/TANH/EXP/SUM_ROWS..., which only sizes the plan; a test graph made only of such ops is planned with 1 thread, so exactness harnesses must include one multi-threaded op (a MUL_MAT) per graph. Diff: scratchpad/coder2/b.diff.

### B6 (coder 2) single-row binary ops
ADD (~2000 per token), MUL, SCALE on 1xN rows: same helper, same bit-identical bar (per-element ops), ~25 ms/token. Then stop.

### T2 (tester) exactness for B and C4, then the full re-profile
- A harness like scratchpad/probe/numa_check.cpp: for each of GET_ROWS (src 128x128x96 f32 with one id; and 110592 x 1), CONCAT ([3,12288,1] + [1,12288,1] dim 0, inputs as strided views like kimi_k3_conv1d), the UNARY op(s) on 1x12288 and 1x7168, SUM_ROWS and RMS_NORM on 1x7168: compute with 1 thread and with 44 threads (ggml_numa_init distribute) and require memcmp-identical outputs, before and after coder 2's changes (before = sanity that the harness runs; after = the real check).
- Then, when C4 and B are in: rebuild build-mm llama-server, re-run the CPU-only decode profile (44 threads, essay prompt, 3 requests, profile the last) and report the same table as T1c plus the new per-token wall time. Then T3 (e2e on the live config) once the planner says so.

### C3 (coder) barrier cost of the ~5200 tiny ops per token (CANCELLED by T1c: barriers are 2-4 us)
With GGML_USE_OPENMP the barrier is `#pragma omp barrier` (GOMP, futex + spin). Measure first: a standalone program (like scratchpad/probe/numa_check.cpp) that runs a graph of 2000 ADD ops on 1x7168 f32 tensors through ggml_graph_compute_with_ctx and reports us per op at 44/22/11 threads, (a) with build-mm (OpenMP) and (b) with a second build `build-mm-noomp` configured like build-mm plus `-DGGML_OPENMP=OFF` (ggml's own spin barrier above, threadpool polling). If (b) is clearly cheaper, also try a hierarchical variant of ggml_barrier for the NUMA-pinned case (threads sync within their node first, then one thread per node crosses the socket), keeping the seq-cst fences. Deliver us per op for each variant and the diff; the tester then runs the real decode profile on the winning build.

### C2 (coder) thread cap for bandwidth-bound GEMV in the NUMA split
In `ggml/src/ggml-cpu/ggml-cpu.c`, the NUMA branch of `ggml_compute_forward_mul_mat` (search `ggml_numa_n_row_split`), when `ne11 == 1`: let only every k-th thread of each node's block take rows (others return at once) for types whose GEMV is bandwidth-bound; `ggml_numa_thread_rows` gets the reduced thread index/count. Env `GGML_NUMA_GEMV_THREADS` (threads per node, default chosen by measurement, 0 = all) for the experiments. Measure with matmul_bench at 44 threads: attn_q, attn_output, ffn_routed_up, ffn_gate_shexp, ffn_gate_inp, and the two iq1_s expert ops (must not slow down). Apply the cap only to the types that gain. The barrier count per op does not change, so the saving must show up as lower op time, not just higher GB/s per active core.

### T2 (tester) verify C1/C2
kcheck (bit-identical), test-quantize-fns, test-backend-ops MUL_MAT + MUL_MAT_ID on CPU, matmul_bench tables, and a `numa_check`-style run (`scratchpad/probe/numa_check.cpp`: `ref` vs `numa` outputs must stay byte-identical; rebuild it against build-mm).

### T3 (tester) end-to-end, only when the planner says so
Stop the live server, rebuild `build` (same options as now: `-DLLAMA_BUILD_UI=ON`), run `scratchpad/bench/serve.sh` + `bench.py` with the launcher defaults (THREADS=44, UB=4096, LM=dio, NUMA) on the 2.6k-token doc + 128 tokens and the essay prompt, compare `tg_tok_s` and the text crc with `scratchpad/bench/results.jsonl` (baseline 0.82-0.87 tok/s; crc must match the latest `ub4096` runs since the kernel is bit-identical), then restore the live server:
`cd /home/aaronzhong/Projects/llama.cpp-master && HOST=127.0.0.1 PORT=18090 nohup kimi-k3/run-server.sh > ~/.local/state/llama.cpp/kimi-server.log 2>&1 &` and check `curl -s localhost:18090/slots`.

### Re-profile result (T2, all changes in): 1256 -> 809.5 ms/token at 44 threads (871 at 22), text byte-identical.
Dense iq4_xs 266 ms (81 GB/s), q5_K 100 (84), iq1_s experts 224 (wall) / 143 (thread-0 compute), f32 router 59, SSM GET_ROWS 21, SUM_ROWS 17, RMS_NORM 13, GDN 10, head 9.
T4: freeing a core for the helpers (taskset + -t 43/42) removes the expert straggler wait (80 -> 30 ms) but the token gets 35% slower (compute threads idle 40%): rejected. Found: one unpinned "llama-server" thread at ~80% duty during decode, floating over cpus 0-87 (HT siblings of busy cores). It is a llama_moe_predictor worker (src/llama-moe-predict.cpp worker_loop: drain_high/prefetch_expert -> numa_willneed = set_mempolicy + posix_madvise(WILLNEED) + set_mempolicy per expert slice), i.e. page-cache walks of pages that are already resident. The compute threads do the same inside every MUL_MAT_ID (ggml_cpu_mul_mat_id_prefetch, GGML_CPU_MOE_PREFETCH=1): 16 used experts over 22 threads per node, so 16 threads start late and the op waits (straggler).

### T5 (tester) cost of the hints when the cache is warm
CPU-only instance, warm cache, 3 requests, profile the third, 44 threads: (a) baseline (GGML_CPU_MOE_PREFETCH=1, LLAMA_MOE_PREDICT=2) = 809 ms; (b) GGML_CPU_MOE_PREFETCH=0; (c) LLAMA_MOE_PREDICT unset (no predictor); (d) both off. Report ms/token, MUL_MAT_ID iq1_s wall vs thread-0 compute, the busy helper's cpu-seconds. If (d) is much faster: the fix is to skip hints for experts hinted recently (per-expert last-hint step/time in the predictor and in the CPU prefetch; a hint repeated within ~60 s is useless since the page cache does not evict that fast here), keeping the cold-cache streaming path intact.

T5 result: predictor on/off = 824 / 708 ms per token (warm cache, identical text); the predictor worker runs at ~80% duty and causes the 80 ms expert straggler wait (31 ms without it); the compute-side GGML_CPU_MOE_PREFETCH costs only 3-6 ms. The predictor is still needed for cold experts (live config streams from disk), so make it cheap when pages are resident.

### C7 (coder 2) skip hints for resident expert slices
In src/llama-moe-predict.cpp prefetch_expert (both the numa_willneed and the posix_madvise paths): before hinting a slice, test residency with mincore() on the first and the last page of the slice (page-aligned addresses, 1 page each; for file mappings mincore reports page-cache residency); if both are resident, skip the hint and count it (a `resident_skips` counter in the stats line). Keep everything else (accounting, drain order, cold path) unchanged. Optionally the same two-page test in ggml_cpu_mul_mat_id_prefetch (ggml-cpu.c) per expert, since it is cheap. Verify: warm-cache CPU-only decode with the predictor ON should approach the 708 ms of T5(c), text identical; a cold-cache request (scratchpad/bench/evict.py drops the model's pages) must still stream with the same disk GB and similar prompt/decode time as before the change (compare against a cold run of the current build). Do not run all-core instances or benchmarks until the planner says the live e2e is finished.

### T3 result (live config, GPU build, launcher defaults, docs/multi-gpu.md 2638 tokens + 128, greedy)
old build (same day): 0.71 / 0.775 tok/s; new build (C1, C4, C5, B2-B4, B6): 1.121 / 1.186 tok/s; prompt 73 s both (PCIe-bound); text crc 597259948 identical. Old libraries kept in build-prev/bin for rollback.
Final build (+ C7, C8, profiler removed), after two page-cache evictions: doc run 0.993 / 1.117 tok/s (343 / 231 GB from disk), crc 597259948; essay (100-token prompt, 128 tokens, 3 requests): 1.002 / 1.144 / 1.238 tok/s (52 / 51 / 41 GB from disk), old build on this prompt 0.79-0.83. The live config still reads ~40 GB per request (prompt expert passes via O_DIRECT plus decode misses), so it sits below the fully warm CPU-only figure (1.33 tok/s, 0 GB).

C7 result (done): ends_resident() = mincore on the first and last page of each slice; skip hint when both resident, counted as resident_skips. Worker 78% -> 10% of a core; warm 825 -> 778 ms/token (predictor off: 736 that day); cold: same 244 GB from disk, same prompt time, identical text. Remaining ~42 ms/token with the predictor on is not the worker any more (T6).

### T6 (tester) the remaining predictor cost on the compute side
With the current build-mm (C7 in), CPU-only instance, 44 threads, warm, essay, 3 requests, profile the third: predictor on vs predictor unset. Diff the per-op tables by node name/op (the 183 extra nodes: CUSTOM moe_hint and whatever else the predictor adds, see src/llama-moe-predict.cpp on_layer / get_hint_ud and where llama-graph or the model file inserts them), and report where the ~42 ms goes: the custom ops' own time, extra barriers, or slower neighbours (e.g. an op that now runs on one thread, or ARGSORT/GET_ROWS of the router). Also run kcheck quick and opcheck 3 on this build.

T6 result: predictor on vs off (C7 in, warm) = +57-66 ms/token: the extra router MUL_MAT f32 (next layer's ffn_gate_inp on this layer's router input, kimi-k3.cpp:395-397 on_moe_selected; 91 x 25.7 MB = 2.3 GB/token) +31 ms; CUSTOM moe_hint (llama_moe_predictor_hint_op, thread 0 only, 70-77 us per layer) +7 ms; MUL_MAT_ID barrier wait +10 ms; small-op jitter the rest. Worker at 10% of a core.

### C8 (coder 2) predictor quiet mode
Goal: warm decode with the predictor on = predictor off (within noise), cold streaming unchanged, graph static (no graph-reuse changes).
1. Quiet detection in llama_moe_predictor: track residency of the HIGH hints (the experts the layer actually selected, on_layer -> enqueue high). With C7 each hint already does ends_resident(); keep a running window (e.g. the last 2 tokens = ~184 layer events, or a count of consecutive all-resident layers). quiet = every selected expert of the window was resident. Any non-resident selected expert (or any disk read accounted by account_load) -> active immediately. Quiet starts false. Expose `bool quiet() const`.
2. When quiet: the worker issues no LOW (guess) hints, and the hint op does no ranking/enqueue (it still must not break the step/layer bookkeeping that drain_high relies on: step_seen/layer_seen must advance). HIGH hints for the selected experts keep running on the worker (they are the residency probe and the cold path), each gated by ends_resident() as in C7.
3. The extra router matmul must cost nothing when quiet. Preferred: move it out of the graph into the hint custom op: the op takes moe_router_in and layers[il_next].ffn_gate_inp as sources; when active and n_tokens is small (<= 8, decode and speculative-size batches; prediction for large prompt batches is not useful since the batch uses nearly all experts: check what on_layer does today for large n_tok and keep that behaviour), it computes the 896 logits per token across all threads (ggml_vec_dot_f32 over the f32 weight rows, split rows over nth, barrier via the op's normal completion) and then ranks on thread 0 as today; when quiet, it returns at once. Remove the ggml_mul_mat at kimi-k3.cpp:395-397. If moving the matmul is too invasive, fall back to: keep the MUL_MAT node but use a q8_0 copy of ffn_gate_inp made at load (590 MB) and skip the hint work when quiet; say which you did and why.
4. Optional, measure only if cheap: worker thread at nice 19 (setpriority on the worker's tid) to reduce the expert-op barrier wait and jitter.
Bar: text identical (crc 2032798946 on the essay, 597259948 on the live doc run); warm on vs off within ~10 ms; cold run (evict.py once, then one request) same disk GB and prompt/decode time as the C7 cold numbers (244 GB, 95 s, ~0.8 tok/s); test-moe-predict passes; prediction quality unchanged when active (the predictor's stats: hit rate / low_done, compare a cold run before/after).

### Round 2 (after T3)
- Per-op profile of one decode token (temporary timer in `ggml_graph_compute_thread`, thread 0, behind an env var; removed afterwards) to split the ~0.3 s non-matmul time.
- Multi-row iq1_s GEMV (2 or 4 rows per call sharing the Q8_K loads) if experts are still below ~70 GB/s at 44 threads.
- iq4_xs / q5_K GEMV review if they stay under ~80 GB/s with the thread cap.
