# PLAN v2: where should the decode matmuls of Kimi-K3 IQ1 run? (measurement only)

Question: with 1 token per step, is any matmul better off on the CPU? Claims to confirm or refute, each with a numeric pass rule:
(1) 1-token matmul time = weight read time: PASS if a path's effective GB/s >= 0.7x that path's measured raw-read ceiling (B2).
(2) CPU reaches only ~half of RAM bandwidth on IQ1_S / Q2_K / IQ4_XS: report per type the ratio CPU-matmul GB/s / measured DDR4
    read GB/s (14 threads). Trace hints: dense IQ4_XS ~0.8x, experts ~0.2x, so the claim is probably wrong in both directions.
(3) GPU-in-VRAM ~40x faster: PASS if ratio >= 20x (pre-check: ~15x). "28 GB does not fit": computed from nvidia-smi free VRAM at
    the server's settings minus KV/recurrent state, divided by per-layer dense bytes -> N layers that fit (feeds B4).
(4) GPU zero-copy over PCIe slower than CPU for dense: PASS if projected all-GPU-zero-copy tok/s < today's tok/s (B4).
The deliverable LEADS with one line ("decode is bound by X; moving Y to the GPU buys Z tok/s") and the B4 table; details follow.
The pivotal comparison: aggregate CPU expert throughput (trace: ~9.7 GB / ~0.77 s compute ~ 12.6 GB/s, A3) vs PCIe zero-copy GB/s
of the real kernel (B1 (c)). Decides go/no-go of the user's design (pinned expert cache, GPU zero-copy, LRU, predictor prefetch).

Machine: E5-2690 v4 14C/28T, 4ch DDR4 (speed: measured in B2), 256 GB, RTX 3080 Ti 12 GB, PCIe Gen3 x16 (link gen sampled, B2).
Build: build/ (CUDA, release). Model: /mnt/k3/kimi-k3-iq1/*.gguf, 93 layers, two layer types:
  KDA (69 of 92 MoE layers; blk.1): attn_q/k/v/output, ssm_g IQ4_XS 46,792,704 B each; ffn_{gate,up,down}_shexp IQ4_XS 30,277,632;
      ffn_routed_{up,down} IQ4_XS 13,647,872; ffn_gate_inp F32 25,690,112; ssm_beta 365,568, ssm_f_a 487,424 (IQ4_XS),
      ssm_f_b 884,736 (IQ4_NL). Dense ~378 MB/layer.
  MLA (every 4th: 3,7,...,91; blk.3): attn_gate, attn_output IQ4_XS 46,792,704; attn_q_a 5,849,088, attn_q_b 15,040,512,
      attn_kv_a_mqa 2,193,408, attn_v_b 3,342,336 (IQ4_XS), attn_k_b 3,538,944 (IQ4_NL); shexp x3 Q5_K 30,277,632;
      routed x2 IQ4_XS; ffn_gate_inp F32. Dense ~267 MB/layer.
  CPU layers 11-92 = 61 KDA + 21 MLA -> ~28.7 GB dense. output.weight Q6_K 963 MB (GPU, 1 matmul/token); token_embd Q6_K
  963 MB (GET_ROWS, 1 row/token, CPU) - both inventoried, output timed in B1, token_embd listed only.
  Experts: ffn_{gate,up}_exps IQ1_S 896 x 2,150,400 B (ne0 3584, ne1 3072); ffn_down_exps Q2_K 896 x 3,612,672 B (layers 1-10)
  / IQ1_S 2,150,400 B (11-92), ne0 3072, ne1 3584. Per token ~9.7 GB; experts of ALL layers 1-92 run on the CPU today.
Implementer inventories shapes/types/nb in C++ with gguf.h (gguf_init_from_file, no_alloc); python gguf needs numpy (absent).

## Part A - from existing traces (python only, kimi-k3/tools/trace_matmul.py, python3 -I, reuse trace_check.load_trace)
Inputs: /tmp/claude-1000/-home-aaronzhong-Projects-llama-cpp/9bbbffdd-0d58-401e-a700-32b343899505/scratchpad/bench/tr/r4/d2.json
(detail 2, 4 tokens, 2.2-2.5 s/token, ~590 ms/token I/O stall) and /home/aaronzhong/kimi-k3-trace-simple.json (8 tokens, 2233-2535 ms).
Events: CPU op = cat "op", tid 100, args.op in {MUL_MAT, MUL_MAT_ID}; GPU device time = cat "op", pid 2, args.stream present;
cat "launch" = host launch (reported, never added to device time); expert work = cat "expert_thread" (dur, args.stall_ns).
Pairing: per layer, MUL_MAT ops sorted by ts, split by layer type (trace counts: KDA layers 15 MUL_MAT ops, MLA layers 14, layer 10: 14, layer 92: 13, layer 0: 11 — the next-layer router
matmul can land on either layer, so anchors drive the pairing, not raw counts; KDA: ops in layers 1,2,4,.. and 11-92 non-MLA; MLA: layers
3,7,10 GPU vs 11,15,..,91 CPU; layer 0 (11 ops) and 92 (13 ops) excluded from pairing, reported alone). Anchor ranks on the named
ops kda_out-L, ffn_moe_logits-L, ffn_shexp-L (same rank in both halves, else fail loudly); map ranks to weights by reading the
layer graph in src/models (kimi-k3) and check sum(bytes of mapped weights) == dense bytes of that layer type. Median + p90 over
tokens and layers.  Tables -> kimi-k3/tools/out/partA.md:
  A1 per layer type and rank: weight, type, MB, CPU us, GPU device us, ratio, CPU GB/s, GPU GB/s
  A2 per layer type: sum CPU MUL_MAT ms vs sum GPU device ms; per token = 61 x KDA + 21 x MLA (actual layer mix, not x82)
  A3 experts per slice type (IQ1_S gate/up, Q2_K down L1-10, IQ1_S down L11-92): per-expert compute us = dur - stall_ns,
     per-thread GB/s, AGGREGATE GB/s = per-layer sum of expert bytes / (MUL_MAT_ID dur - stall share) and x14 per-thread;
     stall seconds per token; this aggregate is the number the go rule uses
  A4 token budget: wall; CPU dense MUL_MAT; MUL_MAT_ID compute; stall; CPU non-matmul ops (~98 ms); GPU ops of layers 0-10 +
     output; GPU ops of CPU layers (KV/recurrent cache, 3153 launches); syncs (5497); copies (1733, ~34 ms); unaccounted.
     Validate "today": today_compute = wall - stall must land within 15% of 1.67 s (step 0 of d2); the 1.37 s all-cached
     figure is NOT in any trace and is reported as unverified unless a trace reproduces it.

## Part B - standalone benchmark on real tensors (kimi-k3/tools/matmul_bench.cpp, bw_probe.cu, build_bench.sh)
Built with g++/nvcc against build/bin/libggml*.so (-rpath); no ggml/llama source change, no CMake target. test-backend-ops perf
rejected (fixed cases, random data, no zero-copy). Weights: gguf.h + pread() O_RDONLY from the shard (no mmap of 540 GB).
Weights timed: all dense weights of blk.50 (KDA) and blk.51 (MLA), output.weight, expert slices gate (IQ1_S, L50), down L50
(IQ1_S), down L5 (Q2_K). Slice e = bytes [e*nb[2], (e+1)*nb[2]) with the tensor's own ne0/ne1 -> plain 2D MUL_MAT.
Cache hotness: GPU L2 6 MB and CPU LLC 35 MB exceed slices, routed, shexp. Per weight use K distinct copies (K = ceil(256 MB /
bytes), K = max(1, ...): a weight >= 256 MB is cold by size; output.weight (963 MB) runs (b), (c), (d) one after another,
freeing VRAM/pinned buffers between paths, so host footprint >= 256 MB and VRAM >= 64 MB), one graph with K MUL_MAT nodes (own src1 / dst, random f32
src1 fixed seed), time whole graph / K after 3 warmups, 20 repeats, median + min. State K per weight class in the output.
Timing: steady_clock around ggml_backend_graph_compute + ggml_backend_synchronize (K-node graphs >= 5 ms); CUDA graph capture
setting (GGML_CUDA_DISABLE_GRAPHS) identical for (b)(c)(d) and recorded; cudaEvents only in bw_probe.cu.
Paths:
 (a) CPU backend, threads 7 / 14 (headline) / 28, ggml_backend_cpu_set_n_threads; weight in CPU buffer (once also cuda-host).
 (b) CUDA0, weight in a CUDA device buffer.
 (c) CUDA0 zero-copy: posix_memalign(4096) host copy; cudaHostRegister(ptr, size, Portable|ReadOnly) directly (retry without
     ReadOnly), check the return (ggml_backend_cuda_register_host_buffer is a no-op without GGML_CUDA_REGISTER_HOST); query
     cudaDevAttrCanUseHostPointerForRegisteredMem, use cudaHostGetDevicePointer() if 0; registration outside timing. Tensor
     created in a real CUDA buffer of the same size, that VRAM cudaMemset 0xFF, then tensor->data = device pointer of the host
     copy. (c) passing NMSE proves it read host memory (0xFF VRAM would fail); bit-identity with (b) is secondary.
     Assert the kernel is mul_mat_vec_q: n=1 quantized src0 (check ggml_cuda_mul_mat dispatch, confirm with nsys once). Concurrency: 4 ggml_backend_cuda_init(0) instances, own graphs/copies, 4 host threads, time first
     launch to last sync; if too fragile, drop it and take concurrency from bw_probe only (state which).
 (d) CUDA0 staged: cudaMemcpyAsync pinned -> VRAM then (b); copy time reported separately.
Ceilings (bw_probe.cu, cudaEvents): CPU multi-thread streaming read of 4 GB (7/14/28 threads -> DDR4 speed); pinned H2D 1 GB;
GPU kernel reading pinned host memory, coalesced sum over 1 GB, grid sweep 1x/4x/16x SMs, 1 and 4 streams, best reported;
VRAM read 1 GB. A background loop samples nvidia-smi pcie.link.gen.current / width during the probe (idle shows gen 1) and the
run records the gen/width seen under load. Not measured, noted: NVMe DMA and GPU zero-copy share the memory controller.
Correctness: reference = CPU f32 dequant (ggml_get_type_traits()->to_float) x f32 dot; NMSE < 5e-4 for (a)(b)(c)(d) on every
weight; (c),(d) also compared bit-wise to (b). A failing path is reported, not ranked.

## Outputs (kimi-k3/tools/out/partB.md + raw JSON) - B4 and the one-line answer come FIRST
 B4 projected s/token and tok/s, formula per row written out, inputs from A4 (overhead, stall) and B1 (t_slice, t_dense):
    today            = overhead + dense_cpu@14t(61 KDA + 21 MLA) + experts_cpu + stall ; must match trace wall within 15%
    today, zero stall= same without stall (what prefetch alone could buy)
    today + N dense layers in VRAM (N from claim (3) free-VRAM math; those layers at (b) times, minus their syncs/copies)
    user's design    = overhead + dense_cpu@14t + 16 x [10 x (2 IQ1_S + Q2_K) + 82 x 3 IQ1_S] x t_slice(c) + stall + extra
                       ping-pong (per-layer sync + copy costs from A4 totals, scaled by added CPU<->GPU hops)
    experts split CPU+GPU concurrently: expert bytes / (aggregate CPU GB/s (A3) + zero-copy GB/s (B1 c)), same ping-pong
    everything GPU zero-copy = overhead - (syncs + copies + GPU-ops-of-CPU-layers ping-pong, A4) + (dense + experts) at (c)
    dense in VRAM listed "does not fit" with the N that does.
 B1 per weight: name, type, MB, K, CPU ms @7/14/28t, VRAM ms, zero-copy ms (1 / 4 instances), staged copy ms + VRAM ms,
    GB/s per path, NMSE per path.  B2 ceilings: DDR4 read GB/s per thread count, H2D, zero-copy read (best grid), VRAM, PCIe
    gen/width under load.  B3 per-token totals: dense 61 x KDA + 21 x MLA + output; experts 16 x [10 x (2 IQ1_S + Q2_K) + 82 x 3 IQ1_S].
 Verdict per claim (1)-(4) with the pass rule above and the numbers.
Go/no-go for the user's design: GO only if B4 "user's design" >= 1.3x today AND zero-copy GB/s of the real kernel (B1 c) >
AGGREGATE CPU expert GB/s (A3; compare per-token expert time, not per-thread) AND the win is not absorbed by stall (the design
moves matmuls, not I/O; stall stays unless prefetch improves). Otherwise NO-GO with the limiter named (CPU dequant, PCIe, or I/O).

## Isolation / safety
No llama-server. Model files O_RDONLY; nothing written under /mnt/k3. GPU: nvidia-smi first, abort if < 3 GiB free; <= 1 GiB VRAM,
<= 1.5 GiB pinned; register one weight class at a time, unregister before the next; free everything on exit (also on error).
Skip runs when load average > 2 (user's server decoding). Outputs under kimi-k3/tools/out/; build artifacts under
/tmp/claude-1000/-home-aaronzhong-Projects-llama-cpp/9bbbffdd-0d58-401e-a700-32b343899505/scratchpad/bench/matmul/.

## Who does what
Opus implementer: trace_matmul.py, matmul_bench.cpp, bw_probe.cu, build_bench.sh (AGENTS.md style: ASCII, few comments).
Sonnet: builds, runs Part A on both traces, Part B 3 times for variance, fills partA.md / partB.md, flags any failed NMSE, any
pairing anchor mismatch, and any "today" projection off the trace wall by > 15%. Reviewers per the global agreement.

## Review history
- v2 approved by both reviewers (CUDA/method, traceability) with two text fixes applied: K = max(1, ...) and sequential paths for
  output.weight (budget), trace MUL_MAT counts per layer type stated; GGML_CUDA_DEBUG dropped (not a real switch; nsys).
- v1 -> v2: traceability review: two layer types (KDA/MLA) with 14/13/11 MUL_MAT, not 15; inventory both, pair per type with named
  anchors, actual 61+21 layer mix; output/token_embd added; numeric pass rules per claim; 1.37 s baseline not in traces -> validate
  against d2 (1.67 s compute) and the 8-token simple trace, A4 split; B4 rows added (N layers in VRAM, split experts, zero stall)
  with ping-pong overhead modelled; B4 + one-line answer first; assert mul_mat_vec_q. CUDA review: direct cudaHostRegister (ggml
  wrapper is env-gated), device pointer query; K copies against L2/LLC hotness; steady_clock timing, CUDA graph setting fixed;
  multi-instance concurrency or bw_probe only; 0xFF VRAM poison proves host read; PCIe link gen sampled under load, grid sweep;
  explicit design formula with unchanged stall and aggregate CPU throughput; expert slice via nb[2] with real ne0/ne1.
