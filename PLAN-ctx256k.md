# PLAN: 256k context for the Kimi-K3 launcher (kimi-k3/run-server.sh)

User request (2026-10-18): "i played with ui, seems context is too small. need at least 256k".

## Measured (IQ1 model, OFFLOAD=1, UBATCH 2048, 12 GB RTX 3080 Ti, 10796 MiB free at start)

| config | CUDA0 model | KV | RS | compute | result |
|---|---|---|---|---|---|
| today: CTX 8192, GPU_LAYERS 11, f16 KV | 4722 | 216 | 443 | 4227 | runs (unaccounted 1050 idle -> 1278 MiB after 6 requests) |
| CTX 262144, GPU_LAYERS 0, q8_0 KV (output head on GPU) | 919 | 3672 | 443 | 5895 | OOM at compute alloc |
| CTX 262144, GPU_LAYERS 0, q8_0 KV, output head on CPU | 0.05 | 3672 | 443 | 5895 | **loads** (818 MiB free idle; no request processed yet). Expect ~590 MiB free at runtime |

- KV is K-only (MLA latent 576 wide, V is a view of K), 24 attention layers; f16 at 256k = 6912 MiB, never fits.
- CUDA FA (DKQ 576 / DV 512 always uses MMA_F16, never VEC) converts q8_0 K to f16 in memory reserved inside the
  compute buffer, sized at reserve time with n_kv = n_ctx (init_full), V reuses it (V_is_K_view). So 5895 MiB is the
  worst case at full context; other runtime pool use is bounded (KV_max, FA fixup by SM count, cuBLAS) — verified by
  the plan review, still to be measured under load (tests 1-3).
- `-ctk q8_0` alone fails ("different K and V cache types") -> pass both.
- `^output\.` matches only output.weight; the head matmul has n_outputs = 1 < 32, so op offload never uploads it.

## Rejected
- KV in host RAM (`-nkvo`): attention runs on the CPU (op offload only applies to weight buffers). MLA attention is
  compute bound: ~0.85 TFLOP per decoded token at 256k -> +1-2 s/token; prompt batches far worse.
- q4_0 KV: quality risk on the MLA latent; not needed.
- Smaller UBATCH: every batch streams all experts once, so prompt time scales with 1/UBATCH.

## Change (run-server.sh only)
Two profiles, chosen by CTX alone (one rule, nothing coupled to GPU_LAYERS):

| | CTX <= 8192 (previous measured config) | CTX > 8192 (new default 262144) |
|---|---|---|
| KV_TYPE default | f16 | q8_0 |
| GPU_LAYERS default (IQ1, OFFLOAD=1 / 0) | 11 / 19 | 0 / 0 |
| GPU_LAYERS default (other models) | 8 | 0 |
| output head | GPU | CPU (`^output\.=CPU` added to -ot) |

1. `CTX` default 8192 -> 262144.
2. `KV_TYPE` env: only f16 | q8_0 accepted (the only measured ones), passed as both `-ctk` and `-ctv` (the model
   rejects different types). Refuse KV_TYPE=f16 with CTX > 131072 (6.75 GiB KV at 256k cannot fit). Refuse
   `-ctk|--cache-type-k|-ctv|--cache-type-v` args (existing normalization covers `_` and `=value` spellings).
3. User-set GPU_LAYERS > 0 with CTX > 8192: allowed, one-line warning ("unmeasured; each dense layer ~345 MiB VRAM,
   ~600 MiB free at 256k with 0 layers — OOM at 'failed to allocate compute pp buffers' means lower it").
4. `CACHE_RAM` env -> `--cache-ram` (MiB). Default 32768 when CTX > 8192, server default otherwise. Why: each context
   checkpoint stores the 443 MiB recurrent state; up to 32 per slot (min spacing 8192) -> ~14 GB at 256k, plus
   KV+RS ~4.1 GB; the server's 8192 MiB default would evict a long chat when the user switches conversations and it
   would be reprocessed from scratch (hours). Host has 256 GB (49 GB pinned dense). Refuse `-cram|--cache-ram` args.
5. Startup echo: "GPU dense layers: none, output head on CPU" when GPU_LAYERS=0 (no more "0..-1"); print KV type and
   cache-ram with ctx.
6. Header comment: new defaults, measured table, costs:
   - decode: dense layers + head on CPU ~+0.09 s/token (0.53 -> ~0.49 tok/s); at long context the GPU also
     dequantizes the whole K cache per token (~20-25 ms at 256k).
   - prompt: ~70-100 s per 2048-token batch near an empty context, plus FA ~15-25 s per batch near full context;
     a fresh 256k prompt takes hours; a chat that grows turn by turn only processes each new turn.
   - host RAM: checkpoints + prompt cache up to CACHE_RAM.

## Tests (dynamic, Sonnet; isolated port 18190 on 127.0.0.1; no other llama process; stop everything after)
Assert numbers:
1. Default start + one real generation: log `n_ctx = 262144`, `K (q8_0)`, CUDA0 model buffer < 1 MiB, KV 3672 MiB;
   `/props` n_ctx 262144; one completion returns tokens; no "converting K and V to f16 instead" / vec-fallback warning.
2. Baseline workload (~/kimi-k3-bench/docB.txt ~1.6k tokens + 128 generated), CHECKPOINTS=0 and default: decode
   >= 0.45 tok/s (baseline 0.53), prompt within +15% of 76 s / 108 s. Shutdown memory breakdown: CUDA0 free >= 300 MiB,
   unaccounted <= 1400 MiB.
3. Long prompt ~32k tokens with a planted needle near the start ("The vault code is 7391-QX"); assert the answer
   contains exactly "7391-QX"; nvidia-smi sampled 1 Hz: peak used recorded, free >= 300 MiB throughout.
4. Web-UI scenario via /v1/chat/completions (stream, cache_prompt true, full history resent each turn), 4 turns:
   turn N+1 `timings.cache_n` >= turn N prompt+generated tokens minus the last few (template re-render), `prompt_n`
   == only the new text (< 200 tokens). Then a second conversation, then back to the first: assert via timings
   (cache_n >= the first conversation's prior total minus template re-render, prompt_n small); secondary: no WRN
   eviction lines ("removing oldest entry", "exceeds cache size limit") in the log (cache-hit lines are trace level).
5. Arg checks (no model load needed where the script dies first): `-ctk f16` refused; `KV_TYPE=q4_0` refused;
   `KV_TYPE=f16 CTX=262144` refused; `--cache-ram 1` refused; `CTX=8192` echo shows GPU layers 0..10, f16, head on GPU
   (and loads: model buffer ~4722 MiB); `CTX=262144 GPU_LAYERS=1` prints the warning (load and report fit/OOM).
6. Soak (background after commit, reported later): one conversation to >= 250k tokens, final generation works, no OOM,
   VRAM free logged. Rollback if it OOMs: report to the user with options (UBATCH 1536 frees ~0.4 GB at +33% prompt
   time; KV q5_1/q4_0 frees 0.4-1.7 GB) — do not silently lower CTX below 256k.

## Review history
- v1: initial.
- v3 (code review): f16 KV size corrected to 6912 MiB.
- v2: runtime review — head placement by CTX not GPU_LAYERS; KV f16 kept for CTX <= 8192; refuse f16 at large CTX;
  runtime VRAM assertions; long-context attention costs; no-vec-fallback check. UX review — "loads" not "runs";
  CACHE_RAM sized for checkpoints at 256k; echo fix; KV_TYPE limited to f16|q8_0; warning for GPU_LAYERS > 0 at large
  CTX; web-UI multi-turn + conversation-switch test; exact needle; defined soak rollback.
