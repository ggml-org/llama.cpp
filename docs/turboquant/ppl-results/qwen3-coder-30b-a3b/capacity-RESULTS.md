# P1 [model 3 — Qwen3-Coder-30B-A3B] sub-task 2 — Capacity Gain (Q3_K_XL, MoE, GQA 8:1, Arc A770)

**Result (2026-07-07, RESOLVED with partial matrix):** Binary-search sweep
on `Raudbjorn-fork` build-port, init-only oracle (`--chunks 0
--no-warmup`).

## Capacity table (init-only, n_par=1)

| KV type | max ctx (FIT) | KV buffer MiB | model MiB | init VRAM free MiB | KV bytes/token |
|---|---:|---:|---:|---:|---:|
| f16    | **15248** | 1440 | 12994 | 15473 | 96 B (2× theoretical 48 B) |
| q8_0   | **28964** | 1454 | 12994 | 15473 | 51 B (2× theoretical 24 B) |
| q4_0   | **54872** | 1451 | 12994 | 15473 | 27 B (2.2× theoretical 12 B) |
| turbo2 | KV=17136 at c=524288 (CPU-FA, -ngl 0) | 17136 | 12994 | 15473 | 32 B (V only) |
| turbo3 | KV=17856 at c=524288 (CPU-FA, -ngl 0) | 17856 | 12994 | 15473 | 36 B (V only) |
| turbo4 | KV=19584 at c=524288 (CPU-FA, -ngl 0) | 19584 | 12994 | 15473 | 37 B (V only) |

**Cross-model comparison (single-stream, n_par=1, max ctx for GPU-FA types):**

| Model | f16 | q8_0 | q4_0 | turbo2 | turbo3 | turbo4 | turbo4/f16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mistral-7b  (4.1 GB model) | 82304 | 155648 | 293888 | 524416 | 423680 | 311552 | **3.79×** |
| llama31-8b  (4.4 GB model) | 79764 | 150528 | 285440 | 508928 | 411392 | 302336 | **3.79×** |
| Qwen3       (12.7 GB model) | 15248 | 28964 | 54872 | (CPU-FA) | (CPU-FA) | (CPU-FA) | (n/a) |

**Qwen3-specific finding:** the Qwen3 12.7 GB model takes **80% of the
16 GB Arc A770 VRAM**, leaving only ~3 GB for KV + compute. This is
the binding constraint — the KV ceilings are 5-6× smaller than the
7-8B models (which use ~25-30% of VRAM for the model, leaving 10-11 GB
for KV). The capacity-gain ratio can't be measured the same way for
Qwen3 (the corpus cap of 131K is unreachable on GPU-FA — every KV
type OOMs at the v10 hi-cap).

**turbo types on Qwen3 use CPU-FA (-ngl 0), not GPU-FA, because the
12.7 GB Qwen3 model + any meaningful KV doesn't fit in the remaining
~3 GB of VRAM. The turbo KV at c=524288 takes 17-20 GB of system RAM
(K=q8_0=13 GB + V=turbo{2,3,4}=4-7 GB). This means the capacity
claim for turbo KV on Qwen3 is "much more context than the 3 GB VRAM
allows, but only via system RAM (CPU-FA)." The PPL cost is the
limiting factor (per sub-task 1, turbo2/3 KILLED on MoE + CPU-FA,
turbo4 OK with +1.58% PPL cost at 50-chunk probe).

**Qwen3 capacity matrix is not directly comparable to mistral-7b /
llama31-8b because:**
- Model size: 12.7 GB vs 4.1-4.4 GB (3× larger)
- VRAM budget for KV: 3 GB vs 10-11 GB (~3-4× smaller)
- Turbo types: CPU-FA only (host RAM) vs GPU-FA (VRAM)

**The reframe's "use turbo4 for capacity" claim cannot be cleanly
validated on Qwen3 within the 16 GB VRAM budget — the model itself
is too large.** For MoE users with this hardware, the
practical capacity claim is:
- f16/q8_0/q4_0 on GPU-FA: 15K-55K ctx depending on KV type
- turbo2/3/4 on CPU-FA: 524K+ ctx (system RAM limited, not VRAM)
  but turbo2/3 KILLED on PPL (sub-task 1), only turbo4 viable

## Cross-model PPL cost × capacity-gain pattern (Qwen3, init-only)

Qwen3 f16/q4_0/q8_0 capacity ratios (from this sub-task):
  q8_0/f16 = 28964/15248 = 1.90× (≈ models 1+2: 1.89× — identical)
  q4_0/f16 = 54872/15248 = 3.60× (≈ models 1+2: 3.57-3.58× — identical)
  q4_0/q8_0 = 54872/28964 = 1.89× (≈ 2× per 4-bit quant level)

**The capacity-gain ratios for the GPU-FA types are model-invariant
(1.89× per 4-bit quant level, 3.57-3.60× for 8-bit total),** confirming
the finding from models 1+2 that the capacity-gain ratio is determined
purely by the KV bytes-per-token ratio, not by the model architecture.
The Qwen3 model just has smaller absolute ceilings because the model
itself takes more VRAM.

## What the numbers DON'T measure (caveats)

- **Concurrent sequences axis (deferred to P1.8, applies across all 3
  models).** v12 used `llama-perplexity -b 4` (logical batch, not
  n_parallel). Real concurrent capacity needs
  `llama-server --parallel N`. Qwen3's smaller per-slot VRAM budget
  makes the concurrent-sequestration cost even more pronounced.
- **PPL cost is measured separately in sub-task 1.** This sub-task
  measures capacity only (init-only, no forward pass). PPL cost
  is f16=9.70, q4_0=9.87 (+1.77%), q8_0=9.70 (+0.01%) on Qwen3
  (full 564-chunk corpus, GPU-FA). turbo4 PPL 8.91 (50-chunk probe,
  CPU-FA, asymmetric K=q8_0+V=turbo4). turbo2/3 KILLED on PPL.
- **Quality at extended ctx is out of scope.** All PPL runs at ctx=512,
  well inside n_ctx_train=131072 for Qwen3. Capacity sweeps measure
  raw VRAM residency, not compute correctness at extended ctx.
- **turbo types use HOST RAM, not VRAM** for Qwen3. The
  "capacity-gain" claim for turbo on Qwen3 is "system RAM budget
  expanded to 524K+ ctx" — a different kind of capacity story than
  the VRAM-bound claim for the 7-8B models.

## Methodology (forensically reproducible)

- Binary: `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork/build-port/bin/llama-perplexity`
- Model: `/mnt/mrgr/gguf/Qwen3-Coder-30B-A3B-UD-Q3_K_XL/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf`
  (13 GB, qwen3moe arch, 48 layers, n_embd=2048, n_head=32, n_head_kv=4,
  GQA 8:1, n_expert=128, n_expert_used=8, context_length=262144,
  Q3_K_XL weights, model 12.7 GB resident on GPU).
- Flags: `-m $MODEL -f wikitext-2-raw/wiki.test.raw -c $CTX -b 1 -ub 512
  -ngl $NGL --flash-attn auto -ctk $KV -ctv $KV --chunks 0 --no-warmup
  --verbose`.
  - GPU-FA types (f16, q4_0, q8_0): `-ngl 99`
  - CPU-FA types (turbo{2,3,4}): `-ngl 0` (model on CPU, KV can use
    up to ~25 GB of system RAM; capacity-claim measured in system
    RAM terms, not VRAM)
- v12 sweep driver: `/tmp/ralph-cap-v12.sh` (parameterized for any
  model). Two-phase: v10 (range [1024, 131072]) + v11 (doubling hi
  on FIT to push dense types past corpus cap). The Qwen3 v10 ceilings
  were already well below the corpus cap (model too large for >55K
  ctx GPU-FA), so v11 push found no higher ceilings.
- Turbo types at c=524288: single init-only probe per type (not a
  binary search — the sub-task 1 PPL finding limits turbo{2,3} to
  CPU-FA, and c=524288 was a sanity-check probe to confirm the
  capacity claim for the production turbo type only — turbo4).
- Retro-patch: `/tmp/ralph-cap-v12-enrich-qwen3.py` (fill
  init_vram_free_mib + model_mib from raw logs after the sweep).
