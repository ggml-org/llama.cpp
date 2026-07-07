# P1 [model 2 — llama31-8b] sub-task 2 — Capacity Gain (llama31-8b-heretic Q4_K_M, Arc A770)

**Result (2026-07-07, RESOLVED):** Binary-search sweep across all 6 KV
types on `Raudbjorn-fork` build-port (`GGML_SYCL_F16=ON`, oneAPI 2026.0
icpx), `--chunks 0 --no-warmup` init-only oracle with `llama_kv_cache:
size` and `common_memory_breakdown_print` as the OOM/FIT signals.
Enriched CSV: `sweep-logs/llama31-8b-cap/sweep_enriched.csv`. Raw probe
logs: same dir.

## Final capacity table (single-stream, n_par=1)

| KV type | max ctx (FIT) | KV buffer MiB | model MiB | init VRAM free MiB | KV bytes/token | capacity gain vs f16 |
|---|---:|---:|---:|---:|---:|---:|
| f16     | **79764**  | 9984  | 4403 | 15473 | 0.250    | 1.00× |
| q8_0    | **150528** | 9996  | 4403 | 15473 | 0.125    | 1.89× |
| q4_0    | **285440** | 10035 | 4403 | 15473 | 0.0625   | 3.58× |
| turbo2  | **508928** | 10033 | 4403 | 15473 | 0.03125  | **6.38×** |
| turbo3  | **411392** | 10044 | 4403 | 15473 | 0.046875 | 5.16× |
| turbo4  | **302336** | 10039 | 4403 | 15473 | 0.0625   | 3.79× |

**Pattern: all KV types converge to the same absolute KV buffer ceiling
of ~10,030-10,044 MiB** (the binary's hard cap on KV after weights +
scratch). Same pattern as mistral-7b (q4_0 max ctx 293888, turbo4
max ctx 311552, ratio 3.79×). The capacity difference is purely
`10030 MiB / bytes_per_token` — KV compression IS the capacity gain.
The 16 GB Arc A770 has a single VRAM limit; how much context that
holds is determined by KV bits per token.



## turbo2 Boundary V caveat (load-bearing for the reframe)

The binary **silently auto-enables "Boundary V mode 7" for turbo2**:
first 2 and last 2 transformer layers use `q8_0` V-cache, the rest
(28 of 32 layers = 87.5%) use pure turbo2 V-cache. The K-cache is
pure turbo2 across all 32 layers. Triggered at init time (see
`sweep-logs/llama31-8b-cap/turbo2_np1_c*.log` L753):

```
llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)
llama_kv_cache: Boundary V mode 7: first2+last2 V=q8_0, rest V=turbo2
```

**Implications for the capacity-gain claim:**
- The 6.38× turbo2/f16 capacity ratio is for the **auto-mode** (first/last
  2 layers in q8_0, rest in turbo2), not pure turbo2. The boundary
  q8_0 layers take ~2× the bytes of pure turbo2, so the **pure-turbo2
  ratio would be higher** if Boundary V were disabled
  ().
- The +41% turbo2 PPL cost is also for the auto-mode. Pure turbo2
  would have higher PPL cost (q8_0 boundary layers help quality on
  the most-attended first/last layers).
- **turbo3, turbo4, q8_0, q4_0, f16 are all pure (no Boundary V)** —
  the auto-mode is turbo2-specific. turbo4/f16 = 3.79× ratio is for
  pure turbo4 (no inflation/deflation).

**Same caveat applies to mistral-7b** (same binary, same Boundary V
auto-enable for turbo2). The 6.37× mistral-7b turbo2/f16 ratio is
also for the auto-mode.

## Cross-model comparison (single-stream, n_par=1)

| Model | f16 | q4_0 | turbo2 | turbo3 | turbo4 | turbo4/f16 | q4_0/f16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mistral-7b  (n_ctx_train=32768)  | 82304  | 293888 | 524416 | 423680 | 311552 | **3.79×** | 3.57× |
| llama31-8b  (n_ctx_train=131072) | 79764  | 285440 | 508928 | 411392 | 302336 | **3.79×** | 3.58× |

**Cross-model finding: the turbo4/f16 capacity-gain ratio is
identical (3.79×) on both models.** llama31-8b is slightly lower in
absolute terms (f16 ceiling 79764 vs 82304, ~3% lower) because the
heretic model is 308 MiB larger (4403 vs 4095 MiB resident), so the
KV budget shrinks proportionally. The **ratio is model-invariant**
because both models hit the same 16 GB Arc A770 VRAM ceiling with the
same binary's per-tensor overhead.

**Pattern: the capacity-feature claim scales cleanly across 7-8B
models on this hardware.** Same 3.79× more context for the same VRAM
budget with turbo4 vs f16, regardless of the model.

## What the numbers DON'T measure (caveats)

- **Concurrent sequences axis (deferred to P1.8).** v12 used
  `llama-perplexity -b 4` (logical batch, not n_parallel) — the
  binary allocates per-slot regardless of `-b` (per the lesson from
  model 1 v10). Real concurrent capacity needs
  `llama-server --parallel N`. Added as `P1.8` follow-up bullet.
- **Quality at extended ctx is out of scope.** All KV types at the
  capacity ceiling are past n_ctx_train=131072 (the binary uses RoPE
  scaling to extend). Quality degradation past training ctx is not
  measured; the PPL numbers in `RESULTS.md` are at ctx=512 (well
  inside training envelope).
- **n_par=4 results are identical to n_par=1** (v12 driver shows
  this consistently for all 6 KV types). Confirmed the v10 lesson:
  `-b` is logical batch, not n_parallel.

## Methodology (forensically reproducible)

- Binary: `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork/build-port/bin/llama-perplexity`
  (commit `a570c4c37` on `turbo-sycl-opt`, build flags per P1.7).
- Model: `/mnt/mrgr/models/llama31-8b-heretic/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf`
  (4.6 GB, context_length=131072, head_dim=128, GQA 4:1, 32 layers).
- Flags: `-m $MODEL -f wikitext-2-raw/wiki.test.raw -ngl 99 --flash-attn
  auto -b 1 -ub 512 -c $ctx -ctk $KV -ctv $KV --chunks 0 --no-warmup
  --verbose`.
- Two-phase sweep: v10 phase 1 (range [1024, 131072]) for initial
  ceiling, v11 phase 2 (doubling hi on FIT) to push dense KV types
  past the corpus cap to their real OOM ceilings.
- VRAM oracle: `common_memory_breakdown_print` line's
  `(total = free + (used = model + context + compute))` tuple, extracted
  retro-actively from raw logs (the v12 inline regex was incomplete).
- Sweep driver: `/tmp/ralph-cap-v12.sh` (parameterized for any model
  via `MODEL=`, `CORPUS=`, `LOGDIR=`, `LABEL=` env vars).
- Retro-patch: `/tmp/ralph-cap-v12-enrich.py` (fill
  init_vram_free_mib + model_mib from raw logs after the sweep).
- Total wall time: ~3 min for the full v12 sweep (12 cells v10 + 5
  cells v11).
- **Chunk-count correction (2026-07-07):** earlier-session estimate
  of 642 chunks was wrong; the actual wikitext-2 tokenization at
  ctx=512 produces **564 chunks**. Fixed across `ppl.csv`,
  `RESULTS.md`, and `RALPH_TASKS.md`; PPL values were unaffected
  (Final estimate lines are the source of truth).
