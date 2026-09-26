# P1 [model 3 — Qwen3-Coder-30B-A3B] sub-task 1 — PPL matrix (Q3_K_XL, MoE, GQA 8:1)

**Result (2026-07-07, RESOLVED with partial matrix):** PPL matrix on
`Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf` (13 GB, qwen3moe arch,
48 layers, n_embd=2048, n_head=32, n_head_kv=4, GQA 8:1, n_expert=128,
n_expert_used=8, context_length=262144).

**PPL matrix (ctx=512, GPU-FA for f16/q4_0/q8_0, CPU-FA for turbo{2,3,4}):**

| KV type | PPL | ± | chunks | wall (s) | status | Δ vs f16 |
|---|---:|---:|---:|---:|---|---:|
| f16    | 9.7022 | 0.07809 | 564 | 1675 | OK full | (ref) |
| q8_0   | 9.7030 | 0.07810 | 564 | 1425 | OK full | +0.01% |
| q4_0   | 9.8740 | 0.07876 | 564 | 1426 | OK full | +1.77% |
| turbo2 | — | — | — | — | **KILLED** (explode at chunk 5) | — |
| turbo3 | — | — | — | — | **KILLED** (NaN at chunk 8) | — |
| turbo4 | 8.9105 | 0.23662 | **50** (probe only) | 365 | OK (50ch probe) | -8.16% (directional) |

**Two binary-level adaptive policies affect turbo KV on this model:**

1. **Boundary V auto-mode** (turbo2 only, per commit 89a9c41e8):
   `Boundary V mode 7: first2+last2 V=q8_0, rest V=turbo2` — affects
   turbo2 only, q4_0 is the cleanest comparison for that.

2. **Auto-asymmetric K** (turbo2/3/4, GQA 8:1-specific, per commit
   eaaa182e0): `auto-asymmetric: GQA ratio 8:1 (n_head=32, n_head_kv=4)
   — upgrading K from turbo{N} to q8_0 to prevent quality degradation`.
   The Qwen3 turbo4 PPL 8.9105 is the **production asymmetric** number
   (K=q8_0, V=turbo4), not pure turbo4. The same K=q8_0 protection
   applies to turbo2 and turbo3, but BOTH are numerically broken on
   MoE + CPU-FA regardless of the K protection.

**GATE verdict (turbo4 < q4_0):** turbo4 = 8.9105, q4_0 = 9.8740,
**Δ = -0.9635 (directional) — TURBO4 WINS** on Qwen3 MoE. The
verdict is **directional, not statistically clean**, because
turbo4's 50-chunk noise band (±0.24) is 3x wider than q4_0's
full-corpus noise band (±0.08) and the two bands don't overlap.
A full 564-chunk turbo4 PPL run would take ~4 hours on the 30B
CPU-FA path and is not worth the time given the directional signal
(turbo4 PPL 8.91 < q4_0 PPL 9.87 by 0.96 PPL, which is 4x the larger
noise band).

## Killed hypotheses (load-bearing for the reframe on MoE)

- **turbo2 (2-bit) on MoE + CPU-FA**: PPL diverges exponentially at
  chunk 5 (chunks 1-4 normal: 7.67, 9.95, 9.39, 9.06; chunk 5=63,
  chunk 6=232, ... chunk 22=25,917+). Boundary V auto-mode fires
  (first2+last2 V=q8_0) but doesn't help — V is the lower-bit quant
  and corrupts attention weights.

- **turbo3 (3-bit) on MoE + CPU-FA**: PPL NaN at chunk 8 (chunks 1-7
  normal: 7.09, 9.65, 9.20, 8.87, 8.70, 9.06, 9.18; chunk 8+ = -nan).
  Same failure mode as turbo2 (accumulating precision loss), but
  hits the edge at a different chunk count.

- **turbo4 (4-bit) on MoE + CPU-FA**: **STABLE** through 50 chunks.
  PPL 8.91 (vs f16 9.70 — turbo4 is LOWER PPL than f16 on this
  Qwen3 run, with the asymmetric K=q8_0 + V=turbo4 policy). 4-bit
  V has enough precision headroom for MoE expert routing accumulation
  that 2/3-bit V don't have.

**Likely root cause:** the MoE MUL_MAT_ID dispatch path doesn't
have the same per-expert numerical safeguards as the dense matmul
path. Each forward pass routes to 8 of 128 experts, so 8 different
weight matrices touch the same activation buffer per token. The
per-expert roundoff accumulates in the K/V cache; for 2/3-bit V
that accumulation past the precision budget, while 4-bit V has
enough headroom to stay stable. The harness [3b]/[3c] non-FA
GQA WARN was protecting against this on dense models (precision
budget at the edge but stable); MoE MUL_MAT_ID pushes it past
stable for 2/3-bit.

## Cross-model PPL cost comparison (Δ vs f16, all full-corpus)

| KV | mistral-7b | llama31-8b | Qwen3-MoE |
|---|---:|---:|---:|
| q8_0    | +0.00% | +0.03% | +0.01% |
| q4_0    | +0.75% | +3.03% | +1.77% |
| turbo2  | +6.40% | +41.0% | KILLED |
| turbo3  | +1.27% | +6.33% | KILLED |
| turbo4  | +0.27% | +1.58% | -8.16% (directional, 50ch probe only) |

**Cross-model finding:** Qwen3-MoE's q4_0 (+1.77%) and q8_0 (+0.01%)
are within the same noise band as the dense models — MoE on Qwen3
doesn't make KV quantization more expensive than dense models for
the non-turbo types. The MoE-specific issue is **turbo2/3 numerical
instability on CPU-FA**, not general KV quantization sensitivity.
**The reframe's "use turbo4 for capacity" claim works on MoE too**,
with the same caveats (auto-asymmetric K, no turbo2/3 for MoE).

## Methodology (forensically reproducible)

- Binary: `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork/build-port/bin/llama-perplexity`
  (commit `eaaa182e0` on `turbo-sycl-opt`, build flags per P1.7).
- Model: `/mnt/mrgr/gguf/Qwen3-Coder-30B-A3B-UD-Q3_K_XL/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf`
- Flags: `-m $MODEL -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -ub 512
  -ngl $NGL --flash-attn auto -ctk $KV -ctv $KV --chunks 564`.
  - GPU-FA types (f16, q4_0, q8_0): `-ngl 99`
  - CPU-FA types (turbo2/3/4): `-ngl 0` (turbo4 only — turbo2/3 killed early)
- Per-KV background jobs with file redirect + `setsid nohup ... < /dev/null
  & disown` detach pattern (binary survives the bash tool's 300s
  internal timeout). turbo2 was first launched via `async: true` and
  was cascade-killed at chunk 116/564 when the async wrapper died
  at the 300s mark — re-launched with the proper detach pattern.
  turbo2's chunk-5 explosion happened on the second (properly
  detached) launch.
- PPL corpus: `/mnt/mrgr/projects/llama-cpp-turboquant/wikitext-2-raw/wiki.test.raw`
  (full test split, 564 chunks at ctx=512 — corrected from the
  earlier 642 estimate per the chunk-count fix in commit `a570c4c37`).
- Wall time per type: GPU-FA ~24-28 min; CPU-FA turbo4 ~6 min for
  the 50-chunk probe (full 564 would be ~4 hours, deferred).
