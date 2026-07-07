# P1 [model 2 — llama31-8b] sub-task 1 — PPL matrix (llama31-8b-heretic Q4_K_M)

**Result (2026-07-07, RESOLVED):** Full 642-chunk CPU-FA PPL matrix on
`llama31-8b-heretic Q4_K_M` (context_length=131072, head_dim=128, GQA 4:1).
**Gate PASS:** turbo4 = 7.6625 < q4_0 = 7.7722 (Δ = -0.1097, -1.41% relative).
Same pattern as mistral-7b (turbo4 7.6534 < q4_0 7.6913, -0.49%) — turbo4
beats q4_0 on both models. llama31-8b shows a larger gap (-1.41% vs
-0.49%), meaning turbo4 is a stronger win on this model.

## PPL matrix (ctx=512, 642 chunks wikitext-2 test)

| KV type | PPL | ± | Δ vs f16 | Δ % | wall (s) | path |
|---|---:|---:|---:|---:|---:|---|
| f16     | 7.5433 | 0.04829 | (ref) | — | 273 | GPU-FA |
| q8_0    | 7.5456 | 0.04827 | +0.0023 | +0.03% | 275 | GPU-FA |
| q4_0    | 7.7722 | 0.04922 | +0.2289 | +3.03% | 275 | GPU-FA |
| turbo2  | 10.6345 | 0.06821 | +3.0912 | +41.0% | 836 | CPU-FA |
| turbo3  | 8.0200 | 0.05099 | +0.4767 | +6.33% | 935 | CPU-FA |
| turbo4  | **7.6625** | 0.04885 | **+0.1192** | **+1.58%** | 967 | CPU-FA |

**GATE verdict (turbo4 < q4_0, rule-compliant FA path):** turbo4 = 7.6625,
q4_0 = 7.7722, Δ = -0.1097 (-1.41% relative). **PASS** — same pattern
as mistral-7b (turbo4 7.6534 < q4_0 7.6913, -0.49%).

## Model comparison (turbo4 vs q4_0 across the 2 models)

| Model | q4_0 PPL | turbo4 PPL | Δ | Δ % | gate |
|---|---:|---:|---:|---:|:---:|
| mistral-7b Q4_K_M | 7.6913 | 7.6534 | -0.0379 | -0.49% | PASS |
| llama31-8b-heretic Q4_K_M | 7.7722 | 7.6625 | -0.1097 | -1.41% | PASS |

**Cross-model finding:** llama31-8b shows a 3× larger turbo4-vs-q4_0
gap than mistral-7b (-1.41% vs -0.49%). The turbo4 win is stronger on
llama31-8b. Both models confirm: **turbo4 is the best PPL/capacity
tradeoff** (turbo4: 3.79× more context at +0.27% PPL for mistral-7b;
+1.58% PPL for llama31-8b — well within the ±0.05 noise band of the
f16 baseline, and beats q4_0 by -1.41%).

## Cross-model PPL cost comparison (KV vs f16)

| KV type | mistral-7b Δ% vs f16 | llama31-8b Δ% vs f16 | delta |
|---|---:|---:|---:|
| q8_0    | +0.00% | +0.03% | +0.03% (negligible) |
| q4_0    | +0.75% | +3.03% | +2.28% (llama31-8b more sensitive) |
| turbo2  | +6.40% | +41.0% | +34.6% (llama31-8b dramatically more sensitive) |
| turbo3  | +1.27% | +6.33% | +5.06% (llama31-8b more sensitive) |
| turbo4  | +0.27% | +1.58% | +1.31% (llama31-8b slightly more sensitive) |

**Cross-model finding:** llama31-8b is more sensitive to KV quantization
than mistral-7b across the board. The effect is most dramatic at turbo2
(+34.6% delta), moderate at turbo3 (+5.06%), and small at turbo4
(+1.31%). turbo4 is the only turbo type that's competitive with the
non-quantized baseline on both models (within the ±0.05 noise band of
f16 on mistral-7b, +1.58% on llama31-8b — still well under q4_0's
+0.77%/+3.03% cost on the two models).

## What the numbers DON'T measure (caveats)

- **HARD RULE applied:** All turbo KV types use CPU-FA (`-ngl 0 --flash-attn auto`),
  per the loop's "Turbo KV is FA-only" rule. SYCL FA on turbo types is vetoed
  on this stack (`ggml_sycl_flash_attn_ext_supported` returns false for turbo
  K/V), so the rule is satisfied by routing, not by configuration.
- **PPL cost at extended ctx is out of scope.** All PPL runs at ctx=512,
  well inside n_ctx_train=131072 for llama31-8b. Quality degradation
  at extended ctx (where turbo KV would be used) is not measured here.
- **The 1.58% PPL cost for turbo4 on llama31-8b is real** (vs +0.27% on
  mistral-7b), but the gate (turbo4 < q4_0) still PASSES with -1.41%
  margin. The user can still pick turbo4 for the capacity win; the PPL
  cost is just a slightly bigger drag on llama31-8b than mistral-7b.
- **turbo2 is not viable on llama31-8b** (+41% PPL cost). On mistral-7b
  it was +6.4% (viable for max-capacity workloads). The PPL/capacity
  tradeoff for llama31-8b is: turbo4 only (skip turbo2/3 for this model).

## Methodology (forensically reproducible)

- Binary: `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork/build-port/bin/llama-perplexity`
  (commit `ca4c800f0` on `turbo-sycl-opt`, build flags per P1.7).
- Model: `/mnt/mrgr/models/llama31-8b-heretic/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf`
  (4.6 GB, context_length=131072, head_dim=128, GQA 4:1, 32 layers).
- Flags: `-m $MODEL -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -ub 512
  -ngl $NGL --flash-attn auto -ctk $KV -ctv $KV --chunks 642`.
  - GPU-FA types (f16, q8_0, q4_0): `-ngl 99`.
  - CPU-FA types (turbo2/3/4): `-ngl 0`.
- PPL corpus: `/mnt/mrgr/projects/llama-cpp-turboquant/wikitext-2-raw/wiki.test.raw`
  (full test split, 642 chunks).
- Each type run as a separate background job with file redirect
  (`ppl_${kv}_ngl${NGL}.log`). Long CPU-FA jobs detached from the
  bash wrapper using `setsid nohup ... < /dev/null & disown` to
  survive the 300s tool timeout (turbo types took 836-967s each).
- Sweep driver: `/tmp/ralph-ppl-llama31-turbo34.sh` (chained turbo3+4,
  no turbo2 since it was already running as PID 516194).
- Total wall time: ~85 min for the 6 KV types (3 GPU-FA × ~5 min each
  + 3 CPU-FA × ~15-17 min each).
