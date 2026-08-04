# memopt-pipeline-w3 integration/compound.md

Single-gene wave (S4 only). No stacking with other genes.

## g1 (S4 low-bit KV paged-attn)
- **owner**: w3-sole
- **champion tip**: 602d795311ca647dc8480aa3246328a41ee1887d
- **champion scores**: peak RSS q4_0 = 8,946,450,432 B (-43 MB vs q8_0 floor
  8,989,589,504 B); tg32 5.72 t/s (+27pct vs q8_0 4.52). Correctness:
  q4_0/q8_0 100pct token-match f16 via flash; paged path broken for all types (SWA bug).
- **standalone gain**: -43 MB peak RSS, +27pct decode t/s, correctness-of-quant verified.
- **verdict**: PARTIAL (promoted to review branch; not a clean ship due to small RSS win + pre-existing paged/SWA bug).
- **integrated-at-SHA**: 6b47815e2 (evolve-review/memopt-pipeline-w3 on main repo).
- **patch path**: integration/patches/g1.patch.

## Skipped / not attempted
- S5 (InfiniGen prefetch): needs host-tier KV cache; S2 did not ship a real win. SKIPPED.
- S6 (MoE disk offload): gemma-4-12B is dense, no payoff. SKIPPED.
- S8 (speculative expert prefetch): extends S6. SKIPPED.

## s2 re-verification (LLAMA_KV_LAZY_CLEAR)
- lazy clear OFF (median): 8,957,345,792 B
- lazy clear ON  (median): 8,863,778,176 B  (-93 MB, -1.04pct)
- Contradicts PREFLIGHT's +0.59 GB regression. Within the session's noise band
  (~40-50 MB run-to-run spread on the floor itself). Left OFF by default.
  Inconclusive; needs a controlled multi-run study to call.

## Canonical stack
Single champion only (g1). The compounding test is moot for a single gene;
g1 stands on its standalone scores. Because the win is small (-0.48pct RSS)
and the end-to-end paged path is broken on this model, the practical
recommendation is: q8_0 (S1, already shipped) remains the sweet spot for
this model until S1's SWA-in-paged bug is fixed and the workload uses a
longer context where the KV fraction is larger.
