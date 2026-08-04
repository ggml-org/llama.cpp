# memopt-pipeline-w4 changes.md

[2026-08-02T22:25:00Z] agent=w4-single gene=g1 gen=0 baseline scores: qwen2.5-0.5B f16 paged=1pct vs flash, gemma-4-12B f16 paged=1pct vs flash, ctest=PASS verdict=baseline reason=baseline-floor-before-fix
[2026-08-02T22:40:00Z] agent=w4-single gene=g1 gen=1 candidate=swa-filter-only scores: qwen_paged=1pct gemma_paged=1pct verdict=pruned reason=SWA-filter-is-a-no-op-for-short-prompts-(n_swa=1024);-did-not-fix-the-bug;-(kept-in-final-diff-as-defensive-long-context-correctness)
[2026-08-02T23:05:00Z] agent=w4-single gene=g1 gen=2 candidate=v_trans-force+swa-filter+gating scores: qwen_paged=100pct gemma_paged=1pct ctest=PASS verdict=live reason=v_trans-fix-resolves-qwen-and-all-non-hybrid-models;-gemma-hybrid-has-separate-undiagnosed-bug
[2026-08-02T23:15:00Z] agent=w4-single gene=g1 freeze champion=dab0d5117 verdict=promoted stacked_on_main=true reason=partial-fix-shipped
