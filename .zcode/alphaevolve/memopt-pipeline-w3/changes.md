# memopt-pipeline-w3 changes log

[iso8601] agent gene gen scores verdict reason
2026-08-01T15:45Z agent=w3-sole gene=g1 gen=1 scores=rss=8.95GB_q4(tg=5.72ts) verdict=live reason="q4_0/q5_0/q2_K fused dequant implemented; q4_0 -43MB vs q8_0 baseline(8.99GB), +27pct decode; baseline paged path broken for f16 on this model (SWA), end-to-end correctness unverifiable"
2026-08-01T15:55Z agent=w3-sole gene=g1 gen=2 scores=correctness verdict=pruned reason="quant validated via flash: q4_0 and q8_0 both 100pct token-match f16 (80/80); paged path broken for ALL types incl f16 (SWA bug); gen1 is champion"
2026-08-01T15:58Z agent=w3-sole gene=s2-reverify gen=0 scores=rss_off=8.96GB_rss_on=8.86GB verdict=note reason="lazy clear ON -93MB vs OFF in this session (median 3-run), opposite of PREFLIGHT +0.59GB; within noise band, inconclusive"
