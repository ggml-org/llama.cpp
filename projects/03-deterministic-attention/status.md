Status — Project 03 Deterministic Attention
==========================================

03A (CUDA Forward)
------------------
- [x] Deterministic dispatcher (vec/tile) and single-block launch policy.
- [x] Tests (batch/run determinism); passing on NVIDIA Ada via container.
- [x] Docs updated with Attention (CUDA).

03B (Coverage & Quantized K/V)
------------------------------
- [x] 03B.0 dispatcher: quantized vec-only (D=128 q4_0/q4_0, q8_0/q8_0) + probes + fallbacks; special head sizes gated OFF by default.
- [x] Tests: quantized determinism + batch invariance; optional FORCE_* smokes; head-size probe tests gated by env.
- [x] Docs/runbook updated.
- [x] 03B.1 tile coverage for D∈{80,96,112} + tests (CUDA Ada/Ampere; single‑column tile path; no logit softcap).
- [x] 03B.2 observability and toggles (one‑time INFO; GGML_DET_ATTENTION_DISABLE_TILE_80_96_112).
- [x] 03B.3 ncols=1 MMA for 80/96/112 (opt-in) + tests. (prototype landed; validated on Ada; Ampere run pending)
- [ ] 03B.4 enable MMA by default for 80/96/112 after soak.
- [ ] 03B.5 576/512 ncols=1 MMA + tests.

03C (KV-Cache + Other Backends)
-------------------------------
- [ ] KV-cache invariance across incremental decode.
- [ ] Metal/Vulkan/OpenCL/HIP deterministic attention policy + tests.
- [ ] Softmax deterministic fallback when FlashAttention not available.
