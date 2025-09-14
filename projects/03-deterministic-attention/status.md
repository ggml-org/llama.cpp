Status — Project 03 Deterministic Attention
==========================================

03A (CUDA Forward)
------------------
- [x] Deterministic dispatcher (vec/tile) and single-block launch policy.
- [x] Tests (batch/run determinism); passing on NVIDIA Ada via container.
- [x] Docs updated with Attention (CUDA).

03B (Coverage & Quantized K/V)
------------------------------
- [x] Dispatcher: deterministic vec path for quantized K/V (D=128, q4_0/q4_0 and q8_0/q8_0). Special head sizes via MMA are gated OFF in det mode pending single-column MMA.
- [x] Tests: quantized K/V determinism + batch invariance; head-size tests disabled by default (enable with RUN_MMA_HEADSIZE_TESTS=1).
- [x] Docs: quantized K/V coverage; clarified that special head sizes are not yet supported in det mode.
- [x] Runbook added (Ada/Ampere via container).

03C (KV-Cache + Other Backends)
-------------------------------
- [ ] KV-cache invariance across incremental decode.
- [ ] Metal/Vulkan/OpenCL/HIP deterministic attention policy + tests.
- [ ] Softmax deterministic fallback when FlashAttention not available.
