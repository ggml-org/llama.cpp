Status — Project 03 Deterministic Attention
==========================================

03A (CUDA Forward)
------------------
- [x] Deterministic dispatcher (vec/tile) and single-block launch policy.
- [x] Tests (batch/run determinism); passing on NVIDIA Ada via container.
- [x] Docs updated with Attention (CUDA).

03B (Coverage & Quantized K/V)
------------------------------
- [ ] Dispatcher: support-probe vec for quantized K/V; allow MMA for D∈{80,96,112,576}.
- [ ] Tests: quantized K/V (D=128, q4_0/q8_0); additional head sizes; skips for unsupported combos.
- [ ] Docs: quantized K/V coverage; special head sizes; caveats.
- [ ] Runbook added (Ada/Ampere via container).

03C (KV-Cache + Other Backends)
-------------------------------
- [ ] KV-cache invariance across incremental decode.
- [ ] Metal/Vulkan/OpenCL/HIP deterministic attention policy + tests.
- [ ] Softmax deterministic fallback when FlashAttention not available.

