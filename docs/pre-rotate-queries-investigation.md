# Pre-Rotate-Queries Investigation Log

## Goal
Move WHT inverse rotation from per-block dequant (O(128) per 4-element access) to graph-level Q forward + V inverse rotation (O(1) per-access amortized). This would reclaim ~77 tok/s speed while maintaining PPL ~6.19.

## Model Under Test
- Qwen3.5-35B-A3B-Q8_0 (MoE)
- n_embd_head_k = 256, n_embd_head_v = 256, n_head = 16, n_head_kv = 2
- WHT rotation group = 128 (QK_TURBO3 = 128)
- Each head has 2 rotation groups (256/128 = 2)

## ggml_mul_mat Semantics (Verified)
- ggml stores 2D tensors column-major: element(i,j) at offset i + j*ne[0]
- Storing a C row-major array M into ggml: ggml sees M^T
- ggml_mul_mat(A, x) computes A^T @ x
- NET EFFECT: storing row-major M and calling mul_mat gives M @ x
- VERIFIED with 2x2 rotation test: stored R row-major, got R@x output

## Rotation Matrix Storage
- TURBO_ROTATION_R: R = diag(s2) * H/sqrt(128) * diag(s1) (forward rotation)
- TURBO_ROTATION_RT: R^T = R^{-1} (inverse rotation)
- For Q forward: store R (TURBO_ROTATION_R) -> mul_mat gives R @ q
- For V inverse: store R^T (TURBO_ROTATION_RT) -> mul_mat gives R^T @ cur = R^{-1} @ cur
- Python verified: R^T @ R = I, round-trip error = 1.2e-15

## Test Results (all on Qwen3.5-35B-A3B, wikitext-2, 8 chunks, turbo3 K+V, flash attn)

### Baseline
| Config | PPL | Notes |
|--------|-----|-------|
| Dequant inverse ON, no graph rot | **6.194** | Known good baseline |
| No dequant inverse, no graph rot | 194 | Fully rotated, no compensation |

### Graph Q Rotation Only (no V inverse)
| Storage | mul_mat gives | PPL | Notes |
|---------|---------------|-----|-------|
| TURBO_ROTATION_RT for Q (R^T@q) | R^T@q = R_inv@q | 157 | Wrong direction for K matching |
| TURBO_ROTATION_R for Q (R@q) | R@q | 157 | Correct direction but V still rotated |

Both give ~157 because V output is still in rotated space (no V inverse). The attention scores differ but V corruption dominates.

### Graph Q + V Rotation (both active)
| Q Storage | V Storage | Q gives | V gives | PPL |
|-----------|-----------|---------|---------|-----|
| RT (R^T@q) | R (R@cur) | R^T@q (wrong) | R@cur (wrong) | 26.6 |
| R (R@q) | RT (R^T@cur) | R@q (correct) | R^T@cur (correct) | **23.5** |

Corrected storage is better (23.5 < 26.6), confirming direction matters. But 23.5 >> 6.19.

### Isolation Tests (dequant inverse ON)
| Graph rotation | PPL | Notes |
|----------------|-----|-------|
| Q rot only (R@q) | 10.5 | Q rotation works - degrades quality vs un-rotated K |
| V inv only (R^T@cur) | 26.6 | V inverse works - degrades quality vs un-rotated V |

### Key Finding
Both rotations mechanically work (modify output, verified with scale(2.0) test). But the full pre-rotate-queries approach (correct Q + correct V) gives PPL 23.5, NOT 6.19.

## Unsolved: Why 23.5 Instead of 6.19?

The math proves the approaches should be equivalent:
- Dequant inverse: x_dequant = R^{-1}(quant(R(x))) ~ x + R^{-1}(epsilon)
- Graph rotation: Q=R(q), K=quant(R(k)), V=quant(R(v)), out=R^{-1}(attn(Q,K,V))
- Error magnitudes are identical (orthogonal rotation preserves norms)

Hypotheses to investigate:
1. **Flash attention precision**: FA kernel may accumulate differently with rotated vs un-rotated values
2. **Block boundary effects**: 128-element rotation groups split across flash attention tiles differently
3. **Metal mul_mat precision**: GPU f32 mul_mat on (128,128)@(128,N) may have precision issues
4. **Non-contiguous tensor handling**: ggml_cont + reshape chain may not preserve data correctly on Metal
5. **Graph optimizer interference**: ggml graph optimizer may simplify/skip the rotation ops

## Current Status
- Dequant inverse rotation RESTORED (PPL = 6.194, speed ~10.7 tok/s)
- Graph rotation code preserved as TODO comments for future investigation
- Virtual method infrastructure (get_turbo_rot_forward/inverse) remains in place
- Rotation tensor allocation and initialization remains in KV cache
