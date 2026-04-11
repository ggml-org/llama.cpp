#version 460
#extension GL_KHR_cooperative_matrix                            : require
#extension GL_KHR_memory_scope_semantics                        : require
#extension GL_KHR_shader_subgroup_basic                         : require
#extension GL_EXT_shader_16bit_storage                          : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8         : require

// =============================================================================
// linear_coop_fp8.glsl — FP8 WMMA via uint8 cooperative matrix (RDNA4)
//
// STATUS: EXPERIMENTAL — RDNA4 SPECIFIC
// This shader uses a technique from the FSR4/Mesa RADV work:
// Since SPIR-V has no float8 type yet, we load weights as uint8 cooperative
// matrices and convert FP8 E4M3FN → fp16 inside the coopmat operation.
// On RDNA4 (gfx1201), the driver (AMD proprietary or Mesa RADV with patches)
// maps uint8 coopmat operations to the hardware FP8 WMMA units when the
// conversion pattern is recognised. This is NOT guaranteed by the spec —
// it's a hardware-specific optimisation the driver may or may not apply.
//
// If the driver doesn't recognise it, the fallback is still correct (just
// fp16 WMMA after the FP8 decode), just not 4x faster.
//
// FP8 E4M3FN FORMAT (used by RDNA4, same as MI350X)
// --------------------------------------------------
// 8 bits: 1 sign, 4 exponent, 3 mantissa
// Exponent bias: 7
// Max value: 448.0
// Special: no infinity, NaN = 0x7F or 0xFF
//
// WEIGHTS
// -------
// Pre-quantised to FP8 E4M3FN offline (separate tooling, see docs).
// Stored as raw uint8 in the weight buffer.
// Per-tensor scale stored separately (not per-block — FP8 has enough range).
//
// THIS IS NOT Q8_0 — it's a different format entirely.
// Q8_0 is symmetric int8 with per-block scale.
// FP8 E4M3FN is a floating point format with implicit exponent.
//
// PERFORMANCE EXPECTATION
// -----------------------
// If driver maps to hardware FP8 WMMA: ~4x fp16 WMMA throughput
// If driver falls back to fp16 WMMA after decode: same as fp16 WMMA
// Either way the weights are half the size of fp16 → bandwidth win
// =============================================================================

layout(local_size_x = 64) in;

// FP8 E4M3FN weights stored as uint8
layout(set = 0, binding = 0) readonly buffer BufW { uint8_t W[]; };  // [M × K] fp8

// FP16 activations
layout(set = 0, binding = 1) readonly buffer BufA { float16_t A[]; };  // [K × N]

// FP16 output
layout(set = 0, binding = 2) writeonly buffer BufC { float16_t C[]; };  // [M × N]

layout(push_constant) uniform PC {
    uint  M, N, K;
    float w_scale;   // per-tensor weight scale
    float a_scale;   // per-tensor activation scale (or 1.0 if activation is unscaled)
} pc;

shared float16_t s_w[16 * 16];   // fp16 decoded weight tile
shared float16_t s_a[16 * 16];   // fp16 activation tile
shared uint8_t   s_w8[16 * 16];  // raw uint8 weight tile (for coopmat load)

// ── FP8 E4M3FN → fp16 decode ────────────────────────────────────────────────
// Standard E4M3FN decode: bias=7, no inf, NaN=0x7F/0xFF
float16_t fp8_e4m3fn_to_fp16(uint8_t v) {
    uint bits = uint(v);

    // Extract fields
    uint sign     = (bits >> 7u) & 0x1u;
    uint exponent = (bits >> 3u) & 0xFu;
    uint mantissa = bits & 0x7u;

    // NaN cases (E4M3FN: 0x7F and 0xFF are NaN)
    if (exponent == 0xFu && mantissa == 0x7u) {
        return float16_t(0.0);   // treat as 0 for stability in inference
    }

    float value;
    if (exponent == 0u) {
        // Denormal: value = (-1)^sign × 0.0... × 2^(-6) × (mantissa/8)
        value = float(mantissa) / 8.0 * exp2(-6.0);
    } else {
        // Normal: value = (-1)^sign × 2^(exp-7) × (1 + mantissa/8)
        value = (1.0 + float(mantissa) / 8.0) * exp2(float(exponent) - 7.0);
    }

    if (sign != 0u) value = -value;
    return float16_t(value);
}

void main() {
    uint tid      = gl_LocalInvocationID.x;
    uint tile_row = gl_WorkGroupID.y;
    uint tile_col = gl_WorkGroupID.x;

    uint out_row_base = tile_row * 16u;
    uint out_col_base = tile_col * 16u;
    if (out_row_base >= pc.M || out_col_base >= pc.N) return;

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc;
    for (uint i = 0u; i < acc.length(); i++) acc[i] = float16_t(0.0);

    bool need_bounds = (out_row_base + 16u > pc.M) || (out_col_base + 16u > pc.N);

    for (uint k = 0u; k < pc.K; k += 16u) {
        bool k_bounds = (k + 16u > pc.K);
        bool bounds   = need_bounds || k_bounds;

        // ── Load fp8 weight tile as uint8 into shared ────────────────────
        for (uint i = 0u; i < 4u; i++) {
            uint elem = tid * 4u + i;
            uint r    = elem / 16u, c = elem % 16u;
            uint gr   = out_row_base + r, gc = k + c;
            s_w8[elem] = (!bounds || (gr < pc.M && gc < pc.K))
                         ? W[gr * pc.K + gc] : uint8_t(0u);
        }
        subgroupBarrier();

        // ── Option A: uint8 coopmat (driver may map to fp8 WMMA on RDNA4) ─
        // Load uint8 coopmat directly, convert inside the matrix operation.
        // The FSR4/RADV approach: load as uint8 coopmat, then bitcast to
        // fp16 coopmat using the element-wise conversion.
        // Whether this actually hits hardware FP8 WMMA depends on the driver.

        coopmat<uint8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_w8;
        coopMatLoad(mat_w8, s_w8, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);

        // Convert uint8 coopmat → fp16 coopmat element-wise
        // This is the technique from the Maister FSR4 RADV article.
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_w;
        for (uint i = 0u; i < mat_w.length(); i++) {
            mat_w[i] = fp8_e4m3fn_to_fp16(mat_w8[i]);
        }

        // ── Load fp16 activation tile ─────────────────────────────────────
        for (uint i = 0u; i < 4u; i++) {
            uint elem = tid * 4u + i;
            uint r    = elem / 16u, c = elem % 16u;
            uint gr   = k + r, gc = out_col_base + c;
            s_a[elem] = (!bounds || (gr < pc.K && gc < pc.N))
                        ? A[gr * pc.N + gc] : float16_t(0.0);
        }
        subgroupBarrier();

        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_a;
        coopMatLoad(mat_a, s_a, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);

        acc = coopMatMulAdd(mat_w, mat_a, acc);
        subgroupBarrier();
    }

    // Apply combined scale factor
    float combined_scale = pc.w_scale * pc.a_scale;
    for (uint i = 0u; i < acc.length(); i++)
        acc[i] = float16_t(float(acc[i]) * combined_scale);

    // Write output
    if (!need_bounds) {
        coopMatStore(acc, C, out_row_base * pc.N + out_col_base, pc.N,
                     gl_CooperativeMatrixLayoutRowMajor);
    } else {
        coopMatStore(acc, s_w, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        subgroupBarrier();
        for (uint i = 0u; i < 4u; i++) {
            uint elem = tid * 4u + i;
            uint r = elem / 16u, c = elem % 16u;
            uint gr = out_row_base + r, gc = out_col_base + c;
            if (gr < pc.M && gc < pc.N) C[gr * pc.N + gc] = s_w[elem];
        }
    }
}
