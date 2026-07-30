// Interleaved Tile640 matmul v2: temporal ALU interleaving.
//
// Keeps cooperative decode (all SIMD groups reconstruct the page together)
// and injects drafter GEMM (P1) + KV cache quantization (P2) instructions
// BETWEEN the memory loads in the dot-product loop. The GPU's out-of-order
// execution overlaps this independent compute with the ~200-400 cycle
// activation memory latency, so P1/P2 work is effectively free.
//
// KV lines are prefetched into threadgroup memory during the decode phase
// to avoid competing for device memory bandwidth during the dot product.
// Drafter weights must fit in L2 cache (~1-5 MB for a 1-2 layer draft model).
//
// P0 output is bit-identical to kernel_TILE640_MATMUL.
// See docs/interleaved-kernel-design.md for the full specification.

#include <metal_stdlib>
using namespace metal;

#define FC_TILE640_INTERLEAVE 1710

constant int FC_tile640i_in_dim  [[function_constant(FC_TILE640_INTERLEAVE + 0)]];
constant int FC_tile640i_out_dim [[function_constant(FC_TILE640_INTERLEAVE + 1)]];
constant int FC_tile640i_packing [[function_constant(FC_TILE640_INTERLEAVE + 2)]];
constant bool FC_tile640i_input_f32 [[function_constant(FC_TILE640_INTERLEAVE + 3)]];

#define T640_PAGE 640
#define T640_LANE 20
#define T640_LANES_PER_PAGE 32
#define T640_WORDS_PER_PAGE 32
#define T640_TOKEN_TILE 4
#define T640_KV_PREFETCH_MAX 128

constant ushort T640_TRIT5_LUT[243] = {
    0x000, 0x001, 0x002, 0x004, 0x005, 0x006, 0x008, 0x009, 0x00a, 0x010, 0x011, 0x012,
    0x014, 0x015, 0x016, 0x018, 0x019, 0x01a, 0x020, 0x021, 0x022, 0x024, 0x025, 0x026,
    0x028, 0x029, 0x02a, 0x040, 0x041, 0x042, 0x044, 0x045, 0x046, 0x048, 0x049, 0x04a,
    0x050, 0x051, 0x052, 0x054, 0x055, 0x056, 0x058, 0x059, 0x05a, 0x060, 0x061, 0x062,
    0x064, 0x065, 0x066, 0x068, 0x069, 0x06a, 0x080, 0x081, 0x082, 0x084, 0x085, 0x086,
    0x088, 0x089, 0x08a, 0x090, 0x091, 0x092, 0x094, 0x095, 0x096, 0x098, 0x099, 0x09a,
    0x0a0, 0x0a1, 0x0a2, 0x0a4, 0x0a5, 0x0a6, 0x0a8, 0x0a9, 0x0aa, 0x100, 0x101, 0x102,
    0x104, 0x105, 0x106, 0x108, 0x109, 0x10a, 0x110, 0x111, 0x112, 0x114, 0x115, 0x116,
    0x118, 0x119, 0x11a, 0x120, 0x121, 0x122, 0x124, 0x125, 0x126, 0x128, 0x129, 0x12a,
    0x140, 0x141, 0x142, 0x144, 0x145, 0x146, 0x148, 0x149, 0x14a, 0x150, 0x151, 0x152,
    0x154, 0x155, 0x156, 0x158, 0x159, 0x15a, 0x160, 0x161, 0x162, 0x164, 0x165, 0x166,
    0x168, 0x169, 0x16a, 0x180, 0x181, 0x182, 0x184, 0x185, 0x186, 0x188, 0x189, 0x18a,
    0x190, 0x191, 0x192, 0x194, 0x195, 0x196, 0x198, 0x199, 0x19a, 0x1a0, 0x1a1, 0x1a2,
    0x1a4, 0x1a5, 0x1a6, 0x1a8, 0x1a9, 0x1aa, 0x200, 0x201, 0x202, 0x204, 0x205, 0x206,
    0x208, 0x209, 0x20a, 0x210, 0x211, 0x212, 0x214, 0x215, 0x216, 0x218, 0x219, 0x21a,
    0x220, 0x221, 0x222, 0x224, 0x225, 0x226, 0x228, 0x229, 0x22a, 0x240, 0x241, 0x242,
    0x244, 0x245, 0x246, 0x248, 0x249, 0x24a, 0x250, 0x251, 0x252, 0x254, 0x255, 0x256,
    0x258, 0x259, 0x25a, 0x260, 0x261, 0x262, 0x264, 0x265, 0x266, 0x268, 0x269, 0x26a,
    0x280, 0x281, 0x282, 0x284, 0x285, 0x286, 0x288, 0x289, 0x28a, 0x290, 0x291, 0x292,
    0x294, 0x295, 0x296, 0x298, 0x299, 0x29a, 0x2a0, 0x2a1, 0x2a2, 0x2a4, 0x2a5, 0x2a6,
    0x2a8, 0x2a9, 0x2aa,
};

static inline uint tile640_trit(uint word, int32_t trit) {
    constexpr uint powers_of_243[4] = { 1u, 243u, 59049u, 14348907u };
    const uint group = (uint) trit / 5u;
    const uint index = (word / powers_of_243[group]) % 243u;
    return (T640_TRIT5_LUT[index] >> (2u * ((uint) trit % 5u))) & 3u;
}

static inline float tile640_load_activation(
        device const uchar * input,
        int64_t index) {
    if (FC_tile640i_input_f32) {
        return ((device const float *) input)[index];
    }
    return float(((device const half *) input)[index]);
}

static inline float4 tile640_load_activation4(
        device const uchar * input,
        int64_t index) {
    if (FC_tile640i_input_f32) {
        return *((device const float4 *) ((device const float *) input + index));
    }
    return float4(*((device const half4 *) ((device const half *) input + index)));
}

struct interleaved_args {
    uint32_t drafter_enabled;
    uint32_t drafter_hidden_dim;
    uint32_t drafter_vocab_slice;
    uint32_t drafter_n_tokens;
    uint32_t kv_enabled;
    uint32_t kv_seq_start;
    uint32_t kv_seq_count;
    uint32_t kv_head_dim;
};

struct tile640_matmul_kargs {
    int32_t ne12;
    int32_t ne13;
    int32_t ne14;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_TILE640_MATMUL_INTERLEAVED(
    constant tile640_matmul_kargs & args,
    device const uint*    packed,
    device const half*    page_scales,
    device const uchar*   lane_scales,
    device const uint*    outlier_row_offsets,
    device const uint*    outlier_cols,
    device const half*    outlier_vals,
    device const uchar*   input,
    device const half*    act_scale,
    device       float*   output,
    constant uint &       modality_id,
    // P1: drafter buffers (working set must fit L2 cache)
    device const half*    drafter_weights,
    device const half*    drafter_bias,
    device const half*    drafter_hidden_state,
    device       float*   drafter_logits,
    // P2: KV cache buffers
    device const half*    kv_cache,
    device       uchar*   kv_quantized,
    device       half*    kv_scales_out,
    constant interleaved_args & iargs,
    uint3 tgp [[threadgroup_position_in_grid]],
    ushort3 tpt [[threads_per_threadgroup]],
    uint  sl  [[thread_index_in_simdgroup]],
    uint  si  [[simdgroup_index_in_threadgroup]])
{
    const uint i = tgp.x;
    const uint j0 = tgp.y * T640_TOKEN_TILE;
    const uint b = tgp.z;
    const int32_t in_dim  = FC_tile640i_in_dim;
    const int32_t out_dim = FC_tile640i_out_dim;
    const int32_t n_tokens = args.ne12;
    if (i >= out_dim || j0 >= (uint) n_tokens) return;

    const int32_t nt            = (in_dim + T640_PAGE - 1) / T640_PAGE;
    const int32_t words_per_row = nt *
        (FC_tile640i_packing != 0 ? 40 : T640_WORDS_PER_PAGE);
    const int32_t pages_per_row = nt;
    const int32_t token_count   = min(T640_TOKEN_TILE, n_tokens - (int32_t) j0);

    device const uint*   row_pack    = packed       + (int64_t)i * words_per_row;
    device const half*   row_ps      = page_scales  + (int64_t)i * pages_per_row;
    device const uchar*  row_ls      = lane_scales  +
        (int64_t)i * pages_per_row * T640_LANES_PER_PAGE;

    threadgroup float decoded_page[T640_PAGE] __attribute__((aligned(16)));
    // KV prefetch buffer: one cache line loaded from device memory during
    // decode, consumed from threadgroup during the dot product (no bandwidth
    // competition with activation loads).
    threadgroup float kv_prefetch[T640_KV_PREFETCH_MAX];

    const int32_t tid      = (int32_t) (si * 32u + sl);
    const int32_t nthreads = (int32_t) (tpt.x * tpt.y * tpt.z);

    // Drafter state: each thread accumulates a partial dot product for one
    // (token, vocab) pair. The pair is assigned by tid and persists across
    // pages; each page contributes a chunk of the hidden_dim reduction.
    float drafter_acc = 0.0f;
    uint drafter_token_idx = 0;
    uint drafter_vocab_idx = 0;
    if (iargs.drafter_enabled) {
        const uint total = iargs.drafter_n_tokens * iargs.drafter_vocab_slice;
        const uint my_tile = (uint) tid % total;
        drafter_token_idx = my_tile / iargs.drafter_vocab_slice;
        drafter_vocab_idx = my_tile % iargs.drafter_vocab_slice;
    }

    // KV state: this threadgroup quantizes one KV line per page iteration.
    // kv_line_idx advances each page; threads cooperatively reduce max_abs.
    uint kv_line_idx = 0;
    float kv_max_abs = 0.0f;

    float acc = 0.0f;

    for (int32_t p = 0; p < nt; ++p) {
        // --- Cooperative decode (identical to base kernel) ---
        const float page_max = float(row_ps[p]);
        for (int32_t col = tid; col < T640_PAGE; col += nthreads) {
            const int32_t qlane = col / T640_LANE;
            const float scale = page_max *
                float(row_ls[p * T640_LANES_PER_PAGE + qlane]) *
                (1.0f / 127.0f);
            uint d;
            if (FC_tile640i_packing != 0) {
                d = (row_pack[p * 40 + col / 16] >> (2 * (col & 15))) & 3u;
            } else {
                d = tile640_trit(row_pack[p * T640_WORDS_PER_PAGE + qlane],
                                 col % T640_LANE);
            }
            decoded_page[col] = d == 1u ? scale : d == 2u ? -scale : 0.0f;
        }

        // --- KV prefetch: load one cache line into threadgroup ---
        if (iargs.kv_enabled && kv_line_idx < iargs.kv_seq_count) {
            const uint seq_idx = iargs.kv_seq_start + kv_line_idx;
            for (uint d = (uint) tid; d < iargs.kv_head_dim; d += (uint) nthreads) {
                kv_prefetch[d] = float(kv_cache[seq_idx * iargs.kv_head_dim + d]);
            }
            kv_max_abs = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Dot product with temporal interleaving ---
        if (si < (uint) token_count) {
            const int32_t page_col0 = p * T640_PAGE;
            const int32_t page_cols = min(T640_PAGE, in_dim - page_col0);
            const int64_t input_base =
                ((int64_t)b * n_tokens + j0 + si) * in_dim + page_col0;

            int32_t k = sl * 4;
            for (; k + 3 < page_cols; k += 128) {
                // Issue activation load (~200-400 cycle latency)
                float4 a4 = tile640_load_activation4(input, input_base + k);

                // --- P1: drafter FMA while a4 is in flight ---
                // One step of the hidden_dim reduction. The drafter weight
                // and hidden state reads hit L2 (small working set), so they
                // do not compete with the activation load for HBM bandwidth.
                if (iargs.drafter_enabled) {
                    const uint h_idx = ((uint) k / 4u + sl) % iargs.drafter_hidden_dim;
                    drafter_acc = fma(
                        float(drafter_weights[drafter_vocab_idx * iargs.drafter_hidden_dim + h_idx]),
                        float(drafter_hidden_state[drafter_token_idx * iargs.drafter_hidden_dim + h_idx]),
                        drafter_acc);
                }
                // --- P2: KV max_abs reduction from threadgroup (no device loads) ---
                else if (iargs.kv_enabled && kv_line_idx < iargs.kv_seq_count) {
                    const uint kv_d = ((uint) k / 4u + sl) % iargs.kv_head_dim;
                    kv_max_abs = fmax(kv_max_abs, fabs(kv_prefetch[kv_d]));
                }

                // --- a4 has arrived; P0 FMA chain ---
                if (act_scale != nullptr) {
                    a4.x *= float(act_scale[page_col0 + k]);
                    a4.y *= float(act_scale[page_col0 + k + 1]);
                    a4.z *= float(act_scale[page_col0 + k + 2]);
                    a4.w *= float(act_scale[page_col0 + k + 3]);
                }
                const float4 d4 = *((threadgroup const float4 *) (decoded_page + k));
                acc = fma(a4.x, d4.x, acc);
                acc = fma(a4.y, d4.y, acc);
                acc = fma(a4.z, d4.z, acc);
                acc = fma(a4.w, d4.w, acc);
            }
            // Scalar tail
            for (; k < page_cols; ++k) {
                float a = tile640_load_activation(input, input_base + k);
                if (act_scale != nullptr) {
                    a *= float(act_scale[page_col0 + k]);
                }
                acc = fma(a, decoded_page[k], acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- P2: write KV quantized output for this line ---
        if (iargs.kv_enabled && kv_line_idx < iargs.kv_seq_count) {
            // Reduce max_abs across all threads
            kv_max_abs = simd_max(kv_max_abs);
            // Cross-SIMD reduction via threadgroup
            threadgroup float kv_reduce[4];
            if (sl == 0) {
                kv_reduce[si] = kv_max_abs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float global_max = kv_reduce[0];
                for (int s = 1; s < (nthreads + 31) / 32; s++) {
                    global_max = fmax(global_max, kv_reduce[s]);
                }
                const uint seq_idx = iargs.kv_seq_start + kv_line_idx;
                const float scale = global_max / 127.0f;
                kv_scales_out[seq_idx] = half(scale);
                kv_reduce[0] = scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float kv_scale = kv_reduce[0];
            if (kv_scale > 0.0f) {
                const uint seq_idx = iargs.kv_seq_start + kv_line_idx;
                for (uint d = (uint) tid; d < iargs.kv_head_dim; d += (uint) nthreads) {
                    kv_quantized[seq_idx * iargs.kv_head_dim + d] =
                        (uchar) clamp((int) round(kv_prefetch[d] / kv_scale), -127, 127);
                }
            }
            kv_line_idx++;
        }
    }

    // --- P1: write drafter logits (reduce across pages) ---
    if (iargs.drafter_enabled) {
        drafter_acc = simd_sum(drafter_acc);
        if (sl == 0 && drafter_token_idx < iargs.drafter_n_tokens) {
            drafter_logits[drafter_token_idx * iargs.drafter_vocab_slice + drafter_vocab_idx] =
                drafter_acc + float(drafter_bias[drafter_vocab_idx]);
        }
    }

    // --- Sparse outlier addback (identical to base kernel) ---
    const int32_t row_off_lo = (int32_t) outlier_row_offsets[i];
    const int32_t row_off_hi = (int32_t) outlier_row_offsets[i + 1];
    const int32_t K_i = row_off_hi - row_off_lo;
    if (si < (uint) token_count) {
        const int64_t input_base =
            ((int64_t)b * n_tokens + j0 + si) * in_dim;
        for (int32_t k = sl; k < K_i; k += 32) {
            const int32_t gk  = row_off_lo + k;
            const int32_t col = (int32_t) outlier_cols[gk];
            if (col < in_dim) {
                if (act_scale != nullptr) {
                    float ov = float(outlier_vals[gk]) * float(act_scale[col]);
                    acc = fma(tile640_load_activation(input, input_base + col), ov, acc);
                } else {
                    acc = fma(tile640_load_activation(input, input_base + col), float(outlier_vals[gk]), acc);
                }
            }
        }
    }

    acc = simd_sum(acc);
    if (si < (uint) token_count && sl == 0) {
        const int64_t output_offset =
            ((int64_t)b * n_tokens + j0 + si) * out_dim + i;
        output[output_offset] = acc;
    }
}
