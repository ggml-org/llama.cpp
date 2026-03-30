//******************************************************************************
// Gated Delta Net F32 Kernel
//
// Implements the gated delta rule recurrence:
//   For each head h, timestep t:
//     1. Gate decay:   S *= exp(g)  (scalar or per-element KDA)
//     2. Delta update: delta[j] = (v[j] - dot(S_row_j, k)) * beta
//     3. Outer product: S_row_j += k * delta[j]
//     4. Attention:    attn[j] = dot(S_row_j, q) * scale
//
// State is stored transposed: s_out[j*S_v + i] = S[i][j]
// Steps 2-4 are fused per row to avoid a scratch buffer.
//
// Parallelized across (head, seq) pairs. Inner loops are 8-wide vectorized.
// S_v (head dimension) must be a multiple of 8.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

struct ggml_et_gated_delta_net_params {
    float* q;           // [S_v, H_q, n_tokens, n_seqs_q]
    float* k;           // [S_v, H_k, n_tokens, n_seqs_k]
    float* v;           // [S_v, H, n_tokens, n_seqs]
    float* g;           // [1 or S_v, H, n_tokens, n_seqs]
    float* beta;        // [1, H, n_tokens, n_seqs]
    float* state_in;    // [S_v, S_v, H, n_seqs]
    float* dst;         // [S_v*H, n_tokens*n_seqs + S_v*n_seqs]
    int32_t S_v;        // head dimension
    int32_t H;          // number of value heads
    int32_t H_q;        // number of Q heads
    int32_t H_k;        // number of K heads
    int32_t n_tokens;   // total tokens
    int32_t n_seqs;     // number of sequences
    int32_t n_seqs_q;   // Q sequence count
    int32_t n_seqs_k;   // K sequence count
    int32_t kda;        // 1 if per-element gate, 0 if scalar
    float   scale;      // 1/sqrt(S_v)
};

// Horizontal sum of 8-wide vector register f10 -> scalar float
static inline float hsum_f10(void) {
    float result;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result)
        :: "t0", "f1", "f2", "f3", "f4", "f5"
    );
    return result;
}

int entry_point(struct ggml_et_gated_delta_net_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    const float* q        = params->q;
    const float* k        = params->k;
    const float* v        = params->v;
    const float* g        = params->g;
    const float* beta     = params->beta;
    const float* state_in = params->state_in;
    float* dst_data       = params->dst;

    const int32_t S_v      = params->S_v;
    const int32_t H        = params->H;
    const int32_t H_q      = params->H_q;
    const int32_t H_k      = params->H_k;
    const int32_t n_tokens = params->n_tokens;
    const int32_t n_seqs   = params->n_seqs;
    const int32_t n_seqs_q = params->n_seqs_q;
    const int32_t n_seqs_k = params->n_seqs_k;
    const int32_t kda      = params->kda;
    const float   scale    = params->scale;

    if (!q || !k || !v || !g || !beta || !state_in || !dst_data) {
        return -1;
    }

    // Output layout: [attn_scores(S_v*H*n_tokens*n_seqs) | new_states(S_v*S_v*H*n_seqs)]
    const int32_t attn_elems = S_v * H * n_tokens * n_seqs;
    float* attn_out_base  = dst_data;
    float* state_out_base = dst_data + attn_elems;

    // Gate dimension 0 size
    const int32_t G0 = kda ? S_v : 1;

    // Contiguous tensor strides (in floats)
    // Q: [S_v, H_q, n_tokens, n_seqs_q]
    const int32_t q_stride_h = S_v;
    const int32_t q_stride_t = S_v * H_q;
    const int32_t q_stride_s = S_v * H_q * n_tokens;
    // K: [S_v, H_k, n_tokens, n_seqs_k]
    const int32_t k_stride_h = S_v;
    const int32_t k_stride_t = S_v * H_k;
    const int32_t k_stride_s = S_v * H_k * n_tokens;
    // V: [S_v, H, n_tokens, n_seqs]
    const int32_t v_stride_h = S_v;
    const int32_t v_stride_t = S_v * H;
    const int32_t v_stride_s = S_v * H * n_tokens;
    // G: [G0, H, n_tokens, n_seqs]
    const int32_t g_stride_h = G0;
    const int32_t g_stride_t = G0 * H;
    const int32_t g_stride_s = G0 * H * n_tokens;
    // Beta: [1, H, n_tokens, n_seqs]
    const int32_t b_stride_t = H;
    const int32_t b_stride_s = H * n_tokens;

    // Scratch for KDA exp(g) values (max S_v = 128)
    float exp_g_buf[128];

    // Parallelize across (head, seq) pairs
    const int32_t total_work = H * n_seqs;

    for (int32_t ir = thread_id; ir < total_work; ir += num_threads) {
        const int32_t head = ir % H;
        const int32_t seq  = ir / H;

        // Head indices for Q and K (GQA head repetition)
        const int32_t h_q = head % H_q;
        const int32_t h_k = head % H_k;
        // Sequence indices for Q and K (sequence repetition)
        const int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        const int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        // State pointers: layout [S_v, S_v, H, n_seqs] contiguous
        const int32_t state_offset = (seq * H + head) * S_v * S_v;
        float* s_out = state_out_base + state_offset;
        const float* s_in = state_in + state_offset;

        // Copy input state to output buffer (work in-place on output)
        for (int32_t idx = 0; idx < S_v * S_v; idx += 8) {
            __asm__ volatile(
                "flw.ps f10, %[src]\n"
                "fsw.ps f10, %[dst]\n"
                : [dst] "=m"(*(float(*)[8])&s_out[idx])
                : [src] "m"(*(const float(*)[8])&s_in[idx])
                : "f10"
            );
        }

        // Attention output pointer for first token of this (head, seq)
        float* attn_data = attn_out_base + (seq * n_tokens * H + head) * S_v;

        for (int32_t t = 0; t < n_tokens; t++) {
            // Input pointers for this timestep
            const float* q_t = q + seq_q * q_stride_s + t * q_stride_t + h_q * q_stride_h;
            const float* k_t = k + seq_k * k_stride_s + t * k_stride_t + h_k * k_stride_h;
            const float* v_t = v + seq * v_stride_s + t * v_stride_t + head * v_stride_h;
            const float* g_t = g + seq * g_stride_s + t * g_stride_t + head * g_stride_h;
            const float  beta_val = beta[seq * b_stride_s + t * b_stride_t + head];

            // ----------------------------------------------------------------
            // Step 1: Gate decay
            // State stored transposed: s_out[j*S_v + i] = S[i][j]
            // KDA: S[i][:] *= exp(g[i])  =>  s_out[j][i] *= exp(g[i]) for all j
            // Scalar: S *= exp(g[0])
            // ----------------------------------------------------------------
            if (kda) {
                // Precompute exp(g[i]) for all i
                for (int32_t i = 0; i < S_v; i++) {
                    exp_g_buf[i] = et_expf(g_t[i]);
                }
                // Apply per-element gate to each row of stored state
                for (int32_t j = 0; j < S_v; j++) {
                    float* row = &s_out[j * S_v];
                    for (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ volatile(
                            "flw.ps f10, %[s_vec]\n"
                            "flw.ps f11, %[g_vec]\n"
                            "fmul.ps f10, f10, f11, rne\n"
                            "fsw.ps f10, %[s_out]\n"
                            : [s_out] "=m"(*(float(*)[8])&row[i])
                            : [s_vec] "m"(*(const float(*)[8])&row[i]),
                              [g_vec] "m"(*(const float(*)[8])&exp_g_buf[i])
                            : "f10", "f11"
                        );
                    }
                }
            } else {
                // Scalar gate: scale entire state by exp(g[0])
                float decay = et_expf(g_t[0]);
                __asm__ volatile("fbc.ps f20, %[d]\n" : : [d] "m"(decay) : "f20");
                for (int32_t idx = 0; idx < S_v * S_v; idx += 8) {
                    __asm__ volatile(
                        "flw.ps f10, %[s_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fsw.ps f10, %[s_out]\n"
                        : [s_out] "=m"(*(float(*)[8])&s_out[idx])
                        : [s_vec] "m"(*(const float(*)[8])&s_out[idx])
                        : "f10"
                    );
                }
            }

            // ----------------------------------------------------------------
            // Steps 2-4 fused per state row:
            //   delta_j = (v[j] - dot(s_row_j, k)) * beta
            //   s_row_j[i] += k[i] * delta_j
            //   attn[j] = dot(s_row_j, q) * scale
            // ----------------------------------------------------------------
            for (int32_t j = 0; j < S_v; j++) {
                float* row = &s_out[j * S_v];

                // -- Compute delta_j: dot(state_row, k) --
                float zero = 0.0f;
                __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                for (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ volatile(
                        "flw.ps f11, %[s_vec]\n"
                        "flw.ps f12, %[k_vec]\n"
                        "fmadd.ps f10, f11, f12, f10\n"
                        :
                        : [s_vec] "m"(*(const float(*)[8])&row[i]),
                          [k_vec] "m"(*(const float(*)[8])&k_t[i])
                        : "f10", "f11", "f12"
                    );
                }

                float dot_sk = hsum_f10();
                float delta_j = (v_t[j] - dot_sk) * beta_val;

                // -- Update state row and compute attention dot product --
                __asm__ volatile(
                    "fbc.ps f20, %[dj]\n"
                    "fbc.ps f10, %[z]\n"
                    :
                    : [dj] "m"(delta_j), [z] "m"(zero)
                    : "f10", "f20"
                );

                for (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ volatile(
                        "flw.ps f11, %[s_vec]\n"        // state[j][i:i+8]
                        "flw.ps f12, %[k_vec]\n"        // k[i:i+8]
                        "flw.ps f13, %[q_vec]\n"        // q[i:i+8]
                        "fmadd.ps f11, f20, f12, f11\n" // state += delta_j * k
                        "fsw.ps f11, %[s_out]\n"        // store updated state
                        "fmadd.ps f10, f11, f13, f10\n" // attn_acc += state * q
                        : [s_out] "=m"(*(float(*)[8])&row[i])
                        : [s_vec] "m"(*(const float(*)[8])&row[i]),
                          [k_vec] "m"(*(const float(*)[8])&k_t[i]),
                          [q_vec] "m"(*(const float(*)[8])&q_t[i])
                        : "f10", "f11", "f12", "f13"
                    );
                }

                attn_data[j] = hsum_f10() * scale;
            }

            attn_data += S_v * H;  // advance to next token
        }
    }

    return 0;
}
