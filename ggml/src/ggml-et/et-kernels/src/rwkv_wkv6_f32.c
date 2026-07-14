//******************************************************************************
// RWKV WKV6 F32 Kernel
//
// Implements the RWKV-6 linear attention recurrence:
//   dst = r @ (time_faaaa * (k @ v) + state)
//   state = time_decay * state + (k @ v)
//
// For each head h, timestep t, row i:
//   kv[j]       = v[j] * k[i]
//   temp[j]     = kv[j] * tf[i] + state[i][j]
//   dst[j]     += temp[j] * r[i]     (accumulated across all i)
//   state[i][j] = state[i][j] * td[i] + kv[j]
//
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_rwkv_wkv6_params {
    float * k;         // src[0]: [S, H, T]  key
    float * v;         // src[1]: [S, H, T]  value
    float * r;         // src[2]: [S, H, T]  receptance
    float * tf;        // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    float * td;        // src[4]: [S, H, T]  time_decay
    float * state_in;  // src[5]: [S*S*H, n_seqs]  initial state
    float * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

int entry_point(struct ggml_et_rwkv_wkv6_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    const float * k        = params->k;
    const float * v        = params->v;
    const float * r        = params->r;
    const float * tf       = params->tf;
    const float * td       = params->td;
    const float * state_in = params->state_in;
    float *       dst_data = params->dst;

    const int32_t C      = params->C;
    const int32_t H      = params->H;
    const int32_t S      = params->S;
    const int32_t T      = params->T;
    const int32_t n_seqs = params->n_seqs;

    if (!k || !v || !r || !tf || !td || !state_in || !dst_data) {
        return -1;
    }

    const int32_t tps       = T / n_seqs;  // tokens per sequence
    float *       state_out = dst_data + C * T;
    float         zero      = 0.0f;

    // Tile j by one cache line so each hart's dst/state writes never share
    // a 64-B line with another hart's writes (the chip is non-coherent).
    // Tiling on j (not i) is required for WKV6 because dst[j] is accumulated
    // across i — splitting i across harts would race on dst writes.
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    const int32_t j_tile         = (S % 16 == 0) ? 16 : S;
    const int32_t tiles_per_head = S / j_tile;
    const int32_t total_units    = H * tiles_per_head;

    // Parallelize across (head, j-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same column slice of state across
    // all timesteps — required for the recurrence to read back its own
    // writes without going through L2.
    for (int32_t u = thread_id; u < total_units; u += num_threads) {
        const int32_t h       = u / tiles_per_head;
        const int32_t tile    = u % tiles_per_head;
        const int32_t j_start = tile * j_tile;
        const int32_t j_end   = j_start + j_tile;

        const int32_t h_off = h * S;      // offset within C for this head
        const int32_t s2d   = h * S * S;  // offset within state for this head

        for (int32_t t = 0; t < T; t++) {
            const int32_t seq       = t / tps;
            const int32_t t_in_seq  = t % tps;
            const int32_t seq_state = seq * S * C;

            const float * s_prev;
            float *       s_cur = state_out + seq_state + s2d;

            if (t_in_seq == 0) {
                s_prev = state_in + seq_state + s2d;
            } else {
                s_prev = s_cur;
            }

            const int32_t th = t * C + h_off;

            // Pointers for this timestep/head
            const float * k_ptr  = k + th;
            const float * v_ptr  = v + th;
            const float * r_ptr  = r + th;
            const float * tf_ptr = tf + h_off;  // tf is per-head, no t offset
            const float * td_ptr = td + th;

            // Zero this hart's slice of dst: dst[th + j_start..th + j_end-1]
            // WKV6 accumulates dst[j] across all i, so must start from zero
            float * dst_row = dst_data + th;
            for (int32_t j = j_start; j < j_end; j += 8) {
                __asm__ volatile(
                    "fbc.ps f10, %[z]\n"
                    "fsw.ps f10, %[dst_vec]\n"
                    : [dst_vec] "=m"(*(float (*)[8]) & dst_row[j])
                    : [z] "m"(zero)
                    : "f10");
            }

            for (int32_t i = 0; i < S; i++) {
                const float * sp_row = s_prev + i * S;  // state_prev row i
                float *       sc_row = s_cur + i * S;   // state_cur  row i

                float k_val  = k_ptr[i];
                float r_val  = r_ptr[i];
                float tf_val = tf_ptr[i];
                float td_val = td_ptr[i];

                // Broadcast k[i], r[i], tf[i], td[i] to vector registers
                __asm__ volatile(
                    "fbc.ps f20, %[kv]\n"   // f20 = k[i] broadcast
                    "fbc.ps f21, %[rv]\n"   // f21 = r[i] broadcast
                    "fbc.ps f22, %[tfv]\n"  // f22 = tf[i] broadcast
                    "fbc.ps f23, %[tdv]\n"  // f23 = td[i] broadcast
                    :
                    : [kv] "m"(k_val), [rv] "m"(r_val), [tfv] "m"(tf_val), [tdv] "m"(td_val)
                    : "f20", "f21", "f22", "f23");

                for (int32_t j = j_start; j < j_end; j += 8) {
                    __asm__ volatile(
                        // Load v[j], state_prev[i][j], dst[j]
                        "flw.ps f10, %[v_vec]\n"  // v[j..j+7]
                        "flw.ps f11, %[s_vec]\n"  // state_prev[i][j..j+7]
                        "flw.ps f12, %[d_vec]\n"  // dst[j..j+7] (accumulated)

                        // kv = v * k_broadcast
                        "fmul.ps f13, f10, f20\n"  // kv = v * k

                        // temp = kv * tf_broadcast + state_prev
                        "fmadd.ps f14, f13, f22, f11\n"  // temp = kv * tf + state

                        // dst[j] += temp * r_broadcast
                        "fmadd.ps f12, f14, f21, f12\n"  // dst += temp * r
                        "fsw.ps f12, %[d_out]\n"         // store updated dst

                        // state_cur[i][j] = state_prev * td_broadcast + kv
                        "fmadd.ps f11, f11, f23, f13\n"  // state = state * td + kv
                        "fsw.ps f11, %[s_out]\n"         // store new state

                        : [d_out] "=m"(*(float (*)[8]) & dst_row[j]), [s_out] "=m"(*(float (*)[8]) & sc_row[j])
                        : [v_vec] "m"(*(const float (*)[8]) & v_ptr[j]), [s_vec] "m"(*(const float (*)[8]) & sp_row[j]),
                          [d_vec] "m"(*(const float (*)[8]) & dst_row[j])
                        : "f10", "f11", "f12", "f13", "f14");
                }
            }
        }
    }

    return 0;
}
