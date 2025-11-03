#pragma once

#include <sycl/sycl.hpp>

template <int Br_MAX = 32, int Bc_MAX = 32>
inline void flash_attn_tiled_kernel(const float * Q,
                                    const float * K,
                                    const float * V,
                                    float *       O,
                                    float *       l,
                                    float *       m,
                                    const int     i_block,
                                    const int     j_block,
                                    const int     Br,
                                    const int     Bc,
                                    const int     N,
                                    const int     d,
                                    const bool    masked,
                                    const float   scale) {
    const int i_start = i_block * Br;
    const int j_start = j_block * Bc;

    float S[Br_MAX][Bc_MAX];
    float P[Br_MAX][Bc_MAX];
    float m_local[Br_MAX];
    float l_local[Br_MAX];

    for (int qi = 0; qi < Br; ++qi) {
        const int q_row = i_start + qi;
        if (q_row >= N) {
            continue;
        }

        for (int kj = 0; kj < Bc; ++kj) {
            const int k_row = j_start + kj;
            if (k_row >= N) {
                S[qi][kj] = -INFINITY;
                continue;
            }

            if (masked && k_row > q_row) {
                S[qi][kj] = -INFINITY;
                continue;
            }

            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[q_row * d + k] * K[k_row * d + k];
            }
            S[qi][kj] = score * scale;
        }
    }

    for (int qi = 0; qi < Br; ++qi) {
        const int q_row = i_start + qi;
        if (q_row >= N) {
            continue;
        }

        m_local[qi] = -INFINITY;
        for (int kj = 0; kj < Bc; ++kj) {
            if (j_start + kj < N) {
                m_local[qi] = sycl::fmax(m_local[qi], S[qi][kj]);
            }
        }

        l_local[qi] = 0.0f;
        for (int kj = 0; kj < Bc; ++kj) {
            if (j_start + kj < N && !sycl::isinf(S[qi][kj])) {
                P[qi][kj] = sycl::exp(S[qi][kj] - m_local[qi]);
                l_local[qi] += P[qi][kj];
            } else {
                P[qi][kj] = 0.0f;
            }
        }
    }

    for (int qi = 0; qi < Br; ++qi) {
        const int q_row = i_start + qi;
        if (q_row >= N) {
            continue;
        }

        const float m_old = m[q_row];
        const float m_new = sycl::fmax(m_old, m_local[qi]);
        const float l_old = l[q_row];
        const float l_new = sycl::exp(m_old - m_new) * l_old + sycl::exp(m_local[qi] - m_new) * l_local[qi];

        const float correction_old = sycl::exp(m_old - m_new);
        const float correction_new = sycl::exp(m_local[qi] - m_new);

        for (int k = 0; k < d; ++k) {
            float pv = 0.0f;
            for (int kj = 0; kj < Bc; ++kj) {
                const int v_row = j_start + kj;
                if (v_row < N) {
                    pv += P[qi][kj] * V[v_row * d + k];
                }
            }

            const int o_idx = q_row * d + k;
            O[o_idx]        = (correction_old * O[o_idx] + correction_new * pv) / l_new;
        }

        l[q_row] = l_new;
        m[q_row] = m_new;
    }
}
