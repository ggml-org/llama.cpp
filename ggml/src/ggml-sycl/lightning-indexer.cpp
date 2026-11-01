#include "lightning-indexer.hpp"
#include "dequantize.hpp"
#include "xmx.hpp"

static void lightning_indexer_f32_sycl(
        const char * q, const char * k, const char * w, const char * m, float * dst,
        int64_t n_embd, int64_t n_head, int64_t n_batch, int64_t n_stream, int64_t n_kv,
        int64_t nem3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3,
        int64_t nbw1, int64_t nbw3,
        int64_t nbm1, int64_t nbm3,
        int64_t nb1, int64_t nb3,
        ggml_type k_type,
        queue_ptr stream) {

    constexpr int64_t LANES = WARP_SIZE;
    constexpr int64_t ELEMS_PER_LANE = 8;
    constexpr int64_t ROWS_PER_BLOCK = 4;
    constexpr int64_t BLOCK_SIZE = ROWS_PER_BLOCK * LANES;

    const int64_t n_rows = n_batch * n_stream * n_kv;
    const int64_t n_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    stream->parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_blocks * BLOCK_SIZE),
            sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int64_t ir   = item.get_global_id(0);
            const int64_t lane = ir % LANES;
            const int64_t row  = ir / LANES;
            if (row >= n_rows) {
                return;
            }

            const int64_t i_bs     = row / n_kv;
            const int64_t i_kv     = row % n_kv;
            const int64_t i_batch  = i_bs / n_stream;
            const int64_t i_stream = i_bs % n_stream;

            // load K row slice into registers (row is contiguous, nbk0 == type size)
            const char * k_base = k + i_kv*nbk2 + i_stream*nbk3;
            float k_local[ELEMS_PER_LANE];
            if (k_type == GGML_TYPE_F16) {
                const sycl::half * k_row = (const sycl::half *) k_base;
#pragma unroll
                for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = static_cast<float>(k_row[lane*ELEMS_PER_LANE + j]);
                }
            } else if (k_type == GGML_TYPE_F32) {
                const float * k_row = (const float *) k_base;
#pragma unroll
                for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    k_local[j] = k_row[lane*ELEMS_PER_LANE + j];
                }
            } else {
                const int64_t lane_base = lane * ELEMS_PER_LANE;
                switch (k_type) {
                    case GGML_TYPE_BF16: {
                        const sycl::ext::oneapi::bfloat16 * k_row = (const sycl::ext::oneapi::bfloat16 *) k_base;
#pragma unroll
                        for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = static_cast<float>(k_row[lane_base + j]);
                        }
                    } break;
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1: {
#pragma unroll
                        for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            const int64_t idx = lane_base + j;
                            const int64_t ib  = idx / QK4_0;
                            const int iqs     = idx % (QK4_0/2);
                            dfloat2 kv;
                            if (k_type == GGML_TYPE_Q4_0) {
                                dequantize_q4_0(k_base, ib, iqs, kv);
                            } else if (k_type == GGML_TYPE_Q4_1) {
                                dequantize_q4_1(k_base, ib, iqs, kv);
                            } else if (k_type == GGML_TYPE_Q5_0) {
                                dequantize_q5_0(k_base, ib, iqs, kv);
                            } else {
                                dequantize_q5_1(k_base, ib, iqs, kv);
                            }
                            k_local[j] = (idx % QK4_0) < (QK4_0/2) ? static_cast<float>(kv.x()) : static_cast<float>(kv.y());
                        }
                    } break;
                    case GGML_TYPE_Q8_0: {
#pragma unroll
                        for (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            const int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_q8_0(k_base, elem0 / QK8_0, elem0 % QK8_0, kv);
                            k_local[2 * pair + 0] = static_cast<float>(kv.x());
                            k_local[2 * pair + 1] = static_cast<float>(kv.y());
                        }
                    } break;
                    case GGML_TYPE_IQ4_NL: {
#pragma unroll
                        for (int64_t pair = 0; pair < ELEMS_PER_LANE / 2; ++pair) {
                            const int64_t elem0 = lane_base + 2 * pair;
                            dfloat2 kv;
                            dequantize_iq4_nl(k_base, elem0 / QK4_NL, elem0 % QK4_NL, kv);
                            k_local[2 * pair + 0] = static_cast<float>(kv.x());
                            k_local[2 * pair + 1] = static_cast<float>(kv.y());
                        }
                    } break;
                    default:
#pragma unroll
                        for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                            k_local[j] = 0.0f;
                        }
                        break;
                }
            }

            const char  * q_base = q + i_batch*nbq2 + i_stream*nbq3;
            const float * w_base = (const float *) (w + i_batch*nbw1 + i_stream*nbw3);

            float score = 0.0f;
            for (int64_t h = 0; h < n_head; ++h) {
                const float * q_row = (const float *) (q_base + h*nbq1);
                float dot = 0.0f;
#pragma unroll
                for (int64_t j = 0; j < ELEMS_PER_LANE; ++j) {
                    const int64_t i = lane*ELEMS_PER_LANE + j;
                    if (i < n_embd) {
                        dot += q_row[i] * k_local[j];
                    }
                }
                dot = sycl::reduce_over_group(item.get_sub_group(), dot, sycl::plus<float>());
                if (lane == 0) {
                    score += sycl::max(dot, 0.0f) * w_base[h];
                }
            }

            if (lane == 0) {
                const sycl::half * m_base = (const sycl::half *) (m + i_batch*nbm1 + (i_stream % nem3)*nbm3);
                // flat-index store: storing through a strided base pointer
                // hangs/misroutes writes on this stack when n_batch*n_stream > 1
                const int64_t dst_idx = i_kv + i_batch*(nb1/sizeof(float)) + i_stream*(nb3/sizeof(float));
                dst[dst_idx] = score + static_cast<float>(m_base[i_kv]);
            }
        });
}

static constexpr int LI_EMBD = 128;

// Reuse each key across heads and score eight keys per sub-group.
static constexpr int LI_VEC_KEYS   = 8;
static constexpr int LI_VEC_SG     = 4;
static constexpr int LI_VEC_LANE_E = LI_EMBD / WARP_SIZE;
static_assert(LI_VEC_LANE_E == 8 && LI_VEC_KEYS == 8, "the transposed reduction maps lane pair 2j, 2j+1 to key j");

template <typename k_t>
static inline void li_load8(const k_t * src, float * out) {
    if constexpr (std::is_same_v<k_t, float>) {
        const sycl::float4 a = *(const sycl::float4 *) src;
        const sycl::float4 b = *(const sycl::float4 *) (src + 4);
        out[0] = a.x(); out[1] = a.y(); out[2] = a.z(); out[3] = a.w();
        out[4] = b.x(); out[5] = b.y(); out[6] = b.z(); out[7] = b.w();
    } else if constexpr (std::is_same_v<k_t, sycl::half>) {
        const sycl::vec<sycl::half, 8> v = *(const sycl::vec<sycl::half, 8> *) src;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[j] = static_cast<float>(v[j]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[j] = static_cast<float>(src[j]);
        }
    }
}

template <typename k_t>
static void lightning_indexer_vec_sycl(
        const char * q, const char * k, const char * w, const char * m, char * dst,
        int n_head, int n_batch, int n_stream, int n_kv, int nem3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        size_t nb1, size_t nb3,
        queue_ptr stream) {

    constexpr int KEYS_PER_WG = LI_VEC_KEYS * LI_VEC_SG;
    const int n_kv_groups = (n_kv + KEYS_PER_WG - 1) / KEYS_PER_WG;

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(n_stream, n_batch, (size_t) n_kv_groups * LI_VEC_SG * WARP_SIZE),
            sycl::range<3>(1, 1, LI_VEC_SG * WARP_SIZE)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const auto sg       = item.get_sub_group();
            const int  lane     = sg.get_local_linear_id();
            const int  i_stream = item.get_group(0);
            const int  i_batch  = item.get_group(1);
            const int  kv0      = item.get_group(2) * KEYS_PER_WG + sg.get_group_linear_id() * LI_VEC_KEYS;

            float k_reg[LI_VEC_KEYS][LI_VEC_LANE_E];
#pragma unroll
            for (int j = 0; j < LI_VEC_KEYS; ++j) {
                if (kv0 + j < n_kv) {
                    const k_t * k_row = (const k_t *) (k + (size_t) (kv0 + j)*nbk2 + (size_t) i_stream*nbk3);
                    li_load8(k_row + lane*LI_VEC_LANE_E, k_reg[j]);
                } else {
#pragma unroll
                    for (int e = 0; e < LI_VEC_LANE_E; ++e) {
                        k_reg[j][e] = 0.0f;
                    }
                }
            }

            const char  * q_base = q + (size_t) i_batch*nbq2 + (size_t) i_stream*nbq3;
            const float * w_base = (const float *) (w + (size_t) i_batch*nbw1 + (size_t) i_stream*nbw3);

            // after the reduction lanes 2j and 2j+1 hold the dot product of key j
            float score = 0.0f;
            for (int h = 0; h < n_head; ++h) {
                float q_reg[LI_VEC_LANE_E];
                li_load8((const float *) (q_base + (size_t) h*nbq1) + lane*LI_VEC_LANE_E, q_reg);

                float dot[LI_VEC_KEYS];
#pragma unroll
                for (int j = 0; j < LI_VEC_KEYS; ++j) {
                    dot[j] = 0.0f;
#pragma unroll
                    for (int e = 0; e < LI_VEC_LANE_E; ++e) {
                        dot[j] += q_reg[e] * k_reg[j][e];
                    }
                }

                // each step halves the values per lane: the lower lane keeps the lower keys
#pragma unroll
                for (int n = LI_VEC_KEYS, o = WARP_SIZE / 2; n > 1; n /= 2, o /= 2) {
                    const bool hi = lane & o;
#pragma unroll
                    for (int j = 0; j < n / 2; ++j) {
                        const float keep = hi ? dot[j + n/2] : dot[j];
                        const float send = hi ? dot[j] : dot[j + n/2];
                        dot[j] = keep + sycl::permute_group_by_xor(sg, send, o);
                    }
                }
                const float sum = dot[0] + sycl::permute_group_by_xor(sg, dot[0], 1);

                score += sycl::max(sum, 0.0f) * w_base[h];
            }

            const int i_kv = kv0 + lane/2;
            if (lane % 2 == 0 && i_kv < n_kv) {
                const sycl::half * m_row = (const sycl::half *) (m + (size_t) i_batch*nbm1 + (size_t) (i_stream % nem3)*nbm3);
                // flat-index store, as in lightning_indexer_f32_sycl
                const size_t dst_idx = i_kv + (size_t) i_batch*(nb1/sizeof(float)) + (size_t) i_stream*(nb3/sizeof(float));
                ((float *) dst)[dst_idx] = score + static_cast<float>(m_row[i_kv]);
            }
        });
}

// Keep the native tile separate from the indexer work-group tuning.
template <typename S, int NSG = (S::TN <= 16 ? 4 : 1), int MT = (S::TM < 32 ? 32 / S::TM : 1), int NT = 4>
struct li_xmx_config : S {
    static constexpr int SUBGROUPS = NSG;
    static constexpr int KEYS      = NSG * S::TN;
    static constexpr int TOK       = MT * S::TM;
    static constexpr int WG_TOK    = NT * TOK;
    static constexpr int THREADS   = NSG * S::SG;
    static constexpr int COLS      = S::TN / S::SG;
    static_assert(NSG > 0 && MT > 0 && NT > 0 && S::TN % S::SG == 0 && LI_EMBD % S::TK == 0, "bad indexer tile");
};

// DIRECT_B: K is stored in the B type, so full key tiles load B from K as column-major and skip the pack
template <typename k_t, typename S, bool DIRECT_B>
static void lightning_indexer_xmx_sycl(
        ggml_backend_sycl_context & ctx,
        const char * q, const char * k, const char * w, const char * m, char * dst,
        int n_head, int n_batch, int n_stream, int n_kv, int nem3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        size_t nb1, size_t nb3,
        queue_ptr stream) {

    namespace mx = sycl::ext::oneapi::experimental::matrix;
    using EA = typename S::EA;
    using EB = typename S::EB;
    // Queries are padded to whole tiles: [stream][head][token][embd].
    const int n_batch_pad = (n_batch + S::TOK - 1) / S::TOK * S::TOK;
    ggml_sycl_pool_alloc<typename S::tsa> q_stage(ctx.pool(), (size_t) n_stream * n_head * n_batch_pad * LI_EMBD);
    typename S::tsa * q_packed = q_stage.get();

    stream->parallel_for(
        sycl::range<3>(n_stream, n_head, (size_t) n_batch_pad * LI_EMBD),
        [=](sycl::id<3> id) {
            const int s = id[0];
            const int h = id[1];
            const int t = id[2] / LI_EMBD;
            const int e = id[2] % LI_EMBD;
            float v = 0.0f;
            if (t < n_batch) {
                v = ((const float *) (q + (size_t) h*nbq1 + (size_t) t*nbq2 + (size_t) s*nbq3))[e];
            }
            q_packed[(((size_t) s*n_head + h)*n_batch_pad + t)*LI_EMBD + e] = EA::cvt(v);
        });

    const int n_kv_groups  = (n_kv + S::KEYS - 1) / S::KEYS;
    const int n_tok_groups = (n_batch + S::WG_TOK - 1) / S::WG_TOK;

    stream->submit([&](sycl::handler & cgh) {
        // Pack K according to the operand storage width.
        sycl::local_accessor<typename S::tsb, 1> k_tile(LI_EMBD * S::KEYS, cgh);
        sycl::local_accessor<float, 1> c_tile(S::SUBGROUPS * S::TOK * S::TN, cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(n_stream, n_tok_groups, (size_t) n_kv_groups * S::THREADS),
                sycl::range<3>(1, 1, S::THREADS)),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                if constexpr (xmx_built<S::SG>()) {
                    const auto sg       = item.get_sub_group();
                    const int  sg_id    = sg.get_group_linear_id();
                    const int  lane     = sg.get_local_linear_id();
                    const int  i_stream = item.get_group(0);
                    const int  t_wg     = item.get_group(1) * S::WG_TOK;
                    const int  kv0      = item.get_group(2) * S::KEYS;

                    const bool direct_b = DIRECT_B && kv0 + S::KEYS <= n_kv;
                    for (int key = item.get_local_linear_id(); !direct_b && key < S::KEYS; key += S::THREADS) {
                        const int i_kv = kv0 + key;
#pragma unroll
                        for (int e = 0; e < LI_EMBD; e += 8) {
                            float kf[8] = {};
                            if (i_kv < n_kv) {
                                const k_t * k_row = (const k_t *) (k + (size_t) i_kv*nbk2 + (size_t) i_stream*nbk3);
                                li_load8(k_row + e, kf);
                            }
#pragma unroll
                            for (int j = 0; j < 8; j += S::VNNI) {
                                if constexpr (S::VNNI == 2) {
                                    auto * kt = (typename EB::pair *) &k_tile[0];
                                    kt[((e + j)/2)*S::KEYS + key] = EB::make(kf[j], kf[j + 1]);
                                } else {
                                    k_tile[(e + j)*S::KEYS + key] = EB::cvt(kf[j]);
                                }
                            }
                        }
                    }
                    if (!direct_b) {
                        sycl::group_barrier(item.get_group());
                    }

                    const auto k_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                 sycl::access::decorated::no>((const k_t *) (k + (size_t) i_stream*nbk3));
                    const size_t k_stride = nbk2 / sizeof(k_t);
                    const auto q_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                 sycl::access::decorated::no>(q_packed);
                    const auto b_ptr = k_tile.template get_multi_ptr<sycl::access::decorated::no>() + sg_id * S::TN * S::VNNI;
                    const auto c_ptr = c_tile.get_multi_ptr<sycl::access::decorated::no>() + sg_id * S::TOK * S::TN;
                    const float * c_sg = &c_tile[sg_id * S::TOK * S::TN];

                    const char * w_s   = w + (size_t) i_stream*nbw3;
                    const int    t_end = sycl::min(t_wg + S::WG_TOK, n_batch);

                    // with_b(kk, f) calls f with the B tile of embedding block kk
                    auto run = [&](auto && with_b) {
                        for (int t0 = t_wg; t0 < t_end; t0 += S::TOK) {
                            float score[S::TOK][S::COLS] = {};
                            for (int h = 0; h < n_head; ++h) {
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, S::TM, S::TN> acc[S::TOK / S::TM];
#pragma unroll
                                for (int mt = 0; mt < S::TOK / S::TM; ++mt) {
                                    mx::joint_matrix_fill(sg, acc[mt], 0.0f);
                                }

                                const size_t a_off = (((size_t) i_stream*n_head + h)*n_batch_pad + t0)*LI_EMBD;
#pragma unroll
                                for (int kk = 0; kk < LI_EMBD / S::TK; ++kk) {
                                    auto mad = [&](auto & sub_b) {
#pragma unroll
                                        for (int mt = 0; mt < S::TOK / S::TM; ++mt) {
                                            mx::joint_matrix<sycl::sub_group, typename S::ta, mx::use::a,
                                                             S::TM, S::TK, mx::layout::row_major> sub_a;
                                            mx::joint_matrix_load(sg, sub_a,
                                                q_ptr + a_off + mt * S::TM * LI_EMBD + kk * S::TK, LI_EMBD);
                                            mx::joint_matrix_mad(sg, acc[mt], sub_a, sub_b, acc[mt]);
                                        }
                                    };
                                    with_b(kk, mad);
                                }

#pragma unroll
                                for (int mt = 0; mt < S::TOK / S::TM; ++mt) {
                                    mx::joint_matrix_store(sg, acc[mt], c_ptr + mt * S::TM * S::TN,
                                                           S::TN, mx::layout::row_major);
                                }
                                sycl::group_barrier(sg);

#pragma unroll
                                for (int t = 0; t < S::TOK; ++t) {
                                    const int   i_batch = sycl::min(t0 + t, n_batch - 1);
                                    const float w_val   = ((const float *) (w_s + (size_t) i_batch*nbw1))[h];
#pragma unroll
                                    for (int c = 0; c < S::COLS; ++c) {
                                        score[t][c] += sycl::max(c_sg[t * S::TN + lane + c*S::SG], 0.0f) * w_val;
                                    }
                                }
                                sycl::group_barrier(sg);
                            }

#pragma unroll
                            for (int c = 0; c < S::COLS; ++c) {
                                const int i_kv = kv0 + sg_id * S::TN + lane + c*S::SG;
                                if (i_kv >= n_kv) {
                                    continue;
                                }
                                const int n_t = sycl::min(S::TOK, n_batch - t0);
#pragma unroll
                                for (int t = 0; t < S::TOK; ++t) {
                                    if (t < n_t) {
                                        const size_t i_batch = t0 + t;
                                        const sycl::half * m_row = (const sycl::half *) (m + i_batch*nbm1 + (size_t) (i_stream % nem3)*nbm3);
                                        const size_t dst_idx = i_kv + i_batch*(nb1/sizeof(float)) + (size_t) i_stream*(nb3/sizeof(float));
                                        ((float *) dst)[dst_idx] = score[t][c] + static_cast<float>(m_row[i_kv]);
                                    }
                                }
                            }
                        }
                    };

                    if constexpr (DIRECT_B) {
                        if (direct_b) {
                            // all B tiles of this work-group stay in registers for every head and token block
                            mx::joint_matrix<sycl::sub_group, typename S::tb, mx::use::b, S::TK, S::TN, mx::layout::col_major> b_reg[LI_EMBD / S::TK];
#pragma unroll
                            for (int kk = 0; kk < LI_EMBD / S::TK; ++kk) {
                                mx::joint_matrix_load(sg, b_reg[kk], k_ptr + (size_t) (kv0 + sg_id * S::TN) * k_stride + kk * S::TK, k_stride);
                            }
                            run([&](int kk, auto && f) { f(b_reg[kk]); });
                            return;
                        }
                    }
                    run([&](int kk, auto && f) {
                        mx::joint_matrix<sycl::sub_group, typename S::tb, mx::use::b, S::TK, S::TN, S::b_layout> sub_b;
                        mx::joint_matrix_load(sg, sub_b, b_ptr + kk * S::TK * S::KEYS, S::KEYS * S::VNNI);
                        f(sub_b);
                    });
                }
            });
    });
}

template <typename k_t>
static void lightning_indexer_fast_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst, int xmx_combo) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];
    const int nem3     = m->ne[3];

    bool launched = false;
    if (xmx_combo >= 0) {
        xmx_visit_combo(xmx_combo, [&](auto combo) {
            using S = li_xmx_config<decltype(combo)>;
            if (n_batch > INT32_MAX - S::WG_TOK || n_batch > INT32_MAX - S::TOK || n_kv > INT32_MAX - S::KEYS) {
                return;
            }
            const size_t n_batch_pad = (n_batch + S::TOK - 1) / S::TOK * S::TOK;
            if ((size_t) n_head > SIZE_MAX / sizeof(typename S::tsa) / LI_EMBD / n_stream / n_batch_pad) {
                return;
            }
            // bf16 measured 2-4% slower from K than from the packed tile; IGC has no column-major B load for TN 64,
            // and SG8 is not tested
            constexpr bool direct_b = std::is_same_v<k_t, sycl::half> && std::is_same_v<typename S::tsb, sycl::half> &&
                                      S::TN == 16 && S::SG == 16;
            lightning_indexer_xmx_sycl<k_t, S, direct_b>(ctx,
                (const char *) q->data, (const char *) k->data, (const char *) w->data, (const char *) m->data, (char *) dst->data,
                n_head, n_batch, n_stream, n_kv, nem3,
                q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3], w->nb[1], w->nb[3], m->nb[1], m->nb[3], dst->nb[1], dst->nb[3],
                ctx.stream());
            launched = true;
        });
    }
    if (!launched) {
        lightning_indexer_vec_sycl<k_t>(
            (const char *) q->data, (const char *) k->data, (const char *) w->data, (const char *) m->data, (char *) dst->data,
            n_head, n_batch, n_stream, n_kv, nem3,
            q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3], w->nb[1], w->nb[3], m->nb[1], m->nb[3], dst->nb[1], dst->nb[3],
            ctx.stream());
    }
}

// f16 16x16x16 (bit 2 of GGML_SYCL_LIGHTNING_INDEXER_XMX_SHAPES): B60, 4 heads 8-19% and 64 heads 33-47% faster than 8x16x16
static constexpr int LI_XMX_PREFER = 1 << 1;

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_Q5_1 || k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q4_1 || k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_IQ4_NL);

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, k, nb);
    GGML_TENSOR_LOCALS(size_t,  nbw, w, nb);
    GGML_TENSOR_LOCALS(int64_t, nem, m, ne);
    GGML_TENSOR_LOCALS(size_t,  nbm, m, nb);
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    // input rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));
    GGML_ASSERT(nb0  == ggml_type_size(dst->type));

    const int64_t n_embd   = neq0;
    const int64_t n_head   = neq1;
    const int64_t n_batch  = neq2;
    const int64_t n_stream = neq3;
    const int64_t n_kv     = nek2;

    GGML_ASSERT(n_embd == WARP_SIZE * 8);

    const bool aligned = reinterpret_cast<uintptr_t>(q->data) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(k->data) % 16 == 0 &&
                         nbq1 % 16 == 0 && nbq2 % 16 == 0 && nbq3 % 16 == 0 &&
                         nbk2 % 16 == 0 && nbk3 % 16 == 0;
    const int64_t max_index = INT32_MAX - 128;
    const bool small_shape = n_head > 0 && n_batch > 0 && n_stream > 0 && n_kv > 0 && nem3 > 0 &&
                             n_head <= max_index && n_batch <= max_index && n_stream <= max_index &&
                             n_kv <= max_index && nem3 <= max_index;
    if (g_ggml_sycl_lightning_indexer && aligned && small_shape) {
        const int type_bit = k->type == GGML_TYPE_F32  ? GGML_SYCL_LIGHTNING_INDEXER_XMX_F32 :
                             k->type == GGML_TYPE_F16  ? GGML_SYCL_LIGHTNING_INDEXER_XMX_F16 :
                             k->type == GGML_TYPE_BF16 ? GGML_SYCL_LIGHTNING_INDEXER_XMX_BF16 : 0;
        const int xmx_combo = n_batch >= 8 && (g_ggml_sycl_lightning_indexer_xmx_types & type_bit) &&
                              g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 ?
            xmx_pick_combo(ctx.stream(), k->type, GGML_PREC_UNDEFINED, g_ggml_sycl_lightning_indexer_xmx_shapes, LI_XMX_PREFER) : -1;
        switch (k->type) {
            case GGML_TYPE_F32:  lightning_indexer_fast_sycl<float>(ctx, dst, xmx_combo); return;
            case GGML_TYPE_F16:  lightning_indexer_fast_sycl<sycl::half>(ctx, dst, xmx_combo); return;
            case GGML_TYPE_BF16: lightning_indexer_fast_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst, xmx_combo); return;
            default: break;
        }
    }

    lightning_indexer_f32_sycl(
            (const char *) q->data, (const char *) k->data,
            (const char *) w->data, (const char *) m->data, (float *) dst->data,
            n_embd, n_head, n_batch, n_stream, n_kv, nem3,
            nbq1, nbq2, nbq3,
            nbk2, nbk3,
            nbw1, nbw3,
            nbm1, nbm3,
            nb1, nb3,
            k->type,
            ctx.stream());
}
