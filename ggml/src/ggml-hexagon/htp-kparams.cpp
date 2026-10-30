#define _USE_MATH_DEFINES

#include "htp-kparams.h"
#include "htp-opnode.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#endif

htp_op_code htp_op_remap(const struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:  return HTP_OP_FLASH_ATTN_EXT;
        case GGML_OP_MUL_MAT:         return htp_is_hadamard(t) ? HTP_OP_HADAMARD : HTP_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID:      return HTP_OP_MUL_MAT_ID;
        case GGML_OP_MUL:             return HTP_OP_MUL;
        case GGML_OP_ADD:             return HTP_OP_ADD;
        case GGML_OP_ADD_ID:          return HTP_OP_ADD_ID;
        case GGML_OP_SUB:             return HTP_OP_SUB;
        case GGML_OP_DIV:             return HTP_OP_DIV;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:            return HTP_OP_CPY;
        case GGML_OP_GET_ROWS:        return HTP_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:        return HTP_OP_SET_ROWS;
        case GGML_OP_SUM:             return HTP_OP_SUM;
        case GGML_OP_SUM_ROWS:        return HTP_OP_SUM_ROWS;
        case GGML_OP_POOL_2D:         return HTP_OP_POOL_2D;
        case GGML_OP_POOL_1D:         return HTP_OP_POOL_1D;
        case GGML_OP_ARGSORT:         return HTP_OP_ARGSORT;
        case GGML_OP_TOP_K:           return HTP_OP_TOP_K;
        case GGML_OP_ARGMAX:          return HTP_OP_ARGMAX;
        case GGML_OP_NORM:            return HTP_OP_NORM;
        case GGML_OP_L2_NORM:         return HTP_OP_L2_NORM;
        case GGML_OP_RMS_NORM:        return HTP_OP_RMS_NORM;
        case GGML_OP_CONCAT:          return HTP_OP_CONCAT;
        case GGML_OP_SCALE:           return HTP_OP_SCALE;
        case GGML_OP_CLAMP:           return HTP_OP_CLAMP;
        case GGML_OP_LEAKY_RELU:      return HTP_OP_LEAKY_RELU;
        case GGML_OP_SQR:             return HTP_OP_SQR;
        case GGML_OP_SQRT:            return HTP_OP_SQRT;
        case GGML_OP_LOG:             return HTP_OP_UNARY_LOG;
        case GGML_OP_SOFT_MAX:        return HTP_OP_SOFTMAX;
        case GGML_OP_SSM_CONV:        return HTP_OP_SSM_CONV;
        case GGML_OP_GATED_DELTA_NET: return HTP_OP_GATED_DELTA_NET;
        case GGML_OP_ROPE:            return HTP_OP_ROPE;
        case GGML_OP_REPEAT:          return HTP_OP_REPEAT;
        case GGML_OP_CUMSUM:          return HTP_OP_CUMSUM;
        case GGML_OP_FILL:            return HTP_OP_FILL;
        case GGML_OP_DIAG:            return HTP_OP_DIAG;
        case GGML_OP_SOLVE_TRI:       return HTP_OP_SOLVE_TRI;
        case GGML_OP_TRI:             return HTP_OP_TRI;
        case GGML_OP_PAD:             return HTP_OP_PAD;
        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;
        case GGML_OP_IM2COL_3D:       return HTP_OP_IM2COL_3D;
        case GGML_OP_CONV_2D:         return HTP_OP_CONV_2D;
        case GGML_OP_CONV_2D_DW:      return HTP_OP_CONV_2D_DW;
        case GGML_OP_ROLL:            return HTP_OP_ROLL;
        case GGML_OP_UPSCALE:         return HTP_OP_UPSCALE;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU:       return HTP_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU:       return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_GELU_QUICK: return HTP_OP_UNARY_GELU_QUICK;
                case GGML_UNARY_OP_GELU_ERF:   return HTP_OP_UNARY_GELU_ERF;
                case GGML_UNARY_OP_SIGMOID:    return HTP_OP_UNARY_SIGMOID;
                case GGML_UNARY_OP_NEG:        return HTP_OP_UNARY_NEG;
                case GGML_UNARY_OP_EXP:        return HTP_OP_UNARY_EXP;
                case GGML_UNARY_OP_SOFTPLUS:   return HTP_OP_UNARY_SOFTPLUS;
                case GGML_UNARY_OP_TANH:       return HTP_OP_UNARY_TANH;
                case GGML_UNARY_OP_ABS:        return HTP_OP_UNARY_ABS;
                case GGML_UNARY_OP_RELU:       return HTP_OP_UNARY_RELU;
                case GGML_UNARY_OP_STEP:       return HTP_OP_UNARY_STEP;
                case GGML_UNARY_OP_CEIL:       return HTP_OP_UNARY_CEIL;
                case GGML_UNARY_OP_SGN:        return HTP_OP_UNARY_SGN;
                default:                       break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(t)) {
                case GGML_GLU_OP_REGLU:        return HTP_OP_GLU_REGLU;
                case GGML_GLU_OP_SWIGLU:       return HTP_OP_GLU_SWIGLU;
                case GGML_GLU_OP_SWIGLU_OAI:   return HTP_OP_GLU_SWIGLU_OAI;
                case GGML_GLU_OP_SWIGLU_CLAMP: return HTP_OP_GLU_SWIGLU_CLAMP;
                case GGML_GLU_OP_GEGLU:        return HTP_OP_GLU_GEGLU;
                case GGML_GLU_OP_GEGLU_QUICK:  return HTP_OP_GLU_GEGLU_QUICK;
                case GGML_GLU_OP_GEGLU_ERF:    return HTP_OP_GLU_GEGLU_ERF;
                default:                       break;
            }
            break;

        default:
            break;
    }
    return HTP_OP_INVALID;
}

bool htp_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (ne[i] == 1) continue;
        if (s == 0) { s = nb[i]; next = s * ne[i]; continue; }
        if (nb[i] != next) return false;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    return true;
}

bool htp_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    size_t s1, sd;
    return (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && htp_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && htp_rows_stride(dst->ne, dst->nb, &sd);
}

ggml_tensor htp_tensor_collapse_rows(const struct ggml_tensor * t) {
    size_t stride = 0;
    htp_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    return c;
}

bool htp_kparams_solve_allreduce(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    struct htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    const uint32_t max_threads = htp_session_n_threads(sess);
    const size_t   vtcm_size   = htp_session_vtcm_size(sess);

    const uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    const uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);
    const bool is_contiguous = ggml_is_contiguous(dst);

    const uint32_t ne0 = (uint32_t) dst->ne[0];
    const uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    const bool use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    if (has_add) {
        if (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            const uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            const uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            const uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } else if (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } else {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } else {
        kparams->n_dsts = (int32_t) n_ranks;
        if (use_1d) {
            const uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            const uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            const uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            const uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } else {
            const uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            const uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            const uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            const uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    if (use_1d) {
        const uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        const uint32_t n_threads  = (std::min)(max_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        if (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        while ((size_t) kparams->vtcm_size > vtcm_size && block_elems > 128) {
            const size_t max_bytes_per_buf = vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            if (block_elems < 128) break;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        if (vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        htp_session_name(sess), kparams->vtcm_size, vtcm_size);
            return false;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        return true;
    } else {
        const uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        const uint32_t n_threads  = (std::min)(max_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(n_ranks, n_threads, has_add, is_row_bcast);

        const uint32_t row_bytes = ne0 * elem_size;
        const uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        const uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        while ((size_t) kparams->vtcm_size > vtcm_size && block_rows > 1) {
            const size_t max_rows_per_buf = vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            if (max_rows_per_buf == 0) break;
        }

        if (vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        htp_session_name(sess), kparams->vtcm_size, vtcm_size);
            return false;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        return true;
    }
}

static bool htp_flash_attn_is_hmx_eligible(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    const struct ggml_tensor * sinks
) {
    if (htp_session_n_hmx(sess) == 0 || opt_fa_select < 2) {
        return false;
    }

    if ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        return false;
    }

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];

    if (DK % 8 != 0 || DV % 8 != 0) {
        return false;
    }

    GGML_UNUSED(sinks);

    if (opt_fa_select > 2) {
        return true;
    }

    const uint32_t M = q->ne[1];
    if (M > 1) {
        return true;
    }

    const uint32_t n_threads     = htp_session_n_threads(sess);
    const uint32_t n_head        = q->ne[2];
    const uint32_t n_kv_heads    = k->ne[2];
    const uint32_t G             = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    const uint32_t S             = k->ne[1];
    const uint32_t total_threads = n_threads * htp_session_mdev_count(sess);

    if (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        return true;
    }

    const bool is_tile_aligned = (G > 0 && (32 % G == 0));
    if (!is_tile_aligned) {
        if (n_threads >= 6) {
            return false;
        }
        return S >= 1024;
    }

    uint32_t s_cross = 512;
    if (DK == 256) {
        s_cross = 4096;
    } else if (DK <= 64) {
        s_cross = (n_threads >= 8) ? 2048 : ((n_threads >= 6) ? 768 : 512);
    } else {
        s_cross = (n_threads >= 8) ? 1024 : 512;
    }

    return S >= s_cross;
}

static bool htp_kparams_solve_flash_attn(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fa_kernel_params * kparams
) {
    if (opt_fa_select < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * mask  = op->src[3];
    const struct ggml_tensor * sinks = op->src[4];
    const struct ggml_tensor * dst   = op;

    if ((q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) ||
        (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (sinks && sinks->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    const uint32_t n_threads = htp_session_n_threads(sess);
    const size_t   vtcm_size = htp_session_vtcm_size(sess);

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nev0 = v->ne[0];

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G          = neq2 / n_kv_heads;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], sizeof(float));
    memcpy(&max_bias,      &op->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &op->op_params[2], sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale         = scale;
    kparams->max_bias      = max_bias;
    kparams->logit_softcap = logit_softcap;
    kparams->head_split    = (opt_fa_split != 0) ? 1 : 0;
    kparams->flags         = 0;
    kparams->G             = G;

    // team-split only when a team has >1 member (n_kv_heads < N_cores)
    uint32_t fa_teams = 1, fa_team_size = 1;
    const uint32_t mdev_count = htp_session_mdev_count(sess);
    if (opt_fa_split && mdev_count > 1) {
        htp_fa_mdev_teams(n_kv_heads, mdev_count, kparams->head_split != 0, &fa_teams, &fa_team_size);
        if (fa_team_size > 1) {
            kparams->flags |= HTP_FA_FLAG_TEAM_SPLIT;
        }
    }

    const uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    if (htp_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        const uint32_t DK_pad = hex_round_up(DK, 64);
        const uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, vtcm_size, n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head, fa_team_size);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            if (logit_softcap == 0.0f) {
                kparams->scale         = scale * (float) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } else {
                kparams->scale         = scale;
                kparams->logit_softcap = logit_softcap * (float) M_LOG2E;
            }
            kparams->Br                  = Br;
            kparams->Bc                  = Bc;
            kparams->n_kv_blocks         = (nek1 + Bc - 1) / Bc;
            kparams->n_threads           = (kparams->n_kv_blocks >= 3 && n_threads >= 2) ? n_threads : 1;
            kparams->u.hmx.g_br          = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline      = (kparams->n_kv_blocks >= 3 && n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size           = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G         = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }
            return true;
        }
    }

    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br          = 1;
    kparams->Bc          = 64;
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads   = n_threads;

    const size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    const size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, n_threads);
    if ((size_t) kparams->vtcm_size > vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams->vtcm_size, vtcm_size);
        return false;
    }

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21        = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1         = init_fastdiv_values(q->ne[1]);
    kparams->broadcast_rk2           = init_fastdiv_values(q->ne[2] / k->ne[2]);
    kparams->broadcast_rk3           = init_fastdiv_values(q->ne[3] / k->ne[3]);
    kparams->broadcast_rv2           = init_fastdiv_values(q->ne[2] / v->ne[2]);
    kparams->broadcast_rv3           = init_fastdiv_values(q->ne[3] / v->ne[3]);
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    return true;
}

static bool htp_kparams_solve_gated_delta_net(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_gdn_kernel_params * kparams
) {
    if (opt_gdn_select < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * beta  = op->src[4];
    const struct ggml_tensor * state = op->src[5];
    const struct ggml_tensor * dst   = op;

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    if (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        return false;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return false;
    }
    if (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        return false;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K ||
        dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    const uint32_t rq3        = (uint32_t) (n_seqs / q->ne[3]);
    const uint32_t rk3        = (uint32_t) (n_seqs / k->ne[3]);
    const uint32_t total_rows = (uint32_t) (H * n_seqs);
    const uint32_t n_threads  = (std::min)(htp_session_n_threads(sess), total_rows);
    const size_t   vtcm_size  = htp_session_vtcm_size(sess);

    const bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (htp_session_n_hmx(sess) > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    struct htp_gdn_hmx_vtcm_layout hmx_layout;
    struct htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    if (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, vtcm_size, n_threads, true, &n_heads_batch)) {
        kparams->kernel_type   = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline      = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size    = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks      = ((uint32_t) n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch = (uint16_t) n_heads_batch;
        kparams->vtcm_size     = (uint32_t) hmx_layout.total_bytes;
    } else {
        htp_gdn_vtcm_layout_build(&hvx_layout, (uint32_t) S_v, n_threads);
        if (hvx_layout.total_bytes > vtcm_size) {
            return false;
        }
        kparams->kernel_type   = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline      = 0;
        kparams->n_heads_batch = 1;
        kparams->vtcm_size     = (uint32_t) hvx_layout.total_bytes;
    }

    kparams->n_threads           = n_threads;
    kparams->S_v                 = (uint32_t) S_v;
    kparams->H                   = (uint32_t) H;
    kparams->n_tokens            = (uint32_t) n_tokens;
    kparams->n_seqs              = (uint32_t) n_seqs;
    kparams->K                   = (uint32_t) K;
    kparams->total_rows          = total_rows;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / sizeof(float));
    kparams->state_size_per_snap = (uint32_t) (S_v * S_v * H * n_seqs);
    kparams->scale               = 1.0f / sqrtf((float) S_v);

    if (H > 0)        kparams->div_H   = init_fastdiv_values((uint32_t) H);
    if (q->ne[1] > 0) kparams->div_q1  = init_fastdiv_values((uint32_t) q->ne[1]);
    if (k->ne[1] > 0) kparams->div_k1  = init_fastdiv_values((uint32_t) k->ne[1]);
    if (rq3 > 0)      kparams->div_rq3 = init_fastdiv_values(rq3);
    if (rk3 > 0)      kparams->div_rk3 = init_fastdiv_values(rk3);
    return true;
}

static bool htp_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    bool is_matmul_id,
    bool is_batched
) {
    GGML_UNUSED(dst);

    if (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        return false;
    }

    const int ne00  = src0->ne[0];
    const int ne01  = src0->ne[1];
    const int ne11  = src1->ne[1];
    const int ne12  = src1->ne[2];
    const int wtype = src0->type;

    if (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        return false;
    }

    if (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        return false;
    }

    if (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        return false;
    }

    const bool is_tiled = htp_tensor_is_tiled(src0);
    if (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        return false;
    }
    if (!is_matmul_id && is_batched && is_tiled) {
        return false;
    }

    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return false;
    }

    const int m = is_matmul_id ? ne12 : ne11;
    if (m <= HTP_MM_HMX_MIN_NROWS) {
        return false;
    }

    return true;
}

static bool htp_kparams_solve_hmx_mm(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    int wtype,
    int ne00_padded,
    int ne01_padded,
    int ne02,
    int ne11,
    int ne12,
    int ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const bool     is_tiled          = htp_tensor_is_tiled(src0);
    const int      aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    const int      n_threads         = (int) htp_session_n_threads(sess);
    const uint32_t mdev_count        = htp_session_mdev_count(sess);
    const int      ne10              = act->ne[0];

    int m_for_solver        = ne11;
    int m_for_solver_padded = ne11_padded;
    if (!is_matmul_id && mdev_count > 1 && ((uint32_t) ne11 >= mdev_count)) {
        const bool dst_row_split = dst ? htp_can_row_partition(dst) : true;
        const bool act_row_split = htp_can_row_partition(act);
        if (dst_row_split && act_row_split) {
            m_for_solver        = (ne11 + (int) mdev_count - 1) / (int) mdev_count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    const bool pipeline       = is_matmul_id ? false : htp_mm_hmx_pipeline(m_for_solver);
    const bool is_batched_val = is_matmul_id ? false : is_batched;
    const int  group_size     = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    bool use_grouped = false;
    int act_threads_selected = 0;

    if (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = true;
        }
    }

    if (!use_grouped) {
        int m_id_rows = 0;
        if (dst && is_matmul_id) {
            const int n_experts = ne02 > 0 ? ne02 : 1;
            const size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            int m_per_expert = (int) ((total_expert_rows + n_experts - 1) / n_experts);
            if (mdev_count > 1 && htp_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (int) mdev_count - 1) / (int) mdev_count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        const uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        if (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            return false;
        }
    }

    kparams->n_hmx              = 1;
    kparams->pipeline           = pipeline ? 1 : 0;
    kparams->m_chunk            = m_chunk;
    kparams->n_chunk            = n_chunk;
    kparams->n_threads          = n_threads;
    kparams->n_act_threads      = act_threads_selected;
    kparams->tile_size          = is_tiled ? htp_tensor_get_tile_size(wtype) : 0;
    kparams->aligned_tile_size  = aligned_tile_size;
    kparams->act_row_size       = htp_mm_tiled_act_row_size(wtype, ne10);
    kparams->vtcm_size          = vtcm_size;
    kparams->div_n_act_threads  = init_fastdiv_values(act_threads_selected);
    kparams->div_ne00_padded    = init_fastdiv_values(ne00_padded);
    kparams->vtcm_bias_size     = (int32_t) src2_size;
    kparams->kernel_type        = (is_batched && !is_matmul_id) ? HTP_MM_KERNEL_HMX_BATCHED : HTP_MM_KERNEL_HMX_2D;
    return true;
}

static void htp_kparams_solve_hvx_mm(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    bool is_tiled,
    int ne02,
    int ne03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    bool is_matmul_id,
    const size_t src2_row_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    if (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    }

    const uint32_t n_threads = htp_session_n_threads(sess);
    kparams->n_hmx     = 0;
    kparams->n_threads = n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    const int src1_nrows = ne11 * ne12 * ne13;

    if (is_tiled) {
        kparams->tile_size         = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        const bool k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        if (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (int) n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            struct htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch    = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, true, false
                );
                if (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                return;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_bias_size = 0;
            return;
        } else {
            if (k_align && opt_mm_select >= 1) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                kparams->kernel_type  = (src1_nrows < (int) n_threads)
                                      ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;

                struct htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch    = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, false, false
                    );
                    if (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }

                uint32_t m_chunk = 0;
                if (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, best_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = best_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_size      = L.total_bytes;
                    kparams->vtcm_bias_size = L.bias_bytes;
                    return;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            return;
        }
    } else if (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    } else if (wtype == GGML_TYPE_F16 || wtype == GGML_TYPE_F32 || wtype == GGML_TYPE_BF16) {
        int32_t ktype = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
        int32_t act_elem = 2;
        if (wtype == GGML_TYPE_F32) {
            ktype    = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            act_elem = 4;
        } else if (wtype == GGML_TYPE_BF16) {
            ktype    = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            act_elem = 4;
        }
        struct htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        if (htp_mm_hvx_solve_vtcm_params(
                ktype, wtype, ne10, src1_nrows, n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type    = ktype;
            kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size   = hex_round_up(ne10 * act_elem, 128);
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->n_prefetch     = 16;
            return;
        }
    }

    kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
}

static bool htp_kparams_solve_matmul_impl(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    const size_t src2_row_size,
    const size_t src2_size,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const bool is_tiled    = htp_tensor_is_tiled(src0);
    const int  wtype       = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    const int  ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    const int  ne11_padded = hex_round_up(ne11, 32);
    const int  ne01_tiled  = hex_round_up(ne01, 32);

    const bool   is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool   is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);
    const size_t vtcm_budget  = htp_session_vtcm_size(sess);

    const bool hmx_enabled = (htp_session_n_hmx(sess) > 0) && (opt_mm_select >= 2);
    if (hmx_enabled && htp_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        if (htp_kparams_solve_hmx_mm(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            goto finalize;
        }
    }

    htp_kparams_solve_hvx_mm(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);

    if (kparams->kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams->vtcm_size > vtcm_budget) {
        HEX_VERBOSE("ggml-hex: %s supported %s VTCM size needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), is_matmul_id ? "MUL_MAT_ID" : "MUL_MAT", kparams->vtcm_size, vtcm_budget);
        return false;
    }
    return true;
}

static bool htp_matmul_check_types(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (is_matmul_id) {
        const struct ggml_tensor * src2 = dst->src[2];
        if (src1->type != GGML_TYPE_F32 || !src2 || src2->type != GGML_TYPE_I32) {
            return false;
        }
        if (ggml_get_op_params_i32(dst, 3) == GGML_PREC_F32) {
            return false;
        }
        switch (src0->type) {
            case GGML_TYPE_Q1_0:
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q2_K:
                if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                    return false;
                }
                if (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                    return false;
                }
                return true;

            case GGML_TYPE_F16:
            case GGML_TYPE_F32:
            case GGML_TYPE_BF16:
                if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                    return false;
                }
                return src0->ne[0] % 32 == 0;

            default:
                return false;
        }
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }
    if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3] ||
        src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
            if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                return false;
            }
            return src0->ne[0] % ggml_blck_size(src0->type) == 0;

        case GGML_TYPE_F32:
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            return src0->nb[1] >= src0->nb[0];

        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return src0->nb[1] >= src0->nb[0];

        default:
            return false;
    }
}

static bool htp_kparams_solve_matmul(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_mm_kernel_params * kparams
) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (!htp_matmul_check_types(src0, src1, dst)) {
        return false;
    }
    if (htp_matmul_can_collapse(src0, src1, dst)) {
        const ggml_tensor src1_collapsed = htp_tensor_collapse_rows(src1);
        const ggml_tensor dst_collapsed  = htp_tensor_collapse_rows(dst);
        return htp_kparams_solve_matmul_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
    }
    return htp_kparams_solve_matmul_impl(sess, src0, src1, dst, 0, 0, kparams);
}

bool htp_matmul_add_can_collapse(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst
) {
    if (!htp_matmul_can_collapse(src0, src1, dst)) {
        return false;
    }
    if (!src2) {
        return true;
    }
    const int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    if (src2_nrows == 1) {
        return true;
    }
    size_t s2;
    return src2->nb[0] == ggml_type_size(src2->type) &&
           ggml_are_same_shape(src2, dst) &&
           htp_rows_stride(src2->ne, src2->nb, &s2);
}

bool htp_kparams_solve_matmul_add(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    if (htp_matmul_add_can_collapse(src0, src1, src2, dst)) {
        const ggml_tensor src1_collapsed = htp_tensor_collapse_rows(src1);
        const ggml_tensor dst_collapsed  = htp_tensor_collapse_rows(dst);
        const ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? htp_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        const struct ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        const size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        return htp_kparams_solve_matmul_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
    }
    const size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    return htp_kparams_solve_matmul_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

bool htp_kparams_solve_mmnx(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    const uint32_t n_threads   = htp_session_n_threads(sess);
    const uint32_t n_hmx       = htp_session_n_hmx(sess);
    const size_t   vtcm_budget = htp_session_vtcm_size(sess);

    memset(kparams, 0, sizeof(*kparams));
    kparams->n_threads = n_threads;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = act->ne[0];
    const int ne11 = act->ne[1];
    const int ne12 = act->ne[2];
    const int ne13 = act->ne[3];

    const bool is_tiled    = htp_tensor_is_tiled(src0);
    const int  wtype       = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    const int  ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    const int  ne11_padded = hex_round_up(ne11, 32);
    const int  ne01_tiled  = hex_round_up(ne01, 32);
    const bool is_batched  = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const bool hmx_enabled = (n_hmx > 0) && (opt_mm_select >= 2);
    if (hmx_enabled && htp_matmul_is_hmx_eligible(src0, act, nullptr, false, is_batched)) {
        if (htp_kparams_solve_hmx_mm(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, false, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            goto finalize;
        }
    }

    if (!is_tiled || act->type != GGML_TYPE_F32 || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return false;
    }

    {
        const int    act_nrows     = ne11 * ne12 * ne13;
        const size_t act_row_size  = htp_mm_tiled_act_row_size(wtype, ne10);
        const size_t src0_row_size = src0->nb[1];

        const uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, n_threads,
                0, src0_row_size, act_row_size, 0, d, false, true
            );
            if (L.total_bytes <= vtcm_budget) {
                best_n_prefetch = d;
                break;
            }
        }

        struct htp_mm_hvx_vtcm_layout L;
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, false, true
        );

        if (opt_mm_select >= 1 && L.total_bytes <= vtcm_budget) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->tile_size         = htp_tensor_get_tile_size(wtype);
            kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
            kparams->act_row_size      = act_row_size;
            kparams->vtcm_bias_size    = 0;
            kparams->vtcm_size         = L.total_bytes;
            kparams->n_prefetch        = best_n_prefetch;
            kparams->n_weights         = n_weights;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            return false;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);
    return (size_t) kparams->vtcm_size <= vtcm_budget;
}

bool htp_kparams_solve_mmidnx(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    const bool ok = htp_kparams_solve_matmul_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
    return ok;
}

static bool htp_kparams_solve_binary(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op_tensor,
    struct htp_binary_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t op = htp_op_remap(op_tensor);
    const struct ggml_tensor * src0 = op_tensor->src[0];
    const struct ggml_tensor * src1 = op_tensor->src[1];
    const struct ggml_tensor * src2 = (op == HTP_OP_ADD_ID) ? op_tensor->src[2] : nullptr;
    const struct ggml_tensor * dst  = op_tensor;

    const bool is_add_id = (op == HTP_OP_ADD_ID);
    if (is_add_id) {
        if (!src2) {
            return false;
        }
        if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
            return false;
        }
        if (!ggml_are_same_shape(src0, dst)) {
            return false;
        }
        if (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
            return false;
        }
        if (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
            return false;
        }
        if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src2->nb[0] != sizeof(int32_t)) {
            return false;
        }
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            return false;
        }
        if (src2->nb[1] < src2->ne[0] * sizeof(int32_t)) {
            return false;
        }
    } else {
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            src1->type != src0->type || dst->type != src0->type) {
            return false;
        }
        if (ggml_is_permuted(src0) || ggml_is_permuted(dst) || !ggml_are_same_shape(src0, dst)) {
            return false;
        }
        if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
            return false;
        }
    }

    const uint32_t max_threads_sess = htp_session_n_threads(sess);
    const size_t   vtcm_size        = htp_session_vtcm_size(sess);

    const size_t elem_size     = ggml_type_size(src0->type);
    const size_t src0_row_size = src0->ne[0] * elem_size;
    const size_t src1_row_size = src1->ne[0] * elem_size;
    const size_t dst_row_size  = dst->ne[0]  * elem_size;

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    if (!htp_binary_strides_safe(src0, src1, dst, elem_size)) {
        return false;
    }

    const bool is_scalar = !is_add_id && src1->ne[0] == 1;
    const bool is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    const bool is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    const bool is_complex          = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    const bool is_contig           = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    const bool is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    if (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        const uint32_t total_elems        = (uint32_t) ggml_nelements(src0);
        const uint32_t max_chunk_elems    = 32768 / elem_size;
        const uint32_t min_chunk_elems    = 256;
        const uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * max_threads_sess) - 1) / (2 * max_threads_sess), 32);
        const uint32_t chunk_size         = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        const uint32_t chunk_bytes        = hex_round_up(chunk_size * elem_size, 128);
        const uint32_t total_chunks       = (total_elems + chunk_size - 1) / chunk_size;
        const uint32_t n_threads          = (total_chunks >= 2) ? (std::min)(max_threads_sess, total_chunks) : 1;

        kparams->kernel_type           = HTP_BINARY_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->n_srcs                = 2;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        struct htp_binary_vtcm_layout L;
        htp_binary_vtcm_layout_build(&L, kparams, vtcm_size);
        if (L.total_bytes == 0 || L.total_bytes > vtcm_size) {
            return false;
        }

        kparams->vtcm_size = L.total_bytes;
        return true;
    }

    enum htp_binary_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    if (is_add_id) {
        kernel_type = HTP_BINARY_KERNEL_ADD_ID;
        src1_size   = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size   = hex_round_up(ggml_nelements(src2) * sizeof(int32_t), 128);
        if (src1_size + src2_size > vtcm_size) {
            return false;
        }
    } else if (is_row_bcast) {
        kernel_type = HTP_BINARY_KERNEL_ROW_BCAST;
        src1_size   = src1_row_size_aligned;
    } else if (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        if (src1_size > vtcm_size) {
            return false;
        }
        kernel_type           = HTP_BINARY_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } else if (is_same_shape) {
        kernel_type = HTP_BINARY_KERNEL_SAME_SHAPE;
    } else if (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        if (src1_size > vtcm_size) {
            return false;
        }
        kernel_type           = HTP_BINARY_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } else {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        if (src1_size > vtcm_size) {
            return false;
        }
        kernel_type           = HTP_BINARY_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    const uint32_t src0_nrows          = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t min_rows_per_thread = (uint32_t) (std::max<size_t>(1, (128 + dst_row_size - 1) / dst_row_size));
    const uint32_t max_threads         = (std::max)(1u, src0_nrows / min_rows_per_thread);

    kparams->kernel_type           = kernel_type;
    kparams->n_threads             = (std::min)(max_threads_sess, max_threads);
    kparams->n_srcs                = 2;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    const bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    const bool src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    const bool dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    const bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    const bool src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    const bool dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    bool split_at_ne01 = false;
    if (src0->ne[2] > 1) {
        if (kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } else if (kernel_type == HTP_BINARY_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } else {
            split_at_ne01 = true;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    bool split_at_ne02 = false;
    if (src0->ne[3] > 1) {
        if (kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } else if (kernel_type == HTP_BINARY_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } else {
            split_at_ne02 = true;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    kparams->div_ne01  = init_fastdiv_values((uint32_t) src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_values((uint32_t) src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_values((uint32_t) (src0->ne[1] * src0->ne[2]));
    kparams->div_ne11  = init_fastdiv_values((uint32_t) src1->ne[1]);
    kparams->div_ne12  = init_fastdiv_values((uint32_t) src1->ne[2]);
    kparams->div_ne13  = init_fastdiv_values((uint32_t) src1->ne[3]);

    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, kparams, vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > vtcm_size) {
        if (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            return htp_binary_solve_chunked(&L, kparams, total_elems, (uint32_t) elem_size,
                                            max_threads_sess, 2, is_scalar_broadcast, vtcm_size);
        }
        return false;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    return true;
}

// collapse contiguous dims of elementwise ops so that rows fill at least one HVX vector
static void htp_collapse_dims(struct ggml_tensor ** tensors, size_t n_tensors, size_t elem_size, uint32_t n_threads) {
    if (n_tensors == 0 || !tensors[0]) {
        return;
    }
    for (size_t t = 0; t < n_tensors; t++) {
        if (!tensors[t] || tensors[t]->nb[0] != elem_size) {
            return;
        }
        if (t > 0 && !ggml_are_same_shape(tensors[0], tensors[t])) {
            return;
        }
    }

    struct ggml_tensor * t0 = tensors[0];

    while (t0->ne[1] > 1 || t0->ne[2] > 1 || t0->ne[3] > 1) {
        const size_t row_size = (size_t) t0->ne[0] * elem_size;
        bool rows_contig = true;
        for (size_t t = 0; t < n_tensors; t++) {
            if (tensors[t]->nb[1] != (size_t) tensors[t]->ne[0] * elem_size) {
                rows_contig = false;
                break;
            }
        }
        bool all_ne1_one = true;
        for (size_t t = 0; t < n_tensors; t++) {
            if (tensors[t]->ne[1] != 1) {
                all_ne1_one = false;
                break;
            }
        }
        const bool can_merge = all_ne1_one ||
                               (rows_contig && (t0->ne[0] == 1 || (row_size < 128 && row_size * t0->ne[1] <= 2048)));
        if (!can_merge) {
            break;
        }
        for (size_t t = 0; t < n_tensors; t++) {
            tensors[t]->ne[0] *= tensors[t]->ne[1];
            tensors[t]->ne[1]  = tensors[t]->ne[2];
            tensors[t]->ne[2]  = tensors[t]->ne[3];
            tensors[t]->ne[3]  = 1;
            tensors[t]->nb[1]  = tensors[t]->nb[2];
            tensors[t]->nb[2]  = tensors[t]->nb[3];
            tensors[t]->nb[3]  = tensors[t]->nb[2] * tensors[t]->ne[2];
        }
    }

    // ne1 too big to merge in full: merge it in chunks of k rows (k divides ne1)
    const size_t row_size = (size_t) t0->ne[0] * elem_size;
    bool rows_contig = true;
    for (size_t t = 0; t < n_tensors; t++) {
        if (tensors[t]->nb[1] != row_size) {
            rows_contig = false;
            break;
        }
    }
    if (t0->ne[1] > 1 && row_size < 128 && rows_contig) {
        int64_t k = 2048 / row_size;
        while (k > 1 && t0->ne[1] % k != 0) {
            k--;
        }
        if (k > 1) {
            for (size_t t = 0; t < n_tensors; t++) {
                tensors[t]->ne[0] *= k;
                tensors[t]->ne[1] /= k;
                tensors[t]->nb[1] *= k;
                tensors[t]->nb[2]  = tensors[t]->nb[1] * tensors[t]->ne[1];
                tensors[t]->nb[3]  = tensors[t]->nb[2] * tensors[t]->ne[2];
            }
        }
    }

    // nrows too small to saturate threads: split ne0 into k rows (k divides ne0)
    const size_t   row_size_curr = (size_t) t0->ne[0] * elem_size;
    const uint32_t nrows         = (uint32_t) (t0->ne[1] * t0->ne[2] * t0->ne[3]);
    rows_contig = true;
    for (size_t t = 0; t < n_tensors; t++) {
        if (tensors[t]->ne[1] != 1 && tensors[t]->nb[1] != row_size_curr) {
            rows_contig = false;
            break;
        }
    }
    if (nrows < n_threads && rows_contig && row_size_curr >= 4096) {
        const size_t vec_elems = 128 / elem_size;
        int64_t k = std::max<int64_t>(row_size_curr / 16384, (n_threads * 4 + nrows - 1) / nrows);
        k = std::min<int64_t>(k, row_size_curr / 2048);
        while (k > 1 && t0->ne[0] % (k * vec_elems) != 0) {
            k--;
        }
        if (k > 1) {
            for (size_t t = 0; t < n_tensors; t++) {
                tensors[t]->ne[0] /= k;
                tensors[t]->ne[1] *= k;
                tensors[t]->nb[1]  = tensors[t]->ne[0] * elem_size;
                tensors[t]->nb[2]  = tensors[t]->nb[1] * tensors[t]->ne[1];
                tensors[t]->nb[3]  = tensors[t]->nb[2] * tensors[t]->ne[2];
            }
        }
    }
}

static bool htp_kparams_solve_unary_impl(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    if (!htp_op_is_unary((htp_op_code) op)) {
        return false;
    }

    if (op == HTP_OP_HADAMARD) {
        const int64_t n = src0->ne[0];
        if (n < 32 || (n & (n - 1)) != 0) {
            return false;
        }
        if (!dst->src[0] || dst->src[0]->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)) {
            return false;
        }
    } else {
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) || dst->type != src0->type) {
            return false;
        }
        if (op == HTP_OP_TRI && (src0->type != GGML_TYPE_F32 || !ggml_is_contiguous(src0))) {
            return false;
        }
        if (src0->type == GGML_TYPE_F16) {
            switch (op) {
                case HTP_OP_NORM:
                case HTP_OP_RMS_NORM:
                case HTP_OP_L2_NORM:
                case HTP_OP_SCALE:
                case HTP_OP_CLAMP:
                case HTP_OP_SQR:
                case HTP_OP_SQRT:
                case HTP_OP_UNARY_LOG:
                case HTP_OP_UNARY_ABS:
                case HTP_OP_UNARY_STEP:
                case HTP_OP_UNARY_CEIL:
                case HTP_OP_UNARY_SGN:
                    break;
                default:
                    return false;
            }
        }
    }

    if (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous(dst) || !ggml_are_same_shape(src0, dst)) {
        return false;
    }

    const size_t elem_size = ggml_type_size(src0->type);

    const size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    const uint32_t src0_nrows          = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t min_rows_per_thread = (uint32_t) (std::max<size_t>(1, (128 + dst_data_row_size - 1) / dst_data_row_size));
    const uint32_t max_threads         = (std::max)(1u, src0_nrows / min_rows_per_thread);
    const uint32_t n_threads           = (std::min)(htp_session_n_threads(sess), max_threads);
    const size_t   vtcm_size           = htp_session_vtcm_size(sess);

    kparams->n_threads = n_threads;

    const bool broadcast_weight = (op == HTP_OP_RMS_NORM_MUL) && (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    kparams->broadcast_weight   = broadcast_weight;

    const bool     src0_contig           = ggml_is_contiguous_1(src0);
    const bool     src1_contig           = (op == HTP_OP_RMS_NORM_MUL && !broadcast_weight) ? ggml_is_contiguous_1(src1) : true;
    const uint32_t src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads;
    const uint32_t min_block_rows        = (uint32_t) (std::max<size_t>(1, 2048 / dst_data_row_size));
    uint32_t       max_block             = (src0_contig && src1_contig) ? src0_nrows : (uint32_t) src0->ne[1];
    if (src0_nrows_per_thread >= 2 * min_block_rows) {
        max_block = (std::min)(max_block, (std::max)(min_block_rows, src0_nrows_per_thread / 4));
    }

    kparams->src0_contig = src0_contig ? 1 : 0;
    kparams->src1_contig = src1_contig ? 1 : 0;

    struct htp_unary_vtcm_layout L;
    uint32_t col_tile            = 0;
    uint32_t vtcm_row_per_thread = 0;

    htp_unary_solve_layout(&L, op, src0->ne[0], dst->ne[0],
                           op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                           max_block, broadcast_weight, n_threads, vtcm_size, elem_size,
                           &col_tile, &vtcm_row_per_thread);

    kparams->col_tile  = col_tile;
    kparams->vtcm_size = L.total_bytes;
    kparams->block     = col_tile ? 0 : vtcm_row_per_thread;

    const uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->tiles_per_row       = tiles_per_row;
    kparams->div_ne01            = init_fastdiv_values(src0->ne[1]);
    kparams->div_ne02            = init_fastdiv_values(src0->ne[2]);
    kparams->div_ne012           = init_fastdiv_values(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr             = init_fastdiv_values(tiles_per_row);

    return (col_tile > 0 || vtcm_row_per_thread > 0) && L.total_bytes <= vtcm_size;
}

static bool htp_kparams_solve_unary(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_unary_kernel_params * kparams
) {
    const htp_op_code opcode = htp_op_remap(op);
    const struct ggml_tensor * src0 = (opcode == HTP_OP_HADAMARD) ? op->src[1] : op->src[0];
    const struct ggml_tensor * src1 = (opcode == HTP_OP_HADAMARD) ? nullptr    : op->src[1];
    return htp_kparams_solve_unary_impl(sess, opcode, src0, src1, op, kparams);
}

bool htp_kparams_solve_rms_norm_mul(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
) {
    return htp_kparams_solve_unary_impl(sess, HTP_OP_RMS_NORM_MUL, src0, src1, dst, kparams);
}

bool htp_kparams_solve_repeat_get_rows(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    if (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K) {
        return false;
    }

    if (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (src0->type == dst->type) {
        if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            return false;
        }
    } else if (src0->type == GGML_TYPE_I32) {
        return false;
    } else if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne10 = src1->ne[0];
    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t nr   = ne10 * ne11 * ne12;

    const bool tiled = htp_tensor_is_tiled(src0);
    const int  wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    if (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } else if (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } else {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    const uint32_t total_tasks          = nr;
    const uint32_t min_tasks_per_thread = (uint32_t) (std::max<size_t>(1, (128 + dst->nb[1] - 1) / dst->nb[1]));
    const uint32_t max_threads          = (std::max)(1u, total_tasks / min_tasks_per_thread);
    const size_t   vtcm_size            = htp_session_vtcm_size(sess);

    kparams->n_threads = (std::min)(htp_session_n_threads(sess), max_threads);

    struct htp_get_rows_vtcm_layout vtcm_layout = {};
    while (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        if (vtcm_layout.total_bytes <= vtcm_size) {
            break;
        }
        --kparams->n_threads;
    }

    if (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size   = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10      = ne10 > 0 ? init_fastdiv_values(ne10) : fastdiv_values{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_values(ne10 * ne11) : fastdiv_values{0, 0};
    kparams->div_ne02      = ne02 > 0 ? init_fastdiv_values(ne02) : fastdiv_values{0, 0};
    kparams->div_ne03      = ne03 > 0 ? init_fastdiv_values(ne03) : fastdiv_values{0, 0};

    if (total_tasks == 0) {
        return true;
    }
    return kparams->n_threads > 0 && (size_t) kparams->vtcm_size <= vtcm_size;
}

static bool htp_kparams_solve_get_rows(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_get_rows_kernel_params * kparams
) {
    return htp_kparams_solve_repeat_get_rows(sess, op->src[0], op->src[1], op, kparams);
}

static bool htp_kparams_solve_set_rows(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    if (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        return false;
    }

    const uint32_t nr = src0->ne[1];

    kparams->n_threads   = (std::min)(htp_session_n_threads(sess), nr);
    kparams->total_tasks = nr;

    struct htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;

    return vtcm_layout.total_bytes <= htp_session_vtcm_size(sess);
}

static bool htp_kparams_solve_softmax(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src2) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src0->ne[2] > 512) {
        return false;
    }
    if (src1) {
        if ((src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            src0->ne[0] != src1->ne[0] || src1->ne[1] < src0->ne[1] ||
            src0->ne[2] % src1->ne[2] != 0 || src0->ne[3] % src1->ne[3] != 0 ||
            !ggml_is_contiguous(src1)) {
            return false;
        }
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const uint32_t src0_nrows          = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t   dst_data_row_size   = op->ne[0] * ggml_type_size(op->type);
    const uint32_t min_rows_per_thread = (uint32_t) (std::max<size_t>(1, (128 + dst_data_row_size - 1) / dst_data_row_size));
    const uint32_t max_threads         = (std::max)(1u, src0_nrows / min_rows_per_thread);
    const uint32_t n_threads           = (std::min)(htp_session_n_threads(sess), max_threads);

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], sizeof(float));
    memcpy(&max_bias, &op->op_params[1], sizeof(float));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    if (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } else {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne10 = src1 ? src1->ne[0] : 1;

    struct htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_threads);
    if (layout.total_bytes > htp_session_vtcm_size(sess)) {
        return false;
    }

    kparams->n_threads = n_threads;
    kparams->vtcm_size = (uint32_t) layout.total_bytes;

    if (!kparams->use_src1) {
        kparams->kernel_type = HTP_SOFTMAX_KERNEL_NOMASK;
    } else if (kparams->use_f16) {
        kparams->kernel_type = HTP_SOFTMAX_KERNEL_MASK_F16;
    } else {
        kparams->kernel_type = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    if (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_values(src0->ne[1]);
    if (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_values(src0->ne[2]);
    const uint32_t ne12 = src1 ? src1->ne[2] : 1;
    const uint32_t ne13 = src1 ? src1->ne[3] : 1;
    if (ne12 > 0) kparams->div_ne12 = init_fastdiv_values(ne12);
    if (ne13 > 0) kparams->div_ne13 = init_fastdiv_values(ne13);

    return true;
}

static bool htp_kparams_solve_rope(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        return false;
    }
    if (src0->ne[0] <= 0) {
        return false;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (src0_nrows == 0) {
        return false;
    }

    const int32_t * op_params = &op->op_params[0];
    const int n_dims = op_params[1];
    const int mode   = op_params[2];
    const int n_offs = op_params[15];

    if (n_dims < 0 || n_dims % 2 != 0) {
        return false;
    }
    if (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        return false;
    }

    float freq_base;
    memcpy(&freq_base, op_params + 5, sizeof(float));
    if (freq_base < 0.0f) {
        return false;
    }

    if (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    const bool is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    if (mode == GGML_ROPE_TYPE_VISION && (n_dims != (int) (src0->ne[0] / 2) || n_offs != 0)) {
        return false;
    }
    if (is_mrope) {
        const int32_t * sections = op_params + 11;
        if (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            return false;
        }
    }

    const int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    if (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        return false;
    }

    if (src2) {
        if (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2) || src2->ne[0] < (n_dims / 2)) {
            return false;
        }
    }

    if (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        src0->nb[1] < src0->ne[0] * sizeof(float) || dst->nb[1] < dst->ne[0] * sizeof(float)) {
        return false;
    }

    const uint32_t n_threads      = (std::min)(htp_session_n_threads(sess), src0_nrows);
    const uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    struct htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    if (layout.total_bytes > htp_session_vtcm_size(sess)) {
        return false;
    }

    kparams->n_threads    = n_threads;
    kparams->vtcm_size    = (uint32_t) layout.total_bytes;
    kparams->div_ne2_ne1  = init_fastdiv_values(dst->ne[2] * dst->ne[1]);
    kparams->div_ne1      = init_fastdiv_values(dst->ne[1]);

    return true;
}

static bool htp_kparams_solve_sum_rows(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_sum_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->op == GGML_OP_ARGMAX) {
        if (dst->type != GGML_TYPE_I32) {
            return false;
        }
    } else if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const uint32_t ne00             = src0->ne[0];
    const uint32_t src0_nrows       = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t max_threads_sess = htp_session_n_threads(sess);

    if (op->op == GGML_OP_SUM) {
        const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        const uint32_t n_threads   = (total_elems >= 1024) ? (std::min)(max_threads_sess, (uint32_t) HTP_MAX_NTHREADS) : 1;
        const uint32_t raw_chunk   = (total_elems + n_threads - 1) / n_threads;

        kparams->n_threads        = n_threads;
        kparams->elems_per_thread = hex_round_up(raw_chunk, 32);
    } else if (op->op == GGML_OP_ARGMAX && src0_nrows <= 1) {
        const uint32_t n_threads = (ne00 >= 1024) ? (std::min)(max_threads_sess, (uint32_t) HTP_MAX_NTHREADS) : 1;
        const uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;

        kparams->n_threads        = n_threads;
        kparams->elems_per_thread = hex_round_up(raw_chunk, 32);
    } else {
        const uint32_t min_rpt     = (src0->nb[1] < 1024) ? 32 : 1;
        const uint32_t max_threads = (std::max)(1u, src0_nrows / min_rpt);

        kparams->n_threads = (std::min)(max_threads_sess, max_threads);
    }

    return true;
}

bool htp_kparams_solve_unary_mul_glu(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_glu_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    if (!htp_op_is_glu((htp_op_code) op)) {
        return false;
    }
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) || dst->type != src0->type) {
        return false;
    }
    if ((op == HTP_OP_GLU_GEGLU_QUICK || op == HTP_OP_GLU_GEGLU_ERF) && src0->type != GGML_TYPE_F32) {
        return false;
    }

    const size_t elem_size = ggml_type_size(src0->type);
    const auto is_valid_src = [elem_size](const struct ggml_tensor * t) {
        return ggml_is_contiguous_1(t) && t->nb[0] == elem_size &&
               (t->ne[1] <= 1 || t->nb[1] >= (size_t) t->ne[0] * elem_size);
    };

    if (!is_valid_src(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1 && (src1->type != src0->type || !ggml_are_same_shape(src0, src1) || !is_valid_src(src1))) {
        return false;
    }

    const size_t   dst_row_size        = dst->ne[0] * elem_size;
    const uint32_t src0_nrows          = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t min_rows_per_thread = (uint32_t) (std::max<size_t>(1, (128 + dst_row_size - 1) / dst_row_size));
    const uint32_t max_threads         = (std::max)(1u, src0_nrows / min_rows_per_thread);
    const uint32_t n_threads           = (std::min)(htp_session_n_threads(sess), max_threads);
    const size_t   vtcm_size           = htp_session_vtcm_size(sess);

    const size_t   row_size_aligned    = hex_round_up(dst_row_size, 128);
    uint32_t       vtcm_row_per_thread = (uint32_t) (vtcm_size / (n_threads * 3 * row_size_aligned));
    if (vtcm_row_per_thread < 2) {
        return false;
    }

    const uint32_t src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads;
    if (src0_nrows_per_thread >= 2 && vtcm_row_per_thread > src0_nrows_per_thread) {
        uint32_t target_rpt = (src0_nrows_per_thread >= 4) ? (src0_nrows_per_thread / 2) & ~1u : 2;
        vtcm_row_per_thread = (std::max)(2u, (std::min)(vtcm_row_per_thread, target_rpt));
    }

    struct htp_glu_vtcm_layout L;
    htp_glu_vtcm_layout_build(&L, dst_row_size, n_threads, vtcm_row_per_thread);

    kparams->n_threads           = n_threads;
    kparams->vtcm_row_per_thread = L.vtcm_row_per_thread;
    kparams->vtcm_size           = (uint32_t) L.total_bytes;

    return true;
}

static bool htp_kparams_solve_glu(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_glu_kernel_params * kparams
) {
    return htp_kparams_solve_unary_mul_glu(sess, htp_op_remap(op), op->src[0], op->src[1], op, kparams);
}

static bool htp_kparams_solve_repeat(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_repeat_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) || src0->type != dst->type) {
        return false;
    }
    if (!ggml_can_repeat(src0, dst) || ggml_is_transposed(src0) || ggml_is_transposed(dst)) {
        return false;
    }

    const uint32_t total_dst_rows      = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const size_t   dst_row_size        = dst->ne[0] * ggml_type_size(dst->type);
    const uint32_t min_rows_per_thread = (uint32_t) (std::max<size_t>(1, (128 + dst_row_size - 1) / dst_row_size));
    const uint32_t max_threads         = (std::max)(1u, total_dst_rows / min_rows_per_thread);
    const uint32_t n_threads           = (std::min)(htp_session_n_threads(sess), max_threads);

    kparams->n_threads = n_threads;
    kparams->nr0       = dst->ne[0] / src0->ne[0];
    kparams->nr1       = dst->ne[1] / src0->ne[1];
    kparams->nr2       = dst->ne[2] / src0->ne[2];
    kparams->nr3       = dst->ne[3] / src0->ne[3];

    kparams->div_ne1  = init_fastdiv_values(dst->ne[1]);
    kparams->div_ne12 = init_fastdiv_values(dst->ne[1] * dst->ne[2]);
    kparams->div_ne01 = init_fastdiv_values(src0->ne[1]);
    kparams->div_ne02 = init_fastdiv_values(src0->ne[2]);
    kparams->div_ne03 = init_fastdiv_values(src0->ne[3]);

    return true;
}

static bool htp_kparams_solve_ssm_conv(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }

    const int64_t d_conv_64  = src1->ne[0];
    const int64_t d_inner_64 = src0->ne[1];
    const int64_t n_t_64     = dst->ne[1];
    const int64_t n_s_64     = dst->ne[2];

    if (d_conv_64 <= 0 || d_conv_64 > 32 || d_inner_64 <= 0) {
        return false;
    }
    if (src0->ne[0] != d_conv_64 - 1 + n_t_64 || src0->ne[1] != d_inner_64 || src0->ne[2] != n_s_64) {
        return false;
    }
    if (src1->ne[0] != d_conv_64 || src1->ne[1] != d_inner_64) {
        return false;
    }
    if (dst->ne[0] != d_inner_64 || dst->ne[1] != n_t_64 || dst->ne[2] != n_s_64) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] != src0->ne[0] * sizeof(float) || src1->nb[1] != src1->ne[0] * sizeof(float)) {
        return false;
    }

    const uint32_t d_conv  = (uint32_t) d_conv_64;
    const uint32_t d_inner = (uint32_t) d_inner_64;
    const uint32_t n_t     = (uint32_t) n_t_64;
    const uint32_t n_s     = (uint32_t) n_s_64;
    const uint32_t ncs     = (uint32_t) src0->ne[0];

    const uint32_t n_threads = (std::min)(htp_session_n_threads(sess), (d_inner + 31) / 32);
    const size_t   vtcm_size = htp_session_vtcm_size(sess);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    const uint32_t raw_rpt              = (d_inner + n_threads - 1) / n_threads;
    const uint32_t d_inner_per_thread   = hex_round_up(raw_rpt, 32);

    if (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;
    } else {
        const uint32_t src1_raw_bytes       = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
        const uint32_t src1_T_bytes         = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
        const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

        const size_t   vtcm_budget     = (vtcm_size > 0 ? vtcm_size / n_threads : (1024 * 1024));
        const uint32_t src0_block_T    = hex_round_up(ncs * 32 * sizeof(float), 128);
        const size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        const size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile          = (std::max)(target_max_tile, 32u);
        target_max_tile          = (std::min)(target_max_tile, 128u);

        const size_t raw_stride_bytes = ncs * sizeof(float) + sizeof(float);
        uint32_t d_inner_tile = (uint32_t) (avail_for_tiles / (2 * (raw_stride_bytes + n_t * sizeof(float))));
        d_inner_tile          = (d_inner_tile / 32) * 32;
        if (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        if (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        if (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;
    }

    struct htp_ssm_conv_vtcm_layout L;
    htp_ssm_conv_vtcm_layout_build(&L, d_inner_per_thread, d_conv, n_t, kparams->d_inner_tile, n_threads);

    kparams->vtcm_size        = (uint32_t) L.total_bytes;
    kparams->div_d_inner_tile = init_fastdiv_values(kparams->d_inner_tile);

    return L.total_bytes <= vtcm_size;
}

static bool htp_kparams_solve_sort(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;
    const bool is_top_k             = (op->op == GGML_OP_TOP_K);

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        return false;
    }

    const uint32_t total_rows    = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t ne00          = src0->ne[0];
    const uint32_t k             = dst->ne[0];
    const uint32_t n_threads_max = htp_session_n_threads(sess) > 0 ? htp_session_n_threads(sess) : 4;
    const size_t   vtcm_budget   = htp_session_vtcm_size(sess) > 0 ? htp_session_vtcm_size(sess) : (8 * 1024 * 1024);

    int32_t order = GGML_SORT_ORDER_DESC;
    if (!is_top_k) {
        order = ((const int32_t *) op->op_params)[0];
    }

    struct htp_sort_vtcm_layout layout;
    if (!htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_top_k)) {
        return false;
    }

    kparams->n_threads     = layout.n_threads;
    kparams->order         = (uint32_t) order;
    kparams->is_top_k      = is_top_k ? 1 : 0;
    kparams->chunk_elems   = layout.chunk_elems;
    kparams->n_chunks      = layout.n_chunks;
    kparams->vtcm_size     = (uint32_t) layout.total_bytes;
    kparams->merge_elems   = layout.merge_elems;
    kparams->n_slots       = layout.n_slots;
    kparams->div_ne01      = init_fastdiv_values((uint32_t) src0->ne[1]);
    kparams->div_ne01_ne02 = init_fastdiv_values((uint32_t) (src0->ne[1] * src0->ne[2]));

    return true;
}

static void htp_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi
) {
    const uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    const int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    if (numer_hi < 0) {
        *ox_hi = 0;
    } else {
        const uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    if (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

static bool htp_kparams_solve_pool_2d(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_pool_2d_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;
    const bool is_pool_1d           = (op->op == GGML_OP_POOL_1D);

    const int32_t * params = dst->op_params;
    if (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (ggml_nbytes(src0) > UINT32_MAX - 256 || ggml_nbytes(dst) > UINT32_MAX - 256) {
        return false;
    }

    const int32_t kernel_x = params[1];
    const int32_t kernel_y = is_pool_1d ? 1 : params[2];
    const int32_t stride_x = params[is_pool_1d ? 2 : 3];
    const int32_t stride_y = is_pool_1d ? 1 : params[4];
    const int32_t pad_x    = params[is_pool_1d ? 3 : 5];
    const int32_t pad_y    = is_pool_1d ? 0 : params[6];

    if (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        return false;
    }

    const uint32_t src_x            = (uint32_t) src0->ne[0];
    const uint32_t src_y            = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    const uint32_t dst_x            = (uint32_t) dst->ne[0];
    const uint32_t dst_y            = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    const uint32_t dst_plane_bytes  = dst_x * dst_y * sizeof(float);
    const uint32_t planes           = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    const uint32_t n_threads        = (std::min)(htp_session_n_threads(sess), planes > 0 ? planes : 1);
    const uint32_t planes_per_chunk = (dst_plane_bytes > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(dst_plane_bytes, HEX_L2_LINE_SIZE)) : 1;

    struct htp_pool_vtcm_layout layout;
    if (!htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, htp_session_vtcm_size(sess))) {
        return false;
    }

    kparams->src_x            = src_x;
    kparams->src_y            = src_y;
    kparams->dst_x            = dst_x;
    kparams->dst_y            = dst_y;
    kparams->kernel_x         = (uint32_t) kernel_x;
    kparams->kernel_y         = (uint32_t) kernel_y;
    kparams->stride_x         = (uint32_t) stride_x;
    kparams->stride_y         = (uint32_t) stride_y;
    kparams->pad_x            = pad_x;
    kparams->pad_y            = pad_y;
    kparams->n_threads        = n_threads;
    kparams->planes           = planes;
    kparams->planes_per_chunk = planes_per_chunk;
    kparams->pool_op          = (uint32_t) params[0];

    const bool exact_tiling = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                               kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y);
    const bool fast_ok      = exact_tiling && (kparams->kernel_x == 1 || kparams->kernel_x == 2);
    const bool narrow_ok    = fast_ok && ((uint64_t) kparams->dst_x * kparams->kernel_x <= 32);
    const bool global_ok    = exact_tiling &&
                              kparams->kernel_x == kparams->src_x && kparams->kernel_y == kparams->src_y &&
                              kparams->dst_x == 1 && kparams->dst_y == 1;
    const bool block_ok     = exact_tiling &&
                              kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                              kparams->kernel_x >= 32;

    if (global_ok) {
        kparams->kernel_type = HTP_POOL_KERNEL_GLOBAL;
    } else if (block_ok) {
        kparams->kernel_type = HTP_POOL_KERNEL_BLOCK;
    } else if (narrow_ok) {
        kparams->kernel_type = HTP_POOL_KERNEL_HVX_NARROW;
    } else if (fast_ok) {
        kparams->kernel_type = HTP_POOL_KERNEL_HVX;
    } else {
        kparams->kernel_type = HTP_POOL_KERNEL_GENERAL;
    }

    kparams->avg_divide_count = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    htp_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                            kparams->stride_x, kparams->pad_x,
                            &kparams->ox_lo, &kparams->ox_hi);
    kparams->inv_kernel_area = 1.0f / (float) (kparams->kernel_x * kparams->kernel_y);

    return true;
}

bool htp_kparams_solve_upscale(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst,
    struct htp_upscale_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const uint32_t src_x = (uint32_t) src0->ne[0];
    const uint32_t src_y = (uint32_t) src0->ne[1];
    const uint32_t src_z = (uint32_t) src0->ne[2];
    const uint32_t src_w = (uint32_t) src0->ne[3];

    const uint32_t dst_x = (uint32_t) dst->ne[0];
    const uint32_t dst_y = (uint32_t) dst->ne[1];
    const uint32_t dst_z = (uint32_t) dst->ne[2];
    const uint32_t dst_w = (uint32_t) dst->ne[3];

    if (src_x == 0 || src_y == 0 || src_z == 0 || src_w == 0 ||
        dst_x == 0 || dst_y == 0 || dst_z == 0 || dst_w == 0) {
        return false;
    }

    const bool is_src_contig_row = (src0->nb[0] == sizeof(float));
    const bool is_src_strided_2d = (src0->nb[0] > sizeof(float)) &&
                                   (src0->nb[0] % sizeof(float) == 0) &&
                                   (src0->nb[1] == (size_t) src_x * src0->nb[0]) &&
                                   ((size_t) src_x * src_y <= 0xFFFFu);

    if ((!is_src_contig_row && !is_src_strided_2d) || dst->nb[0] != sizeof(float)) {
        return false;
    }

    const int32_t mode_flags = dst->op_params[0];
    const uint32_t mode_raw  = (uint32_t) (mode_flags & 0xFF);
    if (mode_raw >= GGML_SCALE_MODE_COUNT) {
        return false;
    }

    const bool align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
    const bool antialias     = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

    float sf0 = (float) dst_x / (float) src_x;
    float sf1 = (float) dst_y / (float) src_y;
    float sf2 = (float) dst_z / (float) src_z;
    float sf3 = (float) dst_w / (float) src_w;
    float pixel_offset = 0.5f;

    if (align_corners) {
        pixel_offset = 0.0f;
        sf0 = (dst_x > 1 && src_x > 1) ? (float) (dst_x - 1) / (float) (src_x - 1) : sf0;
        sf1 = (dst_y > 1 && src_y > 1) ? (float) (dst_y - 1) / (float) (src_y - 1) : sf1;
    }

    const float inv_sf0 = 1.0f / sf0;
    const float inv_sf1 = 1.0f / sf1;
    const float inv_sf2 = 1.0f / sf2;
    const float inv_sf3 = 1.0f / sf3;

    const float filter_radius0 = antialias ? (std::max)(1.0f, inv_sf0) : 0.0f;
    const float filter_radius1 = antialias ? (std::max)(1.0f, inv_sf1) : 0.0f;
    const float invscale0      = antialias ? (1.0f / filter_radius0) : 0.0f;
    const float invscale1      = antialias ? (1.0f / filter_radius1) : 0.0f;

    const uint32_t total_slices = dst_z * dst_w;
    uint32_t       n_threads    = htp_session_n_threads(sess);

    struct htp_upscale_vtcm_layout layout;
    uint32_t dst_nrows_per_chunk = 0;
    while (n_threads > 0) {
        if (htp_upscale_solve_layout(&layout, src_x, src_y, dst_x, dst_y, inv_sf1, filter_radius1, n_threads,
                                     htp_session_vtcm_size(sess), &dst_nrows_per_chunk)) {
            break;
        }
        n_threads--;
    }
    if (n_threads == 0) {
        return false;
    }

    size_t lut_x_size = 0;
    size_t elem_y_sz  = sizeof(uint32_t);
    if (mode_raw == HTP_UPSCALE_MODE_BICUBIC) {
        lut_x_size = hex_round_up(dst_x * sizeof(struct upscale_bicubic_x), 128);
        elem_y_sz  = sizeof(struct upscale_bicubic_y);
    } else if (antialias) {
        lut_x_size = hex_round_up(dst_x * sizeof(struct upscale_antialias_x), 128);
        elem_y_sz  = sizeof(struct upscale_antialias_y);
    } else if (mode_raw == HTP_UPSCALE_MODE_BILINEAR) {
        const uint32_t n_chunks = (dst_x + 31) / 32;
        lut_x_size = hex_round_up(n_chunks * sizeof(struct upscale_bilinear_x_chunk), 128);
        elem_y_sz  = sizeof(struct upscale_bilinear_y);
    } else {
        lut_x_size = hex_round_up(dst_x * sizeof(uint32_t), 128);
    }
    const size_t total_lut_size = lut_x_size + hex_round_up(dst_y * elem_y_sz, 128);
    if (total_lut_size > HTP_DDR_SPAD_SIZE) {
        return false;
    }

    const uint32_t chunks_per_slice = (dst_y + dst_nrows_per_chunk - 1) / dst_nrows_per_chunk;
    const uint32_t total_chunks     = total_slices * chunks_per_slice;

    n_threads = (std::min)(n_threads, total_chunks > 0 ? total_chunks : 1);
    htp_upscale_vtcm_layout_build(&layout, src_x, src_y, dst_x, dst_y, dst_nrows_per_chunk, inv_sf1, filter_radius1, n_threads);

    const uint32_t slice_dst_bytes = dst_x * dst_y * sizeof(float);
    const uint32_t chunk_dst_bytes = dst_x * dst_nrows_per_chunk * sizeof(float);
    const bool     dst_contig      = (dst->nb[1] == (size_t) dst_x * sizeof(float)) &&
                                     (dst_z <= 1 || dst->nb[2] == (size_t) dst_y * dst->nb[1]) &&
                                     (dst_w <= 1 || dst->nb[3] == (size_t) dst_z * dst->nb[2]);

    uint32_t chunks_per_group = 0;
    if (dst_contig) {
        const uint32_t slices_per_group = (slice_dst_bytes > 0)
            ? (HEX_L2_LINE_SIZE / hex_gcd_u32(slice_dst_bytes, HEX_L2_LINE_SIZE))
            : 1;
        if (slices_per_group == 1 && (chunk_dst_bytes % HEX_L2_LINE_SIZE == 0)) {
            chunks_per_group = 1;
        } else {
            chunks_per_group = slices_per_group * chunks_per_slice;
        }
    }

    uint8_t kernel_type = HTP_UPSCALE_KERNEL_BILINEAR;
    if (mode_raw == HTP_UPSCALE_MODE_NEAREST) {
        if (dst_x == 2 * src_x && dst_y == 2 * src_y && !align_corners) {
            kernel_type = HTP_UPSCALE_KERNEL_NEAREST_2X;
        } else if (dst_x == 4 * src_x && dst_y == 4 * src_y && !align_corners) {
            kernel_type = HTP_UPSCALE_KERNEL_NEAREST_4X;
        } else {
            kernel_type = HTP_UPSCALE_KERNEL_NEAREST;
        }
    } else if (mode_raw == HTP_UPSCALE_MODE_BICUBIC) {
        kernel_type = HTP_UPSCALE_KERNEL_BICUBIC;
    } else if (antialias) {
        kernel_type = HTP_UPSCALE_KERNEL_BILINEAR_AA;
    } else if (dst_x == 2 * src_x && dst_y == 2 * src_y && !align_corners && (src_x % 32 == 0)) {
        kernel_type = HTP_UPSCALE_KERNEL_BILINEAR_2X;
    } else {
        kernel_type = HTP_UPSCALE_KERNEL_BILINEAR;
    }

    kparams->kernel_type          = kernel_type;
    kparams->mode                 = (uint8_t) mode_raw;
    kparams->align_corners        = align_corners ? 1 : 0;
    kparams->antialias            = antialias ? 1 : 0;

    kparams->inv_sf0              = inv_sf0;
    kparams->inv_sf1              = inv_sf1;
    kparams->inv_sf2              = inv_sf2;
    kparams->inv_sf3              = inv_sf3;
    kparams->pixel_offset         = pixel_offset;

    kparams->filter_radius0       = filter_radius0;
    kparams->filter_radius1       = filter_radius1;
    kparams->invscale0            = invscale0;
    kparams->invscale1            = invscale1;

    kparams->n_threads            = n_threads;
    kparams->chunks_per_group     = chunks_per_group;

    kparams->dst_nrows_per_chunk  = dst_nrows_per_chunk;
    kparams->chunks_per_slice     = chunks_per_slice;
    kparams->total_chunks         = total_chunks;
    kparams->vtcm_size            = (uint32_t) layout.total_bytes;

    kparams->div_chunks_per_slice = init_fastdiv_values(chunks_per_slice > 0 ? chunks_per_slice : 1);
    kparams->div_dst_z            = init_fastdiv_values(dst_z > 0 ? dst_z : 1);

    return true;
}

static bool htp_kparams_solve_concat(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (!src0 || !src1 || !dst) {
        return false;
    }

    int dim = ((const int32_t *) op->op_params)[0];
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }
    kparams->dim = dim;

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        return false;
    }
    if (src0->type != dst->type || src1->type != dst->type) {
        return false;
    }

    const uint32_t type_size  = ggml_type_size(dst->type);
    const uint32_t n_threads  = htp_session_n_threads(sess);
    const uint32_t mdev_count = htp_session_mdev_count(sess);
    const size_t   vtcm_size  = htp_session_vtcm_size(sess);

    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        const int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        if (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            return false;
        }
    }

    const uint32_t max_gather_ne0 = 128 / type_size;
    const uint32_t total_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    const bool     b_dense        = (src1->nb[0] == type_size);
    const bool     gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    if (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        if (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         n_threads, mdev_count, vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * n_threads;
            return true;
        }
    }

    const bool dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);
    if (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        return true;
    }

    const bool is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    const bool is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    const bool transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    if (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        const uint32_t nth = n_threads > 0 ? n_threads : 8;
        struct htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, nth);

        if (vtcm_size > 0 && layout.total_bytes > vtcm_size) {
            return false;
        }

        kparams->kernel_type          = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads            = nth;
        kparams->vtcm_size            = layout.total_bytes;
        kparams->u.transposed.div_ne2 = init_fastdiv_values((uint32_t) dst->ne[2]);
        return true;
    }

    return false;
}

static bool htp_kparams_solve_cpy(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (!src0 || !dst) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t nelem_src = ggml_nelements(src0);
    const int64_t nelem_dst = ggml_nelements(dst);
    if (nelem_src != nelem_dst || nelem_src < 0) {
        return false;
    }

    const uint32_t src_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    if (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        return true;
    }

    if (nelem_src == 1) {
        if (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            return true;
        }
        if ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            return true;
        }
        return false;
    }

    const bool sametype = (src0->type == dst->type);
    const bool same_extents = ggml_are_same_shape(src0, dst);

    const bool transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    const bool sameshape  = same_extents && !transposed;

    const bool src_is_contiguous = ggml_is_contiguous(src0);
    const bool dst_is_contiguous = ggml_is_contiguous(dst);

    const uint32_t n_threads  = htp_session_n_threads(sess);
    const uint32_t mdev_count = htp_session_mdev_count(sess);
    const size_t   vtcm_size  = htp_session_vtcm_size(sess);

    if (sametype) {
        if (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            return true;
        }

        const uint32_t max_gather_ne0 = 128 / src_type_size;
        const uint32_t total_rows     = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            if (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, false,
                                             src_type_size, total_rows,
                                             n_threads, mdev_count, vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * n_threads;
                return true;
            }
        }

        if (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            return true;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = n_threads > 0 ? (uint8_t) n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_values((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_values((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_values((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_values((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_values((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_values((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        return true;
    }

    if (!sameshape) {
        return false;
    }

    const bool valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    if (!valid_conversion) {
        return false;
    }

    const uint32_t nth = n_threads > 0 ? n_threads : 4;
    struct htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, nth);

    if (vtcm_size > 0 && layout.total_bytes > vtcm_size) {
        return false;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->total_rows              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_threads               = (uint8_t) nth;
    kparams->vtcm_size               = (uint32_t) layout.total_bytes;
    kparams->u.convert.blk_rows      = layout.blk_rows;
    kparams->u.convert.div_ne01      = init_fastdiv_values((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_values((uint32_t) (src0->ne[2] * src0->ne[1]));

    return true;
}

static bool im2col_3d_calc_output_size(
    uint32_t input, uint32_t kernel, int32_t stride, int32_t padding, int32_t dilation, uint32_t * output
) {
    if (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        return false;
    }

    const int64_t numerator = (int64_t) input + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    if (numerator < 0) {
        return false;
    }

    const uint64_t size = (uint64_t) (numerator / stride) + 1;
    if (size == 0 || size > UINT32_MAX) {
        return false;
    }

    *output = (uint32_t) size;
    return true;
}

static bool htp_kparams_solve_im2col(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int32_t * op_params = (const int32_t *) op->op_params;
    const int32_t s0 = op_params[0];
    const int32_t s1 = op_params[1];
    const int32_t p0 = op_params[2];
    const int32_t p1 = op_params[3];
    const int32_t d0 = op_params[4];
    const int32_t d1 = op_params[5];
    const bool is_2D = (op_params[6] == 1);

    if (s0 <= 0 || d0 <= 0 || p0 < 0) {
        return false;
    }
    if (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        return false;
    }

    const uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    const uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    const uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    const uint32_t IW = (uint32_t) src1->ne[0];
    const uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    const uint32_t KW = (uint32_t) src0->ne[0];
    const uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    const uint32_t OW = (uint32_t) dst->ne[1];

    if (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        return false;
    }

    const uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    const uint64_t total_rows_64   = (uint64_t) N * OH;
    if (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        return false;
    }
    const uint32_t patch_stride = (uint32_t) patch_stride_64;
    const uint32_t total_rows   = (uint32_t) total_rows_64;

    if (dst->ne[0] != patch_stride || dst->ne[1] != OW || (is_2D && dst->ne[2] != OH)) {
        return false;
    }

    const uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);

    bool is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    if (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    const uint32_t max_threads    = (sess && htp_session_n_threads(sess) > 0) ? htp_session_n_threads(sess) : 8;
    const uint32_t n_threads      = (total_rows < max_threads) ? total_rows : max_threads;
    const uint32_t vtcm_size      = (sess && htp_session_vtcm_size(sess) > 0) ? (uint32_t) htp_session_vtcm_size(sess) : 0;
    const uint32_t row_size       = (uint32_t) dst->nb[2];
    const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;

    if (is_patchembed) {
        const uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        if (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            struct htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads,
                                         htp_im2col_scratch_bytes(HTP_IM2COL_KERNEL_PATCHEMBED_DMA, KH, patch_stride));

            if (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_size            = (uint32_t) layout.total_bytes;
                kparams->src_row_bytes        = src_row_bytes;
                kparams->dst_row_bytes        = dst_row_bytes;
                kparams->owb                  = OW;
                kparams->wb                   = IW;
                kparams->nbpr                 = 1;
                kparams->N                    = N;
                kparams->IC                   = IC;
                kparams->ID                   = 1;
                kparams->IH                   = IH;
                kparams->IW                   = IW;
                kparams->KD                   = 1;
                kparams->KH                   = KH;
                kparams->KW                   = KW;
                kparams->OD                   = 1;
                kparams->OH                   = OH;
                kparams->OW                   = OW;
                kparams->patch_stride         = patch_stride;
                kparams->total_rows           = total_rows;
                kparams->rows_per_chunk       = rows_per_chunk;
                kparams->div_OH               = init_fastdiv_values(OH);
                kparams->div_OD               = init_fastdiv_values(1);
                return true;
            }
        }
    }

    for (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        const uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        if (Wb > UINT32_MAX) {
            if (owb == 1) break;
            continue;
        }

        const uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        if (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            if (owb == 1) break;
            continue;
        }

        const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        // threads work on output-col blocks, so a single output row (1D) can use them all
        const uint32_t nbpr        = (OW + owb - 1) / owb;
        const uint64_t nblocks     = (uint64_t) total_rows * nbpr;
        const uint32_t blk_threads = (uint32_t) (nblocks < max_threads ? nblocks : max_threads);

        struct htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, blk_threads,
                                     htp_im2col_scratch_bytes(HTP_IM2COL_KERNEL_BLOCKED_DMA, KH, patch_stride));

        if (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->n_threads            = (uint8_t) blk_threads;
            kparams->vtcm_size            = (uint32_t) layout.total_bytes;
            kparams->src_row_bytes        = src_row_bytes;
            kparams->dst_row_bytes        = dst_row_bytes;
            kparams->owb                  = owb;
            kparams->wb                   = (uint32_t) Wb;
            kparams->nbpr                 = nbpr;
            kparams->N                    = N;
            kparams->IC                   = IC;
            kparams->ID                   = 1;
            kparams->IH                   = IH;
            kparams->IW                   = IW;
            kparams->KD                   = 1;
            kparams->KH                   = KH;
            kparams->KW                   = KW;
            kparams->OD                   = 1;
            kparams->OH                   = OH;
            kparams->OW                   = OW;
            kparams->patch_stride         = patch_stride;
            kparams->total_rows           = total_rows;
            kparams->rows_per_chunk       = rows_per_chunk;
            kparams->div_OH               = init_fastdiv_values(OH);
            kparams->div_OD               = init_fastdiv_values(1);
            kparams->div_nbpr             = init_fastdiv_values(nbpr);
            return true;
        }
        if (owb == 1) break;
    }

    return false;
}

static bool htp_kparams_solve_im2col_3d(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int32_t * params = (const int32_t *) op->op_params;
    const int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    const int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    const int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    const int32_t ic = params[9];

    if (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        return false;
    }
    if (src1->ne[3] % ic != 0) {
        return false;
    }

    const uint32_t IC = (uint32_t) ic;
    const uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    if (N == 0) {
        return false;
    }

    uint32_t OW, OH, OD;
    if (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        return false;
    }

    const uint32_t KD = (uint32_t) src0->ne[2];
    const uint32_t KH = (uint32_t) src0->ne[1];
    const uint32_t KW = (uint32_t) src0->ne[0];
    const uint32_t ID = (uint32_t) src1->ne[2];
    const uint32_t IH = (uint32_t) src1->ne[1];
    const uint32_t IW = (uint32_t) src1->ne[0];

    const uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    const uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    const uint64_t dst_ne3_64      = (uint64_t) OD * N;

    if (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        return false;
    }
    const uint32_t patch_stride = (uint32_t) patch_stride_64;
    const uint32_t total_rows   = (uint32_t) total_rows_64;
    const uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    if (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        return false;
    }

    const uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);

    const bool is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    const uint32_t max_threads    = (sess && htp_session_n_threads(sess) > 0) ? htp_session_n_threads(sess) : 8;
    const uint32_t n_threads      = (total_rows < max_threads) ? total_rows : max_threads;
    const uint32_t vtcm_size      = (sess && htp_session_vtcm_size(sess) > 0) ? (uint32_t) htp_session_vtcm_size(sess) : 0;
    const uint32_t row_size       = (uint32_t) dst->nb[2];
    const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;

    if (is_patchembed) {
        const uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        if (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            struct htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads, 256);

            if (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_size            = (uint32_t) layout.total_bytes;
                kparams->src_row_bytes        = src_row_bytes;
                kparams->dst_row_bytes        = dst_row_bytes;
                kparams->owb                  = OW;
                kparams->wb                   = IW;
                kparams->nbpr                 = 1;
                kparams->N                    = N;
                kparams->IC                   = IC;
                kparams->ID                   = ID;
                kparams->IH                   = IH;
                kparams->IW                   = IW;
                kparams->KD                   = KD;
                kparams->KH                   = KH;
                kparams->KW                   = KW;
                kparams->OD                   = OD;
                kparams->OH                   = OH;
                kparams->OW                   = OW;
                kparams->patch_stride         = patch_stride;
                kparams->total_rows           = total_rows;
                kparams->rows_per_chunk       = rows_per_chunk;
                kparams->div_OH               = init_fastdiv_values(OH);
                kparams->div_OD               = init_fastdiv_values(OD);
                return true;
            }
        }
    }

    for (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        const uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        if (Wb > UINT32_MAX) {
            if (owb == 1) break;
            continue;
        }

        const uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        if (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            if (owb == 1) break;
            continue;
        }

        const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        struct htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads, 256);

        if (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_size            = (uint32_t) layout.total_bytes;
            kparams->src_row_bytes        = src_row_bytes;
            kparams->dst_row_bytes        = dst_row_bytes;
            kparams->owb                  = owb;
            kparams->wb                   = (uint32_t) Wb;
            kparams->nbpr                 = (OW + owb - 1) / owb;
            kparams->N                    = N;
            kparams->IC                   = IC;
            kparams->ID                   = ID;
            kparams->IH                   = IH;
            kparams->IW                   = IW;
            kparams->KD                   = KD;
            kparams->KH                   = KH;
            kparams->KW                   = KW;
            kparams->OD                   = OD;
            kparams->OH                   = OH;
            kparams->OW                   = OW;
            kparams->patch_stride         = patch_stride;
            kparams->total_rows           = total_rows;
            kparams->rows_per_chunk       = rows_per_chunk;
            kparams->div_OH               = init_fastdiv_values(OH);
            kparams->div_OD               = init_fastdiv_values(OD);
            return true;
        }
        if (owb == 1) break;
    }

    return false;
}

static bool htp_kparams_solve_pad(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_pad_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int32_t lp0      = ((const int32_t *) op->op_params)[0];
    const int32_t rp0      = ((const int32_t *) op->op_params)[1];
    const bool    circular = (((const int32_t *) op->op_params)[8] != 0);

    if (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        return false;
    }

    const uint32_t type_size      = (uint32_t) ggml_type_size(src0->type);
    const uint32_t total_dst_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    const uint32_t max_threads    = (std::max)(1u, total_dst_rows);
    const uint32_t n_threads      = (std::min)(htp_session_n_threads(sess), max_threads);

    struct htp_pad_vtcm_layout layout;
    htp_pad_vtcm_layout_build(&layout, (uint32_t) src0->ne[0] * type_size, (uint32_t) dst->ne[0] * type_size, n_threads);

    const bool use_dma = (src0->nb[0] == type_size) && (src0->ne[0] >= 512) &&
                         (htp_session_vtcm_size(sess) >= layout.total_bytes);

    kparams->n_threads  = n_threads;
    kparams->total_rows = total_dst_rows;
    kparams->use_dma    = use_dma ? 1 : 0;
    kparams->elem_size  = type_size;
    kparams->vtcm_size  = use_dma ? (uint32_t) layout.total_bytes : 0;

    if (circular && use_dma) {
        kparams->kernel_type = HTP_PAD_KERNEL_HVX_CIRCULAR_DMA;
    } else if (circular) {
        kparams->kernel_type = HTP_PAD_KERNEL_HVX_CIRCULAR;
    } else if (use_dma) {
        kparams->kernel_type = HTP_PAD_KERNEL_HVX_DMA;
    } else {
        kparams->kernel_type = HTP_PAD_KERNEL_HVX;
    }

    kparams->div_ne1     = init_fastdiv_values((uint32_t) dst->ne[1]);
    kparams->div_ne1_ne2 = init_fastdiv_values((uint32_t) (dst->ne[1] * dst->ne[2]));
    kparams->div_ne01    = init_fastdiv_values((uint32_t) src0->ne[1]);
    kparams->div_ne02    = init_fastdiv_values((uint32_t) src0->ne[2]);
    kparams->div_ne03    = init_fastdiv_values((uint32_t) src0->ne[3]);

    return true;
}

static bool htp_kparams_solve_cumsum(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_cumsum_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const uint32_t total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    const uint32_t max_threads = (std::max)(1u, total_rows);
    const uint32_t n_threads   = (std::min)(htp_session_n_threads(sess), max_threads);

    struct htp_cumsum_vtcm_layout layout;
    htp_cumsum_vtcm_layout_build(&layout, (uint32_t) src0->nb[1], (uint32_t) dst->nb[1], n_threads);

    const bool use_dma = (htp_session_vtcm_size(sess) >= layout.total_bytes);

    kparams->n_threads  = n_threads;
    kparams->total_rows = total_rows;
    kparams->use_dma    = use_dma ? 1 : 0;
    kparams->vtcm_size  = use_dma ? (uint32_t) layout.total_bytes : 0;

    return true;
}

static bool htp_kparams_solve_fill(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fill_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * dst = op;

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous_rows(dst) || ggml_is_permuted(dst)) {
        return false;
    }

    const uint32_t ne1 = (uint32_t) dst->ne[1];
    const uint32_t ne2 = (uint32_t) dst->ne[2];
    const uint32_t ne3 = (uint32_t) dst->ne[3];
    const uint32_t nb1 = (uint32_t) dst->nb[1];

    const uint32_t total_rows     = ne1 * ne2 * ne3;
    const uint32_t max_threads    = (std::max)(1u, total_rows);
    const uint32_t n_threads      = (std::min)(htp_session_n_threads(sess), max_threads);
    const uint32_t elem_size      = (uint32_t) ggml_type_size(dst->type);
    const uint32_t rows_per_chunk = (nb1 > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(nb1, HEX_L2_LINE_SIZE)) : 1;

    kparams->n_threads      = n_threads;
    kparams->total_rows     = total_rows;
    kparams->elem_size      = elem_size;
    kparams->kernel_type    = ggml_is_contiguous(dst) ? HTP_FILL_KERNEL_CONTIGUOUS : HTP_FILL_KERNEL_STRIDED;
    kparams->rows_per_chunk = rows_per_chunk;
    kparams->div_ne1        = init_fastdiv_values(ne1);
    kparams->div_ne1_ne2    = init_fastdiv_values(ne1 * ne2);

    return true;
}

static bool htp_kparams_solve_roll(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_roll_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const uint32_t ne0 = (uint32_t) dst->ne[0];
    const uint32_t ne1 = (uint32_t) dst->ne[1];
    const uint32_t ne2 = (uint32_t) dst->ne[2];
    const uint32_t ne3 = (uint32_t) dst->ne[3];

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    uint32_t i0_src0 = 0;
    if (-s0 < 0) {
        i0_src0 = (uint32_t) (-s0 + (int32_t) ne0);
    } else if ((uint32_t) (-s0) >= ne0) {
        i0_src0 = (uint32_t) (-s0) - ne0;
    } else {
        i0_src0 = (uint32_t) (-s0);
    }

    const uint32_t total_rows  = ne1 * ne2 * ne3;
    const uint32_t max_threads = (std::max)(1u, total_rows);
    const uint32_t n_threads   = (std::min)(htp_session_n_threads(sess), max_threads);

    kparams->n_threads  = n_threads;
    kparams->total_rows = total_rows;
    kparams->i0_src0    = i0_src0;
    kparams->n0         = ne0 - i0_src0;

    if (htp_session_mdev_count(sess) <= 1) {
        if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
            kparams->kernel_type = HTP_ROLL_KERNEL_DMA_CONTIGUOUS;
        } else {
            kparams->kernel_type = HTP_ROLL_KERNEL_DMA_STRIDED;
        }
    } else {
        kparams->kernel_type = HTP_ROLL_KERNEL_HVX;
    }

    kparams->div_ne1     = init_fastdiv_values(ne1);
    kparams->div_ne2_ne1 = init_fastdiv_values(ne2 * ne1);

    return true;
}

static bool htp_kparams_solve_diag(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_diag_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[1] != 1 || dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        return false;
    }
    if (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(dst) ||
        ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        return false;
    }

    const uint32_t total_batches  = (uint32_t) (src0->ne[2] * src0->ne[3]);
    const uint32_t max_threads    = (std::max)(1u, total_batches);
    const uint32_t n_threads      = (std::min)(htp_session_n_threads(sess), max_threads);
    const uint32_t dst_batch_size = (uint32_t) (dst->ne[1] * dst->nb[1]);

    bool can_split = (dst->ne[0] == 1 || dst->nb[0] == sizeof(float)) && !ggml_is_permuted(dst);
    uint32_t batches_per_chunk = 1;
    if (can_split) {
        if (dst->ne[2] > 1 && (dst->nb[2] & (HEX_L2_LINE_SIZE - 1)) == 0 &&
            (dst->ne[3] <= 1 || (dst->nb[3] & (HEX_L2_LINE_SIZE - 1)) == 0)) {
            batches_per_chunk = 1;
        } else if (dst->nb[2] == dst_batch_size &&
                   (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
            batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_LINE_SIZE)) : 1;
        } else {
            can_split = false;
        }
    }

    const uint32_t src_batch_size = (uint32_t) (src0->ne[0] * sizeof(float));
    const uint32_t dst_row_size   = (uint32_t) (dst->ne[0] * sizeof(float));

    struct htp_diag_vtcm_layout layout;
    htp_diag_vtcm_layout_build(&layout, src_batch_size, dst_row_size, n_threads);

    const bool use_dma = (htp_session_vtcm_size(sess) >= layout.total_bytes);

    kparams->n_threads         = n_threads;
    kparams->total_batches     = total_batches;
    kparams->use_dma           = use_dma ? 1 : 0;
    kparams->can_split         = can_split ? 1 : 0;
    kparams->batches_per_chunk = batches_per_chunk;
    kparams->vtcm_size         = use_dma ? (uint32_t) layout.total_bytes : 0;
    kparams->div_ne02          = init_fastdiv_values((uint32_t) src0->ne[2]);

    return true;
}

static bool htp_kparams_solve_solve_tri(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_solve_tri_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] != src0->ne[1] || src0->ne[1] != src1->ne[1] ||
        src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3] ||
        !ggml_are_same_shape(src1, dst)) {
        return false;
    }

    const uint32_t max_threads_sess  = htp_session_n_threads(sess);
    const uint32_t k                 = (uint32_t) src1->ne[0];
    const uint32_t col_block         = 32;
    const uint32_t k_chunks          = (k + col_block - 1) / col_block;
    const uint32_t total_batches     = (uint32_t) (src0->ne[2] * src0->ne[3]);
    const bool     batched           = total_batches >= max_threads_sess;
    const uint32_t total_jobs        = batched ? total_batches : (total_batches * k_chunks);
    const uint32_t max_threads       = (std::max)(1u, total_jobs);
    const uint32_t n_threads         = (std::min)(max_threads_sess, max_threads);
    const uint32_t batch_size        = (uint32_t) dst->nb[2];
    const uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(batch_size, HEX_L2_LINE_SIZE)) : 1;

    kparams->n_threads         = n_threads;
    kparams->total_batches     = total_batches;
    kparams->total_jobs        = total_jobs;
    kparams->k_chunks          = k_chunks;
    kparams->col_block         = col_block;
    kparams->batched           = batched ? 1 : 0;
    kparams->batches_per_chunk = batches_per_chunk;
    kparams->div_ne02          = init_fastdiv_values((uint32_t) src0->ne[2]);
    kparams->div_k_chunks      = init_fastdiv_values(k_chunks > 0 ? k_chunks : 1);

    return true;
}

// Implicit GEMM over sliding windows of fp16 input croutons (HMX activation:single), no im2col
static bool htp_kparams_conv2d_win(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    struct htp_conv2d_kernel_params * kp
) {
    if (htp_session_n_hmx(sess) == 0) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);

    const uint32_t KW   = src0->ne[0];
    const uint32_t KH   = src0->ne[1];
    const uint32_t Cin  = src0->ne[2];
    const uint32_t Cout = src0->ne[3];
    const uint32_t IW   = src1->ne[0];
    const uint32_t IH   = src1->ne[1];
    const uint32_t OW   = dst->ne[0];
    const uint32_t OH   = dst->ne[1];
    const uint32_t N    = dst->ne[3];
    const size_t   es   = ggml_type_size(src0->type);

    if (s0 != 1 || s1 != 1 || p0 < 0 || p1 < 0 || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (dst->nb[1] < OW * sizeof(float) || (Cout > 1 && dst->nb[2] < OW * sizeof(float))) {
        return false;
    }

    const bool whcn = src1->nb[0] == sizeof(float) && src1->nb[1] >= IW * sizeof(float) && (Cin == 1 || src1->nb[2] >= IH * src1->nb[1]);
    const bool cwhn = !whcn && src1->nb[2] == sizeof(float) && src1->nb[0] == Cin * sizeof(float) && src1->nb[1] >= IW * src1->nb[0];
    if (!whcn && !cwhn) {
        return false;
    }

    const bool w_rows = src0->nb[0] == es && (KH == 1 || src0->nb[1] == KW * es) && (Cin == 1 || src0->nb[2] == KW * KH * es);
    const bool co_c   = (Cout == 1 || src0->nb[3] == es) && (Cin == 1 || src0->nb[2] >= (std::min)(Cout, 32u) * es);
    if (!w_rows && !co_c) {
        return false;
    }

    // short output rows: fold the kw taps into channels so the virtual row is OW wide and output rows are contiguous
    const bool fold = (whcn || (cwhn && d0 == 1)) && KW > 1 && KW * Cin <= 32 && OW < 64 && KW * KH <= 16 && Cout >= 16;

    const uint32_t T     = KW * KH;
    const uint32_t n_ch  = fold ? KW * Cin : Cin;
    const uint32_t n_cc  = (n_ch + 31) / 32;
    const uint32_t rem   = n_ch - (n_cc - 1) * 32;
    const uint32_t stop  = (rem + 7) / 8 * 8 - 1;
    const uint32_t SR    = fold ? OW : IW + 2 * p0;
    const uint32_t IHS   = IH + 2 * p1;
    const uint32_t n_wt  = fold ? KH : T;
    const uint32_t moff  = (KH - 1) * d1 * SR + (fold ? 0 : (KW - 1) * d0);
    const uint32_t halo  = moff / 32 + 1;
    const uint32_t n_rt  = (Cout + 31) / 32;
    const uint64_t n_q   = (uint64_t) (N * IHS - (IHS - OH) - 1) * SR + OW;
    const uint32_t nth   = kp->n_threads;
    const size_t   tile  = 2048;

    if ((uint64_t) N * IHS * SR + moff + 64 >= (1ull << 31)) {
        return false;
    }
    const uint32_t n_tiles = (uint32_t) ((n_q + 31) / 32);

    // weight packing scratch
    uint32_t w_cs  = 0;
    size_t   wraw  = 0;
    size_t   wscr  = 0;
    if (w_rows) {
        const uint32_t kpw = es == 2 ? 64 : 32;
        for (uint32_t cs = 32; cs >= 2; cs /= 2) {
            w_cs = cs;
            wraw = (size_t) 32 * ((cs * T + kpw - 1) / kpw) * 128;
            if (3 * wraw + 64 * 128 <= 256 * 1024 || (fold && cs == 32)) {
                break;
            }
        }
        // enough packing units for all threads
        const uint32_t kpw2 = es == 2 ? 64 : 32;
        while (!fold && w_cs > 2 && n_rt * ((Cin + 31) / 32) * (32 / w_cs) < kp->n_threads) {
            w_cs /= 2;
        }
        wraw = (std::min)(wraw, (size_t) 32 * ((w_cs * T + kpw2 - 1) / kpw2) * 128);
        wscr = 3 * wraw + 64 * 128 + 128;
    } else {
        w_cs = fold ? KW : (std::min)(T, 16u);
        while (!fold && w_cs > 1 && n_rt * ((Cin + 31) / 32) * ((T + w_cs - 1) / w_cs) < kp->n_threads) {
            w_cs = (w_cs + 1) / 2;
        }
        wscr = 2 * (size_t) w_cs * 32 * 128;
    }

    uint32_t rt_block = n_rt;
    uint32_t ct       = (std::max)(4 * halo, (n_tiles + 3) / 4);
    ct = (std::min)((std::max)(ct, 8u), 512u);
    ct = (std::min)(ct, n_tiles);
    uint32_t seg_b = 8;
    // longer output rows make the output DMA more efficient
    uint32_t seg_t = (std::min)(32u, (std::max)(8u, (OW + 31) / 32));

    while (rt_block > 1 && (size_t) rt_block * n_wt * n_cc * tile > htp_session_vtcm_size(sess) / 2) {
        rt_block = (rt_block + 1) / 2;
    }

    size_t stage = 0, thr = 0, act = 0, out = 0, w_size = 0, total = 0;
    uint32_t n_bc = 0;
    while (true) {
        n_bc   = ct + halo;
        stage  = cwhn ? (size_t) seg_b * 32 * Cin * sizeof(float) + 256 : (size_t) n_ch * seg_b * 32 * sizeof(float);
        if (fold) {
            // raw input rows of the staged pixels with 128 bytes of read slack on both sides
            stage = (size_t) Cin * ((seg_b * 32 + OW - 1) / OW + 1) * IW * sizeof(float) + 256;
        }
        stage  = hex_round_up(stage, 128);
        thr    = hex_round_up((std::max)(2 * stage + (size_t) HTP_CONV2D_WIN_OBUF * 32 * seg_t * 32 * sizeof(float), wscr), tile);
        w_size = (size_t) rt_block * n_wt * n_cc * tile;
        act    = (size_t) n_cc * n_bc * tile;
        out    = (size_t) rt_block * ct * tile;
        struct htp_conv2d_vtcm_layout L;
        htp_conv2d_vtcm_layout_build(&L, rt_block, w_size, act, out, thr, nth);
        total  = L.total_bytes;
        if (total <= htp_session_vtcm_size(sess)) {
            break;
        }
        if (ct > 64) {
            ct /= 2;
        } else if (rt_block > 1) {
            rt_block = (rt_block + 1) / 2;
        } else if (seg_b > 1 || seg_t > 1) {
            seg_b = (std::max)(seg_b / 2, 1u);
            seg_t = (std::max)(seg_t / 2, 1u);
        } else if (ct > 1) {
            ct /= 2;
        } else {
            return false;
        }
    }

    uint32_t n_threads = nth;
    // Keep the fold schedule, where output traffic can dominate input packing.
    if (!fold && rt_block < n_rt && ct > 0 && nth > 0) {
        const uint32_t old_rt  = rt_block;
        const uint32_t ct_max  = (std::min)((std::min)((std::max)(4 * halo, (n_tiles + 3) / 4), 512u), n_tiles);
        const size_t   budget  = htp_session_vtcm_size(sess);
        // Approximate chunk overhead in input crouton work units.
        const size_t   c_chunk = 8;
        const uint32_t tail    = n_tiles % ct;
        const size_t   work =
            (size_t) (n_tiles / ct) * ((ct + halo + nth - 1) / nth) + (tail ? (tail + halo + nth - 1) / nth : 0);
        size_t best_cost =
            (size_t) ((n_rt + rt_block - 1) / rt_block) * (n_cc * work + c_chunk * (n_tiles / ct + (tail != 0)));
        bool found = false;

        // Keep output DMA segments. Try larger kernel blocks to avoid rebuilding input croutons.
        for (uint32_t rb = n_rt; rb > old_rt && !found; rb = (rb + 1) / 2) {
            for (uint32_t nt = nth; nt > 0; nt /= 2) {
                for (uint32_t sb = 8; sb > 0; sb /= 2) {
                    const size_t input_stage = hex_round_up(cwhn ? (size_t) sb * 32 * Cin * sizeof(float) + 256 : (size_t) n_ch * sb * 32 * sizeof(float), 128);
                    const size_t scratch = hex_round_up(
                        (std::max)(2 * input_stage + (size_t) HTP_CONV2D_WIN_OBUF * 32 * seg_t * 32 * sizeof(float),
                                   wscr),
                        tile);
                    if (n_cc == 0 || n_wt > budget / tile / n_cc / rb || halo > budget / tile / n_cc ||
                        scratch > budget / nt) {
                        continue;
                    }
                    const size_t weights = (size_t) rb * n_wt * n_cc * tile;
                    htp_conv2d_vtcm_layout fixed;
                    htp_conv2d_vtcm_layout_build(&fixed, rb, weights, (size_t) n_cc * halo * tile, 0, scratch, nt);
                    if (fixed.total_bytes >= budget) {
                        continue;
                    }
                    const size_t   per_ct        = 2 * ((size_t) n_cc + rb) * tile;
                    const uint32_t cap           = (uint32_t) (std::min)((size_t) ct_max, (budget - fixed.total_bytes) / per_ct);
                    const uint32_t aligned       = (cap + halo) / nt * nt;
                    const uint32_t candidates[2] = { cap, aligned > halo ? aligned - halo : 0 };
                    for (uint32_t i = 0; i < 2; i++) {
                        const uint32_t candidate = candidates[i];
                        if (candidate == 0 || (i > 0 && candidate == cap)) {
                            continue;
                        }
                        if (hex_round_up((size_t) T * sizeof(uint32_t), 128) + 2 * hex_round_up((size_t) candidate, 128) > HTP_DDR_SPAD_SIZE) {
                            continue;
                        }
                        const uint32_t rem        = n_tiles % candidate;
                        const size_t   input_work = (size_t) (n_tiles / candidate) * ((candidate + halo + nt - 1) / nt) + (rem ? (rem + halo + nt - 1) / nt : 0);
                        const size_t   chunks     = n_tiles / candidate + (rem != 0);
                        const size_t   cost       = (size_t) ((n_rt + rb - 1) / rb) * (n_cc * input_work + c_chunk * chunks);
                        if (cost < best_cost) {
                            const size_t input  = (size_t) n_cc * (candidate + halo) * tile;
                            const size_t output = (size_t) rb * candidate * tile;
                            htp_conv2d_vtcm_layout L;
                            htp_conv2d_vtcm_layout_build(&L, rb, weights, input, output, scratch, nt);
                            if (L.total_bytes > budget) {
                                continue;
                            }
                            found     = true;
                            best_cost = cost;
                            rt_block  = rb;
                            ct        = candidate;
                            seg_b     = sb;
                            n_threads = nt;
                            stage     = input_stage;
                            thr       = scratch;
                            act       = input;
                            out       = output;
                            w_size    = weights;
                            total     = L.total_bytes;
                            n_bc      = candidate + halo;
                        }
                    }
                }
            }
        }
    }

    if (hex_round_up((size_t) T * sizeof(uint32_t), 128) + 2 * hex_round_up((size_t) ct, 128) > HTP_DDR_SPAD_SIZE) {
        return false;
    }

    const uint32_t TG   = w_cs;
    const uint32_t CS   = w_cs;
    const uint32_t n_tg = (T + TG - 1) / TG;
    const uint32_t n_cs = (32 + CS - 1) / CS;

    kp->kernel_type        = (uint8_t) HTP_CONV2D_KERNEL_HMX_WINDOW;
    kp->n_threads          = (uint8_t) n_threads;
    kp->cwhn               = cwhn ? 1 : 0;
    kp->w_kmajor           = !w_rows ? 1 : 0;
    kp->n_row_tiles        = n_rt;
    kp->ct_chunk           = ct;
    kp->rt_block           = rt_block;
    kp->stage_size         = stage;
    kp->thr_size           = thr;
    kp->vtcm_size          = total;
    kp->window.fold        = fold;
    kp->window.n_cc        = n_cc;
    kp->window.act_rt_last = 0x780 | 0x2 | (stop << 2);
    kp->window.sr          = SR;
    kp->window.ihs         = IHS;
    kp->window.max_off     = moff;
    kp->window.n_bc        = n_bc;
    kp->window.seg_b       = seg_b;
    kp->window.seg_t       = seg_t;
    kp->window.w_cs        = w_cs;
    kp->window.div_sr      = init_fastdiv_values(SR);
    kp->window.div_ihs     = init_fastdiv_values(IHS);
    kp->window.div_kw      = init_fastdiv_values(KW);
    kp->window.div_tg      = init_fastdiv_values(n_tg);
    kp->window.div_cc      = init_fastdiv_values(n_cc);
    kp->window.div_cs      = init_fastdiv_values(n_cs);
    return true;
}

// Tiled implicit GEMM, MMA on HMX if available, else HVX
static bool htp_kparams_conv2d_tiled(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    struct htp_conv2d_kernel_params * kp
) {
    if (htp_session_n_hmx(sess) == 0) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);

    const uint32_t KW   = src0->ne[0];
    const uint32_t KH   = src0->ne[1];
    const uint32_t Cin  = src0->ne[2];
    const uint32_t Cout = src0->ne[3];
    const uint32_t OW   = dst->ne[0];
    const uint32_t OH   = dst->ne[1];
    const uint32_t N    = dst->ne[3];
    const size_t   es   = ggml_type_size(src0->type);

    if (s0 <= 0 || dst->nb[0] != sizeof(float)) {
        return false;
    }

    // kernel k order must match the staged input: (cin, kh, kw) for WHCN input, (kh, kw, cin) for CWHN input.
    // Kernel is either k-contiguous rows per cout (WHCN only) or cout-contiguous with a uniform k stride.
    const bool   whcn   = src1->nb[0] == sizeof(float);
    const bool   cwhn   = !whcn && src1->nb[2] == sizeof(float) && src1->nb[0] == Cin * sizeof(float);
    const bool   co_c   = Cout == 1 || src0->nb[3] == es;
    const bool   w_rows = whcn && src0->nb[0] == es && (KH == 1 || src0->nb[1] == KW * es) && (Cin == 1 || src0->nb[2] == KW * KH * es);
    const size_t ks     = whcn ? src0->nb[0] : src0->nb[2];
    bool         kmajor = false;
    if (whcn) {
        kmajor = co_c && ks >= Cout * es && (KH == 1 || src0->nb[1] == KW * ks) && (Cin == 1 || src0->nb[2] == KW * KH * ks);
    } else if (cwhn) {
        kmajor = co_c && ks >= Cout * es && src0->nb[0] == Cin * ks && (KH == 1 || src0->nb[1] == KW * Cin * ks);
    }
    if (!w_rows && !kmajor) {
        return false;
    }
    kmajor = kmajor && !w_rows;

    const uint32_t K     = KW * KH * Cin;
    const uint32_t kg    = !cwhn ? K : d0 == 1 ? KW * Cin : Cin;
    const uint32_t n_grp = K / kg;
    const uint32_t kg_p  = hex_round_up(kg, 32);
    const uint32_t k_pad = hex_round_up(n_grp * kg_p, 64);
    const uint32_t n_kt  = k_pad / 32;
    const uint32_t n_rt  = (Cout + 31) / 32;
    const uint32_t n_owt = (OW + 31) / 32;
    const uint32_t n_ct  = N * OH * n_owt;
    const uint32_t wk    = (std::min)(k_pad, 512u);
    const uint32_t max_threads = htp_session_n_threads(sess);
    const size_t   tile  = 2048;

    uint32_t rt_block = n_rt;
    uint32_t ct       = 0;
    uint32_t seg      = 0;
    uint32_t nth      = 0;
    uint32_t n_stage  = 1;
    uint32_t sr       = 0;
    size_t       stage     = 0, thr_size = 0;
    size_t       best_cost = SIZE_MAX;
    const size_t wpack     = kmajor ? 2 * (size_t) wk * 64 * sizeof(float) + 96 * 128 : 2 * (size_t) 32 * wk * (sizeof(float) + sizeof(uint16_t));
    const size_t budget    = htp_session_vtcm_size(sess);
    // Approximate chunk overhead in patch K-tile work units.
    const size_t c_chunk   = 8;

    // Keep the kernel resident to avoid regenerating the input for each rt block.
    // Balance patch work per thread and chunk overhead before reducing rt.
    for (uint32_t rb = n_rt; rb > 0 && ct == 0; rb = rb > 1 ? (rb + 1) / 2 : 0) {
        for (uint32_t nt = max_threads; nt > 0; nt /= 2) {
            for (uint32_t sg = (std::min)(8u, n_owt); sg > 0; --sg) {
                // Keep the unused final stride lane for full-vector loads during packing.
                const uint64_t wb = (uint64_t) 32 * sg * s0 + (uint64_t) (KW - 1) * d0;
                if (wb > budget / sizeof(float) || wb > INT32_MAX) {
                    continue;
                }
                const uint64_t width = cwhn ? wb * Cin : wb;
                if (width > budget / sizeof(float) || width > UINT32_MAX - 63) {
                    continue;
                }
                const uint32_t row  = hex_round_up((uint32_t) width, cwhn ? 64 : 32);
                const uint64_t rows = cwhn ? KH : (uint64_t) Cin * KH;
                if (rows > budget / ((uint64_t) row * sizeof(float))) {
                    continue;
                }
                const size_t input_stage  = cwhn ? (size_t) KH * row * sizeof(float) : (size_t) Cin * KH * row * sizeof(float);
                const size_t output_stage = 2 * 32 * (size_t) sg * 32 * sizeof(float);
                const size_t transpose    = cwhn ? input_stage / 2 + 256 + 64 * 128 : 0;
                const size_t scratch      = hex_round_up((std::max)(input_stage + output_stage + transpose, wpack), tile);
                htp_conv2d_vtcm_layout fixed;
                htp_conv2d_vtcm_layout_build(&fixed, rb, (size_t) rb * n_kt * tile, 0, 0, scratch, nt);
                if (fixed.total_bytes >= budget) {
                    continue;
                }

                // Each spatial tile adds two activation tiles and two output tiles per rt.
                const size_t   per_ct        = 2 * ((size_t) n_kt + rb) * tile;
                const uint32_t nc            = (uint32_t) (std::min)((size_t) (std::min)(n_ct, 64u), (budget - fixed.total_bytes) / per_ct);
                const uint32_t candidates[2] = { nc, nc / nt * nt };
                for (uint32_t i = 0; i < 2; i++) {
                    const uint32_t candidate = candidates[i];
                    if (candidate == 0 || (i > 0 && candidate == candidates[0])) {
                        continue;
                    }
                    const uint32_t per_thread = (candidate + nt - 1) / nt;
                    if (sg > per_thread) {
                        continue;
                    }
                    const size_t work   = (size_t) (n_ct / candidate) * per_thread + (n_ct % candidate + nt - 1) / nt;
                    const size_t chunks = n_ct / candidate + (n_ct % candidate != 0);
                    const size_t cost   = work * n_kt + c_chunk * chunks;
                    if (cost < best_cost) {
                        best_cost = cost;
                        rt_block  = rb;
                        ct        = candidate;
                        seg       = sg;
                        nth       = nt;
                        sr        = row;
                        stage     = input_stage;
                        thr_size  = scratch;
                    }
                }
            }
        }
    }
    if (ct == 0) {
        return false;
    }

    const size_t w_size   = (size_t) rt_block * n_kt * tile;
    const size_t act_size = (size_t) ct * n_kt * tile;
    const size_t out_size = (size_t) rt_block * ct * tile;
    const size_t ostage   = 2 * 32 * (size_t) seg * 32 * sizeof(float);
    const size_t prow     = cwhn ? stage / 2 + 256 + 64 * 128 : 0;
    struct htp_conv2d_vtcm_layout L;
    htp_conv2d_vtcm_layout_build(&L, rt_block, w_size, act_size, out_size, thr_size, nth);
    size_t total = L.total_bytes;

    // Second stage buffer: DMA of the next segment overlaps HVX pack of this one.
    // Skip large stages. They already fill VTCM, and the extra DMA slows HMX.
    {
        const size_t thr2 = hex_round_up((std::max)(2 * stage + ostage + prow, wpack), tile);
        struct htp_conv2d_vtcm_layout L2;
        htp_conv2d_vtcm_layout_build(&L2, rt_block, w_size, act_size, out_size, thr2, nth);
        if (stage <= 128 * 1024 && L2.total_bytes <= htp_session_vtcm_size(sess)) {
            n_stage  = 2;
            thr_size = thr2;
            total    = L2.total_bytes;
        }
    }

    kp->kernel_type        = (uint8_t) HTP_CONV2D_KERNEL_HMX_TILED;
    kp->n_threads          = (uint8_t) nth;
    kp->cwhn               = cwhn ? 1 : 0;
    kp->w_kmajor           = kmajor ? 1 : 0;
    kp->n_row_tiles        = n_rt;
    kp->ct_chunk           = ct;
    kp->rt_block           = rt_block;
    kp->stage_size         = stage;
    kp->thr_size           = thr_size;
    kp->vtcm_size          = total;
    kp->tiled.w_kstride    = ks;
    kp->tiled.k_size       = K;
    kp->tiled.k_group      = kg;
    kp->tiled.k_group_pad  = kg_p;
    kp->tiled.n_groups     = n_grp;
    kp->tiled.n_k_tiles    = n_kt;
    kp->tiled.n_ow_tiles   = n_owt;
    kp->tiled.seg_tiles    = seg;
    kp->tiled.stage_row    = sr;
    kp->tiled.n_stage      = n_stage;
    kp->tiled.wk_chunk     = wk;
    kp->tiled.div_ow_tiles = init_fastdiv_values(n_owt);
    kp->tiled.div_oh       = init_fastdiv_values(OH);
    return true;
}

static bool htp_kparams_solve_conv2d_unchecked(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    struct htp_conv2d_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->total_rows = total_rows;
    kparams->n_threads  = total_rows == 0 ? 1 : (std::min)((uint32_t) htp_session_n_threads(sess), total_rows);

    if (htp_kparams_conv2d_win(sess, dst, kparams)) {
        return true;
    }

    return htp_kparams_conv2d_tiled(sess, dst, kparams);
}

static bool htp_kparams_solve_conv2d(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_conv2d_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[2] != src1->ne[2]) {
        return false;
    }

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t p0 = ggml_get_op_params_i32(op, 2);
    const int32_t p1 = ggml_get_op_params_i32(op, 3);
    const int32_t d0 = ggml_get_op_params_i32(op, 4);
    const int32_t d1 = ggml_get_op_params_i32(op, 5);

    if (s0 <= 0 || s1 <= 0 || d0 <= 0 || d1 <= 0) {
        return false;
    }

    const int64_t ow = (src1->ne[0] + 2 * p0 - d0 * (src0->ne[0] - 1) - 1) / s0 + 1;
    const int64_t oh = (src1->ne[1] + 2 * p1 - d1 * (src0->ne[1] - 1) - 1) / s1 + 1;
    if (ow != dst->ne[0] || oh != dst->ne[1] || src0->ne[3] != dst->ne[2] || src1->ne[3] != dst->ne[3]) {
        return false;
    }

    return htp_kparams_solve_conv2d_unchecked(sess, op, kparams);
}

static bool htp_kparams_solve_conv2d_dw(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_conv2d_dw_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != src1->ne[2]) {
        return false;
    }

    if (src0->ne[0] * src0->ne[1] > 64) {
        return false;
    }

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t p0 = ggml_get_op_params_i32(op, 2);
    const int32_t p1 = ggml_get_op_params_i32(op, 3);
    const int32_t d0 = ggml_get_op_params_i32(op, 4);
    const int32_t d1 = ggml_get_op_params_i32(op, 5);

    if (s0 <= 0 || s1 <= 0 || d0 <= 0 || d1 <= 0) {
        return false;
    }

    const int64_t ow = (src1->ne[0] + 2 * p0 - d0 * (src0->ne[0] - 1) - 1) / s0 + 1;
    const int64_t oh = (src1->ne[1] + 2 * p1 - d1 * (src0->ne[1] - 1) - 1) / s1 + 1;
    if (ow != dst->ne[0] || oh != dst->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    const bool cwhn = ggml_is_contiguous_channels(src1);
    if (!cwhn && !ggml_is_contiguous(src1)) {
        return false;
    }

    if (!cwhn && (d0 != 1 || (s0 != 1 && s0 != 2))) {
        return false;
    }

    const uint32_t c  = (uint32_t) src1->ne[2];
    const uint32_t n  = (uint32_t) src1->ne[3];
    const uint32_t w  = (uint32_t) src1->ne[0];
    const uint32_t h  = (uint32_t) src1->ne[1];
    const uint32_t kw = (uint32_t) src0->ne[0];
    const uint32_t kh = (uint32_t) src0->ne[1];
    const uint32_t ow_u = (uint32_t) ow;
    const uint32_t oh_u = (uint32_t) oh;

    const uint32_t row_src_bytes = cwhn
        ? (w * c * (uint32_t) sizeof(float))
        : (w * (uint32_t) sizeof(float));
    const uint32_t row_dst_bytes = cwhn
        ? (ow_u * c * (uint32_t) sizeof(float))
        : (ow_u * (uint32_t) sizeof(float));

    uint8_t kernel_type = HTP_CONV2D_DW_KERNEL_UNSUPPORTED;
    if (cwhn) {
        kernel_type = HTP_CONV2D_DW_KERNEL_CWHN;
    } else if ((w & 31) == 0 && kw == 3 && kh == 3 && p0 == 1 && p1 == 1 && d0 == 1 && d1 == 1 && s0 == 1 && s1 == 1) {
        kernel_type = HTP_CONV2D_DW_KERNEL_WHCN_S1_K3X3;
    } else if ((w & 31) == 0 && kw == 3 && kh == 3 && p0 == 1 && p1 == 1 && d0 == 1 && d1 == 1 && s0 == 2 && s1 == 2) {
        if (w == 32 && ow_u == 16) {
            kernel_type = HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3_W32;
        } else if (w == 64 && ow_u == 32) {
            kernel_type = HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3_W64;
        } else {
            kernel_type = HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3;
        }
    } else if (s0 == 1) {
        kernel_type = HTP_CONV2D_DW_KERNEL_WHCN_S1;
    } else {
        kernel_type = HTP_CONV2D_DW_KERNEL_WHCN_S2;
    }

    const bool     fast_k3x3      = kernel_type >= HTP_CONV2D_DW_KERNEL_WHCN_S1_K3X3;
    const uint32_t pad_rows       = fast_k3x3 ? 4 : 0;
    const uint32_t pad_top_stride = fast_k3x3 ? w : 0;

    const size_t vtcm_budget = (sess && htp_session_vtcm_size(sess) > 0) ? htp_session_vtcm_size(sess) : 0;
    if (vtcm_budget == 0) {
        return false;
    }

    const uint32_t w_size = (uint32_t) (kw * kh * c * ggml_type_size(src0->type));

    uint32_t max_threads_sess = (uint32_t) htp_session_n_threads(sess);
    uint32_t oh_chunk = oh_u;

    uint32_t n_threads = max_threads_sess;
    struct htp_conv2d_dw_vtcm_layout layout;

    while (true) {
        const uint32_t max_ih_chunk = (std::min)(h, (oh_chunk - 1) * (uint32_t) s1 + (kh - 1) * (uint32_t) d1 + 1);
        const uint32_t max_src_bytes = (max_ih_chunk + pad_rows) * row_src_bytes + 128;
        const uint32_t max_dst_bytes = oh_chunk * row_dst_bytes + 128;

        htp_conv2d_dw_vtcm_layout_build(&layout, max_src_bytes, max_dst_bytes, w_size, n_threads);

        if (layout.total_bytes <= vtcm_budget && layout.total_bytes > 0) {
            break;
        }

        if (oh_chunk > 1) {
            oh_chunk = (oh_chunk + 1) / 2;
        } else if (n_threads > 1) {
            n_threads--;
        } else {
            return false;
        }
    }

    const uint32_t n_chunks = (oh_u + oh_chunk - 1) / oh_chunk;
    const uint32_t total_tasks = cwhn ? (n * n_chunks) : (c * n * n_chunks);
    n_threads = (std::min)(n_threads, total_tasks);
    if (n_threads == 0) {
        n_threads = 1;
    }

    const uint32_t max_ih_chunk = (std::min)(h, (oh_chunk - 1) * (uint32_t) s1 + (kh - 1) * (uint32_t) d1 + 1);
    const uint32_t max_src_bytes = (max_ih_chunk + pad_rows) * row_src_bytes + 128;
    const uint32_t max_dst_bytes = oh_chunk * row_dst_bytes + 128;
    htp_conv2d_dw_vtcm_layout_build(&layout, max_src_bytes, max_dst_bytes, w_size, n_threads);

    kparams->kernel_type    = kernel_type;
    kparams->n_threads      = (uint8_t) n_threads;
    kparams->cwhn           = cwhn ? 1 : 0;
    kparams->is_f16         = (src0->type == GGML_TYPE_F16) ? 1 : 0;
    kparams->total_tasks    = total_tasks;
    kparams->vtcm_size      = (uint32_t) layout.total_bytes;
    kparams->w              = w;
    kparams->h              = h;
    kparams->c              = c;
    kparams->n              = n;
    kparams->kw             = kw;
    kparams->kh             = kh;
    kparams->ow             = ow_u;
    kparams->oh             = oh_u;
    kparams->s0             = s0;
    kparams->s1             = s1;
    kparams->p0             = p0;
    kparams->p1             = p1;
    kparams->d0             = d0;
    kparams->d1             = d1;
    kparams->oh_chunk       = oh_chunk;
    kparams->n_chunks       = n_chunks;
    kparams->row_src_bytes  = row_src_bytes;
    kparams->row_dst_bytes  = row_dst_bytes;
    kparams->pad_top_stride = pad_top_stride;
    kparams->div_c          = init_fastdiv_values(c);
    kparams->div_n_chunks   = init_fastdiv_values(n_chunks > 0 ? n_chunks : 1);

    return true;
}

template <auto SolveFn>
static inline bool htp_kparams_invoke(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    void * kparams
) {
    using KParams = decltype(htp_kparams_type(SolveFn));
    KParams tmp;
    return SolveFn(sess, op, kparams ? (KParams *) kparams : &tmp);
}

void htp_opnode_collapse(const struct ggml_hexagon_session * sess, htp_opnode & node) {
    const uint32_t n_threads = htp_session_n_threads(sess);

    if (htp_op_is_unary(node.opcode) && htp_unary_is_elementwise(node.opcode)) {
        if (node.inputs.empty() || !node.inputs[0] || node.outputs.empty() || !node.outputs[0]) {
            return;
        }
        if (!ggml_are_same_shape(node.inputs[0], node.outputs[0])) {
            return;
        }
        const size_t elem_size = ggml_type_size(node.inputs[0]->type);
        struct ggml_tensor src0_c = *node.inputs[0];
        struct ggml_tensor dst_c  = *node.outputs[0];
        struct ggml_tensor * tensors[2] = { &src0_c, &dst_c };
        htp_collapse_dims(tensors, 2, elem_size, n_threads);
        if (src0_c.ne[0] != node.inputs[0]->ne[0] || src0_c.ne[1] != node.inputs[0]->ne[1]) {
            node.inputs[0]  = node.add_dummy(src0_c);
            node.outputs[0] = node.add_dummy(dst_c);
        }
        return;
    }

    if (htp_op_is_glu(node.opcode)) {
        if (node.inputs.size() < 2 || !node.inputs[0] || !node.inputs[1] || node.outputs.empty() || !node.outputs[0]) {
            return;
        }
        if (!ggml_are_same_shape(node.inputs[0], node.outputs[0]) ||
            !ggml_are_same_shape(node.inputs[1], node.outputs[0])) {
            return;
        }
        const size_t elem_size = ggml_type_size(node.inputs[0]->type);
        struct ggml_tensor src0_c = *node.inputs[0];
        struct ggml_tensor src1_c = *node.inputs[1];
        struct ggml_tensor dst_c  = *node.outputs[0];
        struct ggml_tensor * tensors[3] = { &src0_c, &src1_c, &dst_c };
        htp_collapse_dims(tensors, 3, elem_size, n_threads);
        if (src0_c.ne[0] != node.inputs[0]->ne[0] || src0_c.ne[1] != node.inputs[0]->ne[1]) {
            node.inputs[0]  = node.add_dummy(src0_c);
            node.inputs[1]  = node.add_dummy(src1_c);
            node.outputs[0] = node.add_dummy(dst_c);
        }
        return;
    }

    if (node.opcode == HTP_OP_MUL_MAT) {
        if (node.inputs.size() < 2 || !node.inputs[0] || !node.inputs[1] || node.outputs.empty() || !node.outputs[0]) {
            return;
        }
        const struct ggml_tensor * src0 = node.inputs[0];
        const struct ggml_tensor * src1 = node.inputs[1];
        const struct ggml_tensor * dst  = node.outputs[0];
        if (htp_matmul_can_collapse(src0, src1, dst)) {
            node.inputs[1]  = node.add_dummy(htp_tensor_collapse_rows(src1));
            node.outputs[0] = node.add_dummy(htp_tensor_collapse_rows(dst));
        }
        return;
    }

    if (node.opcode == HTP_OP_MUL_MAT_ADD) {
        if (node.inputs.size() < 2 || !node.inputs[0] || !node.inputs[1] || node.outputs.empty() || !node.outputs[0]) {
            return;
        }
        const struct ggml_tensor * src0 = node.inputs[0];
        const struct ggml_tensor * src1 = node.inputs[1];
        const struct ggml_tensor * src2 = node.inputs.size() > 2 ? node.inputs[2] : nullptr;
        const struct ggml_tensor * dst  = node.outputs[0];
        if (htp_matmul_add_can_collapse(src0, src1, src2, dst)) {
            node.inputs[1]  = node.add_dummy(htp_tensor_collapse_rows(src1));
            node.outputs[0] = node.add_dummy(htp_tensor_collapse_rows(dst));
            if (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
                node.inputs[2] = node.add_dummy(htp_tensor_collapse_rows(src2));
            }
        }
        return;
    }

    if (node.opcode == HTP_OP_UPSCALE) {
        if (node.inputs.empty() || !node.inputs[0] || node.outputs.empty() || !node.outputs[0]) {
            return;
        }
        const struct ggml_tensor * src0 = node.inputs[0];
        const struct ggml_tensor * dst  = node.outputs[0];

        if ((src0->ne[3] > 1 || dst->ne[3] > 1) &&
            src0->ne[2] == dst->ne[2] && src0->ne[3] == dst->ne[3] &&
            src0->nb[3] == (size_t) src0->ne[2] * src0->nb[2] &&
            dst->nb[3]  == (size_t) dst->ne[2]  * dst->nb[2]) {
            struct ggml_tensor src0_c = *src0;
            struct ggml_tensor dst_c  = *dst;
            src0_c.ne[2] *= src0_c.ne[3];
            src0_c.ne[3] = 1;
            src0_c.nb[3] = (size_t) src0_c.ne[2] * src0_c.nb[2];
            dst_c.ne[2]  *= dst_c.ne[3];
            dst_c.ne[3]  = 1;
            dst_c.nb[3]  = (size_t) dst_c.ne[2] * dst_c.nb[2];
            node.inputs[0]  = node.add_dummy(src0_c);
            node.outputs[0] = node.add_dummy(dst_c);
        }
        return;
    }
}

bool htp_kparams_solve(
    const struct ggml_hexagon_session * sess,
    htp_opnode & node
) {
    htp_opnode_collapse(sess, node);

    const ggml_tensor * src0 = node.inputs.empty() ? nullptr : node.inputs[0];
    const ggml_tensor * src1 = node.inputs.size() > 1 ? node.inputs[1] : nullptr;
    const ggml_tensor * dst  = node.dst();

    if (htp_op_is_unary(node.opcode)) {
        return htp_kparams_solve_unary_impl(sess, node.opcode, src0, src1, dst, node.kparams<htp_unary_kernel_params>());
    }
    if (htp_op_is_glu(node.opcode)) {
        return htp_kparams_solve_unary_mul_glu(sess, node.opcode, src0, src1, dst, node.kparams<htp_glu_kernel_params>());
    }
    if (node.opcode == HTP_OP_MUL_MAT) {
        if (!htp_matmul_check_types(src0, src1, dst)) {
            return false;
        }
        return htp_kparams_solve_matmul_impl(sess, src0, src1, dst, 0, 0, node.kparams<htp_mm_kernel_params>());
    }
    if (node.opcode == HTP_OP_MUL_MAT_ADD) {
        const ggml_tensor * src2 = node.inputs.size() > 2 ? node.inputs[2] : nullptr;
        const size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
        return htp_kparams_solve_matmul_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, node.kparams<htp_mm_kernel_params>());
    }
    if (node.opcode == HTP_OP_UPSCALE) {
        return htp_kparams_solve_upscale(sess, src0, dst, node.kparams<htp_upscale_kernel_params>());
    }

    return htp_kparams_solve(sess, node.node, node.kernel_params);
}

bool htp_kparams_solve(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    void * kparams
) {
    const htp_op_code opcode = htp_op_remap(op);
    if (htp_op_is_unary(opcode) || htp_op_is_glu(opcode) || opcode == HTP_OP_UPSCALE) {
        htp_opnode tmp(opcode, op);
        if (!htp_kparams_solve(sess, tmp)) {
            return false;
        }
        if (kparams) {
            memcpy(kparams, tmp.kernel_params, sizeof(tmp.kernel_params));
        }
        return true;
    }

    switch (opcode) {
        case HTP_OP_MUL_MAT:
        case HTP_OP_MUL_MAT_ID:
            return htp_kparams_invoke<htp_kparams_solve_matmul>(sess, op, kparams);

        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_ADD_ID:
        case HTP_OP_SUB:
        case HTP_OP_DIV:
            return htp_kparams_invoke<htp_kparams_solve_binary>(sess, op, kparams);

        case HTP_OP_FLASH_ATTN_EXT:
            return htp_kparams_invoke<htp_kparams_solve_flash_attn>(sess, op, kparams);

        case HTP_OP_GET_ROWS:
            return htp_kparams_invoke<htp_kparams_solve_get_rows>(sess, op, kparams);

        case HTP_OP_SET_ROWS:
            return htp_kparams_invoke<htp_kparams_solve_set_rows>(sess, op, kparams);

        case HTP_OP_ROPE:
            return htp_kparams_invoke<htp_kparams_solve_rope>(sess, op, kparams);

        case HTP_OP_SSM_CONV:
            return htp_kparams_invoke<htp_kparams_solve_ssm_conv>(sess, op, kparams);

        case HTP_OP_SOFTMAX:
            return htp_kparams_invoke<htp_kparams_solve_softmax>(sess, op, kparams);

        case HTP_OP_GATED_DELTA_NET:
            return htp_kparams_invoke<htp_kparams_solve_gated_delta_net>(sess, op, kparams);

        case HTP_OP_ARGSORT:
        case HTP_OP_TOP_K:
            return htp_kparams_invoke<htp_kparams_solve_sort>(sess, op, kparams);

        case HTP_OP_POOL_2D:
        case HTP_OP_POOL_1D:
            return htp_kparams_invoke<htp_kparams_solve_pool_2d>(sess, op, kparams);

        case HTP_OP_CONCAT:
            return htp_kparams_invoke<htp_kparams_solve_concat>(sess, op, kparams);

        case HTP_OP_CPY:
        case HTP_OP_CPY_FENCE:
            return htp_kparams_invoke<htp_kparams_solve_cpy>(sess, op, kparams);

        case HTP_OP_IM2COL:
            return htp_kparams_invoke<htp_kparams_solve_im2col>(sess, op, kparams);

        case HTP_OP_IM2COL_3D:
            return htp_kparams_invoke<htp_kparams_solve_im2col_3d>(sess, op, kparams);

        case HTP_OP_CONV_2D:
            return htp_kparams_invoke<htp_kparams_solve_conv2d>(sess, op, kparams);

        case HTP_OP_CONV_2D_DW:
            return htp_kparams_invoke<htp_kparams_solve_conv2d_dw>(sess, op, kparams);

        case HTP_OP_SUM_ROWS:
        case HTP_OP_SUM:
        case HTP_OP_ARGMAX:
            return htp_kparams_invoke<htp_kparams_solve_sum_rows>(sess, op, kparams);

        case HTP_OP_REPEAT:
            return htp_kparams_invoke<htp_kparams_solve_repeat>(sess, op, kparams);

        case HTP_OP_PAD:
            return htp_kparams_invoke<htp_kparams_solve_pad>(sess, op, kparams);

        case HTP_OP_CUMSUM:
            return htp_kparams_invoke<htp_kparams_solve_cumsum>(sess, op, kparams);

        case HTP_OP_FILL:
            return htp_kparams_invoke<htp_kparams_solve_fill>(sess, op, kparams);

        case HTP_OP_ROLL:
            return htp_kparams_invoke<htp_kparams_solve_roll>(sess, op, kparams);

        case HTP_OP_DIAG:
            return htp_kparams_invoke<htp_kparams_solve_diag>(sess, op, kparams);

        case HTP_OP_SOLVE_TRI:
            return htp_kparams_invoke<htp_kparams_solve_solve_tri>(sess, op, kparams);

        default:
            if (htp_op_is_unary(opcode)) {
                return htp_kparams_invoke<htp_kparams_solve_unary>(sess, op, kparams);
            }
            if (htp_op_is_glu(opcode)) {
                return htp_kparams_invoke<htp_kparams_solve_glu>(sess, op, kparams);
            }
            return false;
    }
}
