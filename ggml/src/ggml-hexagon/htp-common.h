#ifndef HTP_COMMON_H
#define HTP_COMMON_H

#include "htp-opnode.h"
#include "htp-repack.h"

extern int    opt_arch;
extern size_t opt_ndev;
extern size_t opt_nhvx;
extern int    opt_nhmx;
extern size_t opt_vmem;
extern int    opt_etm;
extern int    opt_verbose;
extern int    opt_profile;
extern bool   opt_hostbuf;
extern bool   opt_dma64;
extern int    opt_power_mode;

extern size_t opt_mbuf_dyn;
extern size_t opt_mbuf_static;
extern size_t opt_mbuf_total;

extern int    opt_mm_select;
extern int    opt_fa_select;
extern int    opt_fa_split;
extern int    opt_li_select;
extern int    opt_gdn_select;
extern int    opt_ssm_select;
extern int    opt_ar_select;
extern int    opt_ar_scatter;

extern int    opt_opbatch;
extern int    opt_opqueue;
extern int    opt_optrace;
extern int    opt_oppoll;
extern int    opt_opfusion;
extern int    opt_rt_repack;

#ifndef HEX_VERBOSE
#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)
#endif

static inline uint32_t ggml_hexagon_tensor_get_flags(const struct ggml_tensor * t) {
    return (uint32_t) (uintptr_t) t->extra;
}

static inline void ggml_hexagon_tensor_set_flags(struct ggml_tensor * t, uint32_t flags) {
    t->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(t) | flags);
}

static inline void ggml_hexagon_tensor_clear_flags(struct ggml_tensor * t, uint32_t flags) {
    t->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(t) & ~flags);
}

static inline bool ggml_hexagon_tensor_is_fuseable(const struct ggml_tensor * t) {
    return (ggml_hexagon_tensor_get_flags(t) & HTP_TENSOR_FUSEABLE) != 0;
}

static inline bool ggml_hexagon_tensor_is_tiled(const struct ggml_tensor * t) {
    return (ggml_hexagon_tensor_get_flags(t) & HTP_TENSOR_TILED) != 0;
}

static inline bool ggml_hexagon_tensor_can_tile(const struct ggml_tensor * t) {
    if (opt_rt_repack == 0) {
        return false;
    }
    if (t->buffer) {
        if (ggml_backend_buft_is_host(t->buffer->buft) ||
            ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            return false;
        }
    }
    return htp_is_tiled_weight_tensor(t);
}

static inline bool ggml_hexagon_tensor_will_tile(const struct ggml_tensor * t) {
    return ggml_hexagon_tensor_is_tiled(t) || ggml_hexagon_tensor_can_tile(t);
}

static inline bool htp_tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b1 = b0 + ggml_nbytes(b);

    return a0 < b1 && b0 < a1;
}

static inline bool htp_can_row_partition(const struct ggml_tensor * t) {
    if (t->ne[1] > 1 && (t->nb[1] & 127) != 0) return false;
    if (t->ne[2] > 1 && (t->nb[2] & 127) != 0) return false;
    if (t->ne[3] > 1 && (t->nb[3] & 127) != 0) return false;
    return true;
}

static inline uint32_t htp_mdev_rows_per_chunk(const struct ggml_tensor * t) {
    if ((t->data && ((uintptr_t) t->data & 127) != 0) || (t->view_offs & 127) != 0) return 0;
    const size_t elem_size = ggml_type_size(t->type);
    if (t->ne[0] != 1 && t->nb[0] != elem_size) return 0;
    if (ggml_is_permuted(t)) return 0;
    if (t->ne[2] > 1 && (t->nb[2] & 127) != 0) return 0;
    if (t->ne[3] > 1 && (t->nb[3] & 127) != 0) return 0;
    if (t->ne[1] > 1 && (t->nb[1] & 127) == 0) return 1;
    const size_t row_size = (size_t) t->ne[0] * elem_size;
    if (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * (size_t) t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * (size_t) t->ne[2])) {
        return (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32((uint32_t) row_size, HEX_L2_LINE_SIZE)) : 1;
    }
    return 0;
}

struct ggml_hexagon_session;

const char * htp_session_name(const struct ggml_hexagon_session * sess);
size_t       htp_session_vtcm_size(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_n_hvx(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_n_hmx(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_mdev_count(const struct ggml_hexagon_session * sess);

htp_op_code  htp_op_remap(const struct ggml_tensor * t);

#endif // HTP_COMMON_H
