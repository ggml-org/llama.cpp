#ifndef HTP_COMMON_H
#define HTP_COMMON_H

#include "htp-opnode.h"
#include "htp-repack.h"

struct htp_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

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
extern int    opt_gdn_select;
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

static inline bool htp_tensor_is_fuseable(const struct ggml_tensor * t) {
    if (!t || !t->extra) return false;
    const auto * extra = (const struct htp_tensor_extra *) t->extra;
    return (extra->flags & HTP_TENSOR_FUSEABLE) != 0;
}

static inline bool htp_can_rt_repack(const struct ggml_tensor * t) {
    if (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        return false;
    }
    if (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        return false;
    }
    return opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

static inline bool htp_tensor_is_tiled(const struct ggml_tensor * t) {
    const auto * extra = (const struct htp_tensor_extra *) t->extra;
    if (extra && (extra->flags & HTP_TENSOR_TILED)) {
        return true;
    }
    return htp_can_rt_repack(t);
}

static inline bool htp_needs_repack(const struct ggml_tensor * t) {
    const auto * extra = (const struct htp_tensor_extra *) t->extra;
    if (extra && (extra->flags & HTP_TENSOR_TILED)) {
        return false;
    }
    return htp_can_rt_repack(t);
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

struct ggml_hexagon_session;

const char * htp_session_name(const struct ggml_hexagon_session * sess);
size_t       htp_session_vtcm_size(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_n_threads(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_n_hmx(const struct ggml_hexagon_session * sess);
uint32_t     htp_session_mdev_count(const struct ggml_hexagon_session * sess);

htp_op_code  htp_op_remap(const struct ggml_tensor * t);

#endif // HTP_COMMON_H
