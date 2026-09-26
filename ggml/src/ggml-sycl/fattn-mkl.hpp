#ifndef GGML_SYCL_FATTN_MKL_HPP
#define GGML_SYCL_FATTN_MKL_HPP

#include "ggml.h"

#include <cstdint>

static inline int64_t ggml_sycl_fattn_mkl_batch_offset(int64_t ne3, int64_t nb3, int ib) {
    return ne3 == 1 ? 0 : (int64_t) ib * nb3;
}

static inline bool ggml_sycl_fattn_mkl_can_reuse_k_for_v(
        const ggml_tensor * k,
        const ggml_tensor * v) {
    if (k == nullptr || v == nullptr ||
        k->type != v->type ||
        k->data != v->data ||
        k->view_offs != 0 ||
        v->view_offs != 0) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (k->ne[i] != v->ne[i] || k->nb[i] != v->nb[i]) {
            return false;
        }
    }

    return true;
}

#endif  // GGML_SYCL_FATTN_MKL_HPP
