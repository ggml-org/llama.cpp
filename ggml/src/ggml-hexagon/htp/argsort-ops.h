#ifndef HTP_ARGSORT_OPS_H
#define HTP_ARGSORT_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#include "hex-fastdiv.h"
#include "htp-vtcm.h"

struct htp_sort_kernel_params {
    uint32_t n_hvx;
    uint32_t order;             // GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
    uint32_t is_top_k;          // 1 if TOP_K, 0 if ARGSORT
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t vtcm_size;
    uint32_t merge_elems;
    uint32_t n_slots;
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne01_ne02;
};

struct htp_sort_vtcm_layout {
    size_t   total_bytes;
    size_t   phase1_slot_size;
    size_t   row_values_size;
    size_t   row_indices_size;
    size_t   off_cand;
    size_t   off_phase1;
    size_t   off_merge_values;
    size_t   off_merge_indices;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_hvx;
    uint32_t n_slots;
    uint32_t batch_rows;
};

static inline void htp_sort_vtcm_layout_build(
    struct htp_sort_vtcm_layout * L,
    uint32_t ne00,
    uint32_t total_rows,
    uint32_t chunk_elems,
    uint32_t n_chunks,
    uint32_t merge_elems,
    uint32_t n_hvx,
    uint32_t n_slots,
    uint32_t k,
    bool is_top_k) {

    memset(L, 0, sizeof(*L));
    L->chunk_elems = chunk_elems;
    L->n_chunks    = n_chunks;
    L->merge_elems = merge_elems;
    L->n_hvx       = n_hvx;
    L->n_slots     = n_slots;

    if (is_top_k && k >= ne00) {
        const size_t idx_size = ((k * sizeof(int32_t)) + 127u) & ~(size_t) 127u;
        uint32_t batch_rows = 1u;
        if ((idx_size << 4) <= 65536u) {
            batch_rows = 16u;
        } else if ((idx_size << 2) <= 65536u) {
            batch_rows = 4u;
        }
        if (batch_rows > total_rows && total_rows > 0u) {
            batch_rows = total_rows;
        }
        L->row_indices_size = idx_size;
        L->batch_rows       = batch_rows;
        L->phase1_slot_size = ((idx_size * batch_rows) + 255u) & ~(size_t) 255u;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_phase1, L->phase1_slot_size);
        L->total_bytes = off;
        return;
    }

    uint32_t indices_elems = chunk_elems;
    if (is_top_k) {
        if (total_rows == 1u && n_chunks > 1u) {
            indices_elems = 0u;
        } else if (k < chunk_elems) {
            indices_elems = k;
        }
    }

    uint32_t batch_rows = 1u;
    if (total_rows > 1u && chunk_elems <= 256u) {
        if (total_rows > 7u * n_hvx) {
            batch_rows = 8u;
        } else if (total_rows > 3u * n_hvx) {
            batch_rows = 4u;
        }
    }

    const size_t row_values_size  = ((chunk_elems * sizeof(float)) + 127u) & ~(size_t) 127u;
    const size_t row_indices_size = ((indices_elems * sizeof(int32_t)) + 127u) & ~(size_t) 127u;
    const size_t off_cand         = (row_values_size + row_indices_size) * batch_rows;
    const size_t cand_size        = (is_top_k && chunk_elems > 256u) ? 512u : 0u;

    L->row_values_size  = row_values_size;
    L->row_indices_size = row_indices_size;
    L->off_cand         = off_cand;
    L->batch_rows       = batch_rows;
    L->phase1_slot_size = ((off_cand + cand_size) + 255u) & ~(size_t) 255u;

    size_t off = 0;
    if (total_rows > 1u) {
        VTCM_LAYOUT_ALLOC(off, off_phase1, L->phase1_slot_size * n_slots * n_hvx);
    } else {
        VTCM_LAYOUT_ALLOC(off, off_phase1, L->phase1_slot_size * n_chunks * n_slots);
        const size_t merge_values_size  = ((merge_elems * sizeof(float)) + 127u) & ~(size_t) 127u;
        const size_t merge_indices_size = ((merge_elems * sizeof(int32_t)) + 127u) & ~(size_t) 127u;
        VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_merge_values,  merge_values_size,  n_chunks > 1u);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_merge_indices, merge_indices_size, n_chunks > 1u);
    }
    L->total_bytes = off;
}

static inline bool htp_sort_solve_layout(
    struct htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t total_rows,
    uint32_t k,
    uint32_t n_hvx_max,
    size_t vtcm_budget,
    bool is_top_k) {

    memset(layout, 0, sizeof(*layout));

    if (is_top_k && k >= ne00) {
        htp_sort_vtcm_layout_build(layout, ne00, total_rows, ne00, 1, 0, 1, 1, k, true);
        if (layout->total_bytes > vtcm_budget) {
            return false;
        }
        uint32_t n_hvx = 1u;
        if (total_rows > 1u && (size_t) total_rows * k * sizeof(int32_t) > 16384u) {
            const uint32_t b = layout->batch_rows > 0u ? layout->batch_rows : 1u;
            const uint32_t max_w = (total_rows + b - 1u) / b;
            n_hvx = max_w < n_hvx_max ? max_w : n_hvx_max;
        }
        htp_sort_vtcm_layout_build(layout, ne00, total_rows, ne00, 1, 0, n_hvx, 1, k, true);
        return layout->total_bytes <= vtcm_budget;
    }

    if (total_rows > 1u) {
        uint32_t n_hvx = total_rows < n_hvx_max ? total_rows : n_hvx_max;
        if (ne00 <= 128u) {
            const uint32_t cap = (total_rows >= 4u) ? ((total_rows + 3u) >> 2) : 1u;
            if (n_hvx > cap) {
                n_hvx = cap;
            }
        }

        uint32_t ne00_padded = 0u;
        if (is_top_k && ne00 > 256u) {
            ne00_padded = (ne00 + 31u) & ~31u;
        } else {
            const uint32_t n_vec = (ne00 + 31u) >> 5;
            uint32_t n_vec_pow2 = 1u;
            while (n_vec_pow2 < n_vec) {
                n_vec_pow2 <<= 1;
            }
            ne00_padded = n_vec_pow2 * 32u;
        }

        uint32_t n_slots = 2u;
        htp_sort_vtcm_layout_build(layout, ne00, total_rows, ne00_padded, 1, 0, 1, n_slots, k, is_top_k);
        if (layout->total_bytes > vtcm_budget) {
            n_slots = 1u;
        }

        htp_sort_vtcm_layout_build(layout, ne00, total_rows, ne00_padded, 1, 0, n_hvx, n_slots, k, is_top_k);
        while (n_hvx > 1u && layout->total_bytes > vtcm_budget) {
            n_hvx--;
            htp_sort_vtcm_layout_build(layout, ne00, total_rows, ne00_padded, 1, 0, n_hvx, n_slots, k, is_top_k);
        }
        if (layout->total_bytes > vtcm_budget && n_slots > 1u) {
            n_slots = 1u;
            htp_sort_vtcm_layout_build(layout, ne00, total_rows, ne00_padded, 1, 0, n_hvx, n_slots, k, is_top_k);
        }
        return layout->total_bytes <= vtcm_budget;
    }

    const uint32_t n_vec = (ne00 + 31u) >> 5;
    uint32_t n_vec_pow2 = 1u;
    while (n_vec_pow2 < n_vec) {
        n_vec_pow2 <<= 1;
    }

    if (is_top_k) {
        uint32_t n_chunks = 1u;
        if (ne00 > 2048u) {
            while (n_chunks * 2u <= n_hvx_max && (n_chunks * 2u) * 32u <= n_vec) {
                n_chunks *= 2u;
            }
        }

        while (n_chunks > 1u) {
            const uint32_t chunk_n_vec = (n_vec + n_chunks - 1u) / n_chunks;
            const uint32_t chunk_elems = chunk_n_vec * 32u;
            const uint32_t local_k     = k < chunk_elems ? k : chunk_elems;
            const uint32_t merge_elems = ((n_chunks * local_k) + 31u) & ~31u;

            htp_sort_vtcm_layout_build(layout, ne00, 1, chunk_elems, n_chunks, merge_elems, n_chunks, 1, k, true);
            if (layout->total_bytes <= vtcm_budget) {
                return true;
            }
            n_chunks >>= 1;
        }

        const uint32_t chunk_elems = (ne00 > 256u) ? (n_vec * 32u) : (n_vec_pow2 * 32u);
        htp_sort_vtcm_layout_build(layout, ne00, 1, chunk_elems, 1, 0, 1, 1, k, true);
        return layout->total_bytes <= vtcm_budget;
    }

    uint32_t n_chunks = 1u;
    if (ne00 > 1024u) {
        while (n_chunks * 2u <= n_hvx_max && n_chunks * 2u <= n_vec_pow2) {
            n_chunks *= 2u;
        }
    }

    while (n_chunks > 1u) {
        const uint32_t chunk_elems = (n_vec_pow2 / n_chunks) * 32u;
        const uint32_t merge_elems = n_vec_pow2 * 32u;
        htp_sort_vtcm_layout_build(layout, ne00, 1, chunk_elems, n_chunks, merge_elems, n_chunks, 1, k, false);
        if (layout->total_bytes <= vtcm_budget) {
            return true;
        }
        n_chunks >>= 1;
    }

    htp_sort_vtcm_layout_build(layout, ne00, 1, n_vec_pow2 * 32u, 1, 0, 1, 1, k, false);
    return layout->total_bytes <= vtcm_budget;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
