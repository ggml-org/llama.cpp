#ifndef HTP_ARGSORT_OPS_H
#define HTP_ARGSORT_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#include "hex-fastdiv.h"
#include "htp-vtcm.h"

struct htp_sort_kernel_params {
    uint32_t n_threads;
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
    size_t   off_phase1;
    size_t   off_merge_values;
    size_t   off_merge_indices;
    uint32_t chunk_elems;
    uint32_t n_chunks;
    uint32_t merge_elems;
    uint32_t n_threads;
    uint32_t n_slots;
};

static inline void htp_sort_vtcm_layout_build(
    struct htp_sort_vtcm_layout * L,
    uint32_t total_rows,
    uint32_t chunk_elems,
    uint32_t n_chunks,
    uint32_t merge_elems,
    uint32_t n_threads,
    uint32_t n_slots) {

    memset(L, 0, sizeof(*L));
    L->chunk_elems = chunk_elems;
    L->n_chunks    = n_chunks;
    L->merge_elems = merge_elems;
    L->n_threads   = n_threads;
    L->n_slots     = n_slots;

    const size_t values_size  = ((chunk_elems * sizeof(float)) + 127) & ~127;
    const size_t indices_size = ((chunk_elems * sizeof(int32_t)) + 127) & ~127;
    L->phase1_slot_size       = ((values_size + indices_size) + 255) & ~255;

    size_t off = 0;
    if (total_rows > 1) {
        VTCM_LAYOUT_ALLOC(off, off_phase1, L->phase1_slot_size * n_slots * n_threads);
    } else {
        VTCM_LAYOUT_ALLOC(off, off_phase1, L->phase1_slot_size * n_chunks * n_slots);
        const size_t merge_values_size  = ((merge_elems * sizeof(float)) + 127) & ~127;
        const size_t merge_indices_size = ((merge_elems * sizeof(int32_t)) + 127) & ~127;
        VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_merge_values,  merge_values_size,  n_chunks > 1);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_merge_indices, merge_indices_size, n_chunks > 1);
    }
    L->total_bytes = off;
}

static inline bool htp_sort_solve_layout(
    struct htp_sort_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t total_rows,
    uint32_t k,
    uint32_t n_threads_max,
    size_t vtcm_budget,
    bool is_top_k) {

    memset(layout, 0, sizeof(*layout));

    if (total_rows > 1) {
        uint32_t n_threads = total_rows < n_threads_max ? total_rows : n_threads_max;

        uint32_t n_vec = (ne00 + 31) / 32;
        uint32_t n_vec_pow2 = 1;
        while (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
        uint32_t ne00_padded = n_vec_pow2 * 32;

        size_t values_size  = ((ne00_padded * sizeof(float)) + 127) & ~127;
        size_t indices_size = ((ne00_padded * sizeof(int32_t)) + 127) & ~127;
        size_t spad_per_slot = ((values_size + indices_size) + 255) & ~255;

        uint32_t n_slots = 2;
        if (spad_per_slot * 2 > vtcm_budget) {
            n_slots = 1;
        }

        size_t spad_per_thread = spad_per_slot * n_slots;

        while (n_threads > 1 && (spad_per_thread * n_threads) > vtcm_budget) {
            n_threads--;
        }

        htp_sort_vtcm_layout_build(layout, total_rows, ne00_padded, 1, 0, n_threads, n_slots);
        return layout->total_bytes <= vtcm_budget;
    }

    uint32_t n_vec = (ne00 + 31) / 32;
    uint32_t n_vec_pow2 = 1;
    while (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    uint32_t n_chunks = 1;
    if (ne00 > 1024) {
        while (n_chunks * 2 <= n_threads_max && n_chunks * 2 <= n_vec_pow2) {
            n_chunks *= 2;
        }
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;
    uint32_t merge_elems = 0;

    if (n_chunks > 1) {
        if (is_top_k) {
            uint32_t local_k = k < chunk_elems ? k : chunk_elems;
            uint32_t total_candidates = n_chunks * local_k;
            uint32_t merge_n_vec = (total_candidates + 31) / 32;
            uint32_t merge_n_vec_pow2 = 1;
            while (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
            merge_elems = merge_n_vec_pow2 * 32;
        } else {
            merge_elems = n_vec_pow2 * 32;
        }
    }

    htp_sort_vtcm_layout_build(layout, 1, chunk_elems, n_chunks, merge_elems, n_chunks, 1);

    if (layout->total_bytes > vtcm_budget && n_chunks > 1) {
        n_chunks    = 1;
        chunk_elems = n_vec_pow2 * 32;
        merge_elems = 0;
        htp_sort_vtcm_layout_build(layout, 1, chunk_elems, n_chunks, merge_elems, 1, 1);
    }

    if (layout->total_bytes > vtcm_budget) {
        return false;
    }

    if (n_chunks == 1 && layout->phase1_slot_size * 2 <= vtcm_budget) {
        htp_sort_vtcm_layout_build(layout, 1, chunk_elems, 1, 0, 1, 2);
    }

    return true;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_sort_kernel_params) <= 128, "htp_sort_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ARGSORT_OPS_H
