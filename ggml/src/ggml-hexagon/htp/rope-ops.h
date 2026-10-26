#ifndef HTP_ROPE_OPS_H
#define HTP_ROPE_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"

#define HTP_ROPE_SPAD_BLOCK  8
#define HTP_ROPE_SPAD_NSLOTS 4
#define HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_BLOCK * HTP_ROPE_SPAD_NSLOTS)

struct htp_rope_kernel_params {
    uint32_t n_threads;
    uint32_t vtcm_size;

    struct fastdiv_values div_ne2_ne1;
    struct fastdiv_values div_ne1;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#endif

struct htp_rope_vtcm_layout {
    size_t total_bytes;
    size_t off_spad;
    size_t off_freq_factors;
    size_t bytes_per_thread;
    size_t theta_cache_size_aligned;
    size_t src0_row_size_aligned;
    size_t freq_factors_size_aligned;
};

static inline void htp_rope_vtcm_layout_build(
    struct htp_rope_vtcm_layout * L,
    uint32_t ne00,
    uint32_t n_threads,
    uint32_t n_freq_factors
) {
    const size_t src0_row_size             = ne00 * sizeof(float);
    const size_t src0_row_size_aligned     = hex_round_up((uint32_t) src0_row_size, 128);
    const size_t theta_cache_size_aligned  = hex_round_up((uint32_t) src0_row_size, 256);
    const size_t freq_factors_size_aligned = hex_round_up(n_freq_factors * sizeof(float), 256);

    L->src0_row_size_aligned     = src0_row_size_aligned;
    L->theta_cache_size_aligned  = theta_cache_size_aligned;
    L->freq_factors_size_aligned = freq_factors_size_aligned;
    L->bytes_per_thread          = theta_cache_size_aligned + HTP_ROPE_SPAD_NROWS * src0_row_size_aligned;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_spad, L->bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_freq_factors, freq_factors_size_aligned, n_freq_factors > 0);
    L->total_bytes = off;
}

static inline uint8_t * rope_spad_slot(uint8_t * base, uint32_t slot, size_t row_size_aligned) {
    return base + (slot * HTP_ROPE_SPAD_BLOCK) * row_size_aligned;
}

#endif // HTP_ROPE_OPS_H
