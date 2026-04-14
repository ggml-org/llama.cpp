// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#ifndef HMX_UTILS_H
#define HMX_UTILS_H

#include <hexagon_types.h>
#include <stddef.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

// Initialise aligned 256-byte area with scale vector + zero padding.
static HMX_INLINE_ALWAYS void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
    HVX_Vector *pv = (HVX_Vector *)out_scales;
    *pv++ = v_scale;
    *pv   = Q6_V_vzero();
}

// --- VTCM sequential allocator (from htp-ops-lib/include/dsp/vtcm_mgr.h) ---

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

#endif // HMX_UTILS_H
