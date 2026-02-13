// Precomputed exp2 lookup table for safe-softmax in flash attention.
// Ported from htp-ops-lib/src/dsp/ops/precompute_table.c.
//
// Changes from the original:
//   - vtcm_manager dependency removed.  The caller (htp_iface_start in
//     main.c) passes a VTCM address directly.
//   - Include paths adapted for the merged htp/ directory layout.
//   - Scalar inner loop replaced with HVX vector index generation.
//   - Single table copy (read-only VTCM has no bank conflicts).

#include <HAP_farf.h>
#include <HAP_perf.h>

#include "hmx-hvx-convert.h"
#include "hmx-hvx-math.h"

// Compute the safe-softmax exp2 table and write it into the supplied VTCM
// buffer.  The table contains 32 768 half-float values (64 KB).
static void precompute_safe_softmax_exp2_table(uint8_t *table) {
    const int n               = 32768; // fp16 elements
    const int n_elems_per_vec = HMX_VLEN / sizeof(__fp16); // 64
    const int n_vecs          = n / n_elems_per_vec;       // 512

    HVX_Vector *pv_table = (HVX_Vector *)table;

    int64_t t0 = HAP_perf_get_qtimer_count();

    // Build an initial index vector [0, 1, 2, ..., 63] as uint16.
    _Alignas(128) uint16_t iota_buf[HMX_VLEN / sizeof(uint16_t)];
    for (int j = 0; j < n_elems_per_vec; ++j) {
        iota_buf[j] = (uint16_t)j;
    }
    HVX_Vector v_indices  = hmx_vmem(iota_buf);
    HVX_Vector v_neg_mask = Q6_Vh_vsplat_R(0x8000);
    HVX_Vector v_step     = Q6_Vh_vsplat_R(n_elems_per_vec);

    for (int i = 0; i < n_vecs; ++i) {
        // v_neg_idx = indices | 0x8000  (negative fp16 bit-patterns as uint16)
        HVX_Vector v_neg_idx = Q6_V_vor_VV(v_indices, v_neg_mask);

        // Promote to single-float for higher precision, then convert back.
        HVX_VectorPair vp    = hmx_hvx_vqf16_to_wsf(v_neg_idx);
        HVX_Vector     v0_sf = hmx_hvx_exp2_vsf(Q6_V_lo_W(vp));
        HVX_Vector     v1_sf = hmx_hvx_exp2_vsf(Q6_V_hi_W(vp));
        pv_table[i]          = hmx_hvx_wsf_to_vhf(v1_sf, v0_sf);

        // Advance indices by n_elems_per_vec (64).
        v_indices = Q6_Vh_vadd_VhVh(v_indices, v_step);
    }

    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
    FARF(ALWAYS, "%s: precompute table took %lld us", __func__, elapsed_us);
}

void init_precomputed_tables(uint8_t *vtcm_table) {
    precompute_safe_softmax_exp2_table(vtcm_table);
}
