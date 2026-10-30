#ifndef HTP_TOP_K_MASK_OPS_H
#define HTP_TOP_K_MASK_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// HVX gather results land here (gathers write to VTCM only): 10 vectors per thread
#define HTP_TOP_K_MASK_SCRATCH_BYTES 1280

enum htp_top_k_mask_kernel {
    HTP_TOP_K_MASK_KERNEL_ROWS  = 0, // threads split rows, row inputs streamed per row
    HTP_TOP_K_MASK_KERNEL_COLS  = 1, // threads split (row, col tile) items, row inputs of all rows preloaded
    HTP_TOP_K_MASK_KERNEL_PICKS = 2, // one row with one mask row: threads split the picks, keep flags select mask or -inf
};

struct htp_top_k_mask_shape {
    uint32_t n_kv, elem_size, n_top, n_pick, n_blk, n_extra, total_rows;
    uint32_t has_score, has_blocks, mask_resident; // mask_resident: mask has exactly 1 row
};

struct htp_top_k_mask_vtcm_layout {
    size_t total_bytes;
    size_t off_blocks, off_mask, off_rowin, off_tiles, off_mtiles; // 0 when unused
    size_t off_scratch; // HTP_TOP_K_MASK_SCRATCH_BYTES per thread
    size_t rowin_bytes; // one slot of top_k + extra + score row, parts at htp_top_k_mask_rowin_off_*
    size_t tile_bytes;  // one output tile slot, 128B aligned
};

static inline uint64_t htp_top_k_mask_align128(uint64_t n) {
    return (n + 127) & ~(uint64_t) 127;
}

// Parts of one row input slot: top_k at 0, then extra, then score; shared by the host layout and the NPU
static inline uint64_t htp_top_k_mask_rowin_off_extra(uint32_t n_top) {
    return htp_top_k_mask_align128((uint64_t) n_top * 4);
}

static inline uint64_t htp_top_k_mask_rowin_off_score(uint32_t n_top, uint32_t n_extra) {
    return htp_top_k_mask_rowin_off_extra(n_top) + htp_top_k_mask_align128((uint64_t) n_extra * 4);
}

// returns false on arithmetic overflow; the caller compares total_bytes against the VTCM size
static inline bool htp_top_k_mask_vtcm_layout_build(struct htp_top_k_mask_vtcm_layout * L,
        const struct htp_top_k_mask_shape * S, uint32_t kernel_type, uint32_t n_threads, uint32_t tile_cols) {
    memset(L, 0, sizeof(*L));

    // PICKS streams the mask straight into its row tile
    const bool     picks        = kernel_type == HTP_TOP_K_MASK_KERNEL_PICKS;
    const uint64_t blocks_bytes = S->has_blocks ? htp_top_k_mask_align128((uint64_t) S->n_blk * S->n_pick * 4) : 0;
    const uint64_t mask_bytes   = S->mask_resident && !picks ? htp_top_k_mask_align128((uint64_t) S->n_kv * S->elem_size) : 0;
    const uint64_t rowin_bytes  = htp_top_k_mask_rowin_off_score(S->n_top, S->n_extra) +
                                  (S->has_score ? htp_top_k_mask_align128((uint64_t) S->n_pick * 4) : 0);
    const uint64_t tile_bytes   = htp_top_k_mask_align128((uint64_t) tile_cols * S->elem_size);

    // ROWS streams row inputs into 2 slots per thread, COLS preloads one slot per row. PICKS: one row, a row tile and a flag tile.
    const uint64_t n_rowin      = kernel_type == HTP_TOP_K_MASK_KERNEL_ROWS ? 2 * (uint64_t) n_threads : S->total_rows;
    const uint64_t rowin_total  = n_rowin * rowin_bytes;
    const uint64_t tiles_total  = picks ? 2 * tile_bytes : 2 * (uint64_t) n_threads * tile_bytes;
    const uint64_t mtiles_total = S->mask_resident ? 0 : tiles_total;
    const uint64_t scratch      = (uint64_t) n_threads * HTP_TOP_K_MASK_SCRATCH_BYTES;

    const uint64_t total = blocks_bytes + mask_bytes + rowin_total + tiles_total + mtiles_total + scratch;
    if (total > UINT32_MAX || tile_bytes > UINT32_MAX || rowin_bytes > UINT32_MAX) {
        return false;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_blocks, (size_t) blocks_bytes, blocks_bytes > 0);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_mask,   (size_t) mask_bytes,   mask_bytes > 0);
    VTCM_LAYOUT_ALLOC(off, off_rowin, (size_t) rowin_total);
    VTCM_LAYOUT_ALLOC(off, off_tiles, (size_t) tiles_total);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_mtiles, (size_t) mtiles_total, !S->mask_resident);
    VTCM_LAYOUT_ALLOC(off, off_scratch, (size_t) scratch);

    L->rowin_bytes = (size_t) rowin_bytes;
    L->tile_bytes  = (size_t) tile_bytes;
    L->total_bytes = off;
    return true;
}

struct htp_top_k_mask_kernel_params {
    uint32_t n_threads, kernel_type, tile_cols, n_col_tiles, vtcm_size;
    uint32_t rowin_bytes, tile_bytes;
    uint32_t off_blocks, off_mask, off_rowin, off_tiles, off_mtiles, off_scratch;
    struct fastdiv_values div_ne1, div_ne12, div_m1, div_m2, div_m3, div_n_col_tiles;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_top_k_mask_kernel_params) <= 128, "htp_top_k_mask_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_top_k_mask_kernel_params) <= 128, "htp_top_k_mask_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_TOP_K_MASK_OPS_H
