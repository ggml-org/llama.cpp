#pragma once

#include "common.hpp"

// The legacy implementation uses SLM to implement sorting and top_k selection.
// SLM is limited to 128KB on Xe, which limits how much can be sorted to k<32.
// After a k=8, the radix selection becomes beneficial for most cases, because
// scan-merge has (block + 1) * k pairs of (value, index). Given normal sorting of nlog(n),
// radix-select becomes beneficial quite early. This sets it to 8 - however, the other parameters
// (columns and rows) may also be a driving factor.
// We select the legacy implementation for k below this constant because the overhead of radix select
// exceeds the benefit for very small problems
constexpr int SYCL_TOP_K_SCAN_MERGE_MAX_K = 8;

// Top-k of every row of src, k indices per row into dst_indices, in no particular order.
// Picks between the one-group-per-row and the split-row kernel from the shape and the device.
void ggml_sycl_top_k_radix(
    ggml_backend_sycl_context & ctx,
    const float *   src,
    int32_t *       dst_indices,
    const int64_t   ncols,
    const int64_t   nrows,
    const int       k,
    dpct::queue_ptr main_stream);

// Large-k top-k by radix select on an order-preserving unsigned key.
//
// The k-th largest key of a row is found by four most-significant-first passes over its
// 8-bit digits: histogram the digit over the candidate set, walk the buckets from the
// top, and recurse into the bucket where the running count reaches what is still
// needed. Everything strictly above that bucket is in the top-k. A final pass emits
// every column whose key beats the pivot, then exactly as many pivot-equal columns as
// are still missing, so duplicate keys yield exactly k distinct indices.
//
// SLM holds only the histogram, so unlike the scan-merge kernels the cost does not grow
// with k. One work-group owns a row and runs every pass, so a top-k is one launch and
// needs no pool scratch. The row is re-read once per pass rather than compacted, which
// keeps the candidate set implicit: (key & mask) == prefix.
//
// The output is the set of winning indices in no particular order, which is what the
// reference op provides (it swaps its first two outputs to say so) and what
// test-backend-ops compares.

static constexpr int SYCL_TOP_K_RADIX_BITS       = 8;
static constexpr int SYCL_TOP_K_RADIX_BUCKETS    = 1 << SYCL_TOP_K_RADIX_BITS;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
static constexpr int SYCL_TOP_K_RADIX_HIST_COPIES = 8;
static constexpr int SYCL_TOP_K_RADIX_HIST_SIZE   = SYCL_TOP_K_RADIX_BUCKETS * SYCL_TOP_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
static constexpr int SYCL_TOP_K_RADIX_SLM_WORDS   = SYCL_TOP_K_RADIX_HIST_SIZE + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
static inline uint32_t top_k_radix_key(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if (u == 0x80000000u) {
        u = 0u;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Where a top-k row's values come from. A plain row is a pointer; the QSA indexer fusion
// builds the value from the block score and the mask instead of reading a materialized row.
struct top_k_src_ptr {
    const float * src;

    float operator()(int col) const { return src[col]; }
};

struct top_k_rows_ptr {
    const float * src;
    int64_t       ncols;

    top_k_src_ptr row(int64_t r) const { return { src + r*ncols }; }
};

// value(c, t) = score[idx[c], t] + mask[c, t]: the QSA indexer chain, without the two
// full-size copies the graph would otherwise write.
template <typename mask_t> struct top_k_src_qsa {
    const char *    score;
    const int32_t * idx;
    const mask_t *  mask;
    size_t          nb_blk;

    float operator()(int col) const {
        const float s = *(const float *) (score + (int64_t) idx[col]*nb_blk);
        return s + (float) mask[col];
    }
};

template <typename mask_t> struct top_k_rows_qsa {
    const char *    score;
    const int32_t * idx;
    const char *    mask;
    size_t          nb_blk;
    size_t          nb_score_row;
    size_t          nb_mask_row;

    top_k_src_qsa<mask_t> row(int64_t r) const {
        return { score + r*nb_score_row, idx, (const mask_t *) (mask + r*nb_mask_row), nb_blk };
    }
};

// The selection step lives here because the fused topk-moe kernel reuses it: softmax and
// sigmoid are monotonic, so the routing top-k can be taken on the raw logits.
template <typename Src>
static inline void top_k_radix_select(
    const Src &     src,
    int32_t *       dst_idx,
    const int       ncols,
    const int       k,
    uint32_t *      slm,
    const sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    const int tid        = item_ct1.get_local_id(0);
    const int block_size = item_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_TOP_K_RADIX_HIST_SIZE;
    uint32_t * s_bucket = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 1;
    uint32_t * s_need   = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 2;
    uint32_t * s_cnt_gt = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 3;
    uint32_t * s_cnt_eq = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 4;

    if (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    const int copy = tid & (SYCL_TOP_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t mask   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    for (int shift = 32 - SYCL_TOP_K_RADIX_BITS; shift >= 0; shift -= SYCL_TOP_K_RADIX_BITS) {
        for (int i = tid; i < SYCL_TOP_K_RADIX_HIST_SIZE; i += block_size) {
            hist[i] = 0;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int col = tid; col < ncols; col += block_size) {
            const uint32_t key = top_k_radix_key(src(col));
            if ((key & mask) == prefix) {
                const uint32_t bucket = (key >> shift) & (SYCL_TOP_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_TOP_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        if (tid < SYCL_TOP_K_RADIX_BUCKETS) {
            const uint32_t * h = hist + (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid) * SYCL_TOP_K_RADIX_HIST_COPIES;
            for (int c = 0; c < SYCL_TOP_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        const uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());

        if (tid < SYCL_TOP_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid);
            *s_bucket = cnt;
            *s_need   = need - (incl - cnt);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        const uint32_t digit      = *s_digit;
        const uint32_t bucket_cnt = *s_bucket;
        need   = *s_need;
        prefix |= digit << shift;
        mask   |= (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1) << shift;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        if (bucket_cnt == need) {
            break;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    const uint32_t base_eq = (uint32_t) k - need;

    for (int col = tid; col < ncols; col += block_size) {
        const uint32_t kp = top_k_radix_key(src(col)) & mask;
        if (kp > prefix) {
            const uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = col;
        } else if (kp == prefix) {
            const uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            if (pos < need) {
                dst_idx[base_eq + pos] = col;
            }
        }
    }
}

static inline void top_k_radix_select_f32(
    const float *   src,
    int32_t *       dst_idx,
    const int       ncols,
    const int       k,
    uint32_t *      slm,
    const sycl::nd_item<1> & item_ct1
) {
    top_k_radix_select(top_k_src_ptr{ src }, dst_idx, ncols, k, slm, item_ct1);
}

// CONT(PERMUTE(score)) -> GET_ROWS -> PERMUTE -> CONT -> [f16 mask cast] -> ADD -> TOP_K:
// the QSA indexer chain, taken by the top-k straight off the score and the mask.
bool ggml_sycl_can_fuse_qsa_topk(const ggml_cgraph * cgraph, int node_idx);
int  ggml_sycl_qsa_topk_absorbs(const ggml_cgraph * cgraph, int node_idx);
int  ggml_sycl_fuse_qsa_topk(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);
