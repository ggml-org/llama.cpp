#include "topk-radix.hpp"

#include "ggml-impl.h"

#include "binbcast.hpp"
#include "common.hpp"
#include "getrows.hpp"

#include <algorithm>

template <typename Rows>
static void top_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    const Rows rows,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    const int block_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
    GGML_ASSERT(block_size >= SYCL_TOP_K_RADIX_BUCKETS);

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_TOP_K_RADIX_SLM_WORDS), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);

                top_k_radix_select(
                    rows.row(row), dst_indices + (int64_t) row * k,
                    (int) ncols, k,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item_ct1);
            });
    });
}

// One work-group owns a whole row above, which leaves the device idle whenever a graph
// has fewer rows than it has cores -- the common case at batch size 1, where the
// sparse-attention indexer and the backend sampler both top-k a single row. The kernels
// below spread one row over several groups instead.
//
// A digit pass now needs the whole row's histogram before any group can pick the pivot,
// so the per-pass state moves to global memory and the passes become separate launches:
// a work-group barrier no longer spans the row. Each group still accumulates into SLM
// and contributes 256 global atomics at the end, so global traffic is per-group, not
// per-element. The last group to finish a pass (the one whose fetch_add returns G - 1)
// does the scan for the row and clears the histogram for the next pass, which keeps the
// launch count at one per digit rather than two.
//
// Running all four digits unconditionally costs nothing in correctness: once a bucket
// holds exactly the elements still needed, later digits only extend the prefix, and the
// count of columns above that longer prefix grows by exactly as much as `need` shrinks.
// The emit below therefore stays exact whatever pass the answer settled on.

static constexpr int SYCL_TOP_K_RADIX_ROW_DONE   = SYCL_TOP_K_RADIX_BUCKETS + 0;
static constexpr int SYCL_TOP_K_RADIX_ROW_PREFIX = SYCL_TOP_K_RADIX_BUCKETS + 1;
static constexpr int SYCL_TOP_K_RADIX_ROW_MASK   = SYCL_TOP_K_RADIX_BUCKETS + 2;
static constexpr int SYCL_TOP_K_RADIX_ROW_NEED   = SYCL_TOP_K_RADIX_BUCKETS + 3;
static constexpr int SYCL_TOP_K_RADIX_ROW_CNT_GT = SYCL_TOP_K_RADIX_BUCKETS + 4;
static constexpr int SYCL_TOP_K_RADIX_ROW_CNT_EQ = SYCL_TOP_K_RADIX_BUCKETS + 5;
static constexpr int SYCL_TOP_K_RADIX_ROW_WORDS  = SYCL_TOP_K_RADIX_BUCKETS + 6;

// How wide the split goes is a property of the device, not of the model: enough groups to
// cover the cores, and no more. Past that the extra groups add histogram traffic without
// adding bandwidth (measured on this device: 20 and 40 groups tie, 60 and 160 lose).
//
// nsm is max_compute_units / 16, i.e. it counts an Xe core as 16 EUs. That is a core's
// width on Xe-HPG, but an Xe2 core is 8 XVEs wide, so on Battlemage the field reads half
// the cores actually present (10 for a 20-core B60). The measured curve is flat from one
// group per core to two and only falls off at three, so a factor of two covers the device
// on Xe2 and lands in the flat region on Xe-HPG. It is the one number here that a correct
// core count would remove; it was tuned on Xe2 and has not been measured on Xe-HPG.
static constexpr int SYCL_TOP_K_RADIX_GROUPS_PER_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
static constexpr int SYCL_TOP_K_RADIX_MIN_SPLIT_COLS = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
static constexpr int SYCL_TOP_K_RADIX_MIN_PART_COLS  = 4096;

static int top_k_radix_split_groups(const int device, const int64_t ncols, const int64_t nrows) {
    const int64_t target = (int64_t) SYCL_TOP_K_RADIX_GROUPS_PER_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    if (ncols < SYCL_TOP_K_RADIX_MIN_SPLIT_COLS || nrows >= target) {
        return 1;
    }

    const int64_t by_rows = target / nrows;   // floor: never overshoot a row that is nearly covered
    const int64_t by_cols = ncols / SYCL_TOP_K_RADIX_MIN_PART_COLS;

    return (int) std::max<int64_t>(1, std::min(by_rows, by_cols));
}

using top_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

template <typename Src>
static void top_k_radix_split_pass_f32(
    const Src &     src,
    uint32_t *      state,
    const int       ncols,
    const int       k,
    const int       shift,
    const bool      first,
    const int       part,
    const int       nparts,
    uint32_t *      slm,
    const sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    const int tid        = item_ct1.get_local_id(0);
    const int block_size = item_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_last = slm + SYCL_TOP_K_RADIX_HIST_SIZE;
    uint32_t * s_row  = slm + SYCL_TOP_K_RADIX_HIST_SIZE + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    if (tid == 0) {
        s_row[0] = first ? 0u : state[SYCL_TOP_K_RADIX_ROW_PREFIX];
        s_row[1] = first ? 0u : state[SYCL_TOP_K_RADIX_ROW_MASK];
        s_row[2] = first ? (uint32_t) k : state[SYCL_TOP_K_RADIX_ROW_NEED];
    }

    for (int i = tid; i < SYCL_TOP_K_RADIX_HIST_SIZE; i += block_size) {
        hist[i] = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const uint32_t prefix = s_row[0];
    const uint32_t mask   = s_row[1];
    const uint32_t need   = s_row[2];

    const int copy  = tid & (SYCL_TOP_K_RADIX_HIST_COPIES - 1);
    const int chunk = (ncols + nparts - 1) / nparts;
    const int col0  = part * chunk;
    const int col1  = std::min(ncols, col0 + chunk);

    for (int col = col0 + tid; col < col1; col += block_size) {
        const uint32_t key = top_k_radix_key(src(col));
        if ((key & mask) == prefix) {
            const uint32_t bucket = (key >> shift) & (SYCL_TOP_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_TOP_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    for (int b = tid; b < SYCL_TOP_K_RADIX_BUCKETS; b += block_size) {
        uint32_t sum = 0;
        for (int c = 0; c < SYCL_TOP_K_RADIX_HIST_COPIES; c++) {
            sum += hist[b * SYCL_TOP_K_RADIX_HIST_COPIES + c];
        }
        if (sum) {
            top_k_radix_gatomic(state[b]).fetch_add(sum);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
    if (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> done(state[SYCL_TOP_K_RADIX_ROW_DONE]);
        *s_last = (done.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (*s_last == 0u) {
        return;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    if (tid < SYCL_TOP_K_RADIX_BUCKETS) {
        cnt = top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    const uint32_t incl = sycl::inclusive_scan_over_group(item_ct1.get_group(), cnt, sycl::plus<uint32_t>());

    if (tid < SYCL_TOP_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        const uint32_t digit = (uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1 - tid);
        top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_ROW_PREFIX]).store(prefix | (digit << shift));
        top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_ROW_MASK]).store(
            mask | ((uint32_t) (SYCL_TOP_K_RADIX_BUCKETS - 1) << shift));
        top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_ROW_NEED]).store(need - (incl - cnt));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    for (int b = tid; b < SYCL_TOP_K_RADIX_BUCKETS; b += block_size) {
        top_k_radix_gatomic(state[b]).store(0u);
    }
    if (tid == 0) {
        top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_ROW_DONE]).store(0u);
    }
}

template <typename Src>
static void top_k_radix_split_emit_f32(
    const Src &     src,
    int32_t *       dst_idx,
    uint32_t *      state,
    const int       ncols,
    const int       k,
    const int       part,
    const int       nparts,
    uint32_t *      slm,
    const sycl::nd_item<1> & item_ct1
) {
    using local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    const int tid        = item_ct1.get_local_id(0);
    const int block_size = item_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_base_gt = slm + 2;
    uint32_t * s_base_eq = slm + 3;

    uint32_t * s_row = slm + 4;   // prefix, mask, need

    if (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_row[0] = state[SYCL_TOP_K_RADIX_ROW_PREFIX];
        s_row[1] = state[SYCL_TOP_K_RADIX_ROW_MASK];
        s_row[2] = state[SYCL_TOP_K_RADIX_ROW_NEED];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const uint32_t prefix = s_row[0];
    const uint32_t mask   = s_row[1];
    const uint32_t need   = s_row[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    const uint32_t base_eq = (uint32_t) k - need;

    const int chunk = (ncols + nparts - 1) / nparts;
    const int col0  = part * chunk;
    const int col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    for (int col = col0 + tid; col < col1; col += block_size) {
        const uint32_t kp = top_k_radix_key(src(col)) & mask;
        if (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } else if (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        const uint32_t n_gt = *s_gt;
        const uint32_t n_eq = *s_eq;
        *s_base_gt = n_gt ? top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_ROW_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_base_eq = n_eq ? top_k_radix_gatomic(state[SYCL_TOP_K_RADIX_ROW_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const uint32_t base_gt_g = *s_base_gt;
    const uint32_t base_eq_g = *s_base_eq;

    for (int col = col0 + tid; col < col1; col += block_size) {
        const uint32_t kp = top_k_radix_key(src(col)) & mask;
        if (kp > prefix) {
            dst_idx[base_gt_g + local_atomic(*s_gt).fetch_add(1u)] = col;
        } else if (kp == prefix) {
            const uint32_t pos = base_eq_g + local_atomic(*s_eq).fetch_add(1u);
            if (pos < need) {
                dst_idx[base_eq + pos] = col;
            }
        }
    }
}

template <typename Rows>
static void top_k_radix_split_f32_sycl(
    ggml_backend_sycl_context & ctx,
    const Rows rows,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    const int nparts,
    dpct::queue_ptr main_stream
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    const int block_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
    GGML_ASSERT(block_size >= SYCL_TOP_K_RADIX_BUCKETS);

    const size_t state_words = (size_t) nrows * SYCL_TOP_K_RADIX_ROW_WORDS;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_words);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_stream->memset(state, 0, state_words * sizeof(uint32_t))));

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows * nparts);

    bool first = true;
    for (int shift = 32 - SYCL_TOP_K_RADIX_BITS; shift >= 0; shift -= SYCL_TOP_K_RADIX_BITS) {
        const bool is_first = first;
        first = false;
        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_TOP_K_RADIX_HIST_SIZE + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int g    = item_ct1.get_group(0);
                    const int row  = g / nparts;
                    const int part = g % nparts;

                    top_k_radix_split_pass_f32(
                        rows.row(row),
                        state + (int64_t) row * SYCL_TOP_K_RADIX_ROW_WORDS,
                        (int) ncols, k, shift, is_first, part, nparts,
                        slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item_ct1);
                });
        });
    }

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(8), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int g    = item_ct1.get_group(0);
                const int row  = g / nparts;
                const int part = g % nparts;

                top_k_radix_split_emit_f32(
                    rows.row(row),
                    dst_indices + (int64_t) row * k,
                    state + (int64_t) row * SYCL_TOP_K_RADIX_ROW_WORDS,
                    (int) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    item_ct1);
            });
    });
}

template <typename Rows>
static void top_k_radix_rows(
    ggml_backend_sycl_context & ctx,
    const Rows      rows,
    int32_t *       dst_indices,
    const int64_t   ncols,
    const int64_t   nrows,
    const int       k,
    dpct::queue_ptr main_stream
) {
    const int nparts = top_k_radix_split_groups(ctx.device, ncols, nrows);
    if (nparts > 1) {
        top_k_radix_split_f32_sycl(ctx, rows, dst_indices, ncols, nrows, k, nparts, main_stream);
    } else {
        top_k_radix_f32_sycl(ctx, rows, dst_indices, ncols, nrows, k, main_stream);
    }
}

void ggml_sycl_top_k_radix(
    ggml_backend_sycl_context & ctx,
    const float *   src,
    int32_t *       dst_indices,
    const int64_t   ncols,
    const int64_t   nrows,
    const int       k,
    dpct::queue_ptr main_stream
) {
    top_k_radix_rows(ctx, top_k_rows_ptr{ src, ncols }, dst_indices, ncols, nrows, k, main_stream);
}

// The QSA indexer ends in TOP_K over score[cell_blk[c], t] + mask[c, t]. The graph builds that
// value in full: the gather chain writes one [n_kv, n_tps] copy and the ADD writes another, both
// read once. The top-k re-reads its row on every digit pass anyway, so the value is cheaper to
// rebuild from the score and the mask than to materialize: for a fixed token the score is read
// along its own ne0 and the mask along its ne0, so both stay contiguous in the lane direction.
//
// The whole chain is CONT(PERMUTE(score)) -> GET_ROWS -> PERMUTE -> CONT -> [f16 mask cast] ->
// ADD -> TOP_K. The fused kernel reads the original inputs and writes the TOP_K result.

struct qsa_topk_chain {
    int i_cont_in;
    int i_rows;
    int i_cont_out;
    int i_cast;   // -1 when the mask is already f32
    int i_add;
    int i_topk;
};

static bool ggml_sycl_qsa_topk_shape(const ggml_cgraph * cgraph, int i, qsa_topk_chain * out) {
    if (!ggml_sycl_qsa_gather_shape(cgraph, i)) {
        return false;
    }

    ggml_tensor * cont_out = cgraph->nodes[i + 3];
    if (ggml_node_get_use_count(cgraph, i + 3) != 1) {
        return false;
    }

    // the mask reaches the ADD either through an f16 cast chain or through a single reshape
    int i_cast = -1;
    int i_add  = i + 4;
    int cs     = 0;
    if (i + 4 < cgraph->n_nodes && ggml_sycl_cast_add_shape(cgraph, i + 4, &cs)) {
        i_cast = i + 4;
        i_add  = i + 4 + cs - 1;
    } else if (i + 4 < cgraph->n_nodes && cgraph->nodes[i + 4]->op == GGML_OP_RESHAPE) {
        if (ggml_node_get_use_count(cgraph, i + 4) != 1) {
            return false;
        }
        i_add = i + 5;
    }
    if (i_add + 1 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * add  = cgraph->nodes[i_add];
    ggml_tensor * topk = cgraph->nodes[i_add + 1];
    if (add->op != GGML_OP_ADD || topk->op != GGML_OP_TOP_K) {
        return false;
    }
    for (const ggml_tensor * n : { add, topk }) {
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->view_src) {
            return false;
        }
    }
    if (add->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_is_empty(add) || ggml_is_empty(topk)) {
        return false;
    }

    // only the gathered scores may be the accumulator; src1 is the additive mask and must
    // not alias them, or the kernel would add the scores to themselves
    const ggml_tensor * rhs = add->src[1];
    if (add->src[0] != cont_out || !rhs || rhs == cont_out || rhs->view_src == cont_out) {
        return false;
    }
    if (i_add > i + 4 && rhs != cgraph->nodes[i_add - 1]) {
        return false;
    }
    if (add->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(add, cont_out) || !ggml_are_same_shape(rhs, add)) {
        return false;
    }
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(rhs)) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, i_add) != 1) {
        return false;
    }

    // the f16 the cast would promote is what the kernel reads, so it must share ne0 with the ADD
    if (i_cast >= 0 && cgraph->nodes[i_cast]->src[0]->ne[0] != add->ne[0]) {
        return false;
    }

    const int k = (int) topk->ne[0];
    if (topk->type != GGML_TYPE_I32 || topk->src[0] != add || !ggml_is_contiguous(topk)) {
        return false;
    }
    if (topk->ne[1] != add->ne[1] || topk->ne[2] != add->ne[2] || topk->ne[3] != add->ne[3]) {
        return false;
    }
    // below this k the op picks the scan-merge kernel, which this path does not implement
    if (k <= SYCL_TOP_K_SCAN_MERGE_MAX_K || k > add->ne[0] || add->ne[0] > INT32_MAX) {
        return false;
    }

    if (out) {
        *out = { i, i + 1, i + 3, i_cast, i_add, i_add + 1 };
    }
    return true;
}

bool ggml_sycl_can_fuse_qsa_topk(const ggml_cgraph * cgraph, int i) {
    return g_ggml_sycl_enable_fusion && ggml_sycl_qsa_topk_shape(cgraph, i, nullptr);
}

// Runs the chain matched by ggml_sycl_can_fuse_qsa_topk(); returns the extra nodes consumed.
int ggml_sycl_fuse_qsa_topk(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    qsa_topk_chain c;
    if (!g_ggml_sycl_enable_fusion || !ggml_sycl_qsa_topk_shape(cgraph, i, &c)) {
        return 0;
    }

    const ggml_tensor * src  = cgraph->nodes[c.i_cont_in]->src[0];
    const ggml_tensor * idx  = cgraph->nodes[c.i_rows]->src[1];
    const ggml_tensor * add  = cgraph->nodes[c.i_add];
    ggml_tensor *       topk = cgraph->nodes[c.i_topk];

    const int64_t ncols = add->ne[0];
    const int64_t nrows = ggml_nrows(add);
    const int     k     = (int) topk->ne[0];

    const ggml_tensor * mask = c.i_cast >= 0 ? cgraph->nodes[c.i_cast]->src[0] : add->src[1];

    const char *    score_dd = (const char *) src->data;
    const int32_t * idx_dd   = (const int32_t *) idx->data;
    const char *    mask_dd  = (const char *) mask->data;
    int32_t *       dst_dd   = (int32_t *) topk->data;

    GGML_ASSERT(score_dd && idx_dd && mask_dd && dst_dd);

    const size_t nb_mask_row = (size_t) ncols * ggml_type_size(mask->type);

    if (mask->type == GGML_TYPE_F16) {
        top_k_radix_rows(ctx, top_k_rows_qsa<sycl::half>{ score_dd, idx_dd, mask_dd, src->nb[1], src->nb[0], nb_mask_row },
                         dst_dd, ncols, nrows, k, ctx.stream());
    } else {
        GGML_ASSERT(mask->type == GGML_TYPE_F32);
        top_k_radix_rows(ctx, top_k_rows_qsa<float>{ score_dd, idx_dd, mask_dd, src->nb[1], src->nb[0], nb_mask_row },
                         dst_dd, ncols, nrows, k, ctx.stream());
    }

    return c.i_topk - i;
}
