// tests that the persistent expert pool (ggml_backend_sched_register_expert_pool) serves
// GGML_OP_MUL_MAT_ID results identical to the regular host weight copy path
//
// matrix: quant type (Q2_K/Q3_K/Q4_K/Q6_K/Q8_0/MXFP4) x n_tokens (1/3/8/9) x two pools
// sharing one routing (like fused gate_up + down MoE layers) x cold/evict/hit/graph-rebuild
// rounds. on the CUDA mul_mat_id dispatch the MMVQ batch limit is MMVQ_MAX_BATCH_SIZE = 8
// (and on Ada/Blackwell it applies to every quant type), so nt <= 8 covers the MMVQ path
// at its full batch and nt = 9 is what actually reaches the MMQ ids path (Blackwell: the
// native FP4 kernels) - the path a real prefill takes. the hit/miss telemetry is checked
// after every round against a reference LRU model: each used expert must count exactly
// once per ubatch.
// the expert id backing store lives on the accelerator, like the router output of a real
// model; the reference and pooled MUL_MAT_ID run in the same graph and must match
// bit-exact.
//
// needs an accelerator backend (Metal/CUDA/...); exits with success when there is none

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>

// per-case geometry (n_in must be a multiple of the largest quant block size)
// n_tokens goes up to 9: n_tokens > MMVQ_MAX_BATCH_SIZE (8) is what forces the CUDA MMQ
// mul_mat_id path (decode batches use the mmvq path; on Ada/Blackwell the mmvq batch
// limit applies to every quant type, so nt = 8 alone would never reach MMQ)
static const int n_in     = 512;
static const int n_out    = 64;
static const int n_expert = 32;
static const int n_slots  = 16; // >= the distinct experts any round routes at once
static const int n_used   = 2;
static const int n_tokens_max = 9;

struct tensors {
    ggml_tensor * x = nullptr;      // [n_in, n_tokens_max] F32 host input

    ggml_tensor * w_gu  = nullptr;  // [n_in, n_out, n_expert] host weights
    ggml_tensor * w_dn  = nullptr;  // [n_in, n_out, n_expert] host weights
    ggml_tensor * pool_gu = nullptr;
    ggml_tensor * pool_dn = nullptr;
    ggml_tensor * table_gu = nullptr;
    ggml_tensor * table_dn = nullptr;

    ggml_tensor * ids_wide = nullptr; // [n_expert, n_tokens_max] I32 host tensor
};

// binds a freshly created tensor to a host buffer and fills it with data
template <typename T>
static ggml_tensor * make_host_tensor(
        ggml_context * ctx, ggml_backend_buffer_type_t buft,
        ggml_type type, const int64_t * ne, int ndims, const std::vector<T> & data,
        bool is_input, const char * name) {
    ggml_tensor * t = ndims == 1 ? ggml_new_tensor_1d(ctx, type, ne[0])
                   : ndims == 2 ? ggml_new_tensor_2d(ctx, type, ne[0], ne[1])
                                : ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t));
    t->data = ggml_backend_buffer_get_base(buf);
    t->buffer = buf;
    if constexpr (std::is_same_v<T, uint8_t>) {
        memcpy(t->data, data.data(), data.size());
    } else {
        memcpy(t->data, data.data(), ggml_nbytes(t));
    }
    if (is_input) {
        ggml_set_input(t);
    }
    ggml_format_name(t, "%s", name);
    return t;
}

// quantized weight tensor filled from f32 data, quantizing row by row
static ggml_tensor * make_quant_host_tensor(
        ggml_context * ctx, ggml_backend_buffer_type_t buft, ggml_type type,
        const int64_t * ne, const std::vector<float> & f32_data, const char * name) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t));
    t->data = ggml_backend_buffer_get_base(buf);
    t->buffer = buf;
    const size_t row_bytes = ggml_row_size(type, ne[0]);
    for (int e = 0; e < ne[2]; e++) {
        for (int r = 0; r < ne[1]; r++) {
            const float * src = &f32_data[((size_t) e * ne[1] + r) * ne[0]];
            ggml_quantize_chunk(type, src,
                (uint8_t *) t->data + (size_t) e * t->nb[2] + (size_t) r * row_bytes,
                0, 1, ne[0], nullptr);
        }
    }
    ggml_format_name(t, "%s", name);
    return t;
}

// a device-resident [n_expert, n_tokens_max] I32 tensor whose first n_used rows per
// column hold the expert ids; strided [n_used, n_tokens] views of it feed the graphs,
// like the router output of a real model. the ids must live on the accelerator: the
// pool update reads the routing back from the compute backend in the split prologue,
// so a host ids tensor behind a scheduler input copy would be read before this
// ubatch's upload (stale) rather than as-written
static void make_strided_ids(ggml_context * ctx, tensors & ts, ggml_backend_t accel) {
    const int64_t ne_w[2] = { n_expert, n_tokens_max };
    ts.ids_wide = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne_w[0], ne_w[1]);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(
            ggml_backend_get_default_buffer_type(accel), ggml_nbytes(ts.ids_wide));
    ts.ids_wide->data = ggml_backend_buffer_get_base(buf);
    ts.ids_wide->buffer = buf;
    ggml_format_name(ts.ids_wide, "%s", "ids_wide");
    ggml_backend_buffer_clear(buf, 0);
}

// refresh the expert ids (first n_used rows of each column of the wide store)
static void set_ids(tensors & ts, int n_tokens,
        const std::vector<int32_t> & ids_data) {
    std::vector<int32_t> wide((size_t) n_expert * n_tokens_max, n_expert - 1);
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_used; e++) {
            wide[t * n_expert + e] = ids_data[t * n_used + e];
        }
    }
    ggml_backend_tensor_set(ts.ids_wide, wide.data(), 0, ggml_nbytes(ts.ids_wide));
}

// reference model of the per-pool hit/miss accounting: mirrors the scheduler's LRU
// semantics (used experts processed in ascending id order; hits refresh the stamp,
// misses fill free slots first and otherwise evict the least recently used expert).
// each used expert must be counted exactly once per ubatch - a second update of the
// same pool within one graph execution would count it as a hit again and inflate the
// hit-rate telemetry
struct stats_model {
    std::vector<uint64_t> stamp; // per-expert LRU stamp, 0 = not cached
    uint64_t t = 0;
    int n_cached = 0;
    long long hits = 0;
    long long misses = 0;

    stats_model(int n_expert_, int n_slots_) : stamp(n_expert_, 0), n_slots(n_slots_) {}

    void ubatch(const std::vector<int32_t> & ids, int n_used_, int n_tokens_) {
        const int n_expert_ = (int) stamp.size();
        for (int e = 0; e < n_expert_; e++) {
            bool used = false;
            for (int i = 0; i < n_used_ * n_tokens_; i++) {
                used |= ids[(size_t) i] == e;
            }
            if (!used) {
                continue;
            }
            if (stamp[e] != 0) {
                hits++;
            } else {
                misses++;
                if (n_cached == n_slots) {
                    int victim = -1;
                    for (int c = 0; c < n_expert_; c++) {
                        if (stamp[c] != 0 && (victim == -1 || stamp[c] < stamp[victim])) {
                            victim = c;
                        }
                    }
                    stamp[victim] = 0;
                    n_cached--;
                }
                n_cached++;
            }
            stamp[e] = ++t;
        }
    }

private:
    int n_slots;
};

// build the pooled wrapper chain exactly like llm_graph_context::build_lora_mm_id;
// the table is already [1, n_expert] and is fed to GET_ROWS directly (no reshape view),
// so the remap itself anchors the pooled split and consumes the freshly uploaded table
static ggml_tensor * pooled_ids(ggml_context * ctx, int n_tokens,
        ggml_tensor * table, ggml_tensor * ids_view) {
    ggml_tensor * ids_flat = ggml_cont(ctx, ids_view);
    ggml_tensor * slots = ggml_get_rows(ctx, table, ggml_reshape_1d(ctx, ids_flat, n_used * n_tokens));
    return ggml_reshape_2d(ctx, slots, n_used, n_tokens);
}

static bool run_round(ggml_backend_sched_t sched, ggml_context * ctx,
        tensors & ts, int n_tokens, const std::vector<int32_t> & ids_data, const char * label) {
    set_ids(ts, n_tokens, ids_data);

    ggml_tensor * ids_view = ggml_view_2d(ctx, ts.ids_wide, n_used, n_tokens, ts.ids_wide->nb[1], 0);
    // no input flag: the ids live on the accelerator and must be consumed in place, like
    // a router output; flagging them as input would put a scheduler copy between the
    // write and the use and the split-prologue pool update would read stale memory
    ggml_tensor * x3 = ggml_reshape_3d(ctx,
            ggml_view_2d(ctx, ts.x, n_in, n_tokens, ts.x->nb[1], 0), n_in, 1, n_tokens);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // references through the regular host weight copy path
    ggml_tensor * ref_gu = ggml_mul_mat_id(ctx, ts.w_gu, x3, ids_view);
    ggml_tensor * ref_dn = ggml_mul_mat_id(ctx, ts.w_dn, x3, ids_view);

    // pooled variants sharing one routing (like fused gate_up + down MoE layers)
    ggml_tensor * out_gu = ggml_mul_mat_id(ctx, ts.pool_gu, x3,
            pooled_ids(ctx, n_tokens, ts.table_gu, ids_view));
    ggml_tensor * out_dn = ggml_mul_mat_id(ctx, ts.pool_dn, x3,
            pooled_ids(ctx, n_tokens, ts.table_dn, ids_view));

    ggml_build_forward_expand(gf, ref_gu);
    ggml_build_forward_expand(gf, ref_dn);
    ggml_build_forward_expand(gf, out_gu);
    ggml_build_forward_expand(gf, out_dn);

    // no reserve: sched_reserve would split this very graph, which rewrites the node
    // srcs to the scheduler's copies; the later compute then re-splits against the
    // rewritten graph, treats those copies as already-resident tensors, skips the
    // input uploads and drifts the remap GET_ROWS out of its own split (the pool
    // update then reads stale routing). every round builds a fresh graph, so reset
    // before compute (the decode loop does the same) and let alloc_graph reserve.
    ggml_backend_sched_reset(sched);
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", label);
        return false;
    }

    bool ok = true;
    for (auto & p : { std::make_pair(ref_gu, out_gu), std::make_pair(ref_dn, out_dn) }) {
        const size_t nbytes = ggml_nbytes(p.first);
        std::vector<uint8_t> host_ref(nbytes), host_out(nbytes);
        ggml_backend_tensor_get(p.first, host_ref.data(), 0, nbytes);
        ggml_backend_tensor_get(p.second, host_out.data(), 0, nbytes);

        if (memcmp(host_ref.data(), host_out.data(), nbytes) != 0) {
            ok = false;
            fprintf(stderr, "%s: mismatch between pooled and reference outputs\n", label);
            const float * fr = (const float *) host_ref.data();
            const float * fo = (const float *) host_out.data();
            for (size_t i = 0; i < nbytes / sizeof(float); i++) {
                if (fr[i] != fo[i]) {
                    fprintf(stderr, "  [%zu] ref = %f, pooled = %f\n", i, fr[i], fo[i]);
                    break;
                }
            }
        }
    }

    if (ok) {
        printf("%s: ok\n", label);
    }
    return ok;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    // find an accelerator backend to host the pool
    ggml_backend_dev_t accel_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            accel_dev = dev;
            break;
        }
    }
    if (!accel_dev) {
        printf("no accelerator backend found, skipping\n");
        return 0;
    }

    ggml_backend_t accel = ggml_backend_dev_init(accel_dev, nullptr);
    ggml_backend_buffer_type_t accel_buft = ggml_backend_dev_buffer_type(accel_dev);
    if (!accel) {
        fprintf(stderr, "failed to initialize the accelerator backend\n");
        return 1;
    }
    printf("accelerator: %s\n", ggml_backend_name(accel));

    const ggml_type types[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4 };

    // deterministic pseudo-random weights and activations; the weights differ per expert
    // so that a wrong id -> slot mapping surfaces as a bit-exact mismatch - identical
    // expert data would make any slot mapping compare equal
    std::vector<float> w_data((size_t) n_out * n_in * n_expert);
    std::vector<float> x_data((size_t) n_in * n_tokens_max);
    for (int e = 0; e < n_expert; e++) {
        for (size_t i = 0; i < (size_t) n_out * n_in; i++) {
            w_data[(size_t) e * n_out * n_in + i] =
                float((i * 1103515245u + 12345u) >> 16) / 4096.0f - 8.0f + 0.5f * e;
        }
    }
    for (size_t i = 0; i < x_data.size(); i++) x_data[i] = float((i * 214013u + 2531011u) >> 16) / 8192.0f - 4.0f;

    int n_failed = 0;

    for (ggml_type type : types) {
        printf("== %s ==\n", ggml_type_name(type));

        ggml_backend_t backends[] = { accel, ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) };
        ggml_backend_buffer_type_t bufts[] = {
            accel_buft,
            ggml_backend_get_default_buffer_type(backends[1]),
        };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 4096, false, true);
        // tensors that must stay alive across graph rebuilds
        ggml_init_params ip = { 32 * ggml_tensor_overhead(), NULL, true };
        ggml_context * tctx = ggml_init(ip);
        tensors ts;
        const int64_t ne_w[3] = { n_in, n_out, n_expert };
        ts.w_gu = make_quant_host_tensor(tctx, bufts[1], type, ne_w, w_data, "w_gu");
        ts.w_dn = make_quant_host_tensor(tctx, bufts[1], type, ne_w, w_data, "w_dn");
        const int64_t ne_x[2] = { n_in, n_tokens_max };
        ts.x = make_host_tensor(tctx, bufts[1], GGML_TYPE_F32, ne_x, 2, x_data, true, "x");
        make_strided_ids(tctx, ts, accel);

        ts.pool_gu = ggml_backend_sched_register_expert_pool(sched, ts.w_gu, 0, n_slots, &ts.table_gu);
        ts.pool_dn = ggml_backend_sched_register_expert_pool(sched, ts.w_dn, 0, n_slots, &ts.table_dn);
        if (!ts.pool_gu || !ts.pool_dn) {
            fprintf(stderr, "failed to register the expert pools\n");
            return 1;
        }

        // the pool bookkeeping (and its counters) persists across the n_tokens groups,
        // so the reference model lives alongside the sched; both test pools share one
        // routing, so the sched totals must be exactly twice the single-pool model
        stats_model model(n_expert, n_slots);
        auto check_stats = [&](const char * label, bool & ok) {
            long long hits = -1, misses = -1;
            ggml_backend_sched_get_expert_pool_stats(sched, &hits, &misses);
            const long long exp_hits = 2 * model.hits;
            const long long exp_misses = 2 * model.misses;
            if (hits != exp_hits || misses != exp_misses) {
                fprintf(stderr, "%s: hit-rate telemetry mismatch: sched reports hits=%lld misses=%lld,"
                        " expected hits=%lld misses=%lld\n", label, hits, misses, exp_hits, exp_misses);
                ok = false;
            }
        };

        for (int n_tokens : { 1, 3, 8, n_tokens_max }) {
            ggml_init_params gp = { 64 * ggml_tensor_overhead() + 8 * ggml_graph_overhead(), NULL, true };
            ggml_context * gctx = ggml_init(gp);

            char label[128];
            bool ok = true;

            // n_tokens >= 8 uses the wide id sets that drive the pool through a full
            // 16-slot fill and a full eviction per ubatch; nt = 9 additionally exceeds
            // the mmvq mul_mat_id batch limit and exercises the CUDA MMQ ids path
            // (Blackwell: the native FP4 MMA kernels)
            const bool mmq_case = n_tokens >= 8;

            // round 1: cold pool, low experts are loaded
            snprintf(label, sizeof(label), "%s nt=%d round1 (cold)", ggml_type_name(type), n_tokens);
            const std::vector<int32_t> ids_r1 = mmq_case ?
                    std::vector<int32_t>{0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 0,1} :
                    std::vector<int32_t>{0, 1, 2, 1, 3, 0};
            ok &= run_round(sched, gctx, ts, n_tokens, ids_r1, label);
            model.ubatch(ids_r1, n_used, n_tokens);
            check_stats(label, ok);

            // round 2: unseen experts miss and evict everything
            snprintf(label, sizeof(label), "%s nt=%d round2 (full eviction)", ggml_type_name(type), n_tokens);
            const std::vector<int32_t> ids_r2 = mmq_case ?
                    std::vector<int32_t>{16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 16,17} :
                    std::vector<int32_t>{4, 5, 6, 4, 7, 5};
            ok &= run_round(sched, gctx, ts, n_tokens, ids_r2, label);
            model.ubatch(ids_r2, n_used, n_tokens);
            check_stats(label, ok);

            // round 3: hits plus evicted-expert reloads
            snprintf(label, sizeof(label), "%s nt=%d round3 (hit + reload)", ggml_type_name(type), n_tokens);
            const std::vector<int32_t> ids_r3 = mmq_case ?
                    std::vector<int32_t>{16,17, 0,1, 2,3, 16,17, 0,1, 2,3, 16,17, 0,1, 16,17} :
                    std::vector<int32_t>{4, 0, 1, 4, 0, 1};
            ok &= run_round(sched, gctx, ts, n_tokens, ids_r3, label);
            model.ubatch(ids_r3, n_used, n_tokens);
            check_stats(label, ok);

            // rebuild the graph with fresh tensors: pool contents and bookkeeping must survive
            ggml_free(gctx);
            gctx = ggml_init(gp);
            snprintf(label, sizeof(label), "%s nt=%d round4 (after graph rebuild)", ggml_type_name(type), n_tokens);
            const std::vector<int32_t> ids_r4 = mmq_case ?
                    std::vector<int32_t>{4,5, 4,5, 6,7, 6,7, 4,5, 4,5, 6,7, 6,7, 4,5} :
                    std::vector<int32_t>{2, 3, 2, 3, 2, 3};
            ok &= run_round(sched, gctx, ts, n_tokens, ids_r4, label);
            model.ubatch(ids_r4, n_used, n_tokens);
            check_stats(label, ok);

            if (!ok) {
                n_failed++;
            }
            ggml_free(gctx);
        }

        ggml_free(tctx);
        ggml_backend_free(backends[1]);
        ggml_backend_sched_free(sched);
    }

    ggml_backend_free(accel);

    if (n_failed > 0) {
        fprintf(stderr, "expert pool test FAILED (%d case(s))\n", n_failed);
        return 1;
    }
    printf("expert pool test PASSED\n");
    return 0;
}
