// SYCL MoE Expert Cache.
//
// Keeps the hottest CPU-resident MoE expert weights resident in VRAM so
// decode-time expert matvecs run on the GPU instead of the CPU path.
//
// This implements the ggml_moe_cache_api contract (see
// ggml-backend-moe-cache.h), following the same v1 shape as
// ggml-vulkan-moe-cache.cpp: the scheduler owns one cache session; the CPU
// MUL_MAT_ID path calls begin/plan/dispatch/collect/end per node.
//
// v1 notes:
//   - synchronous fills (no worker thread), same limitation as the Vulkan
//     provider; a background uploader is a natural v2 (moe-cache-common.h's
//     moe_cache_device already carries the queue/worker/inflight fields for
//     it, unused here)
//   - fills per plan are bounded by inserts_per_plan (and queue_mb)
//   - one pool per (expert_size, wtype), allocated lazily in begin()
//   - slabs and scratch buffers are plain USM device allocations
//     (sycl::malloc_device); unlike Vulkan, SYCL needs no staging buffer or
//     UMA/discrete distinction - queue::memcpy handles host<->device
//     directly regardless of device topology
//   - dispatch reuses the existing ggml_sycl_mul_mat_vec_q_id() kernel
//     dispatcher (mmvq.cpp): the slab is addressed exactly like the stacked
//     expert-weight buffer that function already expects, with slot indices
//     standing in for expert ids and expert_size as the inter-expert stride
//   - fused SwiGLU returns NULL (stock CPU path handles the node)
//
// Register by calling ggml_sycl_moe_cache_register() from
// ggml_backend_sycl_reg() after the backend reg struct is set up.

#include "ggml-sycl.h"

#include "common.hpp"
#include "mmvq.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "../ggml-moe-cache-common.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <vector>

// Thread-local session stack (owned by this backend; independent of other backends').
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

// Global session registry for invalidate()/teardown paths.
static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
// Live-session count, so invalidate() can bail before taking g_registry_mu.
static std::atomic<size_t> g_session_count{0};

// Backend registration object this provider was registered under.
static const void * g_moe_cache_owner = nullptr;

// ---------------------------------------------------------------------------
// SYCL device extension
// ---------------------------------------------------------------------------

struct moe_cache_sycl_device : public moe_cache_device {
    moe_cache_sycl_device(int logical, int physical)
        : moe_cache_device(logical, physical) {}

    ~moe_cache_sycl_device() { free_resources(); }

    queue_ptr stream = nullptr;

    // USM device slab per pool (parallel to pools).
    std::vector<void *> pool_slabs;

    // USM device scratch, grown on demand.
    void * d_ids = nullptr;
    size_t d_ids_cap = 0;
    void * d_act = nullptr;
    size_t d_act_cap = 0;
    void * d_out = nullptr;
    size_t d_out_cap = 0;

    // Host scratch for quantized activations (scalar reference quantizer).
    std::vector<char> h_act;

    void free_resources() {
        if (!stream) {
            pool_slabs.clear();
            return;
        }
        try {
            for (void * slab : pool_slabs) {
                if (slab) {
                    sycl::free(slab, *stream);
                }
            }
            if (d_ids) sycl::free(d_ids, *stream);
            if (d_act) sycl::free(d_act, *stream);
            if (d_out) sycl::free(d_out, *stream);
        } catch (...) {
            // Best-effort: the context may already be torn down.
        }
        pool_slabs.clear();
        d_ids = d_act = d_out = nullptr;
        d_ids_cap = d_act_cap = d_out_cap = 0;
        for (auto & pool : pools) {
            pool->slab = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// Quantize activations to Q8_1 (scalar, reference quality; rows are tiny and
// this runs on the CPU thread that already owns them - not a hot path).
// ---------------------------------------------------------------------------

static void sycl_moe_quantize_act_q8_1(const float * src, block_q8_1 * dst,
                                       int64_t n, int64_t padded_n) {
    const int nb = (int)(padded_n / QK8_1);
    for (int ib = 0; ib < nb; ib++) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            const int64_t idx = (int64_t)ib * QK8_1 + i;
            const float v = (idx < n) ? src[idx] : 0.0f;
            amax = std::max(amax, std::fabs(v));
        }
        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            const int64_t idx = (int64_t)ib * QK8_1 + i;
            const float v = (idx < n) ? src[idx] : 0.0f;
            const int8_t q = (int8_t)std::lround(v * id);
            dst[ib].qs[i] = q;
            sum += (float)q * d;
        }
        dst[ib].ds = sycl::half2(sycl::half(d), sycl::half(sum));
    }
}

// Whether ggml_sycl_mul_mat_vec_q_id() (mmvq.cpp) has a dispatch case for
// this type. Kept as a local copy rather than a shared predicate so this
// provider has no cross-file dependency on mmvq.{hpp,cpp} beyond the
// dispatch function itself - keep in sync with that function's switch.
static bool sycl_moe_wtype_dispatchable(ggml_type wtype) {
    switch (wtype) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

static int sycl_moe_query_config(int automatic, size_t budget_mib,
                                 ggml_moe_cache_config * result) {
    if (!result) {
        return 0;
    }

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
        config.min_compute_capability =
            moe_cache_min_compute_capability(config.automatic);
    }
    if (budget_mib > 0) {
        config.budget_mb = budget_mib;
    }
    if (!config.enabled || config.budget_mb > (SIZE_MAX >> 20) ||
        config.reserve_mb > (SIZE_MAX >> 20)) {
        return 0;
    }

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = config.min_compute_capability;
    result->min_devices = 1; // SYCL session uses a single device, like Vulkan's v1
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    return 1;
}

static int sycl_moe_query_device(void * opaque, const ggml_moe_cache_config * config,
                                 ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result || !g_moe_cache_owner) {
        return 0;
    }

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != g_moe_cache_owner) {
        return 0;
    }

    result->logical_device = 0;
    result->physical_device = 0;
    result->compute_capability = 800; // no CC concept on SYCL; matches Vulkan's Ampere-equivalent default
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(800);
    return 1;
}

static int sycl_moe_query_shape(int wtype, int64_t n_in, int64_t n_out,
                                int64_t n_expert, size_t expert_size,
                                ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0) {
        return 0;
    }
    if (!ggml_moe_cache_wtype_supported(wtype) ||
        !sycl_moe_wtype_dispatchable((ggml_type)wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }

    const size_t out_bytes =
        moe_cache_node_rows_max * (size_t)n_out * sizeof(float);
    const size_t act_q8_bytes =
        moe_cache_node_rows_max *
        (size_t)((n_in + QK8_1 - 1) / QK8_1) * sizeof(block_q8_1);
    const size_t scratch_bytes = out_bytes + act_q8_bytes;
    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    if (pool_bytes > SIZE_MAX - scratch_bytes) {
        return 0;
    }
    result->scratch_bytes = scratch_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = scratch_bytes + pool_bytes;
    return 1;
}

// ---------------------------------------------------------------------------
// Pool lifecycle
// ---------------------------------------------------------------------------

static moe_cache_pool * sycl_moe_find_or_create_pool(
        moe_cache_sycl_device & dev, moe_cache_session & session,
        size_t expert_size, int wtype, int64_t n_expert, size_t budget_bytes) {
    const int existing = moe_cache_find_pool(dev, expert_size, wtype);
    if (existing >= 0) {
        return dev.pools[existing].get();
    }

    if (moe_cache_fail(session, "slab")) {
        MOE_CACHE_LOG("[moe-cache] SYCL: skipped %zu KiB expert pool: allocation failed\n",
                expert_size >> 10);
        return nullptr;
    }

    size_t slots = budget_bytes / expert_size;
    if (slots < moe_cache_pool_slots_min) {
        return nullptr;
    }
    if ((uint64_t)n_expert > 0 && slots > (size_t)n_expert) {
        slots = (size_t)n_expert;
    }
    if (slots < moe_cache_pool_slots_min) {
        return nullptr;
    }
    if (slots > (size_t)INT_MAX) {
        slots = INT_MAX;
    }
    const size_t slab_bytes = slots * expert_size;

    void * slab = nullptr;
    try {
        slab = sycl::malloc_device(slab_bytes, *dev.stream);
    } catch (...) {
        slab = nullptr;
    }
    if (!slab) {
        MOE_CACHE_LOG("[moe-cache] SYCL: failed to allocate %zu MiB expert pool\n",
                slab_bytes >> 20);
        return nullptr;
    }

    try {
        std::unique_ptr<moe_cache_pool> pool(new moe_cache_pool());
        pool->expert_size = expert_size;
        pool->wtype = wtype;
        pool->slab = nullptr; // SYCL slabs are device pointers, tracked in pool_slabs
        pool->n_slots = (int)slots;
        pool->covers_all_entries = (uint64_t)slots >= (uint64_t)n_expert;
        pool->slots.resize(slots);
        pool->free_slots.reserve(slots);
        pool->map.reserve(slots);
        for (int index = (int)slots - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        dev.allocated_bytes += slab_bytes;
        dev.pools.push_back(std::move(pool));
        dev.pool_slabs.push_back(slab);
        MOE_CACHE_LOG("[moe-cache] SYCL%d pool[%d]: type=%s expert=%zu KiB slots=%zu entries=%lld coverage=%s total=%zu MiB\n",
                dev.physical, (int)dev.pools.size() - 1,
                ggml_type_name((ggml_type)wtype), expert_size >> 10,
                slots, (long long)n_expert,
                dev.pools.back()->covers_all_entries ? "complete" : "partial",
                slab_bytes >> 20);
        bool expected = false;
        if (session.enabled_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] enabled: first pool allocated on SYCL%d\n",
                    dev.physical);
        }
        return dev.pools.back().get();
    } catch (...) {
        try { sycl::free(slab, *dev.stream); } catch (...) {}
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

static void * sycl_moe_session_create(void * const * backends, int n_backends,
                                      const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            if (supplied_config->budget_bytes % MiB != 0 ||
                supplied_config->reserve_bytes % MiB != 0 ||
                supplied_config->minimum_slab_bytes % MiB != 0 ||
                supplied_config->min_expert_bytes == 0 ||
                supplied_config->min_expert_explicit < 0 ||
                supplied_config->min_expert_explicit > 1 ||
                supplied_config->max_batch < 1 ||
                supplied_config->max_batch > moe_cache_batch_max ||
                supplied_config->min_devices < 1 ||
                supplied_config->min_compute_capability < 0 ||
                supplied_config->min_compute_capability > 999 ||
                supplied_config->overlap_cpu_rows < -1 ||
                supplied_config->overlap_cpu_rows > 8) {
                return nullptr;
            }
            config.enabled = true;
            config.automatic = supplied_config->minimum_slab_bytes > 0;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
        }
        if (!config.enabled) {
            return nullptr;
        }
        // A zero budget can never create a pool; bail before touching the device.
        if (!supplied_config && config.budget_mb == 0) {
            return nullptr;
        }

        // Find the SYCL backend among the scheduler's backends.
        ggml_backend_t sycl_backend = nullptr;
        for (int i = 0; i < n_backends; i++) {
            ggml_backend_t be = (ggml_backend_t)backends[i];
            if (be && ggml_backend_is_sycl(be)) {
                sycl_backend = be;
                break;
            }
        }
        if (!sycl_backend) {
            MOE_CACHE_LOG("[moe-cache] no SYCL device found\n");
            return nullptr;
        }

        ggml_backend_sycl_context * sctx =
            (ggml_backend_sycl_context *)sycl_backend->context;
        if (!sctx) {
            MOE_CACHE_LOG("[moe-cache] SYCL backend has no usable context\n");
            return nullptr;
        }

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unique_ptr<moe_cache_sycl_device> dev(new (std::nothrow)
                moe_cache_sycl_device(sctx->device, sctx->device));
        if (!dev) {
            return nullptr;
        }
        dev->stream = sctx->stream();

        session->devices.push_back(std::move(dev));

        moe_cache_session * result = session.get();
        try {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.store(g_sessions.size(), std::memory_order_release);
        } catch (...) {
            return nullptr;
        }
        MOE_CACHE_LOG("[moe-cache] SYCL session ready (device=%d, budget=%zu MiB)\n",
                sctx->device, result->config.budget_mb);
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] SYCL session creation failed\n");
        return nullptr;
    }
}

// Teardown statistics, same field names as the other backends so the log
// contract is backend-independent.
static void sycl_moe_log_stats(moe_cache_sycl_device & dev) {
    size_t used = 0;
    size_t slots = 0;
    for (const auto & pool_ptr : dev.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
    }
    const long long total = dev.hits + dev.misses;
    MOE_CACHE_LOG("[moe-cache] SYCL%d hits=%lld/%lld (%.1f%%) used=%zu/%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld skips=%lld admission=%lld dispatch-fail=%lld collect-fail=%lld bypass=%lld\n",
            dev.physical, dev.hits, total,
            total ? 100.0 * (double)dev.hits / (double)total : 0.0,
            used, slots, dev.inserts, dev.fills, dev.fill_failures,
            dev.evictions, dev.insert_skips, dev.admission_skips,
            dev.dispatch_failures, dev.collect_failures, dev.contention_bypasses);
}

static void sycl_moe_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }
    {
        std::lock_guard<std::mutex> lock(g_registry_mu);
        g_sessions.erase(session);
        g_session_count.store(g_sessions.size(), std::memory_order_release);
    }
    for (auto & dev_ptr : session->devices) {
        moe_cache_sycl_device & dev =
            static_cast<moe_cache_sycl_device &>(*dev_ptr);
        if (dev.nodes > 0 || dev.dispatch_failures > 0 ||
            dev.collect_failures > 0) {
            sycl_moe_log_stats(dev);
        }
    }
    delete session;
}

static void sycl_moe_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load() || session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    session->active_scopes++;
}

static void sycl_moe_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Begin: find pool, create node. Pools are created lazily on first use.
// ---------------------------------------------------------------------------

static void * sycl_moe_begin(const char * name, const void * host_base,
                             size_t expert_size, int64_t n_in, int64_t n_out,
                             int wtype, int64_t n_expert, int64_t n_tokens,
                             int64_t n_rows) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    if (!name || !host_base || !moe_cache_tensor_name_supported(name) ||
        n_tokens < 1 || expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !ggml_moe_cache_wtype_supported(wtype) ||
        !sycl_moe_wtype_dispatchable((ggml_type)wtype)) {
        return nullptr;
    }
    if (n_rows < n_tokens || n_rows % n_tokens != 0 ||
        n_tokens > session->config.max_batch ||
        n_rows > moe_cache_node_rows_max) {
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }

    if (session->devices.empty()) {
        return nullptr;
    }
    moe_cache_sycl_device & dev =
        static_cast<moe_cache_sycl_device &>(*session->devices[0]);
    // A zero budget can never create a pool; skip GPU setup entirely.
    if (session->config.budget_mb == 0) {
        return nullptr;
    }

    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                dev.dispatch_mu, std::try_to_lock);
    } catch (...) {
        dev.contention_bypasses++;
        return nullptr;
    }
    if (!dispatch_lock.owns_lock()) {
        dev.contention_bypasses++;
        return nullptr;
    }
    if (dev.dead.load()) {
        return nullptr;
    }

    moe_cache_log_configuration(*session);
    const size_t budget_bytes = session->config.budget_mb << 20;
    moe_cache_pool * pool = sycl_moe_find_or_create_pool(
            dev, *session, expert_size, wtype, n_expert, budget_bytes);
    if (!pool) {
        return nullptr;
    }
    const int pool_index = moe_cache_find_pool(dev, expert_size, wtype);
    if (pool_index < 0) {
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        return nullptr;
    }
    node->session = session;
    node->device = &dev;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->n_tokens = n_tokens;
    node->wtype = wtype;
    node->dispatch_lock = std::move(dispatch_lock);

    std::lock_guard<std::mutex> session_lock(session->mu);
    session->active_nodes++;
    return node.release();
}

// ---------------------------------------------------------------------------
// Plan: mark cache hits; sync-fill a bounded number of misses into the slab
// and serve them in this node. slot_indices[i] >= 0 means cache-served.
// ---------------------------------------------------------------------------

static int sycl_moe_plan(void * opaque, const int32_t * ids, int n_ids,
                         int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 ||
        n_ids > moe_cache_node_rows_max || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_sycl_device & dev =
        static_cast<moe_cache_sycl_device &>(*node->device);
    moe_cache_pool & pool = *node->pool;
    int hits = 0;

    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }

    int n_misses = 0;
    int n_fills = 0;
    const size_t max_fill_bytes = session.config.queue_mb << 20;
    if (max_fill_bytes == 0 || node->expert_size > max_fill_bytes) {
        n_fills = 0;
    } else {
        n_fills = std::min((int)session.config.inserts_per_plan,
                           (int)(max_fill_bytes / node->expert_size));
    }
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || dev.dead.load()) {
            continue;
        }
        const moe_cache_key key{node->host_base, expert};
        if (pool.map.find(key) == pool.map.end() ||
            pool.slots[pool.map.at(key)].state != moe_cache_slot_state::valid) {
            n_misses++;
        }
    }
    const int fill_budget = std::min(n_misses, n_fills);

    void * slab = (node->pool_index >= 0 && node->pool_index < (int)dev.pool_slabs.size())
        ? dev.pool_slabs[node->pool_index] : nullptr;

    struct pending_fill { int slot; int index; };
    std::vector<pending_fill> pending;
    pending.reserve(fill_budget > 0 ? fill_budget : 0);

    int fills_done = 0;
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || dev.dead.load()) {
            continue;
        }

        const moe_cache_key key{node->host_base, expert};
        auto found = pool.map.find(key);
        if (found != pool.map.end() &&
            pool.slots[found->second].state == moe_cache_slot_state::valid) {
            const int slot_index = found->second;
            moe_cache_slot & slot = pool.slots[slot_index];
            slot.readers++;
            slot.uses++;
            moe_cache_lru_remove(pool, slot_index);
            moe_cache_lru_push_back(pool, slot_index);
            node->pins[node->n_pins++] = {&pool, slot_index};
            slot_indices[index] = slot_index;
            dev.hits++;
            hits++;
            continue;
        }

        dev.misses++;
        if (moe_cache_fail(session, "insert")) {
            dev.fill_failures++;
            continue;
        }
        if (fills_done >= fill_budget || !slab) {
            continue; // CPU handles this row
        }
        fills_done++;

        int slot_index = -1;
        if (!pool.free_slots.empty()) {
            slot_index = pool.free_slots.back();
            pool.free_slots.pop_back();
        } else {
            int candidate = moe_cache_pick_victim(pool, session.config.hot_uses);
            if (candidate < 0) {
                continue; // all slots pinned; CPU handles this row
            }
            slot_index = candidate;
            const bool sacrificed_hot =
                (int)pool.slots[slot_index].uses > session.config.hot_uses;
            moe_cache_slot_reset(pool, slot_index, false);
            dev.evictions++;
            if (sacrificed_hot) {
                dev.heat_evictions++;
            }
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.key = key;
        slot.generation++;
        slot.state = moe_cache_slot_state::copying;
        const void * source =
            (const char *)node->host_base + (size_t)expert * node->expert_size;
        try {
            pool.map.emplace(key, slot_index);
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            dev.insert_skips++;
            continue;
        }

        try {
            dev.stream->memcpy((char *)slab + (size_t)slot_index * node->expert_size,
                    source, node->expert_size);
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            dev.fill_failures++;
            continue;
        }
        pending.push_back({slot_index, index});
    }

    // One wait for every queued fill this plan(): SYCL's in-order queue
    // executes them in submission order, so a single wait confirms all of
    // them - no per-fill round trip needed.
    if (!pending.empty()) {
        bool copy_ok = true;
        try {
            dev.stream->wait_and_throw();
        } catch (...) {
            copy_ok = false;
        }
        if (copy_ok) {
            for (const pending_fill & fill : pending) {
                moe_cache_slot & pslot = pool.slots[fill.slot];
                pslot.state = moe_cache_slot_state::valid;
                moe_cache_lru_push_back(pool, fill.slot);
                pslot.readers++;
                node->pins[node->n_pins++] = {&pool, fill.slot};
                slot_indices[fill.index] = fill.slot;
                dev.inserts++;
                dev.fills++;
                dev.hits++;
                hits++;
            }
        } else {
            // Slab contents are unknown after a failed transfer; roll back
            // and disable this device's cache so no later node reads stale
            // entries.
            for (const pending_fill & fill : pending) {
                moe_cache_slot_reset(pool, fill.slot, true);
                dev.fill_failures++;
            }
            dev.dead.store(true);
            MOE_CACHE_LOG("[moe-cache] SYCL%d: staged fill transfer failed; "
                          "rolled back %zu fills and disabled the device cache\n",
                          dev.physical, pending.size());
        }
    }

    dev.nodes++;
    return hits;
}

// ---------------------------------------------------------------------------
// Dispatch: run the cached-expert matvec over the hit rows, reusing the
// existing MoE mat-vec-q-id kernel dispatcher (mmvq.cpp). The slab is
// addressed exactly like the stacked expert-weight buffer that dispatcher
// expects: slot indices stand in for expert ids, expert_size is the
// inter-expert stride, and each hit row carries its own activation (a
// nonzero src1_row_stride), unlike the shared-activation MoE-routing case.
// ---------------------------------------------------------------------------

static int sycl_moe_dispatch(void * opaque, int wtype, int64_t n_in, int64_t n_out,
                             int n_hits, const int32_t * slot_indices,
                             const float * const * act_rows) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->planned || !slot_indices || !act_rows ||
        n_hits <= 0 || n_hits > moe_cache_node_rows_max ||
        n_hits != node->n_pins ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out) {
        return 0;
    }

    moe_cache_sycl_device & dev =
        static_cast<moe_cache_sycl_device &>(*node->device);
    if (dev.dead.load() || moe_cache_fail(*node->session, "dispatch")) {
        std::lock_guard<std::mutex> lock(node->session->mu);
        dev.dispatch_failures++;
        return 0;
    }
    const size_t pool_index = (size_t)node->pool_index >= 0
        ? (size_t)node->pool_index : 0;
    if (dev.pool_slabs.empty() || pool_index >= dev.pool_slabs.size()) {
        return 0;
    }
    void * slab = dev.pool_slabs[pool_index];

    const int64_t padded_n_in = ((n_in + QK8_1 - 1) / QK8_1) * QK8_1;
    const int64_t n_blocks = padded_n_in / QK8_1;

    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t);
    const size_t act_bytes = (size_t)n_hits * (size_t)n_blocks * sizeof(block_q8_1);
    const size_t out_bytes = (size_t)n_hits * (size_t)n_out * sizeof(float);

    try {
        if (act_bytes > dev.h_act.size()) {
            dev.h_act.resize(act_bytes);
        }
        if (ids_bytes > dev.d_ids_cap) {
            if (dev.d_ids) sycl::free(dev.d_ids, *dev.stream);
            dev.d_ids = sycl::malloc_device(ids_bytes, *dev.stream);
            dev.d_ids_cap = dev.d_ids ? ids_bytes : 0;
        }
        if (act_bytes > dev.d_act_cap) {
            if (dev.d_act) sycl::free(dev.d_act, *dev.stream);
            dev.d_act = sycl::malloc_device(act_bytes, *dev.stream);
            dev.d_act_cap = dev.d_act ? act_bytes : 0;
        }
        if (out_bytes > dev.d_out_cap) {
            if (dev.d_out) sycl::free(dev.d_out, *dev.stream);
            dev.d_out = sycl::malloc_device(out_bytes, *dev.stream);
            dev.d_out_cap = dev.d_out ? out_bytes : 0;
        }
    } catch (...) {
        dev.dispatch_failures++;
        return 0;
    }
    if (!dev.d_ids || !dev.d_act || !dev.d_out) {
        dev.dispatch_failures++;
        return 0;
    }

    block_q8_1 * act_q8 = (block_q8_1 *)dev.h_act.data();
    for (int i = 0; i < n_hits; i++) {
        sycl_moe_quantize_act_q8_1(act_rows[i], act_q8 + (size_t)i * n_blocks,
                n_in, padded_n_in);
    }

    try {
        dev.stream->memcpy(dev.d_ids, slot_indices, ids_bytes);
        dev.stream->memcpy(dev.d_act, act_q8, act_bytes);
        dev.stream->wait_and_throw();
    } catch (...) {
        dev.dispatch_failures++;
        return 0;
    }

    const size_t row_stride_bytes = (size_t)n_blocks * sizeof(block_q8_1);
    const bool ok = ggml_sycl_mul_mat_vec_q_id(
            (ggml_type)wtype, slab, dev.d_act, (const int32_t *)dev.d_ids,
            (float *)dev.d_out, (int)n_in, (int)n_out, n_hits,
            /*expert_weight_stride=*/ node->expert_size,
            /*dst_row_stride=*/ (size_t)n_out * sizeof(float),
            /*src1_row_stride=*/ row_stride_bytes,
            dev.stream);
    if (!ok) {
        dev.dispatch_failures++;
        return 0;
    }
    try {
        dev.stream->wait_and_throw();
    } catch (...) {
        dev.dispatch_failures++;
        return 0;
    }

    node->dispatched = true;
    return 1;
}

// ---------------------------------------------------------------------------
// Collect: copy results into dst_rows.
// ---------------------------------------------------------------------------

static int sycl_moe_collect(void * opaque, int n_hits, float * const * dst_rows,
                            int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        node->n_pins != n_hits || !dst_rows || n_out != node->n_out) {
        return 0;
    }
    for (int index = 0; index < n_hits; index++) {
        if (!dst_rows[index]) {
            return 0;
        }
    }

    moe_cache_sycl_device & dev =
        static_cast<moe_cache_sycl_device &>(*node->device);
    moe_cache_session & session = *node->session;
    bool ok = !dev.dead.load() && !moe_cache_fail(session, "collect");
    if (ok) {
        const size_t out_bytes = (size_t)n_hits * (size_t)n_out * sizeof(float);
        std::vector<float> h_out;
        try {
            h_out.resize((size_t)n_hits * (size_t)n_out);
            dev.stream->memcpy(h_out.data(), dev.d_out, out_bytes);
            dev.stream->wait_and_throw();
        } catch (...) {
            ok = false;
        }
        if (ok) {
            for (int index = 0; index < n_hits; index++) {
                memcpy(dst_rows[index], h_out.data() + (size_t)index * n_out,
                       (size_t)n_out * sizeof(float));
            }
        }
    }
    node->dispatched = false;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            dev.collect_failures++;
        }
        dev.collect_calls++;
        if (session.config.stats_every > 0 &&
            dev.collect_calls % session.config.stats_every == 0) {
            sycl_moe_log_stats(dev);
        }
    }
    return ok ? 1 : 0;
}

// ---------------------------------------------------------------------------
// End: release slot pins and the node.
// ---------------------------------------------------------------------------

static void sycl_moe_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }
    moe_cache_session & session = *node->session;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Fused SwiGLU - not implemented for v1; stock CPU path handles the node.
// ---------------------------------------------------------------------------

static void * sycl_moe_fused_begin(const ggml_moe_cache_tensor_desc * up,
                                   const ggml_moe_cache_tensor_desc * gate,
                                   int glu_op, float up_min, float up_max,
                                   float gate_min, float gate_max,
                                   const int32_t * ids, int n_rows,
                                   int64_t n_tokens,
                                   const float * const * act_rows,
                                   uint64_t * hit_mask) {
    (void)up; (void)gate; (void)glu_op;
    (void)up_min; (void)up_max; (void)gate_min; (void)gate_max;
    (void)ids; (void)n_rows; (void)n_tokens; (void)act_rows; (void)hit_mask;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Invalidate: drop cached slots whose tensor range overlaps [base, base+size).
// ---------------------------------------------------------------------------

static void sycl_moe_invalidate(const void * base, size_t size) {
    if (!base || size == 0) {
        return;
    }
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (auto & dev_ptr : session->devices) {
            for (auto & pool_ptr : dev_ptr->pools) {
                for (int i = 0; i < pool_ptr->n_slots; i++) {
                    if (pool_ptr->slots[i].key.tensor &&
                        moe_cache_ranges_overlap(
                            pool_ptr->slots[i].key.tensor,
                            pool_ptr->expert_size, base, size)) {
                        moe_cache_slot_reset(*pool_ptr, i, true);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

static void sycl_moe_register(const void * owner) {
    g_moe_cache_owner = owner;
    ggml_moe_cache_api api = {};
    api.owner = owner;
    api.query_config = sycl_moe_query_config;
    api.query_device = sycl_moe_query_device;
    api.query_shape = sycl_moe_query_shape;
    api.session_create = sycl_moe_session_create;
    api.session_destroy = sycl_moe_session_destroy;
    api.session_enter = sycl_moe_session_enter;
    api.session_leave = sycl_moe_session_leave;
    api.begin = sycl_moe_begin;
    api.plan = sycl_moe_plan;
    api.dispatch = sycl_moe_dispatch;
    api.collect = sycl_moe_collect;
    api.end = sycl_moe_end;
    api.fused_begin = sycl_moe_fused_begin;
    api.invalidate = sycl_moe_invalidate;
    ggml_moe_cache_register(&api);
}

// Prior declaration so the definition below does not trip -Wmissing-declarations.
extern "C" void ggml_sycl_moe_cache_register(void * reg);

extern "C" void ggml_sycl_moe_cache_register(void * reg) {
    sycl_moe_register(reg);
}
