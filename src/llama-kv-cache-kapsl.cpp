#include "llama-kv-cache-kapsl.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "ggml-backend.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <algorithm>

namespace {

class llama_kv_cache_kapsl_context : public llama_kv_cache_graph_context {
public:
    explicit llama_kv_cache_kapsl_context(llama_memory_status status) : status(status) {}

    llama_kv_cache_kapsl_context(
            llama_memory_status status,
            llama_kapsl_kv_pool_desc * pool,
            uint32_t n_kv,
            std::vector<llama_ubatch> ubatches = {}) :
        status(status),
        pool(pool),
        n_kv(n_kv),
        ubatches(std::move(ubatches)) {
    }

    ~llama_kv_cache_kapsl_context() override {
        if (block_table_buffer != nullptr) {
            ggml_backend_buffer_free(block_table_buffer);
        }
        if (pool_buffer != nullptr) {
            ggml_backend_buffer_free(pool_buffer);
        }
    }

    bool next() override {
        GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS);
        if (++i_cur >= ubatches.size()) {
            return false;
        }
        return true;
    }

    bool apply() override {
        GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS);
        GGML_ASSERT(i_cur < ubatches.size());
        return true;
    }

    const llama_ubatch & get_ubatch() const override {
        GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS);
        GGML_ASSERT(i_cur < ubatches.size());
        return ubatches[i_cur];
    }

    llama_memory_status get_status() const override {
        return status;
    }

    uint32_t get_n_kv() const override {
        return n_kv;
    }

    ggml_type type_k() const override {
        return GGML_TYPE_F16;
    }

    ggml_type type_v() const override {
        return GGML_TYPE_F16;
    }

    ggml_tensor * get_k(ggml_context * ctx, int32_t) const override {
        return make_pool_tensor(ctx, 0);
    }

    ggml_tensor * get_v(ggml_context * ctx, int32_t) const override {
        return make_pool_tensor(ctx, 1);
    }

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override {
        return ggml_kapsl_kv_write(
                ctx,
                get_k(ctx, il),
                k_cur,
                k_idxs,
                make_block_table_tensor(ctx),
                il,
                (int32_t) pool->block_size,
                (int32_t) pool->block_table_layer_stride);
    }

    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override {
        return ggml_kapsl_kv_write(
                ctx,
                get_v(ctx, il),
                v_cur,
                v_idxs,
                make_block_table_tensor(ctx),
                il,
                (int32_t) pool->block_size,
                (int32_t) pool->block_table_layer_stride);
    }

    ggml_tensor * paged_attn(
            ggml_context * ctx,
            ggml_tensor  * q,
            ggml_tensor  * positions,
            float          scale,
            int32_t        il) const override {
        return ggml_kapsl_paged_attn(
                ctx,
                q,
                make_pool_base_tensor(ctx),
                positions,
                make_block_table_tensor(ctx),
                il,
                (int32_t) pool->block_size,
                (int32_t) pool->block_table_layer_stride,
                (int32_t) pool->num_kv_heads,
                scale);
    }

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override {
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(positions);
        ggml_set_name(positions, "kapsl_k_pos");
        return positions;
    }

    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override {
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(positions);
        ggml_set_name(positions, "kapsl_v_pos");
        return positions;
    }

    ggml_tensor * build_input_k_rot(ggml_context *) const override {
        return nullptr;
    }

    ggml_tensor * build_input_v_rot(ggml_context *) const override {
        return nullptr;
    }

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override {
        set_input_positions(dst, ubatch);
    }

    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override {
        set_input_positions(dst, ubatch);
    }

    void set_input_k_shift(ggml_tensor *) const override {
    }

    void set_input_kq_mask(ggml_tensor *, const llama_ubatch *, bool) const override {
    }

    void set_input_pos_bucket(ggml_tensor *, const llama_ubatch *) const override {
    }

    void set_input_k_rot(ggml_tensor *) const override {
    }

    void set_input_v_rot(ggml_tensor *) const override {
    }

private:
    ggml_backend_buffer_t ensure_pool_buffer() const {
        if (pool == nullptr || pool->device_base == nullptr) {
            GGML_ABORT("llama_kv_cache_kapsl_context has no external pool");
        }
        if (pool_buffer != nullptr) {
            return pool_buffer;
        }

        const size_t pool_bytes =
            (size_t) pool->num_blocks *
            2 *
            pool->num_kv_heads *
            pool->block_size *
            pool->head_dim *
            ggml_type_size(GGML_TYPE_F16);

#if defined(GGML_USE_CUDA)
        pool_buffer = ggml_backend_cuda_buffer_from_device_ptr(
                (int) pool->device_id,
                pool->device_base,
                pool_bytes);
        if (pool_buffer == nullptr) {
            GGML_ABORT("failed to wrap Kapsl CUDA KV pool");
        }
        return pool_buffer;
#else
        GGML_UNUSED(pool_bytes);
        GGML_ABORT("Kapsl external KV pool requires GGML CUDA");
#endif
    }

    ggml_backend_buffer_t ensure_block_table_buffer() const {
        if (pool == nullptr || pool->block_table_device == nullptr) {
            GGML_ABORT("llama_kv_cache_kapsl_context has no Kapsl block table");
        }
        if (block_table_buffer != nullptr) {
            return block_table_buffer;
        }

        const size_t block_table_bytes =
            (size_t) pool->n_layers *
            pool->block_table_layer_stride *
            sizeof(uint32_t);

#if defined(GGML_USE_CUDA)
        block_table_buffer = ggml_backend_cuda_buffer_from_device_ptr(
                (int) pool->device_id,
                pool->block_table_device,
                block_table_bytes);
        if (block_table_buffer == nullptr) {
            GGML_ABORT("failed to wrap Kapsl CUDA block table");
        }
        return block_table_buffer;
#else
        GGML_UNUSED(block_table_bytes);
        GGML_ABORT("Kapsl external KV block table requires GGML CUDA");
#endif
    }

    ggml_tensor * make_pool_tensor(ggml_context * ctx, uint32_t kv_type) const {
        GGML_ASSERT(kv_type < 2);
        ggml_backend_buffer_t buffer = ensure_pool_buffer();

        ggml_tensor * tensor = ggml_new_tensor_4d(
                ctx,
                GGML_TYPE_F16,
                pool->head_dim,
                pool->block_size,
                pool->num_kv_heads,
                pool->num_blocks);

        const size_t kv_type_stride =
            (size_t) pool->num_kv_heads *
            pool->block_size *
            pool->head_dim *
            ggml_type_size(GGML_TYPE_F16);
        void * addr = (char *) pool->device_base + kv_type * kv_type_stride;
        if (ggml_backend_tensor_alloc(buffer, tensor, addr) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("failed to allocate Kapsl external KV tensor");
        }

        return tensor;
    }

    ggml_tensor * make_pool_base_tensor(ggml_context * ctx) const {
        ggml_backend_buffer_t buffer = ensure_pool_buffer();

        const int64_t n_elements =
            (int64_t) pool->num_blocks *
            2 *
            pool->num_kv_heads *
            pool->block_size *
            pool->head_dim;

        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        if (ggml_backend_tensor_alloc(buffer, tensor, pool->device_base) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("failed to allocate Kapsl external base KV tensor");
        }

        return tensor;
    }

    ggml_tensor * make_block_table_tensor(ggml_context * ctx) const {
        ggml_backend_buffer_t buffer = ensure_block_table_buffer();

        ggml_tensor * tensor = ggml_new_tensor_2d(
                ctx,
                GGML_TYPE_I32,
                pool->block_table_layer_stride,
                pool->n_layers);

        if (ggml_backend_tensor_alloc(buffer, tensor, pool->block_table_device) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("failed to allocate Kapsl external block table tensor");
        }

        return tensor;
    }

    static void set_input_positions(ggml_tensor * dst, const llama_ubatch * ubatch) {
        GGML_ASSERT(ubatch != nullptr);
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->ne[0] == ubatch->n_tokens);
        GGML_ASSERT(ubatch->pos != nullptr);
        GGML_ASSERT(ubatch->n_pos > 0);
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

        int32_t * data = (int32_t *) dst->data;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = ubatch->pos[i * ubatch->n_pos];
        }
    }

    llama_memory_status status;
    llama_kapsl_kv_pool_desc * pool = nullptr;
    uint32_t n_kv = 0;
    size_t i_cur = 0;
    std::vector<llama_ubatch> ubatches;
    mutable ggml_backend_buffer_t pool_buffer = nullptr;
    mutable ggml_backend_buffer_t block_table_buffer = nullptr;
};

} // namespace

llama_kv_cache_kapsl::llama_kv_cache_kapsl(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
llama_kapsl_kv_pool_desc *   pool,
                 uint64_t   session_id) :
    pool(pool),
    session_id(session_id) {
    if (pool == nullptr) {
        throw std::runtime_error("llama_kv_cache_kapsl requires a pool descriptor");
    }
    if (pool->dtype != LLAMA_KAPSL_KV_DTYPE_F16 || type_k != GGML_TYPE_F16 || type_v != GGML_TYPE_F16) {
        throw std::runtime_error("llama_kv_cache_kapsl currently supports f16 KV only");
    }
    if (pool->device_base == nullptr || pool->reserve == nullptr || pool->release == nullptr) {
        throw std::runtime_error("llama_kv_cache_kapsl pool descriptor is incomplete");
    }
    if (pool->n_layers != model.hparams.n_layer) {
        throw std::runtime_error("llama_kv_cache_kapsl block table layer count does not match model");
    }
    if (pool->max_blocks_per_seq == 0 || pool->block_table_layer_stride < pool->max_blocks_per_seq) {
        throw std::runtime_error("llama_kv_cache_kapsl block table geometry is invalid");
    }
    if (pool->num_kv_heads != model.hparams.n_head_kv() || pool->head_dim != model.hparams.n_embd_head_k()) {
        throw std::runtime_error("llama_kv_cache_kapsl pool geometry does not match model KV geometry");
    }

    LLAMA_LOG_INFO("%s: using external Kapsl KV pool: device=%u blocks=%u block_size=%u heads=%u head_dim=%u session=%llu\n",
            __func__,
            pool->device_id,
            pool->num_blocks,
            pool->block_size,
            pool->num_kv_heads,
            pool->head_dim,
            (unsigned long long) session_id);
}

llama_kv_cache_kapsl::~llama_kv_cache_kapsl() {
    release_session();
}

bool llama_kv_cache_kapsl::reserve_for_tokens(uint32_t tokens_needed) {
    if (tokens_needed == 0) {
        return true;
    }
    if (has_reservation && tokens_needed <= n_reserved_tokens) {
        if (pool->touch != nullptr) {
            return pool->touch(pool->user_data, session_id);
        }
        return true;
    }

    uint32_t * reserved_table = nullptr;
    uint32_t reserved_blocks = 0;
    if (!pool->reserve(pool->user_data, session_id, tokens_needed, &reserved_table, &reserved_blocks)) {
        LLAMA_LOG_ERROR("%s: Kapsl KV reserve failed: session=%llu tokens=%u\n",
                __func__, (unsigned long long) session_id, tokens_needed);
        return false;
    }
    if (reserved_table == nullptr || reserved_blocks == 0) {
        LLAMA_LOG_ERROR("%s: Kapsl KV reserve returned an empty block table: session=%llu tokens=%u\n",
                __func__, (unsigned long long) session_id, tokens_needed);
        return false;
    }

    block_table_device = reserved_table;
    n_reserved_blocks = reserved_blocks;
    n_reserved_tokens = tokens_needed;
    has_reservation = true;

    pool->block_table_device = reserved_table;
    LLAMA_LOG_DEBUG("%s: reserved Kapsl KV blocks: session=%llu tokens=%u blocks=%u table=%p\n",
            __func__, (unsigned long long) session_id, tokens_needed, reserved_blocks, (void *) reserved_table);
    return true;
}

void llama_kv_cache_kapsl::release_session() {
    if (!has_reservation || pool == nullptr || pool->release == nullptr) {
        return;
    }

    pool->release(pool->user_data, session_id);
    block_table_device = nullptr;
    n_reserved_blocks = 0;
    n_reserved_tokens = 0;
    has_reservation = false;
}

llama_memory_context_ptr llama_kv_cache_kapsl::make_status_context(llama_memory_status status) const {
    return std::make_unique<llama_kv_cache_kapsl_context>(
            status,
            pool,
            n_reserved_tokens);
}

llama_memory_context_ptr llama_kv_cache_kapsl::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    uint32_t tokens_needed = 0;
    while (true) {
        auto ubatch = balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            GGML_ASSERT(ubatch.pos != nullptr);
            GGML_ASSERT(ubatch.n_pos > 0);
            const llama_pos pos = ubatch.pos[i * ubatch.n_pos];
            if (pos >= 0) {
                tokens_needed = std::max(tokens_needed, (uint32_t) pos + 1);
            }
        }
        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens() || ubatches.empty()) {
        return make_status_context(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    if (!reserve_for_tokens(tokens_needed)) {
        return make_status_context(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    LLAMA_LOG_DEBUG("%s: Kapsl external KV reserved %u blocks for %u tokens\n",
            __func__, n_reserved_blocks, tokens_needed);
    return std::make_unique<llama_kv_cache_kapsl_context>(
            LLAMA_MEMORY_STATUS_SUCCESS,
            pool,
            n_reserved_tokens,
            std::move(ubatches));
}

llama_memory_context_ptr llama_kv_cache_kapsl::init_full() {
    return make_status_context(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_kapsl::init_update(llama_context *, bool) {
    return make_status_context(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_kv_cache_kapsl::get_can_shift() const {
    return false;
}

void llama_kv_cache_kapsl::clear(bool) {
    release_session();
}

bool llama_kv_cache_kapsl::seq_rm(llama_seq_id, llama_pos, llama_pos) {
    release_session();
    return true;
}

void llama_kv_cache_kapsl::seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) {
}

void llama_kv_cache_kapsl::seq_keep(llama_seq_id) {
}

void llama_kv_cache_kapsl::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
}

void llama_kv_cache_kapsl::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
}

llama_pos llama_kv_cache_kapsl::seq_pos_min(llama_seq_id) const {
    return -1;
}

llama_pos llama_kv_cache_kapsl::seq_pos_max(llama_seq_id) const {
    return -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kapsl::memory_breakdown() const {
    return {};
}

void llama_kv_cache_kapsl::state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const {
}

void llama_kv_cache_kapsl::state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {
}
