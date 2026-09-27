#include "llama-kv-stream-mtp-proxy.h"

#include "llama-kv-stream-logical-cache.h"
#include "llama-kv-stream-writer.h"
#include "llama-kv-stream-model.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-kv-stream-device.h"

#include <algorithm>
#include <cstring>
#include <vector>

struct llama_kv_stream_mtp_proxy::implementation {
    std::shared_ptr<llama_kv_stream_logical_cache> cache;
    llama_kv_stream_model * target = nullptr;
    const ggml_tensor * pending_k = nullptr;
    ggml_backend_t pending_backend = nullptr;
    ggml_backend_buffer_ptr pending_owner;
    std::shared_ptr<llama_kv_stream_writer> writer;
    ggml_backend_buffer_t writer_scratch = nullptr;
    size_t writer_scratch_bytes = 0;
    using lease_ptr = std::unique_ptr<llama_kv_stream_complete_layer_lease,
        decltype(&llama_kv_stream_complete_layer_lease_free)>;
    lease_ptr provisional{nullptr, llama_kv_stream_complete_layer_lease_free};
    std::unique_ptr<llama_memory_completion> attention_completion;
    llama_kv_stream_population_stats staged;
    bool pending_v = false;
    bool staged_pending = false;
    size_t pending_first = 0;
    uint32_t pending_rows = 0;
    size_t first = 0;
    uint32_t rows = 0;
    size_t calls = 0;
    bool armed = false;
    bool failed = false;
    bool initialized = false;
    bool plane(const ggml_tensor * tensor, bool & value) const {
        llama_kv_stream_host_layer host;
        if (!tensor || !cache->host()->layer(0, host)) return false;
        value = tensor->data == host.v;
        return tensor->data == host.k || value;
    }


    bool supports(const ggml_tensor * op) const {
        if (failed || !op ||
                !op->src[0] || !op->src[1] || !op->src[2]) return false;
        const auto & shape = cache->host()->config().shape;
        if (op->op == GGML_OP_SET_ROWS) {
            bool value = false;
            if (!plane(op->src[2], value)) return false;
            const size_t width = size_t(value ? shape.head_dim_v : shape.head_dim_k)*shape.heads;
            return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == int64_t(width) &&
                op->src[0]->ne[1] > 0 && op->src[0]->ne[1] <= 256 &&
                op->src[1]->type == GGML_TYPE_I64 && ggml_nelements(op->src[1]) == op->src[0]->ne[1];
        }
        if (op->op != GGML_OP_FLASH_ATTN_EXT || !op->src[3] || op->src[0]->ne[1] < 1 ||
                op->src[0]->ne[1] > 256 || (armed && op->src[0]->ne[1] != rows)) return false;
        return
            op->src[1]->type == shape.type_k && op->src[2]->type == shape.type_v &&
            op->src[1]->ne[0] == shape.head_dim_k && op->src[2]->ne[0] == shape.head_dim_v &&
            op->src[1]->ne[2] == shape.heads && op->src[2]->ne[2] == shape.heads &&
            op->src[3]->type == GGML_TYPE_F16;
    }

    ggml_status write(ggml_backend_t backend, ggml_tensor * op) {
        bool value = false;
        if (!plane(op->src[2], value)) return GGML_STATUS_FAILED;
        const size_t count = size_t(op->src[0]->ne[1]);
        const size_t next = cache->tokens();
        std::vector<int64_t> indices(count);
        ggml_backend_tensor_get(op->src[1], indices.data(), 0, count*sizeof(int64_t));
        for (size_t i = 0; i < count; ++i)
            if (indices[i] != int64_t(next + i)) return GGML_STATUS_FAILED;
        if (!value) {
            if (pending_k) return GGML_STATUS_FAILED;
            if (attention_completion && !attention_completion->synchronize()) {
                failed = true;
                return GGML_STATUS_FAILED;
            }
            attention_completion.reset();
            provisional.reset();
            pending_k = op->src[0];
            pending_backend = backend;
            pending_first = next;
            pending_rows = uint32_t(count);
            auto * owner = pending_k->view_src ? pending_k->view_src->buffer : pending_k->buffer;
            pending_owner.reset(ggml_backend_buffer_retain(owner));
            return pending_owner ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
        if (!pending_k || pending_v || op->src[3] != pending_k || pending_first != next || pending_rows != count ||
                (armed && (first != next || rows != count || !target->has_mtp_layer() ||
                    count > target->mtp_reserved_tokens() - next))) return GGML_STATUS_FAILED;
        auto * scratch = target->mtp_writer_workspace();
        if (!scratch) return GGML_STATUS_FAILED;
        const size_t bytes = ggml_backend_buffer_get_size(scratch);
        if (!writer || writer_scratch != scratch || writer_scratch_bytes != bytes) {
            writer.reset();
            writer = std::shared_ptr<llama_kv_stream_writer>(
                llama_kv_stream_writer::create(backend, scratch, bytes,
                    cache->host()->config().shape, 256));
            if (!writer) return GGML_STATUS_FAILED;
            writer_scratch = scratch;
            writer_scratch_bytes = bytes;
        }
        // Both encoded planes are published to host asynchronously. When the
        // retained layer is armed, the same encoded tiles go directly to its
        // protected device spans on this ordered backend stream.
        staged = {};
        const llama_kv_stream_logical_cache::generated_stage publish = armed ?
            [this, backend, next](bool is_value, const ggml_tensor * encoded, size_t row, size_t rows) {
                return target->stage_mtp_tail_async(
                    backend, next, is_value, encoded, row, rows, staged);
            } : llama_kv_stream_logical_cache::generated_stage{};
        if (!cache->begin_generated(writer, pending_k, op->src[0], publish)) {
            failed = true;
            return GGML_STATUS_FAILED;
        }
        pending_v = true;
        staged_pending = armed;
        return GGML_STATUS_SUCCESS;
    }

    bool finish_write() {
        if (!pending_k || !pending_v || !cache->complete_generated()) return false;
        pending_k = nullptr;
        pending_owner.reset();
        pending_backend = nullptr;
        pending_v = false;
        const bool staged_now = staged_pending;
        staged_pending = false;
        return !staged_now || target->advance_mtp_layer_tail_staged(staged);
    }

    // A suffix reset is also the abort path for a half-built K/V pair. Drain
    // both consumers before source buffers, scratch, or the ring guard can move.
    bool cancel_pending(bool & canceled) {
        canceled = false;
        const bool had_pending = pending_k != nullptr;
        try {
            if (attention_completion && !attention_completion->synchronize()) {
                failed = true;
                return false;
            }
            attention_completion.reset();
            provisional.reset();
            if (pending_v) {
                if (!cache->cancel()) {
                    failed = true;
                    return false;
                }
                canceled = true;
            } else if (pending_k) {
                if (!pending_backend) {
                    failed = true;
                    return false;
                }
                ggml_backend_synchronize(pending_backend);
            }
        } catch (...) {
            failed = true;
            return false;
        }
        pending_k = nullptr;
        pending_owner.reset();
        pending_backend = nullptr;
        pending_v = false;
        staged_pending = false;
        staged = {};
        armed = false;
        if (had_pending) {
            writer.reset();
            writer_scratch = nullptr;
            writer_scratch_bytes = 0;
        }
        return true;
    }

    bool complete_publication() {
        if (failed || (pending_k && !finish_write())) {
            failed = true;
            return false;
        }
        if (attention_completion && !attention_completion->synchronize()) {
            failed = true;
            return false;
        }
        attention_completion.reset();
        provisional.reset();
        return !pending_k;
    }

    bool compute_gathered(ggml_backend_t backend, ggml_tensor * op,
            const ggml_kv_stream_partial_ops * ops, ggml_backend_buffer_t workspace) {
        const size_t padded = size_t(op->src[1]->ne[1]);
        const auto & shape = cache->host()->config().shape;
        ggml_kv_stream_layout layout;
        llama_kv_stream_host_layer host;
        if (!ops || ops->version < 4 || !ops->direct || !workspace ||
                padded < cache->tokens() || padded > cache->host()->config().context_tokens ||
                op->src[2]->ne[1] != int64_t(padded) ||
                ggml_kv_stream_layout_make(shape, padded, layout).status != ggml_kv_stream_status::success ||
                layout.bytes > ggml_backend_buffer_get_size(workspace) ||
                !cache->host()->layer(0, host)) return false;
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(workspace));
        if (!base) return false;
        const auto stage = [&](const void * source, size_t offset, size_t bytes) {
            ggml_tensor destination{};
            destination.type = GGML_TYPE_I8;
            destination.buffer = workspace;
            destination.data = base + offset;
            destination.ne[0] = int64_t(bytes);
            destination.nb[0] = 1;
            for (int i = 1; i < 4; ++i) {
                destination.ne[i] = 1;
                destination.nb[i] = bytes;
            }
            ggml_backend_tensor_set(&destination, source, 0, bytes);
        };
        stage(host.k, 0, layout.k_bytes);
        stage(host.v, layout.v_offset, layout.v_bytes);
        ggml_tensor k = *op->src[1], v = *op->src[2], direct = *op;
        k.buffer = v.buffer = workspace;
        k.data = base;
        v.data = base + layout.v_offset;
        k.view_src = v.view_src = nullptr;
        k.view_offs = v.view_offs = 0;
        direct.src[1] = &k;
        direct.src[2] = &v;
        if (!ops->direct(backend, &direct)) return false;
        return true;
    }

    ggml_status compute(ggml_backend_t backend, ggml_tensor * op) {
        if (!supports(op) || !backend || !target) return GGML_STATUS_FAILED;
        if (op->op == GGML_OP_SET_ROWS) return write(backend, op);
        if (pending_k && (!pending_v || (!armed && !finish_write()))) {
            failed = true;
            return GGML_STATUS_FAILED;
        }
        if (armed) {
            if (!pending_k || !staged_pending || cache->tokens() != first ||
                    pending_rows != rows || provisional) return GGML_STATUS_FAILED;
            provisional.reset(target->provisional_mtp_layer(rows, first + rows));
            if (!provisional) return GGML_STATUS_FAILED;
        } else if (!cache->tokens()) {
            return GGML_STATUS_FAILED;
        }
        auto * plan = armed ? llama_kv_stream_complete_layer_lease_plan(provisional.get()) : nullptr;
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            reg ? ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops") : nullptr);
        const auto * ops = get ? get() : nullptr;
        auto * workspace = target->mtp_attention_workspace();
        if (!armed) {
            if (!compute_gathered(backend, op, ops, workspace)) {
                LLAMA_LOG_WARN("%s: MTP gathered attention rejected at %zu tokens\n", __func__, cache->tokens());
                failed = true;
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }
        size_t required = 0;
        const bool planned = plan && ops && ops->version >= 7 && ops->spans &&
            ops->spans_workspace && ops->spans_workspace(backend, op, plan, required);
        const size_t capacity = workspace ? ggml_backend_buffer_get_size(workspace) : 0;
        if (!planned || !workspace || required > capacity) {
            LLAMA_LOG_WARN("%s: MTP TG%u span workspace rejected: planned=%d required=%zu available=%zu\n",
                __func__, rows, int(planned), required, capacity);
            failed = true;
            return GGML_STATUS_FAILED;
        }
        if (!ops->spans(backend, op, plan, workspace)) {
            LLAMA_LOG_WARN("%s: MTP TG%u span kernel rejected after workspace admission\n", __func__, rows);
            failed = true;
            return GGML_STATUS_FAILED;
        }
        // Retain the provisional plan until the attention event completes. The
        // host writer may finish while this kernel and downstream graph work run.
        attention_completion = llama_memory_completion::create(backend);
        if (!attention_completion || !attention_completion->record()) {
            ggml_backend_synchronize(backend);
            failed = true;
            return GGML_STATUS_FAILED;
        }
        ++calls;
        if (calls == 1)
            LLAMA_LOG_INFO("%s: adaptive MTP span attention active (first TG%u, reserved=%zu tokens)\n", __func__, rows, target->mtp_reserved_tokens());
        armed = false;
        return GGML_STATUS_SUCCESS;
    }
};

std::unique_ptr<llama_kv_stream_mtp_proxy> llama_kv_stream_mtp_proxy::create(
        ggml_backend_dev_t device, std::shared_ptr<llama_kv_stream_logical_cache> cache,
        llama_kv_stream_model * target) {
    if (!device || !cache || !target ||
            cache != target->auxiliary_cache() || cache->host()->config().layers != 1)
        return {};
    auto result = std::unique_ptr<llama_kv_stream_mtp_proxy>(new llama_kv_stream_mtp_proxy);
    result->impl = std::make_shared<implementation>();
    auto & state = *result->impl;
    state.cache = std::move(cache);
    state.target = target;
    const ggml_backend_execution_ops ops{
        [](void * p, const ggml_tensor * op) {
            return (**static_cast<std::shared_ptr<implementation> *>(p)).supports(op);
        },
        [](void * p, ggml_backend_t backend, ggml_tensor * op) {
            auto & state = **static_cast<std::shared_ptr<implementation> *>(p);
            try {
                const auto status = state.compute(backend, op);
                if (status != GGML_STATUS_SUCCESS) state.failed = true;
                return status;
            } catch (...) {
                state.failed = true;
                try { ggml_backend_synchronize(backend); } catch (...) {}
                return GGML_STATUS_FAILED;
            }
        },
        [](void *) { return false; },
        [](void *) {},
        [](void * p) { delete static_cast<std::shared_ptr<implementation> *>(p); },
    };
    auto holder = std::make_unique<std::shared_ptr<implementation>>(result->impl);
    result->proxy.reset(ggml_backend_execution_buffer_new(device, state.cache->host()->buffer(),
        ops, holder.get()));
    if (!result->proxy) return {};
    holder.release();
    result->context.reset(ggml_init({16384, nullptr, true}));
    if (!result->context) return {};
    const auto & host = state.cache->host()->config();
    llama_kv_stream_host_layer planes;
    if (!state.cache->host()->layer(0, planes)) return {};
    result->key_root = ggml_new_tensor_2d(result->context.get(), ggml_type(host.shape.type_k),
        host.shape.head_dim_k*host.shape.heads, host.context_tokens);
    result->value_root = ggml_new_tensor_2d(result->context.get(), ggml_type(host.shape.type_v),
        host.shape.head_dim_v*host.shape.heads, host.context_tokens);
    if (!result->key_root || !result->value_root ||
            ggml_backend_tensor_alloc(result->proxy.get(), result->key_root, planes.k) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(result->proxy.get(), result->value_root, planes.v) != GGML_STATUS_SUCCESS)
        return {};
    state.initialized = true;
    return result;
}

llama_kv_stream_mtp_proxy::~llama_kv_stream_mtp_proxy() {
    if (!impl || !impl->initialized) return;
    bool canceled = false;
    try {
        if (impl->cancel_pending(canceled) && impl->target)
            impl->target->release_mtp_layer();
    } catch (...) {
        // Context teardown must not throw; the target will release its owner.
    }
}
ggml_tensor * llama_kv_stream_mtp_proxy::key() const noexcept { return key_root; }
ggml_tensor * llama_kv_stream_mtp_proxy::value() const noexcept { return value_root; }
bool llama_kv_stream_mtp_proxy::complete_publication() {
    return impl && impl->complete_publication();
}

bool llama_kv_stream_mtp_proxy::arm(size_t first, uint32_t rows) {
    if (!impl || impl->failed || !impl->target || !impl->target->has_mtp_layer() ||
            !rows || rows > 4 || first != impl->cache->tokens() ||
            first > impl->target->mtp_reserved_tokens() || rows > impl->target->mtp_reserved_tokens() - first ||
            impl->target->mtp_layer_plan(1) == nullptr) return false;
    impl->first = first;
    impl->rows = rows;
    impl->armed = true;
    return true;
}
void llama_kv_stream_mtp_proxy::disarm() noexcept { if (impl) impl->armed = false; }
bool llama_kv_stream_mtp_proxy::active() const noexcept { return impl && impl->armed; }
size_t llama_kv_stream_mtp_proxy::attention_calls() const noexcept { return impl ? impl->calls : 0; }
bool llama_kv_stream_mtp_proxy::remove_suffix(size_t first, size_t last) {
    if (!impl || !impl->target || !impl->cache || last <= first) return last <= first;
    const size_t frontier = impl->cache->tokens();
    if (first < frontier && last < frontier) return false; // An interior hole is not a suffix.
    bool canceled = false;
    if (!impl->cancel_pending(canceled)) return false;
    // A canceled writer changed the logical generation without changing the
    // committed prefix. Its old lease must not be reused under that identity.
    if (canceled && !impl->target->release_mtp_layer()) {
        impl->failed = true;
        return false;
    }
    if (first >= frontier) {
        if (first == 0 && !impl->target->release_mtp_layer()) {
            impl->failed = true;
            return false;
        }
        if (first == 0) impl->failed = false;
        return true;
    }
    if (!impl->target->truncate_mtp_layer(first)) {
        impl->failed = true;
        return false;
    }
    if (first == 0) impl->failed = false;
    return true;
}
