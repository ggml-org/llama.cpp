#include "ggml-backend-execution.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <memory>
#include <atomic>

static std::atomic<size_t> execution_buffers{0};
bool ggml_backend_execution_buffers_present() { return execution_buffers.load(std::memory_order_acquire) != 0; }

struct execution_storage {
    ggml_backend_buffer_type type{};
    ggml_backend_buffer_ptr backing;
    ggml_backend_execution_ops ops;
    void * context;
};
static execution_storage & storage(ggml_backend_buffer_t buffer) { return *static_cast<execution_storage *>(buffer->context); }
static const char * execution_name(ggml_backend_buffer_type_t) { return "Execution_Host"; }
static bool metadata_only(const ggml_tensor * op) {
    return op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE;
}
static ggml_tensor backing_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor) {
    auto result = *tensor;
    result.buffer = storage(buffer).backing.get();
    // Byte I/O resolves view_src before buffer; detach the routing alias while preserving its absolute address.
    result.view_src = nullptr; result.view_offs = 0;
    return result;
}

// The wrapper is deliberately not a host fallback: only opted-in device dispatch may execute its tensors.
ggml_backend_buffer_t ggml_backend_execution_buffer_new(ggml_backend_dev_t device,ggml_backend_buffer_t backing,
        const ggml_backend_execution_ops & ops,void * context) {
    if (!device || !backing || !ggml_backend_buffer_is_host(backing) || backing->iface.reset || !ops.supports || !ops.compute ||
            !ops.can_write || !ops.modified || !ops.destroy) return nullptr;
    try {
        auto s = std::make_unique<execution_storage>();
        s->backing.reset(ggml_backend_buffer_retain(backing)); s->ops = ops; s->context = context;
        s->type.device = device; s->type.context = s.get();
        s->type.iface.get_name = execution_name;
        s->type.iface.alloc_buffer = [](ggml_backend_buffer_type_t,size_t) -> ggml_backend_buffer_t { return nullptr; };
        s->type.iface.get_alignment = [](ggml_backend_buffer_type_t type) {
            return std::max(size_t(128),ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(static_cast<execution_storage *>(type->context)->backing.get())));
        };
        ggml_backend_buffer_i iface{};
        iface.free_buffer = [](ggml_backend_buffer_t b) {
            auto * s = &storage(b); s->ops.destroy(s->context); delete s;
            execution_buffers.fetch_sub(1,std::memory_order_release);
        };
        iface.get_base = [](ggml_backend_buffer_t b) { return ggml_backend_buffer_get_base(storage(b).backing.get()); };
        iface.init_tensor = [](ggml_backend_buffer_t,ggml_tensor *) { return GGML_STATUS_SUCCESS; };
        iface.set_tensor = [](ggml_backend_buffer_t b,ggml_tensor * t,const void * p,size_t offset,size_t bytes) {
            auto & s = storage(b); GGML_ASSERT(s.ops.can_write(s.context));
            auto alias = backing_tensor(b,t); ggml_backend_tensor_set(&alias,p,offset,bytes); s.ops.modified(s.context);
        };
        iface.get_tensor = [](ggml_backend_buffer_t b,const ggml_tensor * t,void * p,size_t offset,size_t bytes) {
            auto alias = backing_tensor(b,t); ggml_backend_tensor_get(&alias,p,offset,bytes);
        };
        iface.memset_tensor = [](ggml_backend_buffer_t b,ggml_tensor * t,uint8_t value,size_t offset,size_t bytes) {
            auto & s = storage(b); GGML_ASSERT(s.ops.can_write(s.context));
            auto alias = backing_tensor(b,t); ggml_backend_tensor_memset(&alias,value,offset,bytes); s.ops.modified(s.context);
        };
        iface.clear = [](ggml_backend_buffer_t b,uint8_t value) {
            auto & s = storage(b); GGML_ASSERT(s.ops.can_write(s.context));
            ggml_backend_buffer_clear(s.backing.get(),value); s.ops.modified(s.context);
        };
        auto * result = ggml_backend_buffer_init(&s->type,iface,s.get(),ggml_backend_buffer_get_size(backing));
        execution_buffers.fetch_add(1,std::memory_order_release);
        s.release(); return result;
    } catch (const std::bad_alloc &) { return nullptr; }
}

bool ggml_backend_buft_is_execution(ggml_backend_buffer_type_t type) { return type && type->iface.get_name == execution_name; }

// A graph node may use one managed storage owner, never combine unrelated execution domains implicitly.
bool ggml_backend_execution_owner(const ggml_tensor * op,ggml_backend_buffer_t & owner) {
    owner = nullptr;
    if (!op) return false;
    if (!ggml_backend_execution_buffers_present()) return true;
    const auto accept = [&](ggml_backend_buffer_t buffer) {
        while (buffer && !ggml_backend_buft_is_execution(buffer->buft)) buffer = buffer->parent;
        if (!buffer) return true;
        if (owner && owner != buffer) return false;
        owner = buffer; return true;
    };
    if (!accept(op->buffer)) return false;
    if (op->view_src && !accept(op->view_src->buffer)) return false;
    for (auto * source : op->src) if (source &&
            (!accept(source->buffer) || (source->view_src && !accept(source->view_src->buffer)))) return false;
    return true;
}
bool ggml_backend_execution_supports(ggml_backend_buffer_t owner,ggml_backend_dev_t device,const ggml_tensor * op) {
    return owner && op && ggml_backend_buft_is_execution(owner->buft) && owner->buft->device == device &&
        (metadata_only(op) || storage(owner).ops.supports(storage(owner).context,op));
}
ggml_status ggml_backend_execution_compute(ggml_backend_buffer_t owner,ggml_backend_t backend,ggml_tensor * op) {
    if (!backend || !ggml_backend_execution_supports(owner,ggml_backend_get_device(backend),op)) return GGML_STATUS_FAILED;
    if (metadata_only(op)) return GGML_STATUS_SUCCESS;
    return storage(owner).ops.compute(storage(owner).context,backend,op);
}
