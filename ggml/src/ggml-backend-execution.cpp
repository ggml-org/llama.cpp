#include "ggml-backend-execution.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <memory>
#include <atomic>
#include <cstring>
#include <climits>

static std::atomic<size_t> execution_buffers{0};
bool ggml_backend_execution_buffers_present() { return execution_buffers.load(std::memory_order_acquire) != 0; }

static constexpr size_t external_workspace_param = GGML_MAX_OP_PARAMS/sizeof(int32_t)-1;
static_assert(external_workspace_param > 4, "private dispatch metadata must not overlap stock attention parameters");

void ggml_backend_execution_set_external_workspace(ggml_tensor * op, bool external) {
    GGML_ASSERT(op && op->op == GGML_OP_FLASH_ATTN_EXT);
    // Keep dispatch metadata outside the stock arithmetic and sparse-mask words.
    const int32_t value = external ? 1 : 0;
    std::memcpy(op->op_params + external_workspace_param, &value, sizeof(value));
}

bool ggml_backend_execution_has_external_workspace(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    int32_t value = 0;
    std::memcpy(&value, op->op_params + external_workspace_param, sizeof(value));
    return value == 1 || value == 2;
}
void ggml_backend_execution_set_verified_workspace(ggml_tensor * op) {
    GGML_ASSERT(op && op->op == GGML_OP_FLASH_ATTN_EXT);
    const int32_t value=2;
    std::memcpy(op->op_params+external_workspace_param,&value,sizeof(value));
}
bool ggml_backend_execution_has_verified_workspace(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    int32_t value=0;
    std::memcpy(&value,op->op_params+external_workspace_param,sizeof(value));
    return value == 2;
}

void ggml_backend_execution_set_stock_conversion(ggml_tensor * op, bool enabled) {
    GGML_ASSERT(op && op->op == GGML_OP_FLASH_ATTN_EXT);
    const int32_t value=enabled ? 1 : 0;
    std::memcpy(op->op_params+external_workspace_param-1,&value,sizeof(value));
}
bool ggml_backend_execution_has_stock_conversion(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    int32_t value=0;
    std::memcpy(&value,op->op_params+external_workspace_param-1,sizeof(value));
    return value == 1;
}

bool ggml_backend_execution_mask_interval_encode(const ggml_fp16_t * mask, size_t tokens,
        ggml_backend_execution_mask_interval & interval) {
    if (!mask || !tokens || tokens > INT32_MAX) return false;
    ggml_backend_execution_mask_interval next{0,0,int32_t(tokens),1};
    bool started=false, ended=false;
    for (size_t i=0; i<tokens; ++i) {
        if (mask[i] == 0) {
            if (ended) return false;
            if (!started) { next.begin=int32_t(i); started=true; }
            next.end=int32_t(i+1);
        } else {
            if (mask[i] != 0xfc00) return false;
            ended=started;
        }
    }
    interval=next;
    return true;
}

ggml_fp16_t ggml_backend_execution_mask_interval_value(const ggml_backend_execution_mask_interval & interval, int32_t token) {
    const bool valid=interval.version == 1 && interval.tokens > 0 && interval.begin >= 0 &&
        interval.begin <= interval.end && interval.end <= interval.tokens;
    return valid && token >= interval.begin && token < interval.end ? 0 : 0xfc00;
}

bool ggml_backend_execution_mask_metadata(const ggml_tensor * op, ggml_tensor & logical, ggml_tensor & mask) {
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT || op->src[3] || op->src[4] ||
            !op->src[0] || !op->src[1] || !op->src[5]) return false;
    const auto * q=op->src[0], * k=op->src[1], * bounds=op->src[5];
    if (q->ne[1] <= 0 || q->ne[1] > INT32_MAX || k->ne[1] <= 0 || k->ne[1] > INT32_MAX ||
            q->ne[3] != 1 || bounds->type != GGML_TYPE_I32 || bounds->ne[0] != 4 || bounds->ne[1] != q->ne[1] ||
            bounds->ne[2] != 1 || bounds->ne[3] != 1 || !ggml_is_contiguous(bounds)) return false;
    ggml_tensor next{};
    next.type=GGML_TYPE_F16; next.ne[0]=k->ne[1]; next.ne[1]=q->ne[1]; next.ne[2]=next.ne[3]=1;
    next.nb[0]=sizeof(ggml_fp16_t);
    for (int i=1; i<4; ++i) {
        if (size_t(next.ne[i-1]) > SIZE_MAX/next.nb[i-1]) return false;
        next.nb[i]=next.nb[i-1]*size_t(next.ne[i-1]);
    }
    mask=next; logical=*op; logical.src[3]=&mask;
    return true;
}

bool ggml_backend_execution_set_mask_bounds(ggml_tensor * op, ggml_tensor * bounds) {
    if (!op) return false;
    auto candidate=*op;
    candidate.src[5]=bounds;
    ggml_tensor logical{}, mask{};
    if (!ggml_backend_execution_mask_metadata(&candidate,logical,mask)) return false;
    op->src[5]=bounds;
    return true;
}

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
    const bool compact=op->op == GGML_OP_FLASH_ATTN_EXT && op->src[5];
    if (!ggml_backend_execution_buffers_present()) return !compact;
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
    return !compact || owner;
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
