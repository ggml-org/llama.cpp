#import "ggml-metal.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-device.h"
#import "ggml-metal-context.h"
#import "ggml-metal-ops.h"

// globals

// initialized in ggml_backend_metal_reg
static struct ggml_backend_reg    g_ggml_metal_reg;
static struct ggml_backend_device g_ggml_metal_device;

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_free(buffer->context);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    return ggml_metal_buffer_get_base(buffer->context);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_memset_tensor(buffer->context, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_set_tensor(buffer->context, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_get_tensor(buffer->context, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    return false;
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_clear(buffer->context, value);
}

static struct ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer     = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base        = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor      = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor      = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear           = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset           = */ NULL,
};

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_free(buffer->context);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    return ggml_metal_buffer_get_base(buffer->context);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_memset_tensor(buffer->context, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_set_tensor(buffer->context, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_get_tensor(buffer->context, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    return false;
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(!ggml_metal_buffer_is_shared(buffer->context));

    ggml_metal_buffer_clear(buffer->context, value);
}

static struct ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer     = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base        = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor      = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor      = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear           = */ ggml_backend_metal_buffer_private_clear,
    /* .reset           = */ NULL,
};

//
// buffer types
//

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_buffer_t res = ggml_metal_buffer_init(buft->device->context, size, shared);

    struct ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                if (ggml_metal_op_flash_attn_ext_use_vec(tensor)) {
                    res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                }
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_metal_device_get_props(buft->device->context).max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_shared_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_shared_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_shared_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_shared_is_host,
        },
        /* .device  = */ &g_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_metal;
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal_Private";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_metal_device_get_props(buft->device->context).max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_private_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_private_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_private_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_private_is_host,
        },
        /* .device  = */ &g_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_metal;
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    return ggml_metal_device_get_props(buft->device->context).max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(void) {
    // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
    //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_mapped_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_mapped_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_mapped_is_host,
        },
        /* .device  = */ &g_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_mapped_metal;
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    return "Metal";

    GGML_UNUSED(backend);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(backend->context);

    ggml_metal_free(backend->context);

    free(backend);
}

static void ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_synchronize(backend->context);
}

static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_set_tensor_async(backend->context, tensor, data, offset, size);
}

static void ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_get_tensor_async(backend->context, tensor, data, offset, size);
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    return false;

    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return ggml_metal_graph_compute(backend->context, cgraph);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_metal_graph_optimize(backend->context, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_set_n_cb(backend->context, n_cb);

}

static struct ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,

    // the events API is needed only for multi-GPU setups, so likely no need to implement it for Metal
    // in any case, these docs seem relevant if we ever decide to implement it:
    // https://developer.apple.com/documentation/metal/mtlcommandbuffer#Synchronizing-Passes-with-Events
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .optimize_graph          = */ ggml_backend_metal_graph_optimize,
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);

    ggml_metal_t ctx = ggml_metal_init(dev->context);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = malloc(sizeof(struct ggml_backend));

    *backend = (struct ggml_backend) {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_set_abort_callback(backend->context, abort_callback, user_data);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    return ggml_metal_supports_family(backend->context, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_capture_next_compute(backend->context);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    return "Metal";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    return ggml_metal_device_get_props(dev->context).name;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_get_memory(dev->context, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = (struct ggml_backend_dev_caps) {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_metal_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_t ctx = ggml_metal_init(dev->context);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = malloc(sizeof(struct ggml_backend));

    *backend = (struct ggml_backend) {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    const struct ggml_metal_device_props props_dev = ggml_metal_device_get_props(dev->context);

    return props_dev.use_shared_buffers ? ggml_backend_metal_buffer_type_shared() : ggml_backend_metal_buffer_type_private();
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = ggml_metal_buffer_map(dev->context, ptr, size, max_tensor_size);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ggml_metal_device_supports_op(dev->context, op);
}

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name;

    GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= min_batch_size;

    GGML_UNUSED(dev);
    GGML_UNUSED(op);
}

static struct ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return "Metal";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    return &g_ggml_metal_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static struct ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }

    return NULL;

    GGML_UNUSED(reg);
}
static struct ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .device_count     = */ ggml_backend_metal_reg_device_count,
    /* .device_get       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    {
        g_ggml_metal_reg = (struct ggml_backend_reg) {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_metal_reg_i,
            /* .context     = */ NULL,
        };

        g_ggml_metal_device = (struct ggml_backend_device) {
            /* .iface   = */ ggml_backend_metal_device_i,
            /* .reg     = */ &g_ggml_metal_reg,
            /* .context = */ ggml_metal_device_get(),
        };
    }

    return &g_ggml_metal_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
