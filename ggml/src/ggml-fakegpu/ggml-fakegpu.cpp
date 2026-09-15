#include "ggml-fakegpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

static ggml_backend_reg g_ggml_backend_fakegpu_reg;
static ggml_backend_device g_ggml_backend_fakegpu_device;

//
// buffer
//

static void ggml_backend_fakegpu_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * ggml_backend_fakegpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t) buffer->context;

    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *) data;
}

static void ggml_backend_fakegpu_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void ggml_backend_fakegpu_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_fakegpu_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void ggml_backend_fakegpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_fakegpu_buffer_i = {
    /* .free_buffer   = */ ggml_backend_fakegpu_buffer_free,
    /* .get_base      = */ ggml_backend_fakegpu_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_fakegpu_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_fakegpu_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_fakegpu_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ ggml_backend_fakegpu_buffer_clear,
    /* .reset         = */ NULL,
};

//
// buffer type
//

static const char * ggml_backend_fakegpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "FAKEGPU";
}

static ggml_backend_buffer_t ggml_backend_fakegpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_fakegpu_buffer_i, data, size);
}

static size_t ggml_backend_fakegpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static bool ggml_backend_fakegpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

ggml_backend_buffer_type_t ggml_backend_fakegpu_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_fakegpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name       = */ ggml_backend_fakegpu_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_fakegpu_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_fakegpu_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_fakegpu_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_fakegpu_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_fakegpu_buffer_type;
}

//
// backend
//

static const char * ggml_backend_fakegpu_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FAKEGPU";
}

static void ggml_backend_fakegpu_free(ggml_backend_t backend) {
    free(backend);
}

static enum ggml_status ggml_backend_fakegpu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) ||
            node->op == GGML_OP_NONE ||
            node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const size_t nbytes = ggml_nbytes(node);
        if (node->data != NULL && nbytes > 0) {
            memset(node->data, 0, nbytes);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_fakegpu_i = {
    /* .get_name            = */ ggml_backend_fakegpu_name,
    /* .free                = */ ggml_backend_fakegpu_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_fakegpu_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_fakegpu_guid(void) {
    static ggml_guid guid = {
        0x8e, 0x42, 0x89, 0x6d, 0x7a, 0xcd, 0x4e, 0x11,
        0x93, 0x2e, 0x2a, 0x1b, 0x75, 0xc7, 0x68, 0x5f
    };

    return &guid;
}

//
// device
//

static const char * ggml_backend_fakegpu_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FAKEGPU0";
}

static const char * ggml_backend_fakegpu_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Fake GPU backend (zero-output compute)";
}

static void ggml_backend_fakegpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 8ull * 1024ull * 1024ull * 1024ull;
    *total = 8ull * 1024ull * 1024ull * 1024ull;
}

static enum ggml_backend_dev_type ggml_backend_fakegpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_fakegpu_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_fakegpu_device_get_name(dev);
    props->description = ggml_backend_fakegpu_device_get_description(dev);
    props->type = ggml_backend_fakegpu_device_get_type(dev);
    ggml_backend_fakegpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ggml_backend_fakegpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));
    if (backend == NULL) {
        return NULL;
    }

    backend->guid = ggml_backend_fakegpu_guid();
    backend->iface = ggml_backend_fakegpu_i;
    backend->device = dev;
    backend->context = NULL;

    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_fakegpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_fakegpu_buffer_type();
}

static bool ggml_backend_fakegpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return true;
}

static bool ggml_backend_fakegpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft == ggml_backend_fakegpu_buffer_type();
}

static const ggml_backend_device_i ggml_backend_fakegpu_device_i = {
    /* .get_name             = */ ggml_backend_fakegpu_device_get_name,
    /* .get_description      = */ ggml_backend_fakegpu_device_get_description,
    /* .get_memory           = */ ggml_backend_fakegpu_device_get_memory,
    /* .get_type             = */ ggml_backend_fakegpu_device_get_type,
    /* .get_props            = */ ggml_backend_fakegpu_device_get_props,
    /* .init_backend         = */ ggml_backend_fakegpu_device_init,
    /* .get_buffer_type      = */ ggml_backend_fakegpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_fakegpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fakegpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// backend reg
//

static const char * ggml_backend_fakegpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FAKEGPU";
}

static size_t ggml_backend_fakegpu_reg_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_fakegpu_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_ASSERT(index == 0);
    return &g_ggml_backend_fakegpu_device;
}

static const ggml_backend_reg_i ggml_backend_fakegpu_reg_i = {
    /* .get_name         = */ ggml_backend_fakegpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_fakegpu_reg_device_count,
    /* .get_device       = */ ggml_backend_fakegpu_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_fakegpu_reg(void) {
    static bool initialized = false;

    if (!initialized) {
        initialized = true;

        g_ggml_backend_fakegpu_reg.api_version = GGML_BACKEND_API_VERSION;
        g_ggml_backend_fakegpu_reg.iface = ggml_backend_fakegpu_reg_i;
        g_ggml_backend_fakegpu_reg.context = NULL;

        g_ggml_backend_fakegpu_device.iface = ggml_backend_fakegpu_device_i;
        g_ggml_backend_fakegpu_device.reg = &g_ggml_backend_fakegpu_reg;
        g_ggml_backend_fakegpu_device.context = NULL;
    }

    return &g_ggml_backend_fakegpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fakegpu_reg)
