#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-openvino.h"
#include "ggml-openvino/utils.h"
#include "ggml.h"

#include <mutex>
#include <openvino/openvino.hpp>
#include <set>
#include <string>
#include <vector>

#define GGML_OPENVINO_MAX_STREAMS 8

struct ggml_backend_openvino_context {
    int device;                          // the device ID currently in use
    std::string name;                    // context Name
    std::string description;             // context description

    // OpenVINO core components
    ov::Core core;                       // OpenVINO core interface
    std::shared_ptr<ov::CompiledModel> model; // compiled Model
    ov::InferRequest infer_request;      // inference Request

    // OpenVINO Multi-stream support
    static const int MAX_STREAMS = 8;    // define the maximum number of flows
    std::vector<ov::InferRequest> streams; // used to support multi-stream reasoning
    int current_stream;                  // the currently active stream index

    // state Management
    bool is_initialized;                 // initialize

    ggml_backend_openvino_context()
        : device(0), name("OpenVINO"), description("OpenVINO Backend Context"),
          current_stream(0), is_initialized(false) {}
};

static void ggml_backend_openvino_free(ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *)backend->context;
    delete ctx;
    delete backend;
}

static const char * ggml_backend_openvino_get_name(ggml_backend_t backend) {
    return GGML_OPENVINO_NAME;
    GGML_UNUSED(backend);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(backend);
}

static enum ggml_status
ggml_backend_openvino_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph) {
    openvino_frontend_compute(backend, cgraph);

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static const ggml_backend_i ggml_backend_openvino_interface = {
    /* .get_name                = */ ggml_backend_openvino_get_name,
    /* .free                    = */ ggml_backend_openvino_free,
    /* .get_default_buffer_type = */ ggml_backend_openvino_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_openvino_graph_compute,
    /* .supports_op             = */ NULL,
    /* .supports_buft           = */ NULL,
    /* .offload_op              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

int ggml_backend_openvino_get_device_count() {
    return ggml_openvino_info().device_count;
}

static ggml_guid_t ggml_backend_openvino_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

// backend API
GGML_API ggml_backend_t ggml_backend_openvino_init(int device) {
    if (device < 0 || device >= ggml_backend_openvino_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_openvino_context * ctx = new ggml_backend_openvino_context;
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t openvino_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_openvino_guid(),
        /* .interface = */ ggml_backend_openvino_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), device),
        /* .context   = */ ctx,
    };

    return openvino_backend;
}

GGML_API bool ggml_backend_is_openvino(ggml_backend_t backend) {
    GGML_ASSERT(backend->context != nullptr);
    return true;
}

// device buffer
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_buffer_type(int device) {
    GGML_ASSERT(device >= 0);
    return nullptr;
}

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_split_buffer_type(const float *tensor_split) {
    GGML_ASSERT(tensor_split != nullptr);
    return nullptr;
}

// pinned host buffer for use with the CPU backend for faster copies between CPU
// and GPU
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_host_buffer_type(void) { return nullptr;}


struct ggml_backend_openvino_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_openvino_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}
static bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_openvino_buffer_type_get_name;
}


static const char * ggml_backend_openvino_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_OPENVINO_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_openvino_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_openvino_split_buffer_type_get_name;
}

struct ggml_backend_openvino_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_openvino_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_openvino_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *)dev->context;
    return ctx->description.c_str();
}

// TODO
static void ggml_backend_openvino_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(dev->context != nullptr);
    GGML_ASSERT(free != nullptr);
    GGML_ASSERT(total != nullptr);
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *)dev->context;
    // Placeholder
    GGML_ASSERT(ctx->device >= 0);
    // ggml_openvino_set_device(ctx->device);
}

static enum ggml_backend_dev_type ggml_backend_openvino_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_CPU;
    // return GGML_BACKEND_DEVICE_TYPE_GPU_FULL;
}

static void ggml_backend_openvino_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_openvino_device_get_name(dev);
    props->description = ggml_backend_openvino_device_get_description(dev);
    props->type        = ggml_backend_openvino_device_get_type(dev);
    ggml_backend_openvino_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_OPENVINO_NO_PINNED") == nullptr;
#ifdef GGML_OPENVINO_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_openvino_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *)dev->context;
    return ggml_backend_openvino_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *)dev->context;
    return ggml_backend_openvino_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_openvino_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_openvino_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static ggml_backend_buffer_t ggml_backend_openvino_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);

  static const std::set<ggml_op> supported_ops{
      GGML_OP_ADD,       GGML_OP_MUL,      GGML_OP_MUL_MAT, GGML_OP_VIEW,
      GGML_OP_CONT,      GGML_OP_CPY,      GGML_OP_RESHAPE, GGML_OP_PERMUTE,
      GGML_OP_TRANSPOSE, GGML_OP_GET_ROWS, GGML_OP_ROPE,    GGML_OP_RMS_NORM,
      GGML_OP_SCALE,     GGML_OP_SOFT_MAX,
  };
  static const std::set<ggml_unary_op> supported_unary_ops{
      GGML_UNARY_OP_SILU,
  };

  if (op->op == GGML_OP_UNARY) {
      return supported_unary_ops.find(ggml_get_unary_op(op)) !=
             supported_unary_ops.end();
  }
  return supported_ops.find(op->op) != supported_ops.end();
}

static bool ggml_backend_openvino_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_openvino_device_interface = {
    /* .get_name             = */ ggml_backend_openvino_device_get_name,
    /* .get_description      = */ ggml_backend_openvino_device_get_description,
    /* .get_memory           = */ ggml_backend_openvino_device_get_memory,
    /* .get_type             = */ ggml_backend_openvino_device_get_type,
    /* .get_props            = */ ggml_backend_openvino_device_get_props,
    /* .init_backend         = */ ggml_backend_openvino_device_init,
    /* .get_buffer_type      = */ ggml_backend_openvino_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_openvino_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_openvino_device_supports_op,
    /* .supports_buft        = */ ggml_backend_openvino_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct ggml_backend_openvino_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_openvino_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_OPENVINO_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_openvino_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_openvino_info().device_count;
    GGML_UNUSED(reg);

    // TODO
    ggml_backend_openvino_reg_context * ctx = (ggml_backend_openvino_reg_context *)reg->context;

    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_openvino_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_openvino_reg_context * ctx = (ggml_backend_openvino_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
    // GGML_ASSERT(index == 0);

    // static ggml_backend_device ggml_backend_openvino_device = {
    //     /* .iface   = */ ggml_backend_openvino_device_interface,
    //     /* .reg     = */ reg,
    //     /* .context = */ nullptr,
    // };

    // return &ggml_backend_openvino_device;

    // GGML_UNUSED(reg);
    // GGML_UNUSED(index);
}

static void * ggml_backend_openvino_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_openvino_split_buffer_type;
    }
    // if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
    //     return (void *)ggml_backend_openvino_register_host_buffer;
    // }
    // if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
    //     return (void *)ggml_backend_openvino_unregister_host_buffer;
    // }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_openvino_reg_interface = {
    /* .get_name         = */ ggml_backend_openvino_reg_get_name,
    /* .get_device_count = */ ggml_backend_openvino_reg_get_device_count,
    /* .get_device       = */ ggml_backend_openvino_reg_get_device,
    /* .get_proc_address = */ ggml_backend_openvino_get_proc_address,
};

static int get_openvino_device_count() {
    ov::Core core;
    auto devices = core.get_available_devices();
    // return devices.size();
    return 1;
}

static ggml_openvino_device_info ggml_openvino_init() {
    ggml_openvino_device_info info = {};
    // TODO
    info.device_count = get_openvino_device_count();
    return info;
}

const ggml_openvino_device_info & ggml_openvino_info() {
    static ggml_openvino_device_info info = ggml_openvino_init();
    return info;
}

GGML_API ggml_backend_reg_t ggml_backend_openvino_reg(void) {
    static ggml_backend_reg reg;

    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_openvino_reg_context * ctx = new ggml_backend_openvino_reg_context;

            // GGML_LOG_DEBUG("ggml_openvino_info().device_count = %d \n", ggml_openvino_info().device_count);
            for (int i = 0; i < ggml_openvino_info().device_count; i++) {
                ggml_backend_openvino_device_context * dev_ctx = new ggml_backend_openvino_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_OPENVINO_NAME + std::to_string(i);

                // ggml_openvino_set_device(i);
                dev_ctx->description = ov::get_openvino_version().description;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .interface = */ ggml_backend_openvino_device_interface,
                    /* .reg       = */ &reg,
                    /* .context   = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .interface = */ ggml_backend_openvino_reg_interface,
                /* .context   = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}