#include "ggml-openvino.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-openvino/utils.h"

#include <string>
#include <mutex>
#include <vector>
#include <openvino/openvino.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/opsets/opset1.hpp>

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

static void ggml_backend_openvino_add_forward(ggml_tensor * dst) {
    // Step 1: get the input tensor src0 和 src1
    const struct ggml_tensor *src0 = dst->src[0];
    const struct ggml_tensor *src1 = dst->src[1];

    ov::Core core;

    // set the shape and stride of dst
    dst->ne[0] = src0->ne[0];
    dst->ne[1] = src0->ne[1];
    dst->nb[0] = src0->nb[0];
    dst->nb[1] = src0->nb[1];

    if (src0 == nullptr || src1 == nullptr) {
        std::cerr << "Error: src0 or src1 is null." << std::endl;
        return;
    }

    // Step 2: Check that the input tensor types and shapes match
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        std::cerr << "Error: Unsupported tensor type. Only GGML_TYPE_F32 is supported for OpenVINO." << std::endl;
        return;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[1] != src1->ne[1]) {
        std::cerr << "Error: src0 and src1 shapes do not match." << std::endl;
        return;
    }

    ov::Tensor input0 = ov::Tensor(ov::element::f32, {static_cast<size_t>(src0->ne[0]), static_cast<size_t>(src0->ne[1])}, src0->data);
    ov::Tensor input1 = ov::Tensor(ov::element::f32, {static_cast<size_t>(src1->ne[0]), static_cast<size_t>(src1->ne[1])}, src1->data);

    auto input0_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{static_cast<size_t>(src0->ne[0]), static_cast<size_t>(src0->ne[1])});
    auto input1_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{static_cast<size_t>(src0->ne[0]), static_cast<size_t>(src0->ne[1])});
    auto add = std::make_shared<ov::op::v1::Add>(input0_param, input1_param);
    auto model = std::make_shared<ov::Model>(add, ov::ParameterVector{input0_param, input1_param});

    // compile model and store in context
#ifdef  GGML_OPENVINO_GPU
    auto compiled_model = core.compile_model(model, "GPU");
#elif   GGML_OPENVINO_NPU
    auto compiled_model = core.compile_model(model, "NPU");
#else
    auto compiled_model = core.compile_model(model, "CPU");
#endif
    // initialize infer request
    auto infer_request = compiled_model.create_infer_request();

    // Step 4:  set input data, copy src0 and src1 data to OpenVINO input tensors
    infer_request.set_tensor(input0_param, input0);
    infer_request.set_tensor(input1_param, input1);

    // Step 5: execute inference
    infer_request.infer();

    // Step 6: get output data
    ov::Tensor output = infer_request.get_tensor(compiled_model.output());

    // // Allocate memory for dst->data if not already allocated
    // if (dst->data == nullptr) {
    //     dst->data = malloc(dst->nb[0] * dst->ne[0]);
    //     if (dst->data == nullptr) {
    //         std::cerr << "Error: Failed to allocate memory for dst->data." << std::endl;
    //         return;
    //     }
    // }

    std::memcpy(dst->data, output.data(), output.get_byte_size());

    if (dst->ne[0] != src0->ne[0] || dst->ne[1] != src0->ne[1]) {
        std::cerr << "Error: dst tensor shape does not match input tensor shape." << std::endl;
        return;
    }

    // float* dst_data1 = (float*)(dst->data);
    // printf("Output data:");;
    // for (int i = 0; i < (10 < (int)(dst->ne[0]) ? 10 : (int)(dst->ne[0])); ++i) {
    //     printf("%f ", dst_data1[i]);
    // }
    // printf("\n");
    // fflush(stdout);
}

static void ggml_backend_openvino_mul_forward(ggml_tensor * dst) {
    struct ggml_tensor *src0 = dst->src[0];
    struct ggml_tensor *src1 = dst->src[1];

    ov::Core core;

    // define shape
    ov::Shape shape0 = {static_cast<size_t>(src0->ne[1]), static_cast<size_t>(src0->ne[0])};  // For Example: [7, 3072]
    ov::Shape shape1 = {static_cast<size_t>(src1->ne[1]), static_cast<size_t>(src1->ne[0])};  // For Example: [1, 3072] -> broadcast to [7, 3072]

    // create OpenVINO tensor (src0 and src1)
    ov::Tensor tensor0(ov::element::f32, shape0, src0->data);
    ov::Tensor tensor1(ov::element::f32, shape1, src1->data);

    // define input parameters
    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape0);
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape1);

    // create a multiply operation using broadcasting
    auto multiply = std::make_shared<ov::op::v1::Multiply>(input0, input1);

    // create model
    auto model = std::make_shared<ov::Model>(multiply, ov::ParameterVector{input0, input1});
    // compile model and store in context
#ifdef  GGML_OPENVINO_GPU
    ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
#elif   GGML_OPENVINO_NPU
    ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
#else
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
#endif

    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_tensor(input0, tensor0);
    infer_request.set_tensor(input1, tensor1);

    infer_request.infer();

    // get output tensor and copy it back to dst->data
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    std::memcpy(dst->data, output_tensor.data<float>(), src0->ne[0] * src0->ne[1] * sizeof(float));
}

static void ggml_backend_openvino_add(ggml_tensor * dst) {
    // Placeholder for OpenVINO add operation
    // GGML_ASSERT(ctx.device != 0);
    GGML_ASSERT(dst->data != nullptr);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                if (src1->type == GGML_TYPE_F16) {
                    // ggml_backend_openvino_add_forward(ctx, dst, src0, src1);
                } else if (src1->type == GGML_TYPE_F32) {
                    // ggml_compute_forward_add_f16_f32(params, dst);
                } else {
                    GGML_ABORT("fatal error");
                }
            } break;
        case GGML_TYPE_F32:
            {
                if (src1->type == GGML_TYPE_F32) {
                    {
                        ggml_backend_openvino_add_forward(dst);
                    }
                }
                else {
                    GGML_ABORT("fatal error");
                }
            } break;
        default:
            GGML_ABORT("%s: unsupported type %d\n", __func__, src1->type);
    }

}

static void ggml_backend_openvino_mul(ggml_tensor * dst) {
    GGML_ASSERT(dst->data != nullptr);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && "only f32 src1 supported for now");

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_backend_openvino_mul_forward(dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_compute_forward_get_rows_f16(struct ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];
    const struct ggml_tensor *src1 = dst->src[1];

    ov::Core core;

    ov::Shape shape0 = {static_cast<size_t>(src0->ne[1]), static_cast<size_t>(src0->ne[0])};  // [3072, 7]
    ov::Shape shape1 = {static_cast<size_t>(src1->ne[0])};  // [7]

    ov::Tensor tensor0(ov::element::f16, shape0, src0->data);
    ov::Tensor tensor1(ov::element::i32, shape1, src1->data);

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape0);
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape1);

    auto gather = std::make_shared<ov::op::v8::Gather>(input0, input1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));

    auto model = std::make_shared<ov::Model>(gather, ov::ParameterVector{input0, input1});
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_tensor(input0, tensor0);
    infer_request.set_tensor(input1, tensor1);

    infer_request.infer();

    ov::Tensor output_tensor = infer_request.get_output_tensor();
    // Convert output tensor data type from f16 to f32
    ov::Tensor output_tensor_f32 = ov::Tensor(ov::element::f32, output_tensor.get_shape());
    for (size_t i = 0; i < output_tensor.get_size(); ++i) {
        output_tensor_f32.data<float>()[i] = static_cast<float>(output_tensor.data<ov::float16>()[i]);
    }

    // Copy the converted data to dst->data
    std::memcpy(dst->data, output_tensor_f32.data<float>(), output_tensor_f32.get_byte_size());
}

void ggml_compute_forward_get_rows_f32(struct ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];
    const struct ggml_tensor *src1 = dst->src[1];

    ov::Core core;

    ov::Shape shape0 = {static_cast<size_t>(src0->ne[1]), static_cast<size_t>(src0->ne[0])};  // [3072, 7]
    ov::Shape shape1 = {static_cast<size_t>(src1->ne[0])};  // [7]

    ov::Tensor tensor0(ov::element::f32, shape0, src0->data);
    ov::Tensor tensor1(ov::element::i32, shape1, src1->data);

    auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape0);
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape1);

    auto gather = std::make_shared<ov::op::v8::Gather>(input0, input1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));

    auto model = std::make_shared<ov::Model>(gather, ov::ParameterVector{input0, input1});
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_tensor(input0, tensor0);
    infer_request.set_tensor(input1, tensor1);

    infer_request.infer();

    ov::Tensor output_tensor = infer_request.get_output_tensor();

    // Copy the converted data to dst->data
    std::memcpy(dst->data, output_tensor.data<float>(), output_tensor.get_byte_size());
}

void ggml_compute_forward_get_rows(struct ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];
    const struct ggml_tensor *src1 = dst->src[1];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_f32(dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

}

static enum ggml_status ggml_backend_openvino_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE || ggml_is_empty(node)) {
            return GGML_STATUS_SUCCESS;
        }

        switch (node->op) {
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_VIEW:
                break;
            case GGML_OP_ADD:
                {
                    ggml_backend_openvino_add(node);
                } break;
            case GGML_OP_MUL:
                {
                    ggml_backend_openvino_mul(node);
                } break;
            case GGML_OP_MUL_MAT:
                break;
            case GGML_OP_GET_ROWS:
                {
                    ggml_compute_forward_get_rows(node);
                } break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    // openvino_frontend_compute(backend, cgraph);

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
    GGML_UNUSED(ctx);
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

std::set<std::string> get_openvino_available_opsets() {
    ov::Core core;
    std::set<std::string> unique_ops;
    for (const auto& opset  : ov::get_available_opsets()) {
        for (const auto& op : opset.second().get_type_info_set()) {
            unique_ops.insert(op.name).second;
        }
    }
    return unique_ops;
}

static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);
    // ggml_backend_openvino_device_context * dev_ctx = (ggml_backend_openvino_device_context *) dev->context;

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            return true;
        case GGML_OP_ADD:
            return true;
        case GGML_OP_MUL:
            return true;
        case GGML_OP_MUL_MAT:
            return false;
        default:
            return false;
    }
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
