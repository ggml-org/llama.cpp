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
  int end_node = cgraph->n_nodes - 1;
  openvino_frontend_compute(backend, cgraph, 0, end_node);

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

void ggml_backend_openvino_rms_norm_f32(ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];
    assert(src0 != nullptr);

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];

    const size_t input_size = ne0 * ne1 * ne2;

    const float *src_data = static_cast<const float *>(src0->data);
    float *dst_data = static_cast<float *>(dst->data);
    assert(dst_data != nullptr);

    ov::Core core;

    ov::Shape input_shape = {static_cast<size_t>(ne2), static_cast<size_t>(ne1), static_cast<size_t>(ne0)};
    ov::Tensor input_tensor(ov::element::f32, input_shape, const_cast<float *>(src_data));

    auto input_param = std::make_shared<ov::op::v0::Parameter>(
        input_tensor.get_element_type(),
        input_tensor.get_shape()
    );
    assert(input_param != nullptr && "Input parameter creation failed!");

    auto square = std::make_shared<ov::op::v1::Multiply>(input_param, input_param);
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        square,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
        true
    );

    auto mean = std::make_shared<ov::op::v1::Divide>(
        reduce_sum,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {static_cast<float>(ne0)})
    );

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(
            mean,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {eps})
        )
    );

    auto scale = std::make_shared<ov::op::v1::Divide>(
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}),
        rms
    );

    auto normalized_input = std::make_shared<ov::op::v1::Multiply>(input_param, scale);

    ov::ParameterVector parameters = {input_param};
    auto model = std::make_shared<ov::Model>(ov::NodeVector{normalized_input}, parameters);

    // static bool model_saved = false;
    // if (!model_saved) {
    //     std::cout << "\n rms model saved" << std::endl;
    //     ov::save_model(model, "/<Your-Host-Path>/rms_norm_model.xml");
    //     model_saved = true;
    // }

    auto compiled_model = core.compile_model(model, "CPU");

    auto infer_request = compiled_model.create_infer_request();

    infer_request.set_input_tensor(0, input_tensor);

    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    assert(output_tensor.get_size() == input_size);

    std::memcpy(dst_data, output_tensor.data<float>(), input_size * sizeof(float));
}

void ggml_backend_openvino_rms_norm(ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_backend_openvino_rms_norm_f32(dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_backend_openvino_permute(const struct ggml_tensor * dst) {
    // NOP
    GGML_UNUSED(dst);
}

// Extracting valid shapes
std::vector<int64_t> get_effective_shape(const ggml_tensor * t) {
    std::vector<int64_t> shape;
    for (int i = 2; i >= 0; i--) {
        if (t->ne[i] != 1 || t->ne[2] != 1)
            shape.push_back(t->ne[i]);
    }
    return shape;
}

/*
* Construct an index vector for Gather to extract non-contiguous data.
* Parameters:
* - valid_cols: number of valid columns per row (e.g., for src0, valid columns = 96)
* - num_rows: number of rows in each batch (e.g., src0: 32 rows per batch)
* - batch: number of batches (e.g., 32)
* - row_stride: physical row length (in elements), e.g., src0: nb[1]/(element_size) = 6144/2 = 3072
* - batch_stride: physical batch stride (in elements), e.g., src0: nb[2]/(element_size) = 192/2 = 96
*/
std::vector<int64_t> build_indices(int valid_cols, int num_rows, int batch, int row_stride, int batch_stride) {
    std::vector<int64_t> indices;
    indices.reserve(valid_cols * num_rows * batch);
    for (int b = 0; b < batch; b++) {
        for (int r = 0; r < num_rows; r++) {
            for (int c = 0; c < valid_cols; c++) {
                // 计算物理索引 = b * batch_stride + r * row_stride + c
                indices.push_back(b * batch_stride + r * row_stride + c);
            }
        }
    }
    return indices;
}

void ggml_backend_openvino_mul_mat(struct ggml_tensor * dst) {
    assert(dst && dst->src[0] && dst->src[1]);
    const ggml_tensor * src0 = dst->src[0]; // src0 type F16
    const ggml_tensor * src1 = dst->src[1]; // src1 type F32

    if(!ggml_is_contiguous(src1) || dst->src[1]->ne[0] * dst->src[1]->nb[0] != dst->src[1]->nb[1]) {
        int valid_cols_src0 = src0->ne[0];  // 96
        int num_rows_src0   = src0->ne[1];    // 32
        int batch_src0      = src0->ne[2];    // 32

        int valid_cols_src1 = src1->ne[0];  // 96
        int num_rows_src1   = src1->ne[1];    // 7
        int batch_src1      = src1->ne[2];    // 32

        // 对 src0：row_stride = nb[1] / nb[0]
        int row_stride_src0   = src0->nb[1] / src0->nb[0];   // 6144 / 2 = 3072
        int batch_stride_src0 = src0->nb[2] / src0->nb[0];     // 192 / 2 = 96

        // 对 src1：row_stride = nb[1] / nb[0]
        int row_stride_src1   = src1->nb[1] / src1->nb[0];   // 12288 / 4 = 3072
        int batch_stride_src1 = src1->nb[2] / src1->nb[0];     // 384 / 4 = 96

        std::vector<int64_t> indices_src0 = build_indices(valid_cols_src0, num_rows_src0, batch_src0, row_stride_src0, batch_stride_src0);
        std::vector<int64_t> indices_src1 = build_indices(valid_cols_src1, num_rows_src1, batch_src1, row_stride_src1, batch_stride_src1);

        size_t total_src0 = indices_src0.size(); // = 96 * 32 * 32
        size_t total_src1 = indices_src1.size(); // = 96 * 7 * 32

        ov::Shape orig_shape_src0 = { static_cast<size_t>(src0->ne[2]),
                                        static_cast<size_t>(src0->ne[1]),
                                        static_cast<size_t>(src0->ne[0])};
        ov::Shape orig_shape_src1 = { static_cast<size_t>(src1->ne[2]),
                                        static_cast<size_t>(src1->ne[1]),
                                        static_cast<size_t>(src1->ne[0])};

        auto param_src0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, orig_shape_src0);
        auto param_src1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, orig_shape_src1);

        ov::Shape flat_shape_src0 = { total_src0 };
        ov::Shape flat_shape_src1 = { total_src1 };

        auto flatten_src0 = std::make_shared<ov::op::v1::Reshape>(
            param_src0,
            ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{ static_cast<int64_t>(total_src0) }),
            false);
        auto flatten_src1 = std::make_shared<ov::op::v1::Reshape>(
            param_src1,
            ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{ static_cast<int64_t>(total_src1) }),
            false);

        auto indices_const_src0 = ov::op::v0::Constant::create(ov::element::i64, flat_shape_src0, indices_src0);
        auto indices_const_src1 = ov::op::v0::Constant::create(ov::element::i64, flat_shape_src1, indices_src1);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        auto gathered_src0 = std::make_shared<ov::op::v8::Gather>(flatten_src0, indices_const_src0, axis_const);
        auto gathered_src1 = std::make_shared<ov::op::v8::Gather>(flatten_src1, indices_const_src1, axis_const);

        std::vector<int64_t> shape_src0_cont = { batch_src0, num_rows_src0, valid_cols_src0 };
        auto reshape_src0 = std::make_shared<ov::op::v1::Reshape>(
            gathered_src0,
            ov::op::v0::Constant::create(ov::element::i64, { shape_src0_cont.size() }, shape_src0_cont),
            false);

        std::vector<int64_t> shape_src1_cont = { batch_src1, num_rows_src1, valid_cols_src1 };
        auto reshape_src1 = std::make_shared<ov::op::v1::Reshape>(
            gathered_src1,
            ov::op::v0::Constant::create(ov::element::i64, { shape_src1_cont.size() }, shape_src1_cont),
            false);

        auto src0_f32 = std::make_shared<ov::op::v0::Convert>(reshape_src0, ov::element::f32);
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
        auto src0_transposed = std::make_shared<ov::op::v1::Transpose>(src0_f32, transpose_order);

        auto A = src0_transposed;
        auto B = reshape_src1;

        auto batched_matmul = std::make_shared<ov::op::v0::MatMul>(B, A, false, false);

        std::vector<int64_t> final_output_shape = {static_cast<int64_t>(dst->ne[2]),
                                                    static_cast<int64_t>(dst->ne[1]),
                                                    static_cast<int64_t>(dst->ne[0])};

        auto reshape_output = std::make_shared<ov::op::v1::Reshape>(
            batched_matmul,
            ov::op::v0::Constant::create(ov::element::i64, {3}, final_output_shape),
            false);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{ reshape_output },
                                                 ov::ParameterVector{ param_src0, param_src1 });

        ov::Tensor tensor_src0{ ov::element::f16, orig_shape_src0, src0->data };
        ov::Tensor tensor_src1{ ov::element::f32, orig_shape_src1, src1->data };
        ov::Shape output_shape = { static_cast<size_t>(dst->ne[2]),
                                     static_cast<size_t>(dst->ne[1]),
                                     static_cast<size_t>(dst->ne[0]) };
        ov::Tensor tensor_dst(ov::element::f32, output_shape, dst->data);

        ov::Core core;
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(0, tensor_src0);
        infer_request.set_input_tensor(1, tensor_src1);
        infer_request.set_output_tensor(0, tensor_dst);
        infer_request.infer();
        return ;
    }

    int rank = 0;
    if (dst->ne[2] == 1 && dst->ne[3] == 1) {
        rank = 2;
    } else if (dst->ne[3] == 1) {
        rank = 3;
    } else {
        throw std::runtime_error("Only rank 2 or rank 3 are supported in this implementation.");
    }

    std::vector<int64_t> eff_shape_src0 = get_effective_shape(src0);
    std::vector<int64_t> eff_shape_src1 = get_effective_shape(src1);
    std::vector<int64_t> eff_shape_dst = get_effective_shape(dst);

    ov::Shape orig_shape_src0 = { static_cast<size_t>(src0->ne[2]),
                                        static_cast<size_t>(src0->ne[1]),
                                        static_cast<size_t>(src0->ne[0])};
    ov::Shape orig_shape_src1 = { static_cast<size_t>(src1->ne[2]),
                                        static_cast<size_t>(src1->ne[1]),
                                        static_cast<size_t>(src1->ne[0])};
    auto param_src0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, orig_shape_src0);
    auto param_src1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, orig_shape_src1);

    auto reshape_src0 = std::make_shared<ov::op::v1::Reshape>(
        param_src0,
        ov::op::v0::Constant::create(ov::element::i64, { eff_shape_src0.size() }, eff_shape_src0),
        false);
    auto reshape_src1 = std::make_shared<ov::op::v1::Reshape>(
        param_src1,
        ov::op::v0::Constant::create(ov::element::i64, { eff_shape_src1.size() }, eff_shape_src1),
        false);

    auto src0_f32 = std::make_shared<ov::op::v0::Convert>(reshape_src0, ov::element::f32);

    ov::Output<ov::Node> A_for_mul;
    if (rank == 2) {
        auto trans_order = ov::op::v0::Constant::create(ov::element::i64, { 2 }, std::vector<int64_t>{1, 0});
        A_for_mul = std::make_shared<ov::op::v1::Transpose>(src0_f32, trans_order);
    } else if (rank == 3) {
        auto trans_order = ov::op::v0::Constant::create(ov::element::i64, { 3 }, std::vector<int64_t>{0, 2, 1});
        A_for_mul = std::make_shared<ov::op::v1::Transpose>(src0_f32, trans_order);
    } else {
        A_for_mul = src0_f32;
    }

    auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape_src1, A_for_mul, false, false);

    auto matmul_output_shape = matmul->get_output_shape(0);
    std::vector<int64_t> final_output_shape;
    if (matmul_output_shape.size() == 1) {
        final_output_shape = { 1, 1, static_cast<int64_t>(matmul_output_shape[0]) };
    } else if (matmul_output_shape.size() == 2) {
        final_output_shape = { 1, static_cast<int64_t>(matmul_output_shape[0]), static_cast<int64_t>(matmul_output_shape[1]) };
    } else {
        final_output_shape = { static_cast<int64_t>(matmul_output_shape[0]), static_cast<int64_t>(matmul_output_shape[1]), static_cast<int64_t>(matmul_output_shape[2]) };
    }

    auto reshape_output = std::make_shared<ov::op::v1::Reshape>(
        matmul,
        ov::op::v0::Constant::create(ov::element::i64, {3}, final_output_shape),
        false
    );

    auto model = std::make_shared<ov::Model>(ov::NodeVector{ reshape_output },
                                             ov::ParameterVector{ param_src0, param_src1 });

    ov::Tensor tensor_src0{ ov::element::f16, orig_shape_src0, (void *)src0->data };
    ov::Tensor tensor_src1{ ov::element::f32, orig_shape_src1, (void *)src1->data };

    ov::Shape output_shape = { static_cast<size_t>(dst->ne[2]),
                                static_cast<size_t>(dst->ne[1]),
                                static_cast<size_t>(dst->ne[0]) };
    ov::Tensor tensor_dst(ov::element::f32, output_shape, dst->data);

    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(0, tensor_src0);
    infer_request.set_input_tensor(1, tensor_src1);
    infer_request.set_output_tensor(0, tensor_dst);
    infer_request.infer();
}

void ggml_backend_openvino_reshape(ggml_tensor *dst) {

    GGML_UNUSED(dst);
}

void ggml_backend_openvino_view(ggml_tensor *dst) {

    GGML_UNUSED(dst);
}

void ggml_backend_openvino_dup_bytes(struct ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];

    // Validate tensor properties
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    // Determine tensor properties
    const size_t element_size = ggml_type_size(src0->type);

    // Case 1: Both tensors are contiguous
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && (src0->ne[0] * element_size == src0->nb[1])) {
        ov::Shape input_shape = {
            static_cast<size_t>(src0->ne[2]),
            static_cast<size_t>(src0->ne[1]),
            static_cast<size_t>(src0->ne[0])
        };
        size_t num_elements = 1;
        for (auto d : input_shape) {
            num_elements *= d;
        }
        ov::Shape flat_shape = { num_elements };

        ov::Shape dst_shape = {
            static_cast<size_t>(dst->ne[2]),
            static_cast<size_t>(dst->ne[1]),
            static_cast<size_t>(dst->ne[0])
        };

        auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);

        std::vector<int64_t> flat_shape_vec(flat_shape.begin(), flat_shape.end());
        auto flat_reshape_const = ov::op::v0::Constant::create(ov::element::i64, { flat_shape_vec.size() }, flat_shape_vec);
        auto flat_reshape = std::make_shared<ov::op::v1::Reshape>(input_param, flat_reshape_const, false);

        std::vector<int64_t> dst_shape_vec(dst_shape.begin(), dst_shape.end());
        auto dst_reshape_const = ov::op::v0::Constant::create(ov::element::i64, { dst_shape_vec.size() }, dst_shape_vec);
        auto final_reshape = std::make_shared<ov::op::v1::Reshape>(flat_reshape, dst_reshape_const, false);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{ final_reshape }, ov::ParameterVector{ input_param });

        ov::Core core;
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        ov::Tensor input_tensor(ov::element::f32, input_shape, src0->data);
        infer_request.set_input_tensor(0, input_tensor);

        ov::Tensor output_tensor(ov::element::f32, dst_shape, dst->data);
        infer_request.set_output_tensor(0, output_tensor);

        infer_request.infer();
        return;
    }

    // Case 2: Compatible types, dimensions, and strides
    const size_t ne00 = src0->ne[0];
    const size_t ne01 = src0->ne[1];
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb0 = dst->nb[0];

    if (src0->type == dst->type && ne00 == dst->ne[0] && nb00 == element_size && nb0 == element_size) {
        const size_t valid_elems = static_cast<size_t>(src0->ne[0]);  // 3072
        const size_t num_rows    = static_cast<size_t>(src0->ne[1]);  // 7
        const size_t dim2        = static_cast<size_t>(src0->ne[2]);  // 1

        size_t phys_stride = static_cast<size_t>(src0->nb[1]) / element_size; // 9216
        // size_t phys_stride = static_cast<size_t>(src0->ne[0]); // 3072

        ov::Shape input_shape = { dim2, num_rows, phys_stride }; // 如 {1, 7, 9216 }
        ov::Shape logical_shape = { dim2, num_rows, valid_elems }; // {1, 7, 3072}

        // std::cout << "CONT input shape: " << input_shape << std::endl;
        auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);

        // int64_t split_addr = dst->src[0]->view_offs / dst->src[0]->nb[0];
        // std::vector<int64_t> begin = { 0, 0, split_addr };
        // std::vector<int64_t> end   = { static_cast<int64_t>(dim2),
        //                                 static_cast<int64_t>(num_rows),
        //                                 split_addr + static_cast<int64_t>(valid_elems) };

        std::vector<int64_t> begin = { 0, 0, 0 };
        std::vector<int64_t> end   = { static_cast<int64_t>(dim2),
                                        static_cast<int64_t>(num_rows),
                                        static_cast<int64_t>(valid_elems) };
        std::vector<int64_t> strides = { 1, 1, 1 };

        auto begin_const = ov::op::v0::Constant::create(ov::element::i64, { begin.size() }, begin);
        auto end_const   = ov::op::v0::Constant::create(ov::element::i64, { end.size() }, end);
        auto strides_const = ov::op::v0::Constant::create(ov::element::i64, { strides.size() }, strides);

        std::vector<int64_t> begin_mask = {0, 0, 0};
        std::vector<int64_t> end_mask   = {0, 0, 0};
        auto slice = std::make_shared<ov::op::v1::StridedSlice>(
            input_param, 
            begin_const, 
            end_const, 
            strides_const, 
            begin_mask, 
            end_mask
        );

        auto model = std::make_shared<ov::Model>(ov::OutputVector{ slice },
                                                 ov::ParameterVector{ input_param });

        ov::Core core;
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        //[NOTE]: input_shape should be {1, 7, 9216} not the original shap of src0.
        ov::Tensor input_tensor(ov::element::f32, input_shape, src0->data);
        infer_request.set_input_tensor(0, input_tensor);

        ov::Tensor output_tensor(ov::element::f32, logical_shape, dst->data);
        infer_request.set_output_tensor(0, output_tensor);

        infer_request.infer();
        return;
    }

    // Case 3: Non-contiguous source, contiguous destination
    // dst->ne        =[3072,7,1,1],       dst->nb =[4,12288,86016,86016],          dst->type=GGML_TYPE_F32
    // dst->src[0]->ne=[96,32,7,1], dst->src[0]->nb=[4,2688,384,86016],     dst->src[0]->type=GGML_TYPE_F32
    if (ggml_is_contiguous(dst)) {
        size_t valid_i = static_cast<size_t>(src0->ne[0]); // 96
        size_t valid_j = static_cast<size_t>(src0->ne[1]); // 32
        size_t valid_k = static_cast<size_t>(src0->ne[2]); // 7

        ov::Shape src_shape = { valid_k, valid_j, valid_i }; //  {7, 32, 96};
        auto src_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, src_shape);

        ov::Shape input_shape = { valid_j, valid_k, valid_i }; //  {32, 7, 96}
        auto tmp_param = ov::op::v0::Constant::create(ov::element::i64, { input_shape.size() }, input_shape);
        auto input_param = std::make_shared<ov::op::v1::Reshape>(src_param, tmp_param, false);

        // 添加 Transpose 节点，将 {32,7,96} 变换为 {7,32,96}，恢复逻辑顺序
        // 这里交换第 0 与第 1 维，即 permutation = {1, 0, 2}
        std::vector<int64_t> order = {1, 0, 2};
        auto order_const = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input_param, order_const);

        ov::Shape target_shape = { static_cast<size_t>(dst->ne[2]), static_cast<size_t>(dst->ne[1]), static_cast<size_t>(dst->ne[0]) }; // {1, 7, 3072}
        std::vector<int64_t> target_shape_vec = { static_cast<int64_t>(dst->ne[2]),
                                                  static_cast<int64_t>(dst->ne[1]),
                                                  static_cast<int64_t>(dst->ne[0]) };
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, { target_shape_vec.size() }, target_shape_vec);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(transpose, reshape_const, false);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{ reshaped },
                                                 ov::ParameterVector{ src_param });
        ov::Core core;
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        ov::Tensor input_tensor(ov::element::f32, src_shape, src0->data);
        infer_request.set_input_tensor(0, input_tensor);

        ov::Tensor output_tensor(ov::element::f32, target_shape, dst->data);
        infer_request.set_output_tensor(0, output_tensor);

        infer_request.infer();
        return;
    }
}

static void ggml_backend_openvino_transpose(ggml_tensor *dst) {
    // ov::Core core;
    // ov::Shape input_shape{static_cast<size_t>(dst->src[0]->ne[2]), static_cast<size_t>(dst->src[0]->ne[1]), static_cast<size_t>(dst->src[0]->ne[0])};
    // ov::Shape output_shape{static_cast<size_t>(dst->ne[2]), static_cast<size_t>(dst->ne[1]), static_cast<size_t>(dst->ne[0])};
    // auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    
    // //auto res = std::make_shared<ov::op::v1::Transpose>(input_param, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 2, 1})); 
    
    

    // auto new_shape_node = ov::op::v0::Constant::create(ov::element::i64,
    //                                                 ov::Shape{output_shape.size()},
    //                                                 std::vector<int64_t>(output_shape.begin(), output_shape.end()));
    // auto res = std::make_shared<ov::op::v1::Reshape>(input_param, new_shape_node, false);




    // std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(ov::OutputVector{res},
    //                                                                 ov::ParameterVector{input_param});
    // auto compiled_model = core.compile_model(model, "CPU");
    // ov::InferRequest infer_request = compiled_model.create_infer_request();

    // ov::Tensor input_tensor(ov::element::f32, input_shape, dst->src[0]->data);
    // ov::Tensor output_tensor(ov::element::f32, output_shape, dst->data);
    // infer_request.set_input_tensor(0, input_tensor);
    // infer_request.set_output_tensor(0, output_tensor);

    // infer_request.infer();

    // NOP
    GGML_UNUSED(dst);
}

void ggml_backend_openvino_cpy(struct ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];
    const struct ggml_tensor *src1 = dst->src[1];
    assert(src0 != nullptr);
    assert(ggml_nelements(dst) == ggml_nelements(src0));

    // Extract shapes
    ov::Shape src_shape(src0->ne, src0->ne + 4);
    ov::Shape dst_shape(dst->ne, dst->ne + 4);

    // Initialize OpenVINO core
    ov::Core core;

    // Create OpenVINO parameter for the source tensor
    auto src_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, src_shape);

    std::shared_ptr<ov::Model> model;
    if (ggml_is_contiguous(dst)) {
        // Contiguous Case: Flatten src and reshape to dst shape
        ov::Shape flattened_shape = {static_cast<size_t>(ggml_nelements(src0))};
        auto flatten = std::make_shared<ov::op::v1::Reshape>(
            src_input, ov::op::v0::Constant::create(ov::element::i64, {1}, flattened_shape), false);

        auto reshape_to_dst = std::make_shared<ov::op::v1::Reshape>(
            flatten, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape), false);

        auto dst_output = std::make_shared<ov::op::v0::Convert>(reshape_to_dst, ov::element::f16);

        model = std::make_shared<ov::Model>(
            ov::ResultVector{std::make_shared<ov::op::v0::Result>(dst_output)},
            ov::ParameterVector{src_input},
            "ContiguousCopy");
        // Compile and execute the model
        auto compiled_model = core.compile_model(model, "CPU");

        ov::Tensor src_tensor(ov::element::f32, src_shape, src0->data);
        ov::Tensor dst_tensor(ov::element::f16, dst_shape, dst->data);

        auto infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(0, src_tensor);
        infer_request.set_output_tensor(0, dst_tensor);
        infer_request.infer();
    } else {
        int src0_elem_size = ggml_type_size(src0->type);
        int src1_elem_size = ggml_type_size(src1->type);

        int src0_logical_cols = src0->ne[0];
        int src0_logical_rows = src0->ne[1];
        int src1_logical_cols = src1->ne[0];
        int src1_logical_rows = src1->ne[1];

        int src0_phys_cols = src0->nb[0] / src0_elem_size;
        int src0_phys_rows = src0_logical_rows;

        int src1_phys_cols = src1->nb[1] / src1_elem_size;
        int src1_phys_rows = src1_logical_rows;

        ov::Shape src0_phys_shape = {1, static_cast<size_t>(src0_phys_rows), static_cast<size_t>(src0_phys_cols) };
        ov::Shape src1_phys_shape = {1, static_cast<size_t>(src1_phys_rows), static_cast<size_t>(src1_phys_cols) };

        size_t logical_elems = static_cast<size_t>(src0_logical_cols * src0_logical_rows);
        size_t src_flat_size = 1 * src0_phys_cols * src0_phys_rows;
        size_t dst_flat_size = 1 * src1_phys_rows * src1_phys_cols;

        ov::Core core;

        std::vector<int64_t> gather_idx;
        gather_idx.reserve(logical_elems);
        for (int row = 0; row < src0_logical_rows; row++) {
            for (int col = 0; col < src0_logical_cols; col++) {
                gather_idx.push_back(static_cast<int64_t>(row + col * src0_phys_rows));
            }
        }
        ov::Shape gather_idx_shape = { logical_elems };

        std::vector<int64_t> scatter_idx;
        scatter_idx.reserve(logical_elems);
        for (int row = 0; row < src1_logical_rows; row++) {
            for (int col = 0; col < src1_logical_cols; col++) {
                scatter_idx.push_back(static_cast<int64_t>(row * src1_phys_cols + col));
            }
        }
        ov::Shape scatter_idx_shape = { logical_elems, 1 };

        auto param_src0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, src0_phys_shape);
        auto param_src1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, src1_phys_shape);

        auto src_flat_shape_const = ov::op::v0::Constant::create(ov::element::i64, {1},
                                      { static_cast<int64_t>(src_flat_size) });
        auto reshape_src = std::make_shared<ov::op::v1::Reshape>(param_src0, src_flat_shape_const, false);
        auto dst_flat_shape_const = ov::op::v0::Constant::create(ov::element::i64, {1},
                                      { static_cast<int64_t>(dst_flat_size) });
        auto reshape_dst = std::make_shared<ov::op::v1::Reshape>(param_src1, dst_flat_shape_const, false);

        auto gather_indices_const = ov::op::v0::Constant::create(ov::element::i64, gather_idx_shape, gather_idx);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto gathered = std::make_shared<ov::op::v8::Gather>(reshape_src, gather_indices_const, axis_const);
        auto converted = std::make_shared<ov::op::v0::Convert>(gathered, ov::element::f16);

        auto scatter_indices_const = ov::op::v0::Constant::create(ov::element::i64, scatter_idx_shape, scatter_idx);
        auto scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(reshape_dst, scatter_indices_const, converted);

        std::vector<int64_t> dst_phys_shape_vec = {1, static_cast<int64_t>(src1_phys_rows),
                                                    static_cast<int64_t>(src1_phys_cols) };
        auto dst_phys_shape_const = ov::op::v0::Constant::create(ov::element::i64, {3}, dst_phys_shape_vec);
        auto final_output = std::make_shared<ov::op::v1::Reshape>(scatter, dst_phys_shape_const, false);

        ov::ParameterVector params = { param_src0, param_src1 };
        auto model = std::make_shared<ov::Model>(ov::OutputVector{ final_output }, params);
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        ov::Tensor tensor_src(ov::element::f32, src0_phys_shape, src0->data);
        ov::Tensor tensor_dst(ov::element::f16, src1_phys_shape, src1->data);
        infer_request.set_input_tensor(0, tensor_src);
        infer_request.set_input_tensor(1, tensor_dst);

        ov::Tensor out_tensor(ov::element::f16, src1_phys_shape, dst->data);
        infer_request.set_output_tensor(0, out_tensor);

        infer_request.infer();
    }
}

static enum ggml_status ggml_backend_openvino_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Find the indices of GGML_OP_CONT, GGML_OP_CPY nodes, GGML_OP_MUL_MAT and so on.
    std::vector<int> cont_indices;
    std::vector<int> reshape_indices;
    std::vector<int> view_indices;
    std::vector<int> view_indices_prompt;
    std::vector<int> view_split;

    std::vector<int> cpy_indices;
    std::vector<int> cpy_split_16;
    std::vector<int> cpy_split_19;
    std::vector<int> transpose_indices;
    std::vector<int> permute_indices;

    std::vector<int> mul_mat_indices;
    std::vector<int> add_indices;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op == GGML_OP_CONT) {
            cont_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_RESHAPE) {
            reshape_indices.push_back(i);
        // } else if (cgraph->nodes[i]->op == GGML_OP_VIEW) {
        } else if (cgraph->nodes[i]->op == GGML_OP_VIEW) {
            // if (cgraph->nodes[i]->src[0]->ne[0] == 98304 && (cgraph->nodes[i]->ne[0] == 3072 || cgraph->nodes[i]->ne[0] == 1))
            //     continue;
            view_indices.push_back(i);
            if (cgraph->nodes[i]->ne[0] == 32) {
                view_indices_prompt.push_back(i);
            }
            if (i == 18) {
                view_split.push_back(i);
            }
        } else if (cgraph->nodes[i]->op == GGML_OP_CPY) {
            cpy_indices.push_back(i);
            if (i == 16) {
                cpy_split_16.push_back(i);
            }
            if (i == 19) {
                cpy_split_19.push_back(i);
            }
        } else if (cgraph->nodes[i]->op == GGML_OP_TRANSPOSE) {
            transpose_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_PERMUTE) {
            permute_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
            mul_mat_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_ADD) {
            add_indices.push_back(i);
        }
    }


    // Process nodes in order

    if (cgraph->nodes[0]->ne[1] == 1) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (std::find(add_indices.begin(), add_indices.end(), i) != add_indices.end()) {
                ggml_backend_openvino_add_forward(cgraph->nodes[i]);
            } else if (std::find(transpose_indices.begin(), transpose_indices.end(), i) != transpose_indices.end()) {
                ggml_backend_openvino_transpose(cgraph->nodes[i]);
            } else if (std::find(cpy_indices.begin(), cpy_indices.end(), i) != cpy_indices.end()) {
                ggml_backend_openvino_cpy(cgraph->nodes[i]);
            } else if (std::find(view_indices.begin(), view_indices.end(), i) != view_indices.end()) {
                ggml_backend_openvino_view(cgraph->nodes[i]);
            } else if (std::find(cont_indices.begin(), cont_indices.end(), i) != cont_indices.end()) {
                ggml_backend_openvino_dup_bytes(cgraph->nodes[i]);
            } else if (std::find(reshape_indices.begin(), reshape_indices.end(), i) != reshape_indices.end()) {
                ggml_backend_openvino_reshape(cgraph->nodes[i]);
            } else {
                // Process a range of nodes with openvino_frontend_compute
                int start_index = i;
                while (i < cgraph->n_nodes
                        && std::find(add_indices.begin(), add_indices.end(), i) == add_indices.end()
                        && std::find(transpose_indices.begin(), transpose_indices.end(), i) == transpose_indices.end()
                        && std::find(cpy_indices.begin(), cpy_indices.end(), i) == cpy_indices.end()
                        && std::find(view_indices.begin(), view_indices.end(), i) == view_indices.end()
                        && std::find(cont_indices.begin(), cont_indices.end(), i) == cont_indices.end()
                        && std::find(reshape_indices.begin(), reshape_indices.end(), i) == reshape_indices.end()
                        ) {
                    i++;
                }
                if (start_index < i) {
                        openvino_frontend_compute(backend, cgraph, start_index, --i);
                }
            }
        }
    } else {
        int end_node = cgraph->n_nodes - 1;
        openvino_frontend_compute(backend, cgraph, 0, end_node);
    }

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

static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);

#ifdef OPENVINO_OP_DEBUG
static const std::set<std::string>& openvino_ops = []() -> const std::set<std::string>& {
        static const std::set<std::string> ops = get_openvino_available_opsets();
        return ops;
    }();
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
        case GGML_OP_MUL_MAT:
            return false;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op))
            {
                case GGML_UNARY_OP_SILU:
                    return true;
                case GGML_UNARY_OP_ABS: 
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_COUNT:
                    return false;
            }
            return false; 
        default:
            return false;
    }
#else
    static const std::set<std::string>& openvino_ops = []() -> const std::set<std::string>& {
        static const std::set<std::string> ops = get_openvino_available_opsets();
        return ops;
    }();

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

