#include "ggml-backend-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-openvino.h"
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


void ggml_backend_openvino_mul_mat(struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = 0;
    const int nth = 1;

    const enum ggml_type type = src0->type;
    const auto *type_traits = ggml_get_type_traits(type);

    enum ggml_type           const vec_dot_type         = type_traits->vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits->from_float;
    ggml_from_float_to_mat_t const from_float_to_mat    = type_traits->from_float_to_mat;
    int64_t                  const vec_dot_num_rows     = type_traits->nrows;
    int64_t                  const matmul_num_cols      = type_traits->ncols;
    int64_t                  const blck_size_interleave = type_traits->blck_size_interleave;
    ggml_gemv_t              const gemv                 = type_traits->gemv;
    ggml_gemm_t              const gemm                 = type_traits->gemm;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // src1->type = GGML_TYPE_F32, vec_dot_type = GGML_TYPE_F16
    // The main function of this code is to convert the data of src1 from GGML_TYPE_F32 type to vec_dot_type (i.e. GGML_TYPE_F16) and store the result in params->wdata.
    // The code processes data of different dimensions through multiple loops and conditional judgments and uses different conversion functions to complete data conversion.
    std::unique_ptr<char[]> wdata(new char[ne13 * ggml_row_size(vec_dot_type, ne10) * ne11 * ne12]);
    if (src1->type != vec_dot_type) {
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                           (void *)               (wdata.get() + i13*nbw3 + i12*nbw2 + i11*nbw1),
                           ne10);
                }
            }
        }
    }

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;
    // TODO: currently the mmla kernels support only even numbered rows/cols.
    // this check can be removed once they are extended to support odd numbered rows/cols too
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
        num_rows_per_vec_dot = 1;
    }

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        const bool src1_cont = ggml_is_contiguous(src1);

        ggml_vec_dot_t const vec_dot      = type_traits->vec_dot;
        enum ggml_type const vec_dot_type = type_traits->vec_dot_type;

        // broadcast factors
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        // threads with no work simply yield (not sure if it helps)
        if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
            return;
        }

        // const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        assert(ne12 % ne02 == 0);
        assert(ne13 % ne03 == 0);

        // block-tiling attempt
        const int64_t blck_0 = 16;
        const int64_t blck_1 = 16;

        const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

        // attempt to reduce false-sharing (does not seem to make a difference)
        // 16 * 2, accounting for mmla kernels
        float tmp[32];

        for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
            for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
                for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                    const int64_t i13 = (ir1 / (ne12 * ne1));
                    const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                    const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                    // broadcast src0 into src1
                    const int64_t i03 = i13 / r3;
                    const int64_t i02 = i12 / r2;

                    const int64_t i1 = i11;
                    const int64_t i2 = i12;
                    const int64_t i3 = i13;

                    const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                    // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                    //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                    //       the original src1 data pointer, so we should index using the indices directly
                    const char * src1_col = (const char*)wdata.get() +
                        (src1_cont || src1->type != vec_dot_type
                            ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                            : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                    float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                        vec_dot(ne00, &tmp[ir0 - iir0],
                                (num_rows_per_vec_dot > 1 ? 16 : 0),
                                src0_row + ir0 * nb01,
                                (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                src1_col,
                                (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                                num_rows_per_vec_dot);
                    }

                    for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                        memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                    }
                }
            }
        }

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        // current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
        current_chunk++;
    }
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
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        // OpenVINO tensors for src and dst
        // Source is 1D since it's contiguous
        ov::Tensor src_tensor(ov::element::f32, {src0->ne[0]}, src0->data);
        // // Destination is 1D since it's contiguous
        ov::Tensor dst_tensor(ov::element::f32, {dst->ne[0]}, dst->data);

        // Perform the memory copy row by row
        size_t row_size = dst->nb[0];  // Size of one row in destination
        size_t src_stride = src0->nb[0];  // Stride for source tensor

        for (size_t i = 0; i < dst->ne[0]; ++i) {
            std::memcpy((char *)dst_tensor.data()+i*row_size, (char *)src_tensor.data()+i*src_stride, row_size);
        }
        return;
    }

    // Case 2: Compatible types, dimensions, and strides
    const size_t ne00 = src0->ne[0];
    const size_t ne01 = src0->ne[1];
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb0 = dst->nb[0];

    if (src0->type == dst->type && ne00 == dst->ne[0] && nb00 == element_size && nb0 == element_size) {
        for (size_t i01 = 0; i01 < ne01; ++i01) {
            const char *src_row = reinterpret_cast<const char *>(src0->data) + i01 * nb01;
            char *dst_row = reinterpret_cast<char *>(dst->data) + i01 * dst->nb[1];

            ov::Tensor src_row_tensor(ov::element::f32, {ne00}, const_cast<void *>(reinterpret_cast<const void *>(src_row)));
            ov::Tensor dst_row_tensor(ov::element::f32, {ne00}, reinterpret_cast<void *>(dst_row));

            std::memcpy(dst_row_tensor.data<float>(), src_row_tensor.data<float>(), ne00 * sizeof(float));
        }
        return;
    }

    // Case 3: Non-contiguous source, contiguous destination
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    // dst->ne        =[3072,7,1,1],       dst->nb =[4,12288,86016,86016],          dst->type=GGML_TYPE_F32
    // dst->src[0]->ne=[96,32,7,1], dst->src[0]->nb=[4,2688,384,86016],     dst->src[0]->type=GGML_TYPE_F32
    if (ggml_is_contiguous(dst)) {
        const size_t rs = ne00 * element_size; // Row size in bytes for dst

        // Create OpenVINO tensors for source and destination
        // The tensors are reshaped to a 2D structure (num_rows x ne00) for easier iteration and compatibility with the simplified loop.
        ov::Tensor src_tensor(ov::element::f32, ov::Shape{ne03 * ne02 * ne01, ne00}, src0->data);
        ov::Tensor dst_tensor(ov::element::f32, ov::Shape{ne03 * ne02 * ne01, ne00}, dst->data);

        // Perform the copy in a single loop
        const size_t num_rows = ne03 * ne02 * ne01;
        for (size_t row = 0; row < num_rows; ++row) {
            // Calculate the source row pointer based on original strides
            // The source row pointer is calculated based on the combined index row and the strides nb03, nb02, and nb01.
            const char* src0_ptr = (char*)src_tensor.data() +
                                    // Calculates which block of the i03 dimension the current row belongs to
                                   (row / (ne02 * ne01)) * nb03 +   // 0
                                    // Calculates which block of the i02 dimension the current row belongs to within the current i03 block.
                                   ((row / ne01) % ne02) * nb02 +   // 0,   0,......,    0,384,  384,......,  384,768,......, 2304
                                    // Calculates the position within the current i02 block in terms of the i01 index.
                                   (row % ne01) * nb01;             // 0,2688,......,83328,  0, 2688,......,83328,  0,......, 83328

            // Destination row pointer is linear
            // Since dst is contiguous, its rows are accessed linearly using a single stride rs, simplifying the destination pointer calculation.
            char* dst_ptr = (char*)dst_tensor.data() + row * rs;

            // Copy row
            std::memcpy(dst_ptr, src0_ptr, rs);
        }
        return;
    }
    std::cout << "Duplication of bytes completed successfully." << std::endl;
}

static void ggml_backend_openvino_transpose(ggml_tensor *dst) {
    // NOP
    GGML_UNUSED(dst);
}

static void ggml_backend_openvino_permute(const struct ggml_tensor * dst) {
    // NOP
    GGML_UNUSED(dst);
}

void ggml_backend_openvino_cpy(struct ggml_tensor *dst) {
    const struct ggml_tensor *src0 = dst->src[0];
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
        ov::Shape flattened_shape = {ggml_nelements(src0)};
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
        // Non-contiguous case: element-wise copy
        for (int64_t i03 = 0; i03 < dst->ne[3]; ++i03) {
            for (int64_t i02 = 0; i02 < dst->ne[2]; ++i02) {
                for (int64_t i01 = 0; i01 < dst->ne[1]; ++i01) {
                    for (int64_t i00 = 0; i00 < dst->ne[0]; ++i00) {
                        const char *src_ptr = static_cast<const char *>(src0->data) +
                                              i00 * src0->nb[0] + i01 * src0->nb[1] +
                                              i02 * src0->nb[2] + i03 * src0->nb[3];

                        char *dst_ptr = static_cast<char *>(dst->data) +
                                        i00 * dst->nb[0] + i01 * dst->nb[1] +
                                        i02 * dst->nb[2] + i03 * dst->nb[3];

                        *(ggml_fp16_t *)dst_ptr = GGML_FP32_TO_FP16(*(const float *)src_ptr);
                    }
                }
            }
        }
    }
}

static enum ggml_status ggml_backend_openvino_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Find the indices of GGML_OP_CONT, GGML_OP_CPY nodes, GGML_OP_MUL_MAT and so on.
    std::vector<int> cont_indices;
    std::vector<int> reshape_indices;
    std::vector<int> view_indices;

    std::vector<int> cpy_indices;
    std::vector<int> transpose_indices;
    std::vector<int> permute_indices;

    std::vector<int> mul_mat_indices;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op == GGML_OP_CONT) {
            cont_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_RESHAPE) {
            reshape_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_VIEW) {
            view_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_CPY) {
            cpy_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_TRANSPOSE) {
            transpose_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_PERMUTE) {
            permute_indices.push_back(i);
        } else if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
            mul_mat_indices.push_back(i);
        }
    }

    // Process nodes in order
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (std::find(cont_indices.begin(), cont_indices.end(), i) != cont_indices.end()) {
            ggml_backend_openvino_dup_bytes(cgraph->nodes[i]);
        } else if (std::find(reshape_indices.begin(), reshape_indices.end(), i) != reshape_indices.end()) {
            ggml_backend_openvino_reshape(cgraph->nodes[i]);
        } else if (std::find(view_indices.begin(), view_indices.end(), i) != view_indices.end()) {
            ggml_backend_openvino_view(cgraph->nodes[i]);
        } else if (std::find(cpy_indices.begin(), cpy_indices.end(), i) != cpy_indices.end()) {
            ggml_backend_openvino_cpy(cgraph->nodes[i]);
        } else if (std::find(transpose_indices.begin(), transpose_indices.end(), i) != transpose_indices.end()) {
            ggml_backend_openvino_transpose(cgraph->nodes[i]);
        } else if (std::find(permute_indices.begin(), permute_indices.end(), i) != permute_indices.end()) {
            ggml_backend_openvino_permute(cgraph->nodes[i]);
        } else if (std::find(mul_mat_indices.begin(), mul_mat_indices.end(), i) != mul_mat_indices.end()) {
            ggml_backend_openvino_mul_mat(cgraph->nodes[i]);
        } else {
            // Process a range of nodes with openvino_frontend_compute
            int start_index = i;
            while (i < cgraph->n_nodes &&
                    std::find(cont_indices.begin(), cont_indices.end(), i) == cont_indices.end() &&
                    std::find(cpy_indices.begin(), cpy_indices.end(), i) == cpy_indices.end() &&
                    std::find(mul_mat_indices.begin(), mul_mat_indices.end(), i) == mul_mat_indices.end()) {
                i++;
            }
            if (start_index < i) {
                openvino_frontend_compute(backend, cgraph, start_index, --i);
            }
        }
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

std::set<std::string> get_openvino_available_opsets() {
    ov::Core core;
    std::set<std::string> unique_ops;
    for (const auto& opset  : ov::get_available_opsets()) {
        for (const auto& op : opset.second().get_type_info_set()) {
            unique_ops.insert(op.name);
        }
    }
    return unique_ops;
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
            return true;
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

    static const std::map<ggml_op, std::vector<std::string>> op_mapping = {
        {GGML_OP_ACC,                   {"Add"}},
        {GGML_OP_ADD,                   {"Add"}},
        {GGML_OP_ADD1,                  {"Add"}},
        {GGML_OP_ADD_REL_POS,           {"Add", "MatMul", "Reshape"}},
        {GGML_OP_ARANGE,                {"Range"}},
        {GGML_OP_ARGMAX,                {"TopK"}},
        {GGML_OP_ARGSORT,               {"TopK"}},
        {GGML_OP_CLAMP,                 {"Clamp"}},
        {GGML_OP_CONCAT,                {"Concat"}},
        {GGML_OP_CONV_TRANSPOSE_1D,     {"ConvolutionBackpropData"}},
        {GGML_OP_CONV_TRANSPOSE_2D,     {"ConvolutionBackpropData"}},
        {GGML_OP_COS,                   {"Cos"}},
        {GGML_OP_CROSS_ENTROPY_LOSS,    {"Softmax", "Log", "Multiply", "ReduceSum", "Negative"}},
        {GGML_OP_DIAG,                  {"Eye", "Multiply"}},
        {GGML_OP_DIAG_MASK_INF,         {"Eye", "Multiply", "Select", "Broadcast"}},
        {GGML_OP_DIAG_MASK_ZERO,        {"Eye", "Multiply", "Select", "Broadcast"}},
        {GGML_OP_DIV,                   {"Divide"}},
        {GGML_OP_FLASH_ATTN_EXT,        {"ScaledDotProductAttention"}},
        {GGML_OP_GET_ROWS,              {"Gather"}},
        {GGML_OP_GROUP_NORM,            {"GroupNormalization"}},
        {GGML_OP_IM2COL,                {"Custom", "Reshape", "Transpose"}},
        {GGML_OP_LEAKY_RELU,            {"PReLU"}},
        {GGML_OP_LOG,                   {"Log"}},
        {GGML_OP_MEAN,                  {"ReduceMean"}},
        {GGML_OP_MUL,                   {"Multiply"}},
        {GGML_OP_MUL_MAT,               {"MatMul"}},
        {GGML_OP_MUL_MAT_ID,            {"MatMul", "Identity"}},
        {GGML_OP_NORM,                  {"NormalizeL2"}},
        {GGML_OP_OUT_PROD,              {"MatMul", "Reshape"}},
        {GGML_OP_PAD,                   {"Pad"}},
        {GGML_OP_PERMUTE,               {"Transpose"}},
        {GGML_OP_POOL_1D,               {"AvgPool", "MaxPool"}},
        {GGML_OP_POOL_2D,               {"AvgPool", "MaxPool"}},
        {GGML_OP_REPEAT,                {"Tile"}},
        {GGML_OP_RESHAPE,               {"Reshape"}},
        {GGML_OP_RMS_NORM,              {"Multiply", "Divide", "Sqrt"}},
        {GGML_OP_ROPE,                  {"Custom"}},
        {GGML_OP_SCALE,                 {"Multiply", "Constant"}},
        {GGML_OP_SET,                   {"Assign"}},
        {GGML_OP_SIN,                   {"Sin"}},
        {GGML_OP_SOFT_MAX,              {"Softmax"}},
        {GGML_OP_SQR,                   {"Power"}},
        {GGML_OP_SQRT,                  {"Sqrt"}},
        {GGML_OP_SSM_CONV,              {"Custom"}},
        {GGML_OP_SSM_SCAN,              {"Custom"}},
        {GGML_OP_SUB,                   {"Subtract"}},
        {GGML_OP_SUM,                   {"ReduceSum"}},
        {GGML_OP_SUM_ROWS,              {"ReduceSum", "Squeeze", "Unsqueeze"}},
        {GGML_OP_TIMESTEP_EMBEDDING,    {"Range", "Power", "Multiply", "Sin", "Cos", "Concat"}},
        {GGML_OP_TRANSPOSE,             {"Transpose"}},
        {GGML_OP_UPSCALE,               {"Interpolate"}},
        {GGML_OP_VIEW,                  {"Reshape"}},
        {GGML_OP_WIN_PART,              {"StridedSlice", "Concat", "Reshape", "Custom"}},
        {GGML_OP_WIN_UNPART,            {"Reshape", "Transpose", "Custom"}},
    };

    auto it = op_mapping.find(op->op);
    if (it == op_mapping.end()) {
        return false;
    }

    for (const std::string& op_name : it->second) {
        if (openvino_ops.count(op_name) == 0) {
            return false;
        }
    }

    return true;
#endif
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
