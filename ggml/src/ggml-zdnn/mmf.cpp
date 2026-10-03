#include "ggml.h"
#include "mmf.hpp"

// Cache the PARMBLKFORMAT_1 availability check (z16+ feature)
static bool g_has_parmblkformat_1_checked = false;
static bool g_has_parmblkformat_1 = false;

static bool has_parmblkformat_1() {
    if (!g_has_parmblkformat_1_checked) {
        g_has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        g_has_parmblkformat_1_checked = true;
    }
    return g_has_parmblkformat_1;
}

void ggml_zdnn_mul_mat_f(
    const ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const enum ggml_type type = src0->type;

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

    const ggml_tensor * weights = src0;
    const ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;
    ggml_backend_zdnn_buffer * inputs_extra  = (ggml_backend_zdnn_buffer *)inputs->extra;
    ggml_backend_zdnn_buffer * output_extra  = (ggml_backend_zdnn_buffer *)output->extra;
    ggml_backend_zdnn_buffer * bias_extra    = (ggml_backend_zdnn_buffer *)output_extra->extra;

    const int64_t weights_rows = ne01;
    const int64_t weights_cols = ne00;
    const int64_t inputs_rows  = ne11;
    const int64_t inputs_cols  = ne10;

    assert(inputs_cols == weights_cols);

    const int64_t output_rows = ne1;
    const int64_t output_cols = ne0;

    // GGML_LOG_INFO("%s: tensor '%s' tensor dimensions: [%ld, %ld, %ld, %ld] pre_tfm_desc dimensions: [%ld, %ld, %ld, %ld]\n",
    //               __func__, weights_extra->name,
    //               weights->ne[3], weights->ne[2], weights->ne[1], weights->ne[0],
    //               weights_extra->pre_tfm_desc.dim1,
    //               weights_extra->pre_tfm_desc.dim2,
    //               weights_extra->pre_tfm_desc.dim3,
    //               weights_extra->pre_tfm_desc.dim4);

    // GGML_LOG_INFO("%s: tensor '%s' tensor dimensions: [%ld, %ld, %ld, %ld] pre_tfm_desc dimensions: [%ld, %ld, %ld, %ld]\n",
    //               __func__, inputs_extra->name,
    //               inputs->ne[3], inputs->ne[2], inputs->ne[1], inputs->ne[0],
    //               inputs_extra->pre_tfm_desc.dim1,
    //               inputs_extra->pre_tfm_desc.dim2,
    //               inputs_extra->pre_tfm_desc.dim3,
    //               inputs_extra->pre_tfm_desc.dim4);

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[1]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim1  == inputs->ne[0]  && "inputs_extra->pre_tfm_desc.dim1 must match inputs->ne[0]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim2  == inputs->ne[1]  && "inputs_extra->pre_tfm_desc.dim2 must match inputs->ne[1]");

    // zdnn_matmul_transpose_op requires PARMBLKFORMAT_1 (z16+)
    // For z15, we would need to use zdnn_matmul_op with pre-transposed weights
    if (!has_parmblkformat_1()) {
        GGML_LOG_ERROR("%s: zdnn_matmul_transpose_op requires z16+ hardware (PARMBLKFORMAT_1)\n", __func__);
        GGML_LOG_ERROR("%s: z15 support with zdnn_matmul_op is not yet implemented\n", __func__);
        GGML_ABORT("z15 matmul not supported - requires z16+ hardware");
    }

    // Reset destination tensor if already transformed (required by zDNN)
    if (output_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&output_extra->ztensor);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        false, true, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->data));

    GGML_UNUSED(ctx);
    GGML_UNUSED(weights_rows);
    GGML_UNUSED(weights_cols);
    GGML_UNUSED(inputs_rows);
    GGML_UNUSED(inputs_cols);
    GGML_UNUSED(output_rows);
    GGML_UNUSED(output_cols);
}
