#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-device.h"

#include <cassert>
#include <algorithm>

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_base(ggml_metal_t ctx, ggml_op op) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD_ID: op_str = "add_id"; break;
        case GGML_OP_CONCAT: op_str = "concat"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_cpy(ggml_metal_t ctx, ggml_type tsrc, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_pool_2d(ggml_metal_t ctx, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_get_rows(ggml_metal_t ctx, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_set_rows(ggml_metal_t ctx, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_set_rows_%s", ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

int ggml_metal_op_concat(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_base(ctx_enc->ctx, GGML_OP_CONCAT);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_repeat(ggml_metal_t ctx, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_unary(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    const int64_t n = ggml_nelements(op);

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_SCALE:      op_str = "scale";      break;
        case GGML_OP_CLAMP:      op_str = "clamp";      break;
        case GGML_OP_SQR:        op_str = "sqr";        break;
        case GGML_OP_SQRT:       op_str = "sqrt";       break;
        case GGML_OP_SIN:        op_str = "sin";        break;
        case GGML_OP_COS:        op_str = "cos";        break;
        case GGML_OP_LEAKY_RELU: op_str = "leaky_relu"; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_str = "tanh";        break;
                case GGML_UNARY_OP_RELU:        op_str = "relu";        break;
                case GGML_UNARY_OP_SIGMOID:     op_str = "sigmoid";     break;
                case GGML_UNARY_OP_GELU:        op_str = "gelu";        break;
                case GGML_UNARY_OP_GELU_ERF:    op_str = "gelu_erf";    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_str = "gelu_quick";  break;
                case GGML_UNARY_OP_SILU:        op_str = "silu";        break;
                case GGML_UNARY_OP_ELU:         op_str = "elu";         break;
                case GGML_UNARY_OP_NEG:         op_str = "neg";         break;
                case GGML_UNARY_OP_ABS:         op_str = "abs";         break;
                case GGML_UNARY_OP_SGN:         op_str = "sgn";         break;
                case GGML_UNARY_OP_STEP:        op_str = "step";        break;
                case GGML_UNARY_OP_HARDSWISH:   op_str = "hardswish";   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_str = "hardsigmoid"; break;
                case GGML_UNARY_OP_EXP:         op_str = "exp";         break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * suffix = "";
    if (n % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_%s_%s%s", op_str, ggml_type_name(op->src[0]->type), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_glu(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_sum_rows(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_SUM_ROWS:
            op_str = "sum_rows"; break;
        case GGML_OP_MEAN:
            op_str = "mean"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_soft_max(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    const ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_ssm_conv(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_ssm_conv_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_ssm_scan(ggml_metal_t ctx, const ggml_tensor * op)  {
    char base[256];
    char name[256];

    if (op->src[3]->ne[0] == 1) {
        snprintf(base, 256, "kernel_ssm_scan_group_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_ssm_scan_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_rwkv(ggml_metal_t ctx, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    switch (op->op) {
        case GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mv_ext(ggml_metal_t ctx, ggml_type tsrc0, ggml_type tsrc1, int r1ptg) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mm(ggml_metal_t ctx, ggml_type tsrc0, ggml_type tsrc1) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 8192);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mv(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

                nsg = 1;
                nr0 = 1;
                nr1 = 4;
                if (ne00 == 4) {
                    nr0 = 32;
                    suffix = "_c4";
                }
            } break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = 1;
                nr0 = 1;
                if (op->src[1]->type == GGML_TYPE_F32) {
                    if (ne00 == 4) {
                        nr0 = 32;
                        nr1 = 4;
                        suffix = "_c4";
                    } else if (ne11 * ne12 < 4) {
                        suffix = "_1row";
                    } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                        suffix = "_l4";
                        nr1 = ne11;
                    } else {
                        nr1 = 4;
                    }
                } else {
                    nr1 = 4;
                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_nr0 (res, nr0);
    ggml_metal_pipeline_set_nr1 (res, nr1);
    ggml_metal_pipeline_set_nsg (res, nsg);
    ggml_metal_pipeline_set_smem(res, smem);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mm_id_map0(ggml_metal_t ctx, int ne02, int ne20) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    const size_t smem = (size_t) ne02*ne20*sizeof(uint16_t);

    ggml_metal_pipeline_set_smem(res, smem);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mm_id(ggml_metal_t ctx, ggml_type tsrc0, ggml_type tsrc1) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 8192);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mv_id(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
                nsg = 1;
                nr0 = 1;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
                nsg = 1;
                nr0 = 1;
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
                nsg = 1;
                nr0 = 1;
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_nr0 (res, nr0);
    ggml_metal_pipeline_set_nr1 (res, nr1);
    ggml_metal_pipeline_set_nsg (res, nsg);
    ggml_metal_pipeline_set_smem(res, smem);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_argmax(ggml_metal_t ctx, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*(sizeof(float) + sizeof(int32_t)));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_argsort(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_flash_attn_ext(
        ggml_metal_t ctx,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "kernel_%s_%s_dk%d_dv%d_mask=%d_sinks=%d_bias=%d_scap=%d_ns10=%d_ns20=%d_nsg=%d",
            "flash_attn_ext",
            ggml_type_name(op->src[1]->type),
            dk,
            dv,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();

    ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
    ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
    ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
    ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);

    ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
    ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
    ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

    res = ggml_metal_compile_pipeline(ctx, base, name, cv);

    ggml_metal_cv_free(cv);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_flash_attn_ext_vec(
        ggml_metal_t ctx,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext_vec",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "kernel_%s_%s_dk%d_dv%d_mask=%d_sink=%d_bias=%d_softcap=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            "flash_attn_ext_vec",
            ggml_type_name(op->src[1]->type),
            dk,
            dv,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();

    ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
    ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
    ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
    ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);

    ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
    ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
    ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
    ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

    res = ggml_metal_compile_pipeline(ctx, base, name, cv);

    ggml_metal_cv_free(cv);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_t ctx,
        const ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "kernel_flash_attn_ext_vec_reduce_dv=%d_nwg=%d", dv, nwg);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    ggml_metal_cv_t cv = ggml_metal_cv_init();

    ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
    ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

    res = ggml_metal_compile_pipeline(ctx, base, name, cv);

    ggml_metal_cv_free(cv);

    return res;

    GGML_UNUSED(op);
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_bin(
        ggml_metal_t ctx,
        ggml_op op,
        int32_t n_fuse,
        bool row) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD:   op_str = "add";   break;
        case GGML_OP_SUB:   op_str = "sub";   break;
        case GGML_OP_MUL:   op_str = "mul";   break;
        case GGML_OP_DIV:   op_str = "div";   break;
        default: GGML_ABORT("fatal error");
    };

    if (row) {
        snprintf(base, 256, "kernel_%s_row_c4_fuse_%d", op_str, n_fuse);
    } else {
        snprintf(base, 256, "kernel_%s_fuse_%d", op_str, n_fuse);
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_rms_norm(ggml_metal_t ctx, const ggml_tensor * op, int32_t n_fuse) {
    assert(op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(op->src[0]->ne[0] % 4 == 0);
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    switch (n_fuse) {
        case 1: snprintf(base, 256, "kernel_rms_norm_f32");         break;
        case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32");     break;
        case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32"); break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_l2_norm(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_L2_NORM);

    GGML_ASSERT(op->src[0]->ne[0] % 4 == 0);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_l2_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_group_norm(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_norm(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_NORM);

    GGML_ASSERT(op->src[0]->ne[0] % 4 == 0);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    ggml_metal_pipeline_set_smem(res, 32*sizeof(float));

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_rope(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROPE);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[2];

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } else if (is_mrope && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } else if (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_im2col(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IM2COL);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_conv_transpose_1d(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_upscale(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_UPSCALE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_upscale_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_pad(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_pad_reflect_1d(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD_REFLECT_1D);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_arange(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARANGE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_timestep_embedding(ggml_metal_t ctx, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    res = ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    return res;
}

int ggml_metal_op_repeat(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_repeat(ctx_enc->ctx, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separete kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_cpy(ctx_enc->ctx, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_graph_encoder_concurrency_reset(ctx_enc);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_bin(ctx_enc->ctx, GGML_OP_ADD, 1, false);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int ggml_metal_op_scale(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float scale;
    float bias;
    memcpy(&scale, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((const int32_t *) op->op_params) + 1, sizeof(float));

    ggml_metal_kargs_scale args = {
        /*.scale =*/ scale,
        /*.bias  =*/ bias,
    };

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_unary(ctx_enc->ctx, op);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_clamp(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float min;
    float max;
    memcpy(&min, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&max, ((const int32_t *) op->op_params) + 1, sizeof(float));

    ggml_metal_kargs_clamp args = {
        /*.min =*/ min,
        /*.max =*/ max,
    };

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_unary(ctx_enc->ctx, op);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_unary(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_unary(ctx_enc->ctx, op);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         1);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_glu(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_glu(ctx_enc->ctx, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2);

    //[encoder setComputePipelineState:pipeline];
    //[encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
    //if (src1) {
    //    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
    //} else {
    //    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
    //}
    //[encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
    //[encoder setBytes:&args length:sizeof(args) atIndex:3];

    //[encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_sum_rows(ctx_enc->ctx, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    //[encoder setComputePipelineState:pipeline];
    //[encoder setBytes:&args length:sizeof(args) atIndex:0];
    //[encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
    //[encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
    //[encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

    //[encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_get_rows(ctx_enc->ctx, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne10, ne11, ne12, 32, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_set_rows(ctx_enc->ctx, op->type);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_soft_max(ctx_enc->ctx, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes(ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_ssm_conv(ctx_enc->ctx, op);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes(ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op), 3);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne1, ne02, 1, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_scan(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.nb13         =*/ nb13,
        /*.nb21         =*/ nb21,
        /*.nb22         =*/ nb22,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.nb53         =*/ nb53,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_ssm_scan(ctx_enc->ctx, op);

    const size_t sms = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[6]), 7);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         8);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, sms, 0);

    if (ne30 == 1) {
        // Mamba-2
        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, d_inner, n_head, n_seqs, d_state, 1, 1);
    } else {
        GGML_ASSERT(d_inner == 1);
        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, n_head, n_seqs, 1, d_state, 1, 1);
    }

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_rwkv(ctx_enc->ctx, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_cpy(ctx_enc->ctx, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    // TODO: support
    //const int32_t nk00 = ne00/ggml_blck_size(op->type);
    const int32_t nk00 = ne00;

    int nth = 32; // SIMD width

    while (nth < nk00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk00) {
            nrptg = (nth + nk00 - 1)/nk00;
            nth   = nk00;

            if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min(nth, nk00);

    ggml_metal_kargs_cpy args = {
        /*.ne00 =*/ nk00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_2d(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_pool_2d(ctx_enc->ctx, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_mul_mat(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be someting cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_mul_mv_ext(ctx_enc->ctx, op->src[0]->type, op->src[1]->type, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
            /*.nsg   =*/ nsg,
            /*.nxpsg =*/ nxpsg,
            /*.r1ptg =*/ r1ptg,
        };

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        ctx_enc->props_dev->has_simdgroup_mm &&
        op->src[1]->type == GGML_TYPE_F32 &&
        ne00 % 32 == 0 && ne00 >= 64 &&
        (ne11 > ne11_mm_min || (ggml_is_quantized(op->src[0]->type) && ne12 > 1))) {
        //printf("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        switch (op->src[0]->type) {
            case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
            case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
            case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
            default: break;
        }

        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_mul_mm(ctx_enc->ctx, op->src[0]->type, op->src[1]->type);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

        ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
    } else {
        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_mul_mv(ctx_enc->ctx, op);

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        const int nr0 = ggml_metal_pipeline_get_nr0(pipeline);
        const int nr1 = ggml_metal_pipeline_get_nr1(pipeline);
        const int nsg = ggml_metal_pipeline_get_nsg(pipeline);

        const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

        if (op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    return 1;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

int ggml_metal_op_mul_mat_id(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const int ne21_mm_id_min = 32;

    if (ctx_enc->props_dev->has_simdgroup_mm &&
        ne00 % 32 == 0 && ne00 >= 64 &&
        (ne21 >= ne21_mm_id_min)) {
        GGML_ASSERT(ne00 % 4 == 0);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        switch (op->src[0]->type) {
            case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
            case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
            case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
            default: break;
        }

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_mul_mm_id_map0(ctx_enc->ctx, ne02, ne20);

            const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= ctx_enc->props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
            ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_graph_encoder_concurrency_reset(ctx_enc);

        {
            ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_mul_mm_id(ctx_enc->ctx, op->src[0]->type, GGML_TYPE_F16);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
            ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_dst,  5);

            const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

            ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne21 + 31)/32, (ne01 + 63)/64, ne02, 128, 1, 1);
        }
    } else {
        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_mul_mv_id(ctx_enc->ctx, op);

        const int nr0 = ggml_metal_pipeline_get_nr0(pipeline);
        const int nr1 = ggml_metal_pipeline_get_nr1(pipeline);
        const int nsg = ggml_metal_pipeline_get_nsg(pipeline);

        const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes(ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, bid_src0, 1);
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, bid_src1, 2);
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, bid_dst,  3);
        ggml_metal_encoder_set_buffer(ctx_enc->encoder, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

        if (op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_base(ctx_enc->ctx, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t nwg = 32;

    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne03 = op->src[0]->ne[3];
    const int64_t ne20 = op->src[2]->ne[0];

    // temp buffer for writing the results from each workgroup
    // - ne20: the size of the Value head
    // -  + 2: the S and M values for each intermediate result
    return ggml_type_size(GGML_TYPE_F32)*(ne01*ne02*ne03*nwg*(ne20 + 2));
}

int ggml_metal_op_flash_attn_ext(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4  == 0);
    GGML_ASSERT(ne11 % 32 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= GGML_PAD(op->src[0]->ne[1], 8) &&
            "the Flash-Attention Metal kernel requires the mask to be padded to 8 and at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int64_t nqptg = 8;  // queries per threadgroup    !! sync with kernel template arguments !!
        const int64_t ncpsg = 64; // cache values per simdgroup !! sync with kernel template arguments !!

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

        //int64_t nsgmax = 4;
        //
        //if (is_q) {
        //    nsgmax = 2;
        //    while (true) {
        //        const size_t smem = FATTN_SMEM(nsgmax);
        //        if (smem > props_dev->max_theadgroup_memory_size) {
        //            break;
        //        }
        //        nsgmax *= 2;
        //    }
        //    nsgmax /= 2;
        //}

        // simdgroups per threadgroup (a.k.a. warps)
        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
        int32_t nsg = 4;

        const size_t smem = FATTN_SMEM(nsg);

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_flash_attn_ext(ctx_enc->ctx, op, has_mask, has_sinks, has_bias, has_scap, nsg);

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), 3);
        if (op->src[3]) {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[3]), 4);
        } else {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 4);
        }
        if (op->src[4]) {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[4]), 5);
        } else {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 5);
        }
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         6);

        ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
#undef FATTN_SMEM
    } else {
        // half4x4 kernel
        const int64_t nqptg = 1;  // queries per threadgroup    !! sync with kernel template arguments !!
        const int64_t ncpsg = 32; // cache values per simdgroup !! sync with kernel template arguments !!
        const int64_t nkpsg = 1*ncpsg;

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 1  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(GGML_PAD(ne00, 128) + 4*ncpsg*(nsg)) + 2*GGML_PAD(ne20, 128)*(nsg))*(sizeof(float)/2), 16))

        int64_t nsgmax = 2;
        while (true) {
            const size_t smem = FATTN_SMEM(nsgmax);
            // avoid using more than half of the threadgroup memory - can cause slow downs especially for large head sizes
            if (smem > ctx_enc->props_dev->max_theadgroup_memory_size/2) {
                break;
            }
            nsgmax *= 2;
        }
        nsgmax /= 2;

        // simdgroups per threadgroup (a.k.a. warps)
        //const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32)));
        const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) 1024/32)));

        int64_t nsg = 1;
        while (nsg <= nsgt) {
            nsg *= 2;
        }
        nsg /= 2;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (false) {
            // for small KV caches, we could launch a single workgroup and write the results directly to dst/
            // however, this does not lead to significant improvement, so disabled
            nwg = 1;
            nsg = 4;
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*nkpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_flash_attn_ext_vec(ctx_enc->ctx, op, has_mask, has_sinks, has_bias, has_scap, nsg, nwg);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
        ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), 3);
        if (op->src[3]) {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[3]), 4);
        } else {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 4);
        }
        if (op->src[4]) {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[4]), 5);
        } else {
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 5);
        }

        const size_t smem = FATTN_SMEM(nsg);

        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, ctx_enc->props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
        GGML_ASSERT(smem <= ctx_enc->props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, ggml_metal_get_buffer_id(op), 6);

            ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

            // write the results from each workgroup into a temp buffer
            ggml_metal_buffer_id bid_tmp = bid_dst;
            bid_tmp.offs += ggml_nbytes(op);
            ggml_metal_encoder_set_buffer(ctx_enc->encoder, bid_tmp, 6);

            ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_graph_encoder_concurrency_reset(ctx_enc);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                ggml_metal_pipeline_t pipeline0 = ggml_metal_op_get_pipeline_flash_attn_ext_vec_reduce(ctx_enc->ctx, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline0);
                ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
#undef FATTN_SMEM
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_tensor ** ops = ggml_graph_nodes(gf) + idx;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    bool bcast_row = false;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (ctx_enc->use_fusion) {
        fops[0] = GGML_OP_ADD;
        fops[1] = GGML_OP_ADD;
        fops[2] = GGML_OP_ADD;
        fops[3] = GGML_OP_ADD;
        fops[4] = GGML_OP_ADD;
        fops[5] = GGML_OP_ADD;
        fops[6] = GGML_OP_ADD;
        fops[7] = GGML_OP_ADD;

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        for (n_fuse = 0; n_fuse <= 6 && idx + n_fuse + 1 < ctx_enc->idx_end; ++n_fuse) {
            if (!ggml_can_fuse(gf, idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            if (ops[n_fuse] != ops[n_fuse + 1]->src[0]) {
                break;
            }

            // b[0] === b[1] === ...
            if (!ggml_are_same_layout(ops[n_fuse]->src[1], ops[n_fuse + 1]->src[1])) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            ggml_metal_buffer_id bid_fuse = ggml_metal_get_buffer_id(ops[n_fuse + 1]->src[1]);
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[n_fuse + 1] = bid_fuse.offs;
        }

        ++n_fuse;

        if (ctx_enc->debug_fusion > 1 && n_fuse > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    ggml_metal_pipeline_t pipeline = nullptr;

    if (ggml_nelements(op->src[1]) == ne10 && ggml_is_contiguous(op->src[1]) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(op->src[0]));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        pipeline = ggml_metal_op_get_pipeline_bin(ctx_enc->ctx, op->op, n_fuse, true);

        bcast_row = true;
    } else {
        pipeline = ggml_metal_op_get_pipeline_bin(ctx_enc->ctx, op->op, n_fuse, false);
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ops[n_fuse - 1]);

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_graph_encoder_concurrency_check(ctx_enc, ops[i])) {
                ggml_metal_graph_encoder_concurrency_reset(ctx_enc);

                break;
            }
        }
    }

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_dst,  3);

    if (bcast_row) {
        const int64_t n = ggml_nelements(op)/4;

        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, n, 1, 1, 1, 1, 1);
    } else {
        int nth = 32;

        while (16*nth < ne0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_rms_norm(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    ggml_tensor ** ops = ggml_graph_nodes(gf) + idx;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_rms_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = rms_norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (ctx_enc->use_fusion) {
        fops[0] = GGML_OP_RMS_NORM;
        fops[1] = GGML_OP_MUL;
        fops[2] = GGML_OP_ADD;

        for (n_fuse = 0; n_fuse <= 1 && idx + n_fuse + 1 < ctx_enc->idx_end; ++n_fuse) {
            if (!ggml_can_fuse(gf, idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            if (ops[n_fuse] != ops[n_fuse + 1]->src[0]) {
                break;
            }

            if (ops[n_fuse + 1]->src[1]->ne[0] != op->ne[0]) {
                break;
            }

            if (!ggml_is_contiguous_rows(ops[n_fuse + 1]->src[1])) {
                break;
            }

            if (ops[n_fuse + 1]->type != GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            bid_fuse[n_fuse] = ggml_metal_get_buffer_id(ops[n_fuse + 1]->src[1]);

            args.nef1[n_fuse + 1] = ops[n_fuse + 1]->src[1]->ne[1];
            args.nef2[n_fuse + 1] = ops[n_fuse + 1]->src[1]->ne[2];
            args.nef3[n_fuse + 1] = ops[n_fuse + 1]->src[1]->ne[3];

            args.nbf1[n_fuse + 1] = ops[n_fuse + 1]->src[1]->nb[1];
            args.nbf2[n_fuse + 1] = ops[n_fuse + 1]->src[1]->nb[2];
            args.nbf3[n_fuse + 1] = ops[n_fuse + 1]->src[1]->nb[3];
        }

        ++n_fuse;

        if (ctx_enc->debug_fusion > 1 && n_fuse > 1) {
            if (n_fuse == 2) {
                GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL\n", __func__);
            }
            if (n_fuse == 3) {
                GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL + ADD\n", __func__);
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ops[n_fuse - 1]);

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_graph_encoder_concurrency_check(ctx_enc, ops[i])) {
                ggml_metal_graph_encoder_concurrency_reset(ctx_enc);

                break;
            }
        }
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_rms_norm(ctx_enc->ctx, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, bid_dst, 4);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_l2_norm(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    int nth = 32; // SIMD width

    ggml_metal_kargs_l2_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.eps    =*/ eps,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_l2_norm(ctx_enc->ctx, op);

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    const int64_t nrows = ggml_nrows(op->src[0]);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_group_norm(ctx_enc->ctx, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.eps    =*/ eps,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_norm(ctx_enc->ctx, op);

    int nth = 32; // SIMD width
    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00/4);

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    const int64_t nrows = ggml_nrows(op->src[0]);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_rope(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_rope(ctx_enc->ctx, op);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;


    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_im2col(ctx_enc->ctx, op);

    const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
    const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, quotient * CHW, OH, OW, n_threads, 1, 1);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_conv_transpose_1d(ctx_enc->ctx, op);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const float sf0 = (float)ne0/op->src[0]->ne[0];
    const float sf1 = (float)ne1/op->src[0]->ne[1];
    const float sf2 = (float)ne2/op->src[0]->ne[2];
    const float sf3 = (float)ne3/op->src[0]->ne[3];

    ggml_metal_kargs_upscale args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0 =*/ ne0,
        /*.ne1 =*/ ne1,
        /*.ne2 =*/ ne2,
        /*.ne3 =*/ ne3,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
        /*.sf0 =*/ sf0,
        /*.sf1 =*/ sf1,
        /*.sf2 =*/ sf2,
        /*.sf3 =*/ sf3
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_upscale(ctx_enc->ctx, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_pad(ctx_enc->ctx, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_pad_reflect_1d(ctx_enc->ctx, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_arange(ctx_enc->ctx, op);

    //[encoder setComputePipelineState:pipeline];
    //[encoder setBuffer:id_dst  offset:offs_dst  atIndex:0];
    //[encoder setBytes:&args length:sizeof(args) atIndex:1];

    //[encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_timestep_embedding(ctx_enc->ctx, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_argmax(ctx_enc->ctx, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = ggml_metal_pipeline_get_smem(pipeline);

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    // bitonic sort requires the number of elements to be power of 2
    int64_t ne00_padded = 1;
    while (ne00_padded < ne00) {
        ne00_padded *= 2;
    }

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_argsort(ctx_enc->ctx, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(ne00_padded*sizeof(int32_t), 16);

    ggml_metal_kargs_argsort args = {
        /*.ncols =*/ ne00,
        /*.ncols_pad =*/ ne00_padded
    };

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(ctx_enc->encoder, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, 1, nrows, 1, ne00_padded, 1, 1);

    return 1;
}

int ggml_metal_op_leaky_relu(ggml_metal_graph_encoder_t ctx_enc, int idx) {
    ggml_cgraph * gf = ctx_enc->gf;
    ggml_tensor * op = ggml_graph_node(gf, idx);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint32_t, nb,  op,         nb);

    float slope;
    memcpy(&slope, op->op_params, sizeof(float));

    ggml_metal_kargs_leaky_relu args = {
        /*.slope =*/ slope
    };

    ggml_metal_pipeline_t pipeline = ggml_metal_op_get_pipeline_unary(ctx_enc->ctx, op);

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_encoder_set_pipeline(ctx_enc->encoder, pipeline);
    ggml_metal_encoder_set_bytes   (ctx_enc->encoder, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx_enc->encoder, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(ctx_enc->encoder, n, 1, 1, 1, 1, 1);

    return 1;
}
