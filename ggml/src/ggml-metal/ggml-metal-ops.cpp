#include "ggml-metal-ops.h"

#include "ggml-metal-impl.h"

#include <cassert>

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

ggml_metal_pipeline_t ggml_metal_op_flash_attn_ext_get_pipeline(
        ggml_tensor * op,
        ggml_metal_t ctx,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg) {
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

ggml_metal_pipeline_t ggml_metal_op_flash_attn_ext_vec_get_pipeline(
        ggml_tensor * op,
        ggml_metal_t ctx,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg,
        int32_t nwg) {
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

ggml_metal_pipeline_t ggml_metal_op_flash_attn_ext_vec_reduce_get_pipeline(
        ggml_tensor * op,
        ggml_metal_t ctx,
        int32_t dv,
        int32_t nwg) {
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

ggml_metal_pipeline_t ggml_metal_op_bin_get_pipeline(
        enum ggml_op op,
        ggml_metal_t ctx,
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

    return ggml_metal_compile_pipeline(ctx, base, name, nullptr);
}

ggml_metal_pipeline_t ggml_metal_op_rms_norm_get_pipeline(
        ggml_tensor * op,
        ggml_metal_t ctx,
        int32_t n_fuse) {
    char base[256];
    char name[256];

    switch (n_fuse) {
        case 1: snprintf(base, 256, "kernel_rms_norm");         break;
        case 2: snprintf(base, 256, "kernel_rms_norm_mul");     break;
        case 3: snprintf(base, 256, "kernel_rms_norm_mul_add"); break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_t res = ggml_metal_get_pipeline(ctx, name);
    if (res) {
        return res;
    }

    return ggml_metal_compile_pipeline(ctx, base, name, nullptr);

    GGML_UNUSED(op);
}
