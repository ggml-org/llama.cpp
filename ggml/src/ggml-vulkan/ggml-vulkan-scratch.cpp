#include "ggml-vulkan-common.h"

size_t ggml_vk_scratch_offset(const vk_device & device, size_t nbytes) {
    return ggml_vk_align_size(nbytes, device->properties.limits.minStorageBufferOffsetAlignment);
}

vk_fa_dequant_kv ggml_vk_fa_dequant_kv(const vk_device & device, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v) {
    auto is_dense_kv_cache = [](const ggml_tensor * t) {
        return t->nb[0] == ggml_type_size(t->type) &&
               t->nb[2] == ggml_row_size(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };

    const uint32_t HSK  = (uint32_t)k->ne[0];
    const uint32_t HSV  = (uint32_t)v->ne[0];
    const int64_t  neq1 = q->ne[1];

    const bool k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    const bool v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    const bool kv_raw  = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    const bool kv_raw_strided = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_size(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_size(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    const size_t scratch_ts = kv_raw_strided ? ggml_type_size(k->type) : sizeof(ggml_fp16_t);
    const bool use = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                     is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                     (uint64_t)ggml_nelements(k) * scratch_ts <= device->properties.limits.maxStorageBufferRange &&
                     (uint64_t)ggml_nelements(v) * scratch_ts <= device->properties.limits.maxStorageBufferRange &&
                     (kv_raw_strided ||
                      (device->pipeline_dequant_transpose[k->type] != nullptr &&
                       device->pipeline_dequant_transpose[v->type] != nullptr)) &&
                     // coopmat2 path does not benefit from the f16 scratch
                     !device->coopmat2 &&
                     // Intel Xe1 regresses, see PR 25494
                     (device->vendor_id != VK_VENDOR_ID_INTEL ||
                      (device->coopmat_support && device->architecture != vk_device_architecture::INTEL_XE1));

    return { use, kv_raw_strided, scratch_ts };
}

size_t ggml_vk_op_scratch_size(const vk_device & device, const ggml_tensor * dst) {
    size_t scratch = 0;
    switch (dst->op) {
        case GGML_OP_FLASH_ATTN_EXT: {
            const vk_fa_dequant_kv dkv = ggml_vk_fa_dequant_kv(device, dst->src[0], dst->src[1], dst->src[2]);
            if (dkv.use) {
                const size_t k_sz = (size_t)ggml_nelements(dst->src[1]) * dkv.scratch_ts;
                const size_t v_sz = (size_t)ggml_nelements(dst->src[2]) * dkv.scratch_ts;
                scratch = ggml_vk_scratch_offset(device, k_sz) + v_sz;
            }
        } break;
        default:
            break;
    }
    if (scratch != 0 &&
        ggml_vk_scratch_offset(device, ggml_nbytes(dst)) + scratch > device->max_memory_allocation_size) {
        return 0;
    }
    return scratch;
}

size_t ggml_vk_tensor_alloc_size(const vk_device & device, const ggml_tensor * dst) {
    const size_t nbytes = ggml_nbytes(dst);
    if (dst->view_src != nullptr) {
        return nbytes;
    }
    const size_t scratch = ggml_vk_op_scratch_size(device, dst);
    if (scratch == 0) {
        return nbytes;
    }
    return ggml_vk_scratch_offset(device, nbytes) + scratch;
}

bool ggml_vk_scratch_available(const ggml_backend_vk_context * ctx, const ggml_tensor * dst) {
    if (dst->view_src != nullptr) {
        return false;
    }
    if (dst->buffer == nullptr || !ggml_backend_buffer_is_vk(dst->buffer)) {
        return false;
    }
    const size_t scratch = ggml_vk_op_scratch_size(ctx->device, dst);
    if (scratch == 0) {
        return false;
    }
    // meta backend (-sm tensor) reserves for the full tensor, slices may not fit
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    return vk_tensor_offset(dst) + ggml_vk_scratch_offset(ctx->device, ggml_nbytes(dst)) + scratch <= buf_ctx->dev_buffer->size;
}

vk_subbuffer ggml_vk_scratch_subbuffer(const ggml_backend_vk_context * ctx, const ggml_tensor * dst, size_t offset, size_t size) {
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    const size_t base = vk_tensor_offset(dst) + ggml_vk_scratch_offset(ctx->device, ggml_nbytes(dst)) + offset;
    return { buf_ctx->dev_buffer, base, size };
}
