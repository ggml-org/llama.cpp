#include "ggml-hrx-ops.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>

bool ggml_hrx_supports_flash_attn_ext(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask = op->src[3];
    if (q == nullptr || k == nullptr || v == nullptr) {
        return false;
    }
    // Attention sinks and kv_rows are not supported.
    if (op->src[4] != nullptr || op->src[5] != nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }
    if (mask != nullptr && mask->type != GGML_TYPE_F16) {
        return false;
    }
    // ALiBi and logit softcapping are not supported.
    if (ggml_get_op_params_f32(op, 1) != 0.0f || ggml_get_op_params_f32(op, 2) != 0.0f) {
        return false;
    }

    const int64_t d_k  = k->ne[0];
    const int64_t d_v  = v->ne[0];
    const int64_t n_kv = k->ne[1];

    // The scalar kernel holds at most 256 channels per head.
    if (d_k < 1 || d_k > 256 || d_v < 1 || d_v > 256 || q->ne[0] != d_k || v->ne[1] != n_kv) {
        return false;
    }
    if (k->ne[2] < 1 || k->ne[3] < 1 || v->ne[2] < 1 || v->ne[3] < 1) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || q->ne[3] % k->ne[3] != 0 || q->ne[3] % v->ne[3] != 0) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t)) {
        return false;
    }
    if (op->ne[0] != d_v || op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1] || op->ne[3] != q->ne[3] ||
        !ggml_is_contiguous(op)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(q, s) || !ggml_hrx_elem_strides(k, s) || !ggml_hrx_elem_strides(v, s)) {
        return false;
    }
    if (mask != nullptr) {
        if (mask->nb[0] != sizeof(ggml_fp16_t) || mask->ne[0] < n_kv || mask->ne[1] < q->ne[1] ||
            mask->ne[2] < 1 || mask->ne[3] < 1 || q->ne[2] % mask->ne[2] != 0 || q->ne[3] % mask->ne[3] != 0 || !ggml_hrx_elem_strides(mask, s)) {
            return false;
        }
        if (ggml_nbytes(mask) / 2 >= INT_MAX) {
            return false;
        }
    }
    // Element indices and spans are 32-bit launch scalars.
    return ggml_nelements(op) < INT_MAX && ggml_nbytes(q) / 4 < INT_MAX && ggml_nbytes(k) / 2 < INT_MAX &&
           ggml_nbytes(v) / 2 < INT_MAX;
}

// The WMMA kernel gets wave64 providers for gfx11 and gfx12 from motifs/flash_attn.loom.
// Other architectures use the scalar kernel.
static bool ggml_hrx_flash_attn_ext_wmma_arch(const ggml_hrx_device_context * dev) {
    const ggml_hrx_matrix_op op = dev->arch->matrix_op;
    return (op == GGML_HRX_MATRIX_OP_WMMA_GFX11 || op == GGML_HRX_MATRIX_OP_WMMA_GFX12) && dev->arch->wave64;
}

// The WMMA kernel loads 16 x 16 K and V fragments directly from the views.
// The HRX base address keeps the 16-byte alignment of the binding offset.
static bool ggml_hrx_flash_attn_ext_use_wmma(const ggml_hrx_device_context * dev, const ggml_tensor * k, const ggml_tensor * v) {
    if (!ggml_hrx_flash_attn_ext_wmma_arch(dev)) {
        return false;
    }
    const int64_t d_k = k->ne[0];
    const int64_t d_v = v->ne[0];
    if (d_k % 16 != 0 || d_v % 16 != 0 || d_k > 256 || d_v > 256 || k->ne[1] < 16) {
        return false;
    }
    if (k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3]) {
        return false;
    }
    for (const ggml_tensor * t : { k, v }) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
            return false;
        }
        // Fragment addresses in one 16-token window must fit in 32 bits.
        if (t->nb[1] / 2 > 16777216) {
            return false;
        }
    }
    return true;
}

static int64_t ggml_hrx_flash_attn_ext_largest_divisor(int64_t n, int64_t limit) {
    int64_t best = 1;
    for (int64_t d = 2; d <= std::min(n, limit); ++d) {
        if (n % d == 0) {
            best = d;
        }
    }
    return best;
}

// More heads per tile share each K/V load. Fewer heads per tile give more workgroups.
static int64_t ggml_hrx_flash_attn_ext_heads_per_tile(const ggml_hrx_tune * tune, int64_t n_q, int64_t n_head, int64_t ratio,
                                                      int64_t n_seq) {
    auto n_wg = [&](int64_t hg) {
        return (n_q * hg + 15) / 16 * (n_head / hg) * n_seq;
    };
    const int64_t target = std::min(tune->fa_target_workgroups, n_wg(1));
    int64_t       best   = 1;
    for (int64_t hg = 2; hg <= std::min<int64_t>(ratio, 16); ++hg) {
        if (ratio % hg == 0 && n_wg(hg) >= target) {
            best = hg;
        }
    }
    return best;
}

// The waves of a tile split its KV. Few tiles get more waves.
static int64_t ggml_hrx_flash_attn_ext_waves(const ggml_hrx_tune * tune, int64_t n_tiles, int64_t n_kv) {
    const int64_t n_sub = (n_kv + 15) / 16;
    if (n_tiles >= tune->fa_many_tiles) {
        return 4;
    }
    if (n_sub >= tune->fa_long_kv_subblocks) {
        return 16;
    }
    return n_sub >= tune->fa_mid_kv_subblocks ? 8 : 4;
}

bool ggml_hrx_op_flash_attn_ext(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * q    = node->src[0];
    const ggml_tensor * k    = node->src[1];
    const ggml_tensor * v    = node->src[2];
    const ggml_tensor * mask = node->src[3];

    int64_t qs[GGML_MAX_DIMS];
    int64_t ks[GGML_MAX_DIMS];
    int64_t vs[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(q, qs);
    ggml_hrx_elem_strides(k, ks);
    ggml_hrx_elem_strides(v, vs);

    // Without a mask, the kernel does not read the mask binding. Q fills that binding.
    int64_t ms[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    int64_t m_ne2  = 1;
    int64_t m_ne3  = 1;
    int64_t m_span = 1;
    if (mask != nullptr) {
        ggml_hrx_elem_strides(mask, ms);
        m_ne2  = mask->ne[2];
        m_ne3  = mask->ne[3];
        m_span = (int64_t) (ggml_nbytes(mask) / 2);
    }

    const float scale = ggml_get_op_params_f32(node, 0);
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    if (ggml_hrx_flash_attn_ext_use_wmma(ctx->device, k, v)) {
        const int64_t n_q    = q->ne[1];
        const int64_t n_head = q->ne[2];
        const int64_t n_seq  = q->ne[3];
        const int64_t n_kv   = k->ne[1];
        const int64_t ratio  = n_head / k->ne[2];
        const ggml_hrx_tune * tune = ctx->device->tune;
        const int64_t n_sub  = (n_kv + 15) / 16;
        const bool    small  = n_q <= tune->fa_small_batch;
        int64_t       hg     = ggml_hrx_flash_attn_ext_heads_per_tile(tune, n_q, n_head, ratio, n_seq);
        int64_t       n_x    = (n_q * hg + 15) / 16;
        int64_t       waves  = ggml_hrx_flash_attn_ext_waves(tune, n_x * (n_head / hg) * n_seq, n_kv);
        int64_t       chunk  = n_sub;
        if (small) {
            // Small batches split the KV across workgroups. All heads of a KV head share one tile.
            // The split depends only on n_kv, so a row has the same reduction order for each batch size.
            hg    = ggml_hrx_flash_attn_ext_largest_divisor(ratio, 16);
            n_x   = (n_q * hg + 15) / 16;
            waves = tune->fa_split_waves;
            chunk = std::max(tune->fa_min_split_subblocks, (n_sub + tune->fa_max_splits - 1) / tune->fa_max_splits);
        }
        const int64_t n_split = (n_sub + chunk - 1) / chunk;

        const ggml_tensor * part      = q;
        int64_t             part_span = 1;
        if (n_split > 1) {
            part_span = n_split * (ggml_nelements(node) + 2 * ggml_nelements(node) / v->ne[0]);
            part      = ggml_hrx_scratch(ctx, (size_t) part_span * sizeof(float));
            if (part == nullptr || part_span >= INT_MAX) {
                return false;
            }
        }

        ggml_hrx_kernel_desc desc;
        desc.file   = "flash_attn/flash_attn_ext_f32_f16_wmma.loom";
        desc.symbol = "ggml_flash_attn_ext_f32_f16_wmma";
        desc.libs   = { "motifs/flash_attn.loom" };
        desc.config = { { "ggml.flash_attn.waves", std::to_string(waves) } };
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

        if (!ggml_hrx_dispatch(ctx, kernel, { n_x, n_head / hg, n_seq * n_split },
                {
                    n_q, n_head, k->ne[0], v->ne[0], n_kv, hg, ratio, n_seq / k->ne[3],
                    qs[1], qs[2], qs[3], (int64_t) (ggml_nbytes(q) / 4),
                    ks[1], ks[2], ks[3], vs[1], vs[2], vs[3],
                    mask != nullptr ? 1 : 0, m_ne2, m_ne3, ms[1], ms[2], ms[3], m_span,
                    ggml_nelements(node), (int64_t) scale_bits, n_split, chunk, part_span,
                },
                { q, k, v, mask != nullptr ? mask : q, node, part })) {
            return false;
        }
        if (n_split == 1) {
            return true;
        }

        ggml_hrx_kernel_desc rdesc;
        rdesc.file   = "flash_attn/flash_attn_ext_reduce_f32.loom";
        rdesc.symbol = "ggml_flash_attn_ext_reduce_f32";
        ggml_hrx_kernel * reduce = ggml_hrx_get_kernel(ctx->device, rdesc);
        return ggml_hrx_dispatch(ctx, reduce, { ggml_nelements(node) / v->ne[0] },
            { n_split, v->ne[0], ggml_nelements(node), part_span }, { part, node });
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "flash_attn/flash_attn_ext_f32_f16.loom";
    desc.symbol = "ggml_flash_attn_ext_f32_f16";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n_q    = q->ne[1];
    const int64_t n_head = q->ne[2];
    const int64_t n_seq  = q->ne[3];
    return ggml_hrx_dispatch(ctx, kernel, { n_q, n_head, n_seq },
        {
            n_q, n_head, n_seq, k->ne[0], v->ne[0], k->ne[1],
            qs[1], qs[2], qs[3], (int64_t) (ggml_nbytes(q) / 4),
            n_head / k->ne[2], n_seq / k->ne[3], ks[1], ks[2], ks[3], (int64_t) (ggml_nbytes(k) / 2),
            n_head / v->ne[2], n_seq / v->ne[3], vs[1], vs[2], vs[3], (int64_t) (ggml_nbytes(v) / 2),
            mask != nullptr ? 1 : 0, m_ne2, m_ne3, ms[1], ms[2], ms[3], m_span,
            ggml_nelements(node), (int64_t) scale_bits,
        },
        { q, k, v, mask != nullptr ? mask : q, node });
}
