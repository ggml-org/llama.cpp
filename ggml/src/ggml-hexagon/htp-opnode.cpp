#include "htp-opnode.h"

#include "htp/allreduce-ops.h"
#include "htp/argsort-ops.h"
#include "htp/binary-ops.h"
#include "htp/cumsum-ops.h"
#include "htp/diag-ops.h"
#include "htp/fill-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/get-rows-ops.h"
#include "htp/glu-ops.h"
#include "htp/matmul-ops.h"
#include "htp/pad-ops.h"
#include "htp/repeat-ops.h"
#include "htp/roll-ops.h"
#include "htp/softmax-ops.h"
#include "htp/solve-tri-ops.h"
#include "htp/ssm-conv.h"
#include "htp/sum-rows-ops.h"
#include "htp/unary-ops.h"
#include "htp/upscale-ops.h"

#include <stdio.h>

std::string htp_opnode::name() const {
    switch (opcode) {
        case HTP_OP_MDEV_GROUP:    return "MDEV_GROUP";
        case HTP_OP_CPY_FENCE:     return "CPY+FENCE";
        case HTP_OP_FENCE:         return (node && node->op_params[1] == 0) ? "FENCE_WAIT" : "FENCE_SIGNAL";
        case HTP_OP_ALLREDUCE:     return "ALLREDUCE";
        case HTP_OP_ALLREDUCE_ADD: return "ALLREDUCE+ADD";
        case HTP_OP_HADAMARD:      return "MUL_MAT_HADAMARD";
        case HTP_OP_MUL_MAT_NX:    return "MUL_MAT_NX";
        case HTP_OP_MUL_MAT_ID_NX: return "MUL_MAT_ID_NX";
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_MUL:
        case HTP_OP_DIV:
            if (!fused.empty()) {
                const auto * kp = kparams<htp_binary_kernel_params>();
                if (kp->kernel_type != HTP_BINARY_KERNEL_SCALAR) {
                    switch (opcode) {
                        case HTP_OP_ADD: return "ADD_NX";
                        case HTP_OP_SUB: return "SUB_NX";
                        case HTP_OP_MUL: return "MUL_NX";
                        case HTP_OP_DIV: return "DIV_NX";
                        default: break;
                    }
                }
            }
            break;
        default:
            break;
    }

    if (!node) {
        return "";
    }
    if (htp_is_hadamard(node)) {
        return "MUL_MAT_HADAMARD";
    }

    std::string res = ggml_op_desc(node);
    for (const auto * t : fused) {
        res += "+";
        res += ggml_op_desc(t);
    }
    return res;
}

static const char * format_tensor_name(const struct ggml_tensor * t) {
    return t ? t->name : "NONE";
}

static std::string format_tensor_dims(const struct ggml_tensor * t) {
    if (!t) {
        return "NONE";
    }
    char buf[64];
    if (t->ne[2] == 1 && t->ne[3] == 1) {
        snprintf(buf, sizeof(buf), "%d:%d", (int) t->ne[0], (int) t->ne[1]);
    } else {
        snprintf(buf, sizeof(buf), "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
    }
    return buf;
}

static const char * format_tensor_type(const struct ggml_tensor * t) {
    return t ? ggml_type_name(t->type) : "NONE";
}

static std::string format_tensor_strides(const struct ggml_tensor * t) {
    if (!t) {
        return "NONE";
    }
    char buf[64];
    const char * c = ggml_is_contiguous(t) ? "" : "!";
    if (t->ne[2] == 1 && t->ne[3] == 1) {
        snprintf(buf, sizeof(buf), "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
    } else {
        snprintf(buf, sizeof(buf), "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
    }
    return buf;
}

static const char * format_tensor_buff(const struct ggml_tensor * t) {
    if (t && t->buffer) {
        return ggml_backend_buffer_name(t->buffer);
    }
    return "NONE";
}

template <typename Fn>
static std::string format_op_tensors(const htp_opnode & node, Fn fn) {
    std::string res;
    const auto & inputs  = node.get_inputs();
    const auto & outputs = node.get_outputs();

    if (!inputs.empty()) {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (i > 0) res += " x ";
            res += fn(inputs[i]);
        }
        res += " -> ";
    }

    if (outputs.empty()) {
        res += fn(node.dst());
    } else {
        for (size_t i = 0; i < outputs.size(); i++) {
            if (i > 0) res += " x ";
            res += fn(outputs[i]);
        }
    }
    return res;
}

static std::string format_kernel_params(const htp_opnode & node) {
    char str[128];
    const size_t max_size = sizeof(str);

    if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
        node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX ||
        node.opcode == HTP_OP_MUL_MAT_ADD) {
        const auto * kparams = node.kparams<htp_mm_kernel_params>();
        const char * path = "unknown";
        int32_t type = kparams->kernel_type;
        if (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
            path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
        } else if (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                   type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
            path = "hvx-flat";
        } else if (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
            path = "hvx-tiled";
        }
        snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
        const auto * kparams = node.kparams<htp_fa_kernel_params>();
        const char * path = "unknown";
        int32_t type = kparams->kernel_type;
        if (type == HTP_FA_KERNEL_HMX) {
            path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
        } else if (type == HTP_FA_KERNEL_HVX) {
            path = "hvx";
        }
        snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
    } else if (htp_op_is_unary(node.opcode)) {
        const auto * kparams = node.kparams<htp_unary_kernel_params>();
        snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_MDEV_GROUP && node.node) {
        snprintf(str, max_size, "idx %d count %d", (int) node.node->op_params[0], (int) node.dst()->ne[1]);
    } else if ((node.opcode == HTP_OP_FENCE || node.opcode == HTP_OP_CPY_FENCE) && node.node) {
        snprintf(str, max_size, "seq 0x%x", (uint32_t) node.node->op_params[0]);
    } else if (node.opcode == HTP_OP_ALLREDUCE && node.node) {
        snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) node.node->op_params[0], (uint32_t) node.node->op_params[1]);
    } else if (node.opcode == HTP_OP_SSM_CONV) {
        const auto * kparams = node.kparams<htp_ssm_conv_kernel_params>();
        snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_SOFTMAX) {
        const auto * kparams = node.kparams<htp_softmax_kernel_params>();
        snprintf(str, max_size, "k%d nth %d vtcm %d", (int) kparams->kernel_type, (int) kparams->n_threads, (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_GATED_DELTA_NET) {
        const auto * kparams = node.kparams<htp_gdn_kernel_params>();
        const char * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
        snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                 (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID ||
               node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV) {
        const auto * kparams = node.kparams<htp_binary_kernel_params>();
        snprintf(str, max_size, "vtcm %u", (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K) {
        const auto * kparams = node.kparams<htp_sort_kernel_params>();
        snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                 node.opcode == HTP_OP_TOP_K ? "top_k" : "argsort",
                 (int) kparams->n_threads, (int) kparams->n_chunks,
                 (int) kparams->chunk_elems, (int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_GET_ROWS) {
        const auto * kparams = node.kparams<htp_get_rows_kernel_params>();
        const char * ktype_str = "unknown";
        switch (kparams->kernel_type) {
            case HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; break;
            case HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; break;
            case HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; break;
        }
        snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned int) kparams->vtcm_size);
    } else if (htp_op_is_glu(node.opcode)) {
        const auto * kparams = node.kparams<htp_glu_kernel_params>();
        snprintf(str, max_size, "nth %d vtcm %u", (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_PAD) {
        const auto * kparams = node.kparams<htp_pad_kernel_params>();
        snprintf(str, max_size, "k%d nth %d vtcm %u", (int) kparams->kernel_type, (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_CUMSUM) {
        const auto * kparams = node.kparams<htp_cumsum_kernel_params>();
        snprintf(str, max_size, "%s nth %d vtcm %u", kparams->use_dma ? "dma" : "hvx", (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_FILL) {
        const auto * kparams = node.kparams<htp_fill_kernel_params>();
        snprintf(str, max_size, "%s nth %d", kparams->kernel_type == HTP_FILL_KERNEL_CONTIGUOUS ? "contig" : "strided", (int) kparams->n_threads);
    } else if (node.opcode == HTP_OP_ROLL) {
        const auto * kparams = node.kparams<htp_roll_kernel_params>();
        snprintf(str, max_size, "k%d nth %d", (int) kparams->kernel_type, (int) kparams->n_threads);
    } else if (node.opcode == HTP_OP_DIAG) {
        const auto * kparams = node.kparams<htp_diag_kernel_params>();
        snprintf(str, max_size, "%s nth %d vtcm %u", kparams->use_dma ? "dma" : "hvx", (int) kparams->n_threads, (unsigned int) kparams->vtcm_size);
    } else if (node.opcode == HTP_OP_SOLVE_TRI) {
        const auto * kparams = node.kparams<htp_solve_tri_kernel_params>();
        snprintf(str, max_size, "%s nth %d", kparams->batched ? "batched" : "chunked", (int) kparams->n_threads);
    } else if (node.opcode == HTP_OP_UPSCALE) {
        const auto * kparams = node.kparams<htp_upscale_kernel_params>();
        const char * ktype_str = "unknown";
        switch (kparams->kernel_type) {
            case HTP_UPSCALE_KERNEL_NEAREST:     ktype_str = "nearest";     break;
            case HTP_UPSCALE_KERNEL_NEAREST_2X:  ktype_str = "nearest-2x";  break;
            case HTP_UPSCALE_KERNEL_NEAREST_4X:  ktype_str = "nearest-4x";  break;
            case HTP_UPSCALE_KERNEL_BILINEAR:    ktype_str = kparams->align_corners ? "bilinear-ac" : "bilinear"; break;
            case HTP_UPSCALE_KERNEL_BILINEAR_2X: ktype_str = "bilinear-2x"; break;
            case HTP_UPSCALE_KERNEL_BILINEAR_AA: ktype_str = "bilinear-aa"; break;
            case HTP_UPSCALE_KERNEL_BICUBIC:     ktype_str = "bicubic";     break;
        }
        snprintf(str, max_size, "%s nth %d chk %d vtcm %u", ktype_str, (int) kparams->n_threads, (int) kparams->dst_nrows_per_chunk, (unsigned int) kparams->vtcm_size);
    } else {
        snprintf(str, max_size, "----");
    }
    return str;
}

static void append_section(std::string & res, const std::string & sec) {
    if (!res.empty()) {
        res += "|";
    }
    res += sec;
}

std::string htp_format_tensor(const ggml_tensor * t, uint32_t flags) {
    std::string res;
    if (flags & HTP_OPFMT_NAMES)   append_section(res, format_tensor_name(t));
    if (flags & HTP_OPFMT_DIMS)    append_section(res, format_tensor_dims(t));
    if (flags & HTP_OPFMT_TYPES)   append_section(res, format_tensor_type(t));
    if (flags & HTP_OPFMT_STRIDES) append_section(res, format_tensor_strides(t));
    if (flags & HTP_OPFMT_BUFFS)   append_section(res, format_tensor_buff(t));
    return res;
}

std::string htp_format_op(const htp_opnode & node, uint32_t flags) {
    std::string res;
    if (flags & HTP_OPFMT_OP)      append_section(res, node.name());
    if (flags & HTP_OPFMT_NAMES)   append_section(res, format_op_tensors(node, format_tensor_name));
    if (flags & HTP_OPFMT_DIMS)    append_section(res, format_op_tensors(node, format_tensor_dims));
    if (flags & HTP_OPFMT_TYPES)   append_section(res, format_op_tensors(node, format_tensor_type));
    if (flags & HTP_OPFMT_STRIDES) append_section(res, format_op_tensors(node, format_tensor_strides));
    if (flags & HTP_OPFMT_BUFFS)   append_section(res, format_op_tensors(node, format_tensor_buff));
    if (flags & HTP_OPFMT_KPARAMS) append_section(res, format_kernel_params(node));
    return res;
}
