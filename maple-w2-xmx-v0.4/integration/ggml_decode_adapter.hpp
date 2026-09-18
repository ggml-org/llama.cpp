// SPDX-License-Identifier: MIT
#pragma once
#include "maple_w2a16.hpp"
#include "ggml.h"

// This header supplies real shape/type checks and maps the plain ggml operation
// to the asynchronous kernel API. It does NOT allocate/import/synchronize Vulkan
// memory and is not installed into a user's dispatcher automatically.
namespace maple_w2 {
struct ImportedPointers {
    const uint8_t *weight=nullptr;
    const void *activation=nullptr;
    const int32_t *ids=nullptr;
    float *output=nullptr;
    int32_t *status=nullptr;
};
// Returns nullptr on success, otherwise a stable fallback reason.
// Start with plain Q=1 MUL_MAT_ID. Speculative Q>1 and fused graph operations
// deliberately remain in Vulkan until their full semantics are integrated.
inline const char *make_ggml_decode_view(const ggml_tensor *node,
        const ImportedPointers& pointers, Layout gpu_layout,
        bool weights_validated, bool allow_f32_to_f16,
        bool dispatcher_has_fused_following_nodes, DeviceProblem& out) {
    if(!node || node->op!=GGML_OP_MUL_MAT_ID) return "not plain MUL_MAT_ID";
    if(dispatcher_has_fused_following_nodes) return "fused bias/scale/epilogue unsupported by adapter";
    const auto *w=node->src[0],*x=node->src[1],*ids=node->src[2];
    if(!w||!x||!ids) return "missing input";
    if(w->type!=GGML_TYPE_TQ2_0 || node->type!=GGML_TYPE_F32 || ids->type!=GGML_TYPE_I32)
        return "unsupported tensor type";
    if(x->type!=GGML_TYPE_F32 && x->type!=GGML_TYPE_F16) return "activation must be F16/F32";
    if(x->type==GGML_TYPE_F32 && !allow_f32_to_f16) return "activation rounding not explicitly allowed";
    if(!weights_validated) return "ternary codes and finite scales not validated at model load";
    if(!ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(ids)||!ggml_is_contiguous(node))
        return "non-contiguous view: use Vulkan fallback";
    if(w->ne[0]<=0||w->ne[0]>UINT32_MAX||w->ne[1]<=0||w->ne[1]>UINT32_MAX||w->ne[2]<=0||w->ne[2]>UINT32_MAX)
        return "dimension overflow";
    if(w->ne[0]%256 || w->ne[1]%8 || w->ne[3]!=1) return "unsupported weight shape";
    if(ids->ne[0]<=0 || ids->ne[0]>w->ne[2] || ids->ne[1]!=1 || ids->ne[2]!=1 || ids->ne[3]!=1)
        return "pilot adapter accepts one token only";
    if(x->ne[0]!=w->ne[0] || (x->ne[1]!=1 && x->ne[1]!=ids->ne[0]) || x->ne[2]!=1 || x->ne[3]!=1)
        return "unsupported activation shape";
    if(node->ne[0]!=w->ne[1] || node->ne[1]!=ids->ne[0] || node->ne[2]!=1 || node->ne[3]!=1)
        return "unsupported output shape";
    if(!pointers.weight||!pointers.activation||!pointers.ids||!pointers.output||!pointers.status)
        return "Vulkan buffers have not been imported into matching SYCL context";
    out={};out.shape={uint32_t(w->ne[0]),uint32_t(w->ne[1]),uint32_t(w->ne[2])};
    out.tokens=1;out.topk=uint32_t(ids->ne[0]);out.per_selection=x->ne[1]!=1;
    out.layout=gpu_layout;out.w0=pointers.weight;out.x=pointers.activation;
    out.ids=pointers.ids;out.y0=pointers.output;out.status=pointers.status;
    out.activation_type=x->type==GGML_TYPE_F16?ActivationType::f16:ActivationType::f32;
    return nullptr;
}
} // namespace maple_w2
