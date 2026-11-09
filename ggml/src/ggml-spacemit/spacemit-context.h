#pragma once

#include "ggml.h"

#include <spert.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

struct spacemit_opnode;  // forward declaration (defined in spacemit-opnode.h)

namespace ggml::spacemit {

inline constexpr size_t cache_line_size_f32 = 64 / sizeof(float);

struct context {
    spert::Context *        runtime       = nullptr;
    uint32_t                ith           = 0;
    uint32_t                nth           = 0;
    void *                  workspace     = nullptr;
    size_t                  workspace_size = 0;
    spert::SharedBufferView shared        = {};

    void reset(spert::Context & runtime_in, uint32_t ith_in, uint32_t nth_in,
               void * workspace_in, size_t workspace_size_in, spert::SharedBufferView shared_in) {
        runtime        = &runtime_in;
        ith            = ith_in;
        nth            = nth_in;
        workspace      = workspace_in;
        workspace_size = workspace_size_in;
        shared         = shared_in;
    }

    void clear() {
        runtime        = nullptr;
        ith            = 0;
        nth            = 0;
        workspace      = nullptr;
        workspace_size = 0;
        shared         = {};
    }

    void sync() { runtime->sync(); }
};

class tensor_traits_base {
  public:
    virtual ~tensor_traits_base() = default;

    virtual bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const = 0;
    virtual bool compute_forward(context & ctx, ggml_tensor * op) const                 = 0;
    // Fused multi-op dispatch, keyed on spacemit_op_code. Default is "not supported".
    virtual bool compute_forward_fused(context & ctx, const spacemit_opnode & node) const {
        GGML_UNUSED(ctx);
        GGML_UNUSED(node);
        return false;
    }
    virtual int  repack(ggml_tensor * tensor, const void * data, size_t size) const     = 0;

    // Override for repacked tensors; the default copies raw bytes.
    virtual void unpack(const ggml_tensor * tensor, void * data, size_t size) const {
        memcpy(data, tensor->data, size);
    }
};

}  // namespace ggml::spacemit
