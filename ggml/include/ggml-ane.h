#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Apple Neural Engine backend (experimental, uses the private AppleNeuralEngine framework)

GGML_BACKEND_API ggml_backend_t     ggml_backend_ane_init(void);
GGML_BACKEND_API bool               ggml_backend_is_ane(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ane_reg(void);

// collect mode: while enabled, graphs are only translated (nothing is compiled or run, outputs are undefined) and their
// shapes are added to the programs; each program is then compiled once with all collected shapes at its first real use.
// Run the shapes an app will use (e.g. prompt and decode batch sizes) in collect mode at load time: one compile per
// program, and the same programs on every launch (OS compile cache hits).
GGML_BACKEND_API void ggml_backend_ane_collect_shapes(bool collect);

// KV length buckets used by attention (default: powers of 2 from 256); each bucket is one more function per program,
// and a program holds ~5 functions at most, so keep (batch sizes) x (buckets) <= 4. The largest bucket should be >= the
// KV cache size. Call before building graphs.
GGML_BACKEND_API void ggml_backend_ane_set_kv_buckets(const int * sizes, int n);

#ifdef __cplusplus
}
#endif
