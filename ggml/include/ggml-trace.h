#pragma once

#include "ggml.h"

// Lightweight tracing of tensor-parallel allreduces and RPC traffic.
//
// Enabled with GGML_TRACE=1. Output path is taken from GGML_TRACE_HTML
// (default: llama-trace.html), detail depth from GGML_TRACE_MAX_STEPS.
//
// All entry points are no-ops when tracing is disabled.

#ifdef __cplusplus
extern "C" {
#endif

    GGML_API bool ggml_trace_enabled(void);

    // a step is one decode call; events recorded outside a step are bucketed separately
    GGML_API void ggml_trace_step_begin(const char * label, int n_tokens);
    GGML_API void ggml_trace_step_end  (void);

    // returns an id to pass to ggml_trace_allreduce_end, or -1 if not recorded
    GGML_API int  ggml_trace_allreduce_begin(int subgraph, int n_devices, const char * name,
                                             const char * type, const int64_t * ne, size_t nbytes);
    GGML_API void ggml_trace_allreduce_end  (int id, const char * method);

    GGML_API void ggml_trace_transfer(const char * phase, const char * src_name, const char * dst_name,
                                      const char * name, const int64_t * ne, size_t nbytes);

    GGML_API void ggml_trace_rpc(const char * endpoint, int device, const char * cmd,
                                 const char * name, const int64_t * ne,
                                 size_t bytes_out, size_t bytes_in);

    // safe to call more than once
    GGML_API void ggml_trace_write_html(const char * path);

    // uses GGML_TRACE_HTML, called automatically at exit
    GGML_API void ggml_trace_write_html_default(void);

#ifdef __cplusplus
}
#endif
