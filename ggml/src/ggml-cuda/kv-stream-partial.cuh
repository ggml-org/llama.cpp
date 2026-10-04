#pragma once
#include "../ggml-kv-stream-device.h"
#include "../ggml-kv-stream-copy.h"
#include "kv-stream-attention-plan.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Optional synchronous partial export/merge adapter; not an ordinary attention dispatcher.
const ggml_kv_stream_partial_ops * ggml_cuda_kv_stream_partial_ops();
const ggml_kv_stream_copy_ops * ggml_cuda_kv_stream_copy_ops();
// Private metadata query; no payload or physical KV allocation is needed to describe a launch.
ggml_cuda_kv_stream_plan_status ggml_cuda_kv_stream_attention_plan_query(
        ggml_backend_t backend, const ggml_tensor * op, const ggml_kv_stream_span_plan_view * view,
        ggml_cuda_kv_stream_execution_style style, ggml_cuda_kv_stream_attention_plan & output);
#endif
