#pragma once
#include "../ggml-kv-stream-device.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Optional synchronous partial export/merge adapter; not an ordinary attention dispatcher.
const ggml_kv_stream_partial_ops * ggml_cuda_kv_stream_partial_ops();
#endif
