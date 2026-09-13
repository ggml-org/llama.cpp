#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-backend-impl.h"

// Test-only access to the backend's actual primary stream; no native handles enter the llama API.
void kv_capture_test_begin(ggml_backend_t backend) {
    ggml_backend_synchronize(backend);
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    GGML_ASSERT(cudaSetDevice(ctx->device) == cudaSuccess);
    const auto stream = ctx->streams[ctx->device][ctx->curr_stream_no];
    GGML_ASSERT(stream && cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) == cudaSuccess);
}
bool kv_capture_test_end(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    cudaGraph_t graph = nullptr;
    const auto status = cudaStreamEndCapture(ctx->streams[ctx->device][ctx->curr_stream_no],&graph);
    if (graph) GGML_ASSERT(cudaGraphDestroy(graph) == cudaSuccess);
    return status == cudaSuccess;
}
