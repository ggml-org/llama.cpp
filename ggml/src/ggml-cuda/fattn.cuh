#include "common.cuh"
#include "kv-stream-span.h"
#include "kv-stream-attention-plan.h"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// Adapt the existing stock selector without another architecture preference table.
ggml_cuda_kv_stream_kernel_family ggml_cuda_flash_attn_ext_kernel_family(int device, const ggml_tensor * dst);
size_t ggml_cuda_flash_attn_ext_selected_alloc_size(ggml_cuda_kv_stream_kernel_family family, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_selected(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_kv_stream_kernel_family family);

// Return caller-owned device scratch required by one span-aware F16 MMA launch.
size_t ggml_cuda_flash_attn_ext_mma_f16_spans_workspace(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, size_t spans);
bool ggml_cuda_flash_attn_ext_mma_f16_spans_requirements(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, size_t spans,
        ggml_cuda_kv_stream_workspace_requirements & output, ggml_cuda_kv_stream_plan_status * failure = nullptr);
// Execute the stock F16 MMA topology over an ordered logical K/V span table.
bool ggml_cuda_flash_attn_ext_mma_f16_spans(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_kv_span * spans, size_t count,
        void * workspace, size_t workspace_bytes);

bool ggml_cuda_flash_attn_ext_mma_convert_rows(
        ggml_backend_cuda_context & ctx, ggml_type type,
        const char * source, half * destination, int rows, int64_t source_stride, int64_t destination_stride);
