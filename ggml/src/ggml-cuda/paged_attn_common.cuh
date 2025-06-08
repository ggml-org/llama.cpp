#ifndef PAGED_ATTN_COMMON_CUH
#define PAGED_ATTN_COMMON_CUH

#include "common.cuh" // For ggml_type, half, etc.
// Placeholder for page mapping info (conceptual, defined in ggml-cuda.cu or a shared header)
// Ensure this matches the definition visible to ggml-cuda.cu
struct paged_kv_token_mapping_gpu {
    int page_idx;                    // Index of the page in the page_pool_gpu array
    int offset_in_page_elements;     // Offset in terms of elements from the start of the page
                                     // This offset points to the beginning of the K data for this token (all heads for K for this layer).
};

struct paged_kv_sequence_view_gpu {
    const paged_kv_token_mapping_gpu* token_mappings_gpu;
    const void* const* page_pool_gpu;

    int32_t num_tokens_in_logical_sequence;
    ggml_type dtype;

    uint16_t k_head_size_elements;           // Dimension of a single K head in elements (e.g., n_embd_head_k)
    uint16_t v_head_size_elements;           // Dimension of a single V head in elements (e.g., n_embd_head_v)
    uint16_t num_k_heads_total;              // Total number of K heads for this layer (e.g., n_head_kv from model)
    uint16_t num_v_heads_total;              // Total number of V heads for this layer (usually same as K)
    uint16_t element_size_bytes;             // sizeof(element type), e.g. sizeof(half) for F16
    // Byte offset from the start of a token's K-V item block to the start of its V data block.
    // This is typically: num_k_heads_total * k_head_size_elements * element_size_bytes.
    uint32_t v_block_start_offset_bytes;
};

// Device helper to get a pointer to the data for a specific head of a specific token in a paged KV cache.
// This assumes the paged_kv_sequence_view_gpu is for a single layer.
template<typename T> // T should match the data type stored (e.g., half for F16)
__device__ __forceinline__ const T* get_paged_kv_data_ptr_cuda(
    const paged_kv_sequence_view_gpu* view, // Pass by pointer to avoid copying struct to registers
    int logical_token_idx,                  // The token's logical position in the full sequence
    int head_idx_in_tensor,                 // The head index *within the K or V tensor part* (0 to num_k_heads_total-1 or num_v_heads_total-1)
    bool is_value_tensor)                   // True if requesting V data, false for K data
{
    // Basic bounds check for token index
    if (logical_token_idx < 0 || logical_token_idx >= view->num_tokens_in_logical_sequence) {
        // This can happen if q_len > kv_len (e.g. first token). Kernels might handle this by not reading.
        // Or, for robustness, ensure host never asks for out-of-bounds tokens for paged cache.
        // Returning nullptr might cause crashes if not checked by caller.
        // A safer alternative might be to point to a "zero page" if out of bounds.
        // For performance, often rely on upstream logic to not request out-of-bounds reads.
        // printf("Accessing token %d out of bounds %d\n", logical_token_idx, view->num_tokens_in_logical_sequence);
        return nullptr;
    }

    const paged_kv_token_mapping_gpu mapping = view->token_mappings_gpu[logical_token_idx];

    if (mapping.page_idx < 0) {
        // Page index is invalid (e.g., token not resident, though for FA all needed tokens should be)
        // printf("Invalid page_idx %d for token %d\n", mapping.page_idx, logical_token_idx);
        return nullptr;
    }

    const uint8_t* page_base_ptr_u8 = (const uint8_t*)view->page_pool_gpu[mapping.page_idx];

    // mapping.offset_in_page_elements is the offset from page start to the K-V item for this token for this layer.
    size_t token_item_start_offset_in_page_bytes = (size_t)mapping.offset_in_page_elements * view->element_size_bytes;
    const uint8_t* token_item_base_ptr_u8 = page_base_ptr_u8 + token_item_start_offset_in_page_bytes;

    size_t specific_head_data_start_bytes;

    if (is_value_tensor) {
        // Data for this V head starts at:
        // (start of V block for this token) + (head_idx * size_of_one_v_head)
        specific_head_data_start_bytes = view->v_block_start_offset_bytes +
                                         ((size_t)head_idx_in_tensor * view->v_head_size_elements * view->element_size_bytes);
    } else { // Key tensor
        // Data for this K head starts at:
        // (start of K block for this token, which is offset_in_page) + (head_idx * size_of_one_k_head)
        specific_head_data_start_bytes = (size_t)head_idx_in_tensor * view->k_head_size_elements * view->element_size_bytes;
    }

    return (const T*)(token_item_base_ptr_u8 + specific_head_data_start_bytes);
}

#endif // PAGED_ATTN_COMMON_CUH
