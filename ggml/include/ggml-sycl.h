//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#define GGML_SYCL_NAME        "SYCL"
#define GGML_SYCL_MAX_DEVICES 48

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_sycl_init(int device);

GGML_BACKEND_API bool ggml_backend_is_sycl(ggml_backend_t backend);

// devide buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);

// KV cache buffer type (enforces safe per-allocation cap for chunking)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_kv_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);

// tensor parallel buffer type (Megatron-style column/row parallel with all-reduce)
// Initializes TP system on first call. Pass device_ids=NULL for auto-detection.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_tp_buffer_type(int n_devices, const int * device_ids);

// Get the TP world size (number of devices in TP group, 1 if TP not enabled)
GGML_BACKEND_API int ggml_backend_sycl_get_tp_world_size(void);

// Get the TP rank for this process (0 if TP not enabled or single-process mode)
GGML_BACKEND_API int ggml_backend_sycl_get_tp_rank(void);

// Check if running in multi-process TP mode
GGML_BACKEND_API bool ggml_backend_sycl_is_multiprocess_tp(void);

// KV buffer type for a backend device (falls back to default buffer type if not SYCL)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_kv_buffer_type_from_dev(ggml_backend_dev_t device);

// Get the byte offset for reading this rank's shard from GGUF file
// For column-parallel tensors, this is the offset into the tensor data
// For row-parallel tensors, returns 0 (requires special handling due to interleaved data)
// tensor_name: the tensor name to check TP layer type
// tensor_ne: original tensor dimensions [ne0, ne1, ne2, ne3]
// tensor_type: ggml_type of the tensor
GGML_BACKEND_API size_t ggml_backend_sycl_get_tp_data_offset(const char *    tensor_name,
                                                             const int64_t * tensor_ne,
                                                             enum ggml_type  tensor_type);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);

// Host compute buffer type - uses SYCL host memory (malloc_host) with SYCL buffer interface
// This is used for TP compute buffers to allow cross-device data sharing.
// Unlike host_buffer_type, this uses the SYCL buffer interface so it works with SYCL kernels.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_compute_buffer_type(int device);

// CPU-offload compute buffer: host-pinned memory with SYCL interface and is_host=true.
// Eliminates staging overhead for CPU-dispatched layers in cpu-dispatch.cpp.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_cpu_offload_compute_buffer_type(int device);
GGML_BACKEND_API bool ggml_backend_sycl_cpu_offload_available(void);

// Weight residency policy: true when dense weights should remain host-backed and streamed via unified cache.
GGML_BACKEND_API bool ggml_backend_sycl_weights_evictable(void);
// Set unified cache budget as a percentage of free VRAM (clamped 1..100).
GGML_BACKEND_API void ggml_backend_sycl_set_unified_cache_budget_pct(int pct);
// Set unified host cache budget as a percentage of total system RAM (clamped 1..100).
GGML_BACKEND_API void ggml_backend_sycl_set_unified_cache_host_budget_pct(int pct);

// Set per-tensor oneDNN pack M dimension for ONEDNN_PACKED/ONEDNN_WOQ layouts.
// pack_m <= 0 clears the override (falls back to GGML_SYCL_ONEDNN_PACK_M).
GGML_BACKEND_API void ggml_backend_sycl_set_onednn_pack_m(struct ggml_tensor * tensor, int64_t pack_m);

// Register a host-backed weight tensor for SYCL layout metadata/accessors.
GGML_BACKEND_API void ggml_backend_sycl_register_host_weight_tensor(ggml_backend_dev_t   dev,
                                                                    struct ggml_tensor * tensor);

// Cache identity for weights and MoE experts (no pointers, layout handled separately).
// model_id: unique per model load
// has_gguf/file_idx/file_offs/nbytes: GGUF-backed weights
// name_hash/type/ne: non-GGUF weights (fallback identity)
// aux_id: reserved for non-GGUF/MoE uniqueness (e.g., cache_uuid)
struct ggml_sycl_cache_id {
    bool           valid;
    uint64_t       model_id;
    bool           has_gguf;
    uint16_t       file_idx;
    size_t         file_offs;
    size_t         nbytes;
    uint64_t       name_hash;
    enum ggml_type type;
    int64_t        ne[GGML_MAX_DIMS];
    bool           tp_sharded;
    int            tp_rank;
    int            tp_world_size;
    int64_t        tp_local_ne[GGML_MAX_DIMS];
    int64_t        tp_offset_ne[GGML_MAX_DIMS];
    uint64_t       aux_id;
};

// Register GGUF metadata for stable weight identity in the unified cache.
// tensor: GGUF tensor
// model_id: unique per model load
// file_idx: GGUF split index
// file_offs: byte offset in the GGUF file
// tensor_nbytes: GGUF tensor byte size
GGML_BACKEND_API void ggml_backend_sycl_register_weight_identity(const struct ggml_tensor * tensor,
                                                                 uint16_t                   file_idx,
                                                                 size_t                     file_offs,
                                                                 size_t                     tensor_nbytes,
                                                                 uint64_t                   model_id);

// Weight usage categories for layout selection.
enum ggml_backend_sycl_tensor_usage {
    GGML_SYCL_TENSOR_USAGE_UNKNOWN = 0,
    GGML_SYCL_TENSOR_USAGE_ATTENTION_WEIGHT,
    GGML_SYCL_TENSOR_USAGE_FFN_WEIGHT,
    GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT,
    GGML_SYCL_TENSOR_USAGE_MOE_GATE,
    GGML_SYCL_TENSOR_USAGE_EMBEDDING,
    GGML_SYCL_TENSOR_USAGE_NORM,
};

// Register per-tensor usage metadata (used for layout selection).
GGML_BACKEND_API void ggml_backend_sycl_register_weight_usage(const char *                        tensor_name,
                                                              enum ggml_backend_sycl_tensor_usage usage);

// Tensor inventory for tiered memory placement
struct ggml_sycl_tensor_info {
    const char * name;
    size_t       size;
};

struct ggml_sycl_tensor_inventory {
    struct ggml_sycl_tensor_info * tensors;
    size_t                         count;
    size_t                         total_size;
    int                            n_expert;       // Total experts per layer (0 for dense models)
    int                            n_expert_used;  // Experts activated per token (0 for dense models)
};

// Set tensor inventory for tiered memory placement.
// Must be called after model metadata parsing, before tensor allocation.
// This enables automatic VRAM/host placement based on tensor priority.
GGML_BACKEND_API void ggml_backend_sycl_set_tensor_inventory(ggml_backend_t                            backend,
                                                             const struct ggml_sycl_tensor_inventory * inventory);

// Check if tiered memory mode is enabled for this backend.
// Returns true if the unified cache system is active (always true by default).
// Note: This does NOT indicate whether the model exceeds VRAM - use
// ggml_backend_sycl_model_exceeds_vram() for that check.
GGML_BACKEND_API bool ggml_backend_sycl_is_tiered_enabled(ggml_backend_t backend);

// Check if the model size exceeds available VRAM budget.
// Returns true if model weights need to be streamed from host memory.
// Use this to decide whether warmup should be skipped (warmup is harmful
// when weights are streamed, as it triggers unnecessary host->device copies).
GGML_BACKEND_API bool ggml_backend_sycl_model_exceeds_vram(ggml_backend_t backend);

// Get VRAM budget available for weights (budget minus runtime reservations).
// Returns 0 if backend is NULL or cache is not initialized.
GGML_BACKEND_API size_t ggml_backend_sycl_get_vram_budget(ggml_backend_t backend);

// Get free margin in bytes after weights + runtime allocations.
// Returns 0 if budget is exceeded. Used by llama_params_fit.
GGML_BACKEND_API size_t ggml_backend_sycl_get_vram_margin(ggml_backend_t backend);

// Check if the unified tensor cache instance is available.
// Returns true if tiered mode is enabled AND the cache has been created.
GGML_BACKEND_API bool ggml_backend_sycl_has_tensor_cache(ggml_backend_t backend);

// Get cache hit/miss statistics for the unified tensor cache.
// Used for monitoring tiered dispatch effectiveness.
// hits/misses may be NULL if caller doesn't need that stat.
GGML_BACKEND_API void ggml_backend_sycl_get_cache_stats(ggml_backend_t backend, uint64_t * hits, uint64_t * misses);

// Get a stable cache identity for a weight tensor on the specified device.
GGML_BACKEND_API struct ggml_sycl_cache_id ggml_backend_sycl_get_weight_cache_key(const struct ggml_tensor * tensor,
                                                                                  int                        device);

// Get a cache identity for ANY tensor (weight or non-weight) on the specified device.
// Uses data pointer as unique identifier for non-GGUF tensors. Suitable for caching
// mmap'd non-weight tensors like MoE ids, get_rows indices, etc.
GGML_BACKEND_API struct ggml_sycl_cache_id ggml_backend_sycl_get_tensor_cache_key(const struct ggml_tensor * tensor,
                                                                                  int                        device);

GGML_BACKEND_API void ggml_backend_sycl_print_sycl_devices(void);
GGML_BACKEND_API void ggml_backend_sycl_get_gpu_list(int * id_list, int max_len);
GGML_BACKEND_API void ggml_backend_sycl_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API int  ggml_backend_sycl_get_device_count();
GGML_BACKEND_API void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total);
GGML_BACKEND_API void ggml_backend_sycl_set_debug(int level);

// Device-to-host memcpy using the SYCL backend queue for the tensor's buffer.
// This avoids mixing queues/contexts in tests.
GGML_BACKEND_API void ggml_backend_sycl_memcpy_d2h(const struct ggml_tensor * tensor, void * dst, size_t size);

// SYCL doesn't support registering host memory, keep here for reference
// GGML_BACKEND_API bool ggml_backend_sycl_register_host_buffer(void * buffer, size_t size);
// GGML_BACKEND_API void ggml_backend_sycl_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_sycl_reg(void);

// Flash attention sequence IDs for multi-sequence batching
// Set host pointers for seq_ids arrays (called from llama layer before graph execution)
// These are stored in thread-local storage and used by fattn kernel
// The pointers must be valid USM host memory (allocated by SYCL_Host buffer)
GGML_BACKEND_API void ggml_backend_sycl_set_seq_ids_host(const int32_t * q_seq_ids,
                                                         size_t          q_count,
                                                         const int32_t * kv_seq_ids,
                                                         size_t          kv_count);

// Clear the seq_ids host pointers (called after graph execution)
GGML_BACKEND_API void ggml_backend_sycl_clear_seq_ids_host(void);

// Pipeline parallelism (vLLM-style layer split with chunked prefill)
// Initialize PP with specified devices and layer count. layers_per_stage can be NULL for even distribution.
GGML_BACKEND_API void ggml_backend_sycl_pp_init(const int * device_ids,
                                                int         n_devices,
                                                int         total_layers,
                                                const int * layers_per_stage);

// Clean up PP resources
GGML_BACKEND_API void ggml_backend_sycl_pp_free(void);

// Check if PP is enabled
GGML_BACKEND_API bool ggml_backend_sycl_pp_enabled(void);

// Get number of PP stages
GGML_BACKEND_API int ggml_backend_sycl_pp_num_stages(void);

// Get device ID for a given layer
GGML_BACKEND_API int ggml_backend_sycl_pp_get_device_for_layer(int layer);

// Set chunked prefill configuration
GGML_BACKEND_API void ggml_backend_sycl_pp_set_chunked_prefill(int32_t chunk_size, bool enabled);

// ===========================================================================
// GPU Sampling API (Multi-step decode support)
// These functions run sampling entirely on GPU, avoiding CPU sync overhead
// ===========================================================================

// GPU sampler handle (opaque pointer)
typedef struct ggml_sycl_sampler * ggml_sycl_sampler_t;

// Create a GPU sampler for the given backend
// n_vocab: vocabulary size (determines work buffer sizes)
// seed: RNG seed for probabilistic sampling
GGML_BACKEND_API ggml_sycl_sampler_t ggml_backend_sycl_sampler_create(ggml_backend_t backend,
                                                                      int            n_vocab,
                                                                      uint32_t       seed);

// Free GPU sampler resources
GGML_BACKEND_API void ggml_backend_sycl_sampler_free(ggml_sycl_sampler_t sampler);

// Sample a token from logits tensor on GPU
// logits_tensor: must be a SYCL tensor of shape [n_batch, n_vocab]
// temp: temperature (0 = greedy, 1 = no scaling)
// Returns: sampled token ID
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token(ggml_sycl_sampler_t sampler,
                                                        ggml_tensor *       logits_tensor,
                                                        float               temp);

// Sample from a specific index in a batched logits tensor
// logits_tensor: SYCL tensor of shape [n_batch, n_vocab]
// idx: batch index to sample from (0 to n_batch-1)
// temp: temperature (0 = greedy, 1 = no scaling)
// Returns: sampled token ID
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token_idx(ggml_sycl_sampler_t sampler,
                                                            ggml_tensor *       logits_tensor,
                                                            int                 idx,
                                                            float               temp);

// Sample from a specific index with full sampling parameters
// logits_tensor: SYCL tensor of shape [n_batch, n_vocab]
// idx: batch index to sample from (0 to n_batch-1)
// temp: temperature (0 = greedy, >0 = probabilistic)
// top_k: top-k filtering (0 = disabled)
// top_p: top-p/nucleus filtering (1.0 = disabled)
// min_p: min-p filtering (0.0 = disabled)
// Returns: sampled token ID
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token_full(ggml_sycl_sampler_t sampler,
                                                             ggml_tensor *       logits_tensor,
                                                             int                 idx,
                                                             float               temp,
                                                             int                 top_k,
                                                             float               top_p,
                                                             float               min_p);

// Async version - submit sampling kernels without waiting
// Use ggml_backend_sycl_sample_token_get() to retrieve result later
GGML_BACKEND_API void ggml_backend_sycl_sample_token_async(ggml_sycl_sampler_t sampler,
                                                           ggml_tensor *       logits_tensor,
                                                           float               temp);

// Get result from async sampling (blocks until complete)
GGML_BACKEND_API int32_t ggml_backend_sycl_sample_token_get(ggml_sycl_sampler_t sampler);

// ===========================================================================
// Multi-step GPU Sampling API
// For generating multiple tokens without CPU sync between steps
// Tokens are stored in a device-side ring buffer
// ===========================================================================

// Reset the token buffer before starting a new multi-step generation
// Call this at the beginning of each multi-step batch
GGML_BACKEND_API void ggml_backend_sycl_sampler_reset_buffer(ggml_sycl_sampler_t sampler);

// Sample a token and store it in the device-side ring buffer
// This does NOT sync to host - use for multi-step decode
// Returns the index in the buffer where the token was written
GGML_BACKEND_API int ggml_backend_sycl_sample_token_to_device(ggml_sycl_sampler_t sampler,
                                                              ggml_tensor *       logits_tensor,
                                                              float               temp);

// Sample a token with full parameters and store in device-side ring buffer
// Supports top-k, top-p, min-p filtering for multi-step decode
// Returns the index in the buffer where the token was written
GGML_BACKEND_API int ggml_backend_sycl_sample_token_to_device_full(ggml_sycl_sampler_t sampler,
                                                                   ggml_tensor *       logits_tensor,
                                                                   float               temp,
                                                                   int                 top_k,
                                                                   float               top_p,
                                                                   float               min_p);

// Get device pointer to token at specific buffer index (for embedding lookup)
// Returns NULL if index is out of range or buffer not allocated
GGML_BACKEND_API int32_t * ggml_backend_sycl_get_sampled_token_ptr(ggml_sycl_sampler_t sampler, int index);

// Get device pointer to the most recently sampled token
// Convenience function for feeding back to next decode step
GGML_BACKEND_API int32_t * ggml_backend_sycl_get_current_token_ptr(ggml_sycl_sampler_t sampler);

// Copy tokens from device buffer to host (sync point)
// tokens: host buffer to receive tokens
// max_tokens: size of host buffer
// Returns: number of tokens copied (min of token_count and max_tokens)
GGML_BACKEND_API int ggml_backend_sycl_get_sampled_tokens(ggml_sycl_sampler_t sampler,
                                                          int32_t *           tokens,
                                                          int                 max_tokens);

// Get the number of tokens currently in the buffer (since last reset)
GGML_BACKEND_API int ggml_backend_sycl_get_token_count(ggml_sycl_sampler_t sampler);

// Get the maximum buffer size (number of tokens that can be stored)
GGML_BACKEND_API int ggml_backend_sycl_get_token_buffer_size(void);

// ===========================================================================
// Speculative Decoding Verification API
// For verifying draft tokens against model logits entirely on GPU
// ===========================================================================

// Verify draft tokens against batched logits from multi-token decode
// all_logits: SYCL tensor of shape [n_outputs, n_vocab] - logits for all batch positions
// draft_tokens: array of draft token IDs [n_draft] (on host)
// n_draft: number of draft tokens to verify
// logits_offset: index into all_logits to start verification (typically 1, since logits[0] is for the base token)
// Returns: number of accepted tokens (longest matching prefix where argmax == draft)
// Note: Compares draft[i] with argmax(logits[logits_offset + i]) for i = 0..n_draft-1
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative(ggml_sycl_sampler_t sampler,
                                                          ggml_tensor *       all_logits,
                                                          const int32_t *     draft_tokens,
                                                          int                 n_draft,
                                                          int                 logits_offset);

// Extended version that also returns the sampled tokens (argmax at each position)
// This is needed for llama-lookup and other speculative decoding tools
// sampled_tokens_out: [n_draft] host array to receive the argmax token at each position
// Returns: number of accepted tokens (same as ggml_backend_sycl_verify_speculative)
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative_with_tokens(ggml_sycl_sampler_t sampler,
                                                                      ggml_tensor *       all_logits,
                                                                      const int32_t *     draft_tokens,
                                                                      int32_t *           sampled_tokens_out,
                                                                      int                 n_draft,
                                                                      int                 logits_offset);

// Version that takes raw GPU pointer - for use with persistent GPU logits buffer
// gpu_logits: GPU pointer to logits data [n_outputs, n_vocab] in row-major order
// n_vocab: vocabulary size (inner dimension)
// n_outputs: number of output positions (outer dimension)
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative_from_ptr(ggml_sycl_sampler_t sampler,
                                                                   const float *       gpu_logits,
                                                                   int                 n_vocab,
                                                                   int                 n_outputs,
                                                                   const int32_t *     draft_tokens,
                                                                   int32_t *           sampled_tokens_out,
                                                                   int                 n_draft,
                                                                   int                 logits_offset);

// Version that takes HOST logits - copies to GPU internally (safest API)
// host_logits: HOST pointer to logits data [n_outputs, n_vocab] in row-major order
// draft_tokens: HOST pointer to draft token IDs [n_draft]
// sampled_tokens_out: HOST pointer to receive sampled tokens [n_draft]
GGML_BACKEND_API int ggml_backend_sycl_verify_speculative_from_host(ggml_sycl_sampler_t sampler,
                                                                    const float *       host_logits,
                                                                    int                 n_vocab,
                                                                    int                 n_outputs,
                                                                    const int32_t *     draft_tokens,
                                                                    int32_t *           sampled_tokens_out,
                                                                    int                 n_draft,
                                                                    int                 logits_offset);

// ===========================================================================
// Device Memory Utilities
// ===========================================================================

// Check if a buffer is a SYCL buffer
GGML_BACKEND_API bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer);

// Copy data from device memory to a tensor's device memory
// Used for multi-step decode where token IDs are kept on GPU
// src_device_ptr: SYCL device pointer (source)
// tensor: destination tensor (must be on SYCL backend)
// size: number of bytes to copy
GGML_BACKEND_API void ggml_backend_sycl_copy_device_to_tensor(void * src_device_ptr, ggml_tensor * tensor, size_t size);

// Copy data from tensor to GPU buffer with offset (device-to-device)
// Used for multi-ubatch logits accumulation on GPU
// backend: SYCL backend
// src_tensor: source tensor on SYCL backend
// dst_buffer: destination GPU buffer (allocated via ggml_backend_buft_alloc_buffer)
// dst_offset: byte offset into dst_buffer
// size: number of bytes to copy
GGML_BACKEND_API void ggml_backend_sycl_copy_tensor_to_buffer(ggml_backend_t        backend,
                                                              ggml_tensor *         src_tensor,
                                                              ggml_backend_buffer_t dst_buffer,
                                                              size_t                dst_offset,
                                                              size_t                size);

// Synced version: synchronizes compute before copying
// Use this when the tensor data may still be computing asynchronously
GGML_BACKEND_API void ggml_backend_sycl_copy_tensor_to_buffer_sync(ggml_backend_t        backend,
                                                                   ggml_tensor *         src_tensor,
                                                                   ggml_backend_buffer_t dst_buffer,
                                                                   size_t                dst_offset,
                                                                   size_t                size);

// Get device pointer from a SYCL buffer
// Returns the base pointer that can be used for GPU operations
GGML_BACKEND_API void * ggml_backend_sycl_buffer_get_ptr(ggml_backend_buffer_t buffer);

// Async copy from GPU buffer to host memory (device-to-host)
// Used for deferred logits copy to reduce sync overhead
// buffer: source SYCL GPU buffer
// src_ptr: pointer within the buffer (from ggml_backend_buffer_get_base or similar)
// dst: destination host memory
// offset: byte offset in src_ptr
// size: number of bytes to copy
// The copy is async and will complete when ggml_backend_synchronize is called
GGML_BACKEND_API void ggml_backend_sycl_buffer_get_async(ggml_backend_buffer_t buffer,
                                                         const void *          src_ptr,
                                                         void *                dst,
                                                         size_t                offset,
                                                         size_t                size);

// ===========================================================================
// Pending Device Token API (for multi-step GPU decode)
// ===========================================================================

// Set a pending device token to be used by the next decode operation
// The token_ptr must point to device memory containing n_tokens int32_t tokens
// Call this before llama_decode() when using multi-step GPU decode
GGML_BACKEND_API void ggml_backend_sycl_set_pending_device_token(void * token_ptr, size_t n_tokens);

// Clear the pending device token after decode completes
GGML_BACKEND_API void ggml_backend_sycl_clear_pending_device_token(void);

// ===========================================================================
// Continuous Batching API (Multi-sequence GPU Sampling)
// For processing multiple sequences in parallel with all operations on GPU
// ===========================================================================

// Multi-sequence sampler handle (opaque pointer)
typedef struct ggml_sycl_multi_seq_sampler * ggml_sycl_multi_seq_sampler_t;

// Create a multi-sequence GPU sampler
// backend: SYCL backend to use
// max_seqs: maximum number of concurrent sequences (up to 64)
// n_vocab: vocabulary size
// seed: base RNG seed (each sequence gets seed + seq_id)
GGML_BACKEND_API ggml_sycl_multi_seq_sampler_t ggml_backend_sycl_multi_seq_sampler_create(ggml_backend_t backend,
                                                                                          int            max_seqs,
                                                                                          int            n_vocab,
                                                                                          uint32_t       seed);

// Free multi-sequence sampler resources
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_sampler_free(ggml_sycl_multi_seq_sampler_t sampler);

// Add a sequence to the sampler
// seq_id: unique sequence identifier (0 to max_seqs-1)
// temp: temperature for this sequence
// Returns: true if added successfully, false if seq_id out of range or already active
GGML_BACKEND_API bool ggml_backend_sycl_multi_seq_add(ggml_sycl_multi_seq_sampler_t sampler, int seq_id, float temp);

// Remove a sequence from the sampler
// seq_id: sequence to remove
// Returns: true if removed, false if not active
GGML_BACKEND_API bool ggml_backend_sycl_multi_seq_remove(ggml_sycl_multi_seq_sampler_t sampler, int seq_id);

// Update temperature for a sequence
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_set_temp(ggml_sycl_multi_seq_sampler_t sampler,
                                                           int                           seq_id,
                                                           float                         temp);

// Update all sampling parameters for a sequence
// seq_id: sequence identifier
// temp: temperature (0 = greedy)
// top_k: top-k filtering (0 = disabled)
// top_p: nucleus sampling threshold (1.0 = disabled)
// min_p: minimum probability threshold (0.0 = disabled)
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_set_params(ggml_sycl_multi_seq_sampler_t sampler,
                                                             int                           seq_id,
                                                             float                         temp,
                                                             int                           top_k,
                                                             float                         top_p,
                                                             float                         min_p);

// Get number of active sequences
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_active_count(ggml_sycl_multi_seq_sampler_t sampler);

// Sample tokens for all active sequences from batched logits
// batched_logits: device pointer to [n_active, n_vocab] float array
//                 rows must be in same order as seq_ids from get_active_seq_ids()
// greedy: if true, use argmax; if false, use probabilistic sampling
// Returns: number of tokens sampled (equals n_active)
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_sample(ggml_sycl_multi_seq_sampler_t sampler,
                                                        float *                       batched_logits,
                                                        bool                          greedy);

// Sample tokens for specific sequences from a logits tensor with batch indices
// logits_base: device pointer to logits tensor base [n_batch, n_vocab]
// seq_ids: array of sequence IDs to sample for
// batch_indices: array of batch indices (logits row for each seq_id)
// n_seqs: number of sequences to sample
// Uses per-sequence parameters (temp, top_k, top_p, min_p) set via set_params()
// Returns: number of tokens sampled
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_sample_indexed(ggml_sycl_multi_seq_sampler_t sampler,
                                                                float *                       logits_base,
                                                                const int *                   seq_ids,
                                                                const int *                   batch_indices,
                                                                int                           n_seqs);

// Get sampled tokens to host
// tokens_out: host buffer to receive tokens [n_active]
// seq_ids_out: optional host buffer to receive corresponding seq_ids [n_active]
// max_tokens: size of output buffers
// Returns: number of tokens copied
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_tokens(ggml_sycl_multi_seq_sampler_t sampler,
                                                            int32_t *                     tokens_out,
                                                            int *                         seq_ids_out,
                                                            int                           max_tokens);

// Get device pointer to sampled token for a sequence (for token ring buffer)
// seq_id: sequence ID
// Returns: device pointer to int32_t, or NULL if seq_id not active
GGML_BACKEND_API int32_t * ggml_backend_sycl_multi_seq_get_token_ptr(ggml_sycl_multi_seq_sampler_t sampler, int seq_id);

// Get list of active sequence IDs (for ordering batched logits)
// seq_ids_out: host buffer to receive sequence IDs
// max_seqs: size of output buffer
// Returns: number of active sequences
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_active_seq_ids(ggml_sycl_multi_seq_sampler_t sampler,
                                                                    int *                         seq_ids_out,
                                                                    int                           max_seqs);

// Reset token ring buffer for a sequence (call before starting generation)
GGML_BACKEND_API void ggml_backend_sycl_multi_seq_reset_buffer(ggml_sycl_multi_seq_sampler_t sampler, int seq_id);

// Get tokens from ring buffer for a sequence
// seq_id: sequence ID
// tokens_out: host buffer to receive tokens
// max_tokens: size of output buffer
// Returns: number of tokens in buffer
GGML_BACKEND_API int ggml_backend_sycl_multi_seq_get_ring_tokens(ggml_sycl_multi_seq_sampler_t sampler,
                                                                 int                           seq_id,
                                                                 int32_t *                     tokens_out,
                                                                 int                           max_tokens);

// ===========================================================================
// Batched Logits Management
// For extracting logits from batched decode without per-sequence D2H copies
// ===========================================================================

// Extract logits pointer for a specific sequence from batched output
// This returns a device pointer - NO copy is performed
// ctx: llama context (needs access to logits tensor)
// batch_idx: index of sequence in batch (0 to n_tokens-1 for sequences needing logits)
// Returns: device pointer to float[n_vocab], or NULL on error
GGML_BACKEND_API float * ggml_backend_sycl_get_batch_logits_ptr(void * ctx, int batch_idx);

// Get number of sequences with logits in current batch
GGML_BACKEND_API int ggml_backend_sycl_get_batch_logits_count(void * ctx);

// ===========================================================================
// KV Cache Synchronization API
// For multi-ubatch processing where KV cache writes must complete before reads
// ===========================================================================

// Submit a barrier after graph execution and return immediately.
// The barrier ensures all prior commands complete before subsequent ones.
// This is lighter-weight than full queue sync - it just ensures ordering.
// Call this AFTER each ubatch's graph_compute.
GGML_BACKEND_API void ggml_backend_sycl_submit_barrier(ggml_backend_t backend);

// Wait for the previously submitted barrier to complete.
// Call this BEFORE the next ubatch's graph_compute if needed.
// Returns immediately if no barrier was submitted.
GGML_BACKEND_API void ggml_backend_sycl_wait_barrier(ggml_backend_t backend);

// ===========================================================================
// Weight Streaming Control API
// ===========================================================================

// Signal model load phase to SYCL backend
// When loading=true: weight caching is disabled to avoid OOM on large models
// When loading=false: weight caching is enabled for inference
// Use this to bracket model loading to prevent cache allocation during load
GGML_BACKEND_API void ggml_backend_sycl_set_model_loading(bool loading);

// Release all host-backed weight extras (layout metadata, accessors, etc.)
// Call this when unloading a model to free SYCL resources associated with tensors.
// This is safe to call multiple times. Tensor->extra pointers are cleared before
// freeing to prevent use-after-free.
// Note: Extras are NOT automatically released when backends are freed, because
// tensors may outlive backends (e.g., temporary backends during model loading).
GGML_BACKEND_API void ggml_backend_sycl_release_host_weight_extras(void);

#ifdef __cplusplus
}
#endif
