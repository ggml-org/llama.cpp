#pragma once

//
// tessera-metal.h
//
// Metal compute acceleration for the Tessera quantize pipeline.
//
// The Tessera quantizer is memory-bandwidth-bound: ts_scale_clip_ternarize_fused
// and the reconstruction MSE pass each stream the full weight tensor through
// the shared CPU bus. On Apple Silicon the GPU has its own ~68 GB/s memory
// path and thousands of idle ALU cores during quantization. This module
// offloads the elementwise phases (scale, clip, ternarize, dequant, MSE) to
// Metal compute kernels that stream the weight tensor through GPU bandwidth
// instead of competing with CPU threads for the shared bus.
//
// Architecture: during the GA the SAME weight tensor is evaluated hundreds of
// times with different alpha/clip candidates. Upload the weight tensor to a
// Metal buffer ONCE per layer (ts_metal_upload_weights); each candidate eval
// then dispatches Metal kernels that read from the GPU-resident buffer. Only
// small results (MSE scalar, best alpha) come back to the CPU.
//
// The whole path is gated by ts_metal_available() so non-Apple builds and
// headless hosts still compile and run the existing vDSP / scalar fallback.
//

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the global Metal dispatch state (device, queue, library,
// cached pipelines). Created lazily by ts_metal_init and torn down by
// ts_metal_shutdown. Thread-safe: the MTLCommandQueue is safe to encode
// against from multiple GA threads concurrently.
typedef struct ts_metal_context ts_metal_context_t;

// Opaque handle to a GPU-resident weight buffer (plus its cached scale
// buffers). Returned by ts_metal_upload_weights, consumed by the eval
// entry points. Lives until ts_metal_release_weights.
typedef struct ts_metal_weights ts_metal_weights_t;

// True if a Metal device was found and the kernel library compiled. Call
// before any other ts_metal_* function. Cheap (atomic flag read after the
// first init). On non-Apple builds or when init failed, returns 0 and the
// quantizer falls back to the vDSP path.
int ts_metal_available(void);

// Initialize the global Metal context (device + command queue + library).
// Idempotent and thread-safe. Returns 0 on success, non-zero if no Metal
// device is available or the kernel library failed to compile. After the
// first success ts_metal_available() returns 1 for the process lifetime.
int ts_metal_init(void);

// Release the global context. Safe to call when not initialized.
void ts_metal_shutdown(void);

// Upload a weight tensor (out_dim x in_dim row-major float) to a GPU-resident
// Metal buffer and return a handle. The data is copied once; subsequent
// candidate evals reuse the buffer. Returns nullptr on failure (caller falls
// back to CPU). The handle must be released with ts_metal_release_weights.
//
// act_scales (length in_dim, may be nullptr) is uploaded alongside so AWQ
// scaling can be applied on the GPU without a round-trip per candidate.
ts_metal_weights_t * ts_metal_upload_weights(const float * weights,
                                             const float * act_scales,
                                             int64_t out_dim,
                                             int64_t in_dim);

void ts_metal_release_weights(ts_metal_weights_t * w);

// -----------------------------------------------------------------------
// Kernel 1: fused scale + clip + ternarize (replaces FUSE A)
// -----------------------------------------------------------------------
//
// Reads the GPU-resident weight tensor, applies the per-channel weight scale
// (wscale, length in_dim), optionally clips per row at limit = row_maxabs *
// clip, computes global_amp = mean(|ws|), and ternarizes against that
// threshold. Writes ws (scaled), core (clipped), ternary, and global_amp.
//
// Mirrors ts_scale_clip_ternarize_fused but on the GPU. Output buffers are
// host pointers; the dispatch allocates matching shared Metal buffers,
// runs the kernel, syncs, and copies back. (For the GA hot path these can be
// kept GPU-resident; this C entry point is the drop-in replacement for the
// fused CPU helper and so returns host-visible data.)
//
// Returns 0 on success. If Metal is unavailable or dispatch fails, returns
// non-zero (caller should fall back to the CPU path).
int ts_metal_scale_clip_ternarize(ts_metal_weights_t * w,
                                  const float * wscale,
                                  float clip,
                                  float * ws_out,
                                  float * core_out,
                                  int8_t * ternary_out,
                                  float * global_amp_out);

// -----------------------------------------------------------------------
// Kernel 2: fused dequant + outlier-restore + MSE + recon (replaces FUSE B)
// -----------------------------------------------------------------------
//
// Given an already-ternarized, packed Tile640 layout, dequantizes it, restores
// outliers at the provided flat indices (outlier_idx, length n_outliers) by
// overwriting the dequant with the corresponding ws value, computes the
// reconstruction MSE = mean((ws - deq)^2) over all non-outlier elements, and
// builds recon[col] = deq[col] * input_scale[col].
//
// Mirrors the fused recon+MSE block at the end of ts_quantize_2d. page_scales
// (uint16 f16) and lane_scales (int8) are the Tile640 fit scales; input_scale
// (length in_dim) is the per-channel AWQ inverse.
int ts_metal_dequant_mse_recon(ts_metal_weights_t * w,
                               const int8_t * ternary,
                               const uint16_t * page_scales,
                               const int8_t * lane_scales,
                               const int32_t * outlier_idx,
                               int64_t n_outliers,
                               const float * ws,
                               const float * input_scale,
                               float * recon_out,
                               float * mse_out);

// -----------------------------------------------------------------------
// Kernel 3: batched AWQ grid search (the big win)
// -----------------------------------------------------------------------
//
// Replaces the n_grid-iteration AWQ alpha grid search with a single Metal
// dispatch. For each alpha in `grid` (length n_grid), the kernel scales the
// weight columns by the AWQ scale, dequantizes via the simple uniform
// ternarize (sign-only, against mean(|ws|)), and accumulates an
// importance-weighted MSE against the original weights. The per-alpha MSEs
// are reduced in threadgroup shared memory and returned in `mse_out`
// (length n_grid). The caller picks argmin.
//
// act_scales (length in_dim) drives the AWQ normalization; act2 (length
// in_dim, = act_scales^2) provides the importance weights. Both are taken
// from the weights handle when present.
//
// This is the kernel that justifies the Metal effort: it cuts the grid search
// from n_grid full-tensor passes to one.
int ts_metal_awq_grid_search(ts_metal_weights_t * w,
                             const float * grid,
                             int64_t n_grid,
                             float * mse_out);

#ifdef __cplusplus
}
#endif
