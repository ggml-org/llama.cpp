// tessera: real forward-pass activation capture for the clip graph.
//
// v2 of the multimodal activation capture (M1 used a numpy synthetic
// forward pass; this module replaces it). The architecture is the
// same shape as the imatrix tap (src/llama-context.cpp:1168-1188 for
// the text path): we load the clip model, run the forward pass for
// each input, and tap the per-tensor activation data after the
// compute. The same per-tensor stats formulas (kurtosis / eff_rank /
// rms / mean_abs / tail_ratio / p99) the v1 Python side produced are
// computed in C++ so the values are byte-equivalent to a future
// Python implementation, modulo the per-tensor stat arithmetic
// drift (this drift is asserted in test-clip-capture.cpp; any future
// change that breaks it must update the test).
//
// The capture module is a sibling of the imatrix capture: it lives
// in tools/mtmd/ (where the clip graph is), not in
// tools/quantize/tessera/ (which the M0a/M0b workers own). The
// activation tap reads data from a built graph (the imatrix tap
// reads from a built graph too), and it does so in a way that
// preserves the imatrix's memory discipline (mmap, chunk, peak-RSS
// budget).
//
// The standalone binary ``llama-clip-capture`` is the surface the
// Python side invokes via subprocess (the imatrix CLI is the
// precedent). The header also exposes a per-modality entry point so
// vision / audio are clean. The mm_projector is captured via the
// vision / audio path (the projector's activation envelope is the
// one the vision / audio tower feeds it; the dispatch already
// routes mm.* tensors on the text lane).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// The mode tag selects the modality-specific forward-pass driver
// (vision image preprocess + clip graph, or audio mel preprocess +
// clip graph). The mm_projector role is captured by driving the
// vision / audio tower it consumes (the dispatch routes mm.*
// tensors on the text lane; the calibration side does the same).
typedef enum ts_clip_capture_mode {
    TS_CLIP_CAPTURE_MODE_VISION = 0,
    TS_CLIP_CAPTURE_MODE_AUDIO  = 1,
} ts_clip_capture_mode;

// Run the clip graph on a list of inputs and emit per-tensor
// activation statistics to a JSON file. The JSON shape mirrors
// what the v1 synthetic pass produces so the Python side can
// drop the bytes into the same tensor_stats rows without
// reshaping:
//
//   {
//     "tool": "llama-clip-capture",
//     "mode": "vision" | "audio",
//     "model": "<clip_model_path>",
//     "n_inputs": <int>,
//     "n_activations": <int>,
//     "peak_rss_bytes_approx": <int>,
//     "wall_clock_ms": <int>,
//     "tensors": [
//       {
//         "name": "v.blk.0.attn_out-0",
//         "n_elements": <int>,
//         "kurtosis": <float>,
//         "eff_rank": <float>,
//         "rms": <float>,
//         "mean_abs": <float>,
//         "tail_ratio": <float>,
//         "p99": <float>,
//         "n_samples": <int>
//       },
//       ...
//     ]
//   }
//
// Returns 0 on success, non-zero on failure. On failure, ``err`` is
// populated with a human-readable message. On success, ``err`` is
// left empty (or contains stderr progress that the CLI also logs).
//
// The capture accumulates activations across the input list: each
// per-tensor stat is a single value computed over the union of the
// per-input activation buffers. This matches the v1 synthetic
// path's "synthesise one activation envelope per tensor" model;
// the real-data path produces a single envelope per tensor by
// concatenation (not by averaging; averaging loses the heavy-tail
// and the eff-rank signal both).
//
// Memory discipline: the orchestrator can pass ``peak_rss_budget_bytes`` to
// refuse a capture that would exceed the host's peak-RSS budget.
// The CLI reads this from the --peak-rss-budget-gb flag.
int ts_clip_capture_activations(
        const char * clip_model_path,
        const std::vector<std::string> & input_paths,
        ts_clip_capture_mode mode,
        const char * output_json_path,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

// Per-modality entry points. These are thin wrappers around
// ``ts_clip_capture_activations`` and exist so the C-API surface is
// explicit about which modality the call drives.
int ts_clip_capture_activations_vision(
        const char * clip_model_path,
        const std::vector<std::string> & image_paths,
        const char * output_json_path,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

int ts_clip_capture_activations_audio(
        const char * clip_model_path,
        const std::vector<std::string> & audio_paths,
        const char * output_json_path,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

#ifdef __cplusplus
}
#endif
