// tessera: real forward-pass activation capture for the clip graph.
//
// The architecture is the same shape as the imatrix tap
// (src/llama-context.cpp:1168-1188 for the text path): we load the
// clip model, run the forward pass for each input, and tap the
// per-tensor activation data after the compute. The per-tensor
// stat formulas (kurtosis / eff_rank / rms / mean_abs /
// tail_ratio / p99) are computed in C++ so the values are
// byte-equivalent to a future Python implementation, modulo the
// per-tensor stat arithmetic drift (this drift is asserted in
// test-clip-capture.cpp; any future change that breaks it must
// update the test).
//
// The capture module is a sibling of the imatrix capture: it lives
// in tools/mtmd/ (where the clip graph is), not in
// tools/quantize/tessera/ (which the M0a/M0b workers own). The
// activation tap reads data from a built graph (the imatrix tap
// reads from a built graph too), and it does so in a way that
// preserves the imatrix's memory discipline (mmap, chunk,
// peak-RSS budget).
//
// The standalone binary ``llama-clip-capture`` is the surface the
// Python side invokes via subprocess (the imatrix CLI is the
// precedent). The header also exposes a per-modality entry point
// so vision / audio / mm_projector are clean. The mm_projector is
// captured via a separate first-class path: the C++ capture runs
// the upstream tower (vision or audio) to produce the
// multimodal embedding, then runs the projector's forward pass
// on that embedding, then captures activations at the
// ``mm.`` tensors. The mm_projector dispatch and the calibration
// side agree on the prefix scheme: ``v.`` / ``a.`` / ``mm.``
// drive the model_role stamp on the row.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// The mode tag selects the modality-specific forward-pass driver
// (vision image preprocess + clip graph, audio mel preprocess +
// clip graph, or mm_projector via the vision / audio tower's
// multimodal embedding). The mm_projector paths run the
// upstream tower first, then run the projector's own forward
// pass on the embedding; the captured activations are the
// ``mm.``-prefixed tensors the projector emits.
typedef enum ts_clip_capture_mode {
    TS_CLIP_CAPTURE_MODE_VISION             = 0,
    TS_CLIP_CAPTURE_MODE_AUDIO              = 1,
    TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION = 2,
    TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_AUDIO  = 3,
} ts_clip_capture_mode;

// Run the clip graph on a list of inputs and emit per-tensor
// activation statistics to a JSON file. The JSON shape mirrors
// what the calibration side consumes:
//
//   {
//     "tool": "llama-clip-capture",
//     "mode": "vision" | "audio" |
//             "mm_projector_via_vision" | "mm_projector_via_audio",
//     "model": "<clip_model_path>",
//     "mm_projector_model": "<projector_gguf>" or null,
//     "n_inputs": <int>,
//     "n_chunks": <int>,
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
// Returns 0 on success, non-zero on failure. On failure, ``err``
// is populated with a human-readable message. On success, ``err``
// is left empty (or contains stderr progress the CLI also logs).
//
// Batching: the input list is folded into a single forward pass
// per batch via the canonical ``clip_image_f32_batch`` path
// (vision) or the audio preprocessor's batched path (audio). The
// ``batch_size`` parameter controls the chunk size for the inner
// batched forward call: when the input list is larger than
// ``batch_size``, the capture chunks the inputs into multiple
// forward calls and accumulates the per-tensor stats. Per-tensor
// stats are accumulated over all chunks and all inputs. The
// output JSON reports the total ``n_inputs`` and ``n_chunks``.
//
// Dead-node handling: the clip graph contains layout views
// (``(view)``, ``(permuted)``, ``(cont)``, ``(reshaped)``,
// ``(transposed)``) and inter-backend copies (``(copy)``) that
// the scheduler creates. The layout views share storage with
// their source; capturing them is redundant (the source's
// per-tensor stats already cover the same data). The
// inter-backend copies are dead nodes that the scheduler's
// split logic may not compute, leaving uninitialised data. The
// capture excludes both at the graph level (before the
// per-tensor stats are computed) and prints a stderr warning
// listing the excluded tensors and why. The JSON output never
// silently loses rows.
//
// Memory discipline: the orchestrator can pass
// ``peak_rss_budget_bytes`` to refuse a capture that would
// exceed the host's peak-RSS budget. The CLI reads this from
// the ``--peak-rss-budget-gb`` flag.
int ts_clip_capture_activations(
        const char * clip_model_path,
        const char * mm_projector_path,  // may be null for vision/audio
        const std::vector<std::string> & input_paths,
        ts_clip_capture_mode mode,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

// Per-modality entry points. These are thin wrappers around
// ``ts_clip_capture_activations`` and exist so the C-API surface
// is explicit about which modality the call drives.
int ts_clip_capture_activations_vision(
        const char * clip_model_path,
        const std::vector<std::string> & image_paths,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

int ts_clip_capture_activations_audio(
        const char * clip_model_path,
        const std::vector<std::string> & audio_paths,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

int ts_clip_capture_activations_mm_projector(
        const char * tower_model_path,
        const char * projector_model_path,
        const std::vector<std::string> & input_paths,
        bool via_vision,  // true=vision tower, false=audio tower
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err);

#ifdef __cplusplus
}
#endif
