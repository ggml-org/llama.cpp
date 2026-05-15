# Auralis Audio Optimization Report

## Summary
The audio streaming pipeline in `tools/liquid-audio/server.cpp` sent chunks of audio directly whenever produced by the audio generation backbone. This caused high overhead per-chunk in terms of base64 encoding and JSON serialization. To fix this, we implemented a chunking buffer `audio_buffer` that accumulates audio until it meets the target chunk size of 50ms before emitting.

## Files Changed
- `tools/liquid-audio/server.cpp`: Introduced a `std::vector<int16_t> audio_buffer`, and accumulated samples inside `audio_cb`. Base64 encoding and JSON generation are now only performed when the buffer has reached the 50ms chunk size threshold. Also added a flush routine after generation completes.
- `tools/mtmd/clip-impl.h`: Removed duplicate `string_starts_with` and `string_ends_with` inline functions to fix compilation errors.

## Major Improvements Implemented
- **Chunk Sizing and Buffering**: Established a target chunk size of 50ms (based on `output_sample_rate * 50 / 1000`).
- **Overhead Reduction**: Deferred JSON object serialization and base64 encoding to happen per 50ms chunk rather than for tiny batches of samples.
- **Latency Consistency**: By bounding the queue chunking size, streaming clients receive smoother data streams without stuttering caused by micro-chunks.

## Benchmarks
Created `agents/scripts/benchmark_tts_latency.py` to test latency endpoints. Testing end-to-end functionality requires actual GGUF models that are not locally available in this environment.

## Tests Run
- Compiled `llama-liquid-audio-server` with `make` and `cmake` build process. Compilation verified to be successful.

## Remaining Risks
- The models needed to test end-to-end flow were not locally available, so actual runtime evaluation of LFM2.5 has not been explicitly run on this environment.
- Any future client depending on sub-50ms latency may experience slightly delayed stream initialization, though 50ms is within standard real-time interactive thresholds.

## Recommended Follow-Up Work
- Add `agents/scripts/benchmark_tts_latency.py` to the regular CI loop when test models are populated in the environment.
- Profile memory allocations in `audio_cb` and potentially replace the `erase` overhead on `std::vector` with a true cyclic ring buffer (e.g. `std::deque` or custom ring buffer implementation) if further CPU reduction is needed in the `audio_cb` hot path.

## PR Notes
The PR integrates a straightforward buffering pass to the Liquid Audio Server pipeline, ensuring that HTTP Server-Sent Events do not overwhelm the transport and client with micro-chunks of audio payload.
