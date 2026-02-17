#pragma once

/**
 * voxtral-stream.h - Streaming audio infrastructure for Voxtral Realtime
 *
 * Provides:
 * 1. Ring buffer for audio samples
 * 2. Microphone capture using miniaudio (cross-platform: Windows, macOS, Linux)
 * 3. Streaming mel spectrogram computation
 * 4. Integration with mtmd for real-time audio encoding + text decoding
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <atomic>
#include <functional>

// Forward declarations
struct mtmd_context;

// ============================================================================
// Mel Chunk (self-contained, compatible with mtmd_audio_mel layout)
// ============================================================================

struct voxtral_mel_chunk {
    int n_len     = 0;  // number of mel frames
    int n_len_org = 0;  // original length (before padding)
    int n_mel     = 0;  // number of mel bins

    // Data layout: [n_mel x n_len], mel_bin-major
    // i.e., data[mel_bin * n_len + frame_idx]
    std::vector<float> data;
};

// ============================================================================
// Audio Ring Buffer
// ============================================================================

struct voxtral_ring_buffer {
    std::vector<float> buffer;
    size_t capacity = 0;
    std::atomic<size_t> head{0};  // write position
    std::atomic<size_t> tail{0};  // read position

    void init(size_t cap);
    void reset();

    // Write samples into the ring buffer (called from audio callback thread)
    // Returns number of samples actually written
    size_t write(const float * samples, size_t n_samples);

    // Read samples from the ring buffer (called from main thread)
    // Returns number of samples actually read
    size_t read(float * out, size_t max_samples);

    // Number of samples available for reading
    size_t available() const;
};

// ============================================================================
// Microphone Capture (using miniaudio)
// ============================================================================

struct voxtral_mic_capture {
    voxtral_ring_buffer ring;
    void * device = nullptr;  // opaque: ma_device*
    int sample_rate = 16000;
    bool is_running = false;

    // Initialize and start capturing from the default microphone
    // sample_rate: target sample rate (16000 for Voxtral)
    // buffer_seconds: ring buffer duration in seconds
    // Returns 0 on success, -1 on failure
    int start(int sample_rate = 16000, float buffer_seconds = 30.0f);

    // Stop capturing and release resources
    void stop();

    // Read captured samples into output buffer
    // Returns number of samples read
    size_t read_samples(float * out, size_t max_samples);

    // Number of samples available
    size_t available() const;

    ~voxtral_mic_capture();
};

// ============================================================================
// Streaming Mel Spectrogram Preprocessor
// ============================================================================

struct voxtral_streaming_mel {
    // Audio parameters (from model hparams)
    int n_fft       = 400;
    int hop_length  = 160;
    int n_mel       = 128;
    int sample_rate = 16000;

    // Internal state
    std::vector<float> sample_buffer;  // accumulated PCM samples
    size_t total_frames_produced = 0;  // total mel frames produced so far

    // Internal cache
    struct cache_data;
    cache_data * cache = nullptr;

    // Initialize with explicit parameters
    void init(int n_fft, int hop_length, int n_mel, int sample_rate);

    // Feed new PCM samples and get mel spectrogram chunks
    // Returns true if new mel chunks are available
    // chunk_frames: number of mel frames per output chunk (e.g., 3000 for 30s)
    //   Use 0 to get all available frames as a single chunk
    bool process(const float * samples, size_t n_samples,
                 std::vector<voxtral_mel_chunk> & output,
                 size_t chunk_frames = 3000);

    // Flush remaining samples (call at end of stream)
    // Pads with zeros to produce final chunk
    bool flush(std::vector<voxtral_mel_chunk> & output,
               size_t chunk_frames = 3000);

    // Reset state for a new stream
    void reset();

    ~voxtral_streaming_mel();
};

// ============================================================================
// Streaming Transcription Session
// ============================================================================

// Callback for new transcription text
using voxtral_text_callback = std::function<void(const char * text, bool is_partial)>;

struct voxtral_stream_session {
    // Components
    voxtral_mic_capture mic;
    voxtral_streaming_mel mel_proc;

    // Configuration
    int chunk_duration_ms    = 5000;   // mel chunk duration in ms (default 5s)
    int poll_interval_ms     = 100;    // how often to poll for new audio
    int max_tokens_per_chunk = 200;    // max tokens to generate per audio chunk

    // State
    bool is_running = false;
    size_t total_audio_ms = 0;

    // Initialize the streaming session
    // Returns 0 on success
    int init(int sample_rate, int n_fft = 400, int hop_length = 160, int n_mel = 128);

    // Start streaming from microphone
    int start_mic();

    // Process a chunk of audio from a file (for testing)
    // Returns number of mel chunks produced
    int process_file_chunk(const float * samples, size_t n_samples,
                           std::vector<voxtral_mel_chunk> & mel_chunks);

    // Stop streaming
    void stop();
};
