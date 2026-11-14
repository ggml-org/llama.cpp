// Native M5-TTS engine: the whole inference path behind one small C++ API.
//
// The Java engine this replaces owned the tokenizer, the segmenter, the depth
// heads, the codec recurrence and the WAV writer. All of that now lives here;
// the Java side (NativeEngine) is a thin shell and the JNI layer only marshals
// arguments. The two stage subprocesses (backbone, depth) are unchanged from
// the app's point of view: they are still separate processes, but they talk
// over shared memory whenever the binary supports it.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace m5 {

struct Options {
    std::string model_dir;
    std::string backbone_cmd = "backbone-resident backbone-Q8_0.gguf HTP0 --pipe";
    // --depth-chain runs the sixteen-book head chain inside the stage process
    // (one pipe round trip per frame instead of sixteen). Measured a wash on
    // the SM6650 (-1.5%: the heads are weight-bandwidth-bound, not latency-
    // bound), so the pipe default stays; chain is opt-in and the engine falls
    // back automatically when the child is an older build.
    std::string depth_cmd = "depth-resident depth-Q4_0.gguf HTP0 --depth-pipe";
    std::string codec_model = "codec_decoder_cached.onnx";
    std::string workdir = ".";
    int voice = 0;
    // Selects the cached clone by name instead of index (overrides `voice`).
    // Names made only of digits are treated as indices.
    std::string voice_name;
    // Free-text style/emotion description for the <|caption_begin|> slot.
    std::string caption;
    // Defaults to <model_dir>/voices.json; points at another clone cache.
    std::string voices_path;
    int threads = 3;
    int first_frames = 2;
    int chunk_frames = 4;
    long seed = 1234;
    bool overlap = true;
    // Shared-memory stage transport. Measured slower than pipes on the SM6650
    // (the HTP submission path degrades when the parent sleeps on a futex), so
    // it is opt-in rather than the default. See m5-tts-port-notes.md.
    bool ring = false;
    bool verbose = false;  // per-step diagnostics on stderr
    std::string dump_codes;  // debug: append the generated codes to this file
    bool stages = true;      // start the stage processes (false for --encode)
};

struct Stats {
    double elapsed = 0, audio = 0, rtf = 0;
    double backbone = 0, depth = 0, codec = 0, first_pcm = 0;
    int frames = 0, segments = 0;
};

// Receives emitted PCM as it becomes available (float, mono, 24 kHz).
using PcmSink = std::function<void(const float * samples, size_t count)>;

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    bool open(const Options & options, std::string & error);
    bool is_open() const;
    void close();

    // Re-reads the clone cache and switches to `selector` (name or index).
    // Only the small voices.json file is re-read; the stages keep running.
    bool set_voice(const std::string & selector, std::string & error);

    // Applies a transient voice straight from codes ([T][16], values in
    // [0, 2047]) without touching any cache file. Used by in-request clones.
    bool set_voice_codes(const std::string & reference_text,
                         const std::vector<std::vector<long>> & codes,
                         std::string & error);

    // Sets the free-text style/emotion caption (empty = unset).
    void set_caption(const std::string & caption);

    // Splits and conditions the text the way M5Engine does.
    std::vector<std::vector<long>> encode_segments(const std::string & text) const;
    std::string condition(const std::string & text) const;

    // Codec batching policy. Small chunks (2/4) keep the first packet short for
    // streaming; large chunks (e.g. 8/16) amortise the ONNX call overhead for
    // whole-file rendering. Call between renders (the render thread applies it).
    void set_chunking(int first_frames, int chunk_frames);

    // Renders one already-conditioned token sequence (one segment).
    // wav_path: file destination; empty + wav_bytes == nullptr renders without
    // keeping audio (streaming sinks only). wav_bytes, when set, receives the
    // full WAV container in memory and wav_path is ignored.
    bool render_tokens(const std::vector<long> & tokens, const std::string & wav_path,
                       const PcmSink & sink, Stats & stats, std::string & error,
                       std::string * wav_bytes = nullptr);

    // Splits, conditions and renders every segment into one WAV.
    bool render_text(const std::string & text, const std::string & wav_path,
                     const PcmSink & sink, Stats & stats, std::string & error,
                     std::string * wav_bytes = nullptr);

    // Asks a running render to stop; the next chunk boundary returns false.
    void stop() { stopped_.store(true); }
    void clear_stop() { stopped_.store(false); }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> stopped_{false};
};

}  // namespace m5
