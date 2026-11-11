#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
    int64_t n_len;
    int64_t n_len_org;
    int64_t n_mel;

    std::vector<float> data;
};

struct mtmd_audio_mel_filters {
    int64_t n_mel;
    int64_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(uint32_t n);

    void fill_hann_window(uint32_t length, bool periodic);

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int64_t n_mel,
                                    int64_t n_fft,
                                    int   sample_rate,               // e.g. 16000
                                    float fmin             = 0.0f,   // e.g. 0.0
                                    float fmax             = -1.0f,  // e.g. sr/2; pass -1 for auto
                                    bool  slaney_area_norm = true,
                                    float scale            = 1.0f,
                                    bool  use_htk          = false
    );
};

// compute one log10-mel frame from n_fft windowed samples, cache must be initialized for n_fft
void mtmd_audio_log_mel_frame(const mtmd_audio_cache & cache, const float * samples, int n_fft, int n_mel, float * out);

struct mtmd_audio_mel_config {
    int64_t n_mel            = 0;
    int64_t n_fft_bins       = 0;
    int32_t hann_window_size = 0;
    int32_t hop_length       = 0;
    int32_t sample_rate      = 0;
    bool    no_padding       = false;
    bool    center_padding   = false;
    float   preemph          = 0.f;
    bool    use_natural_log  = false;
    bool    norm_per_feature = false;
    bool    use_magnitude    = false;  // |X| instead of |X|^2
    float   mel_floor        = 5.960464477539063e-08f;
    bool    mel_floor_add    = false;  // log(x + floor) instead of log(max(x, floor))
    bool    std_eps_after_sqrt = false;  // std + eps instead of sqrt(var + eps)
    int     n_threads        = 1;

    const mtmd_audio_cache * cache = nullptr;
    ggml_backend_sched_t     sched = nullptr;
};

struct mtmd_audio_preproc_out {
    std::vector<mtmd_image_f32> entries; // one per chunk, empty on failure
    std::vector<float> samples;          // input of compute(), empty if entries already hold the data

    bool needs_compute() const {
        return !samples.empty() && (entries.empty() || entries[0].is_placeholder());
    }
};

struct mtmd_audio_preprocessor {
    const mtmd_hparams & hparams;
    mtmd_audio_mel_config mel_config; // filled by initialize() of the preprocessors that use the log-mel graph

    // mel_config.sched must be set before the first compute(), see mtmd_context::init_audio()
    mtmd_audio_preprocessor(const mtmd_model * model): hparams(model->hparams) {}
    mtmd_audio_preprocessor(const mtmd_hparams & hparams, ggml_backend_sched_t sched): hparams(hparams) {
        mel_config.sched = sched;
    }

    // mel_config points to the cache of this instance
    mtmd_audio_preprocessor(const mtmd_audio_preprocessor &) = delete;
    mtmd_audio_preprocessor & operator=(const mtmd_audio_preprocessor &) = delete;

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe

    // entries have the final shapes, but no data if needs_compute()
    mtmd_audio_preproc_out preprocess(const float * samples, size_t n_samples) const;

    // run it on the thread that encodes, it uses mel_config.sched
    bool compute(mtmd_audio_preproc_out & out) const;

  protected:
    // shape_only skips the log-mel graph: the shapes are right, the data is zero-filled
    virtual bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_whisper(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_muse_spark_audio : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_muse_spark_audio(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override {}
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_conformer(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    mtmd_audio_preprocessor_conformer(const mtmd_hparams & hparams, ggml_backend_sched_t sched) : mtmd_audio_preprocessor(hparams, sched) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

// same as conformer, the audio is cut to 30 s and padded to 0.5 s (d1-omni audio.py)
struct mtmd_audio_preprocessor_d1omni : mtmd_audio_preprocessor_conformer {
    using mtmd_audio_preprocessor_conformer::mtmd_audio_preprocessor_conformer;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;
};

struct mtmd_audio_preprocessor_lfm3a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_lfm3a(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_granite_speech : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_granite_speech(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4a : mtmd_audio_preprocessor {
    // each segment is padded and encoded on its own
    static constexpr int segment_seconds = 30;

    mtmd_audio_preprocessor_gemma4a(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4ua : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4ua(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;
};

struct mtmd_audio_preprocessor_qwen3a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3a(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_dots3note : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_dots3note(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_mimo_audio : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_mimo_audio(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_qwen3tts_spk : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3tts_spk(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    mtmd_audio_cache cache;
};

// mimi convolves the waveform directly, so this only pads it to a whole number of frames
struct mtmd_audio_preprocessor_pockettts : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_pockettts(const mtmd_model * model) : mtmd_audio_preprocessor(model) {}
    void initialize() override {}
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;
};

// chatterbox reference front-end: one clip gives five entries, used by position in mtmd.cpp
//   0: CAMPPlus kaldi fbank            [80 x n_frames]  (mel major)
//   1: s3 tokenizer log-mel, flow cap  [n_mel x n_frames] (mel major)
//   2: s3 tokenizer log-mel, t3 cap    [n_mel x n_frames] (mel major)
//   3: s3gen prompt features           [n_frames x 80]  (frame major, 24 kHz mel rate)
//   4: voice encoder power mel         [n_frames x 40]  (frame major, trimmed, partial grid)
// turbo loudness-normalizes the clip to -27 LUFS and caps the t3 clip at 15 s instead of 6 s
struct mtmd_audio_preprocessor_chatterbox_ref : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_chatterbox_ref(const mtmd_model * model, bool is_mtl) : mtmd_audio_preprocessor(model), is_mtl(is_mtl) {}
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  private:
    bool             is_mtl;
    mtmd_audio_cache fbank_cache;  // entry 0, the window slot holds the povey window
    mtmd_audio_cache s3tok_cache;  // entries 1 and 2
    mtmd_audio_cache matcha_cache; // entry 3
    mtmd_audio_cache ve_cache;     // entry 4
};

struct mtmd_audio_preprocessor_parakeet : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_parakeet(const mtmd_model * model) : mtmd_audio_preprocessor(model) { }
    void initialize() override;
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;

  protected:
    bool log_mel(const float * samples, size_t n_samples, mtmd_audio_mel & out, bool shape_only) const;

  private:
    mtmd_audio_cache cache;
};

// one chunk per recording: the raw log-mel of its valid frames
struct mtmd_audio_preprocessor_nemotron_transcribe : mtmd_audio_preprocessor_parakeet {
    mtmd_audio_preprocessor_nemotron_transcribe(const mtmd_model * model) : mtmd_audio_preprocessor_parakeet(model) { }
    bool process(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output, bool shape_only) const override;
};

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct mtmd_audio_streaming_istft {
    mtmd_audio_streaming_istft(int n_fft, int hop_length);

    // reset streaming state
    void reset();

    // process a single STFT frame (streaming)
    // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
    // returns: up to hop_length samples
    std::vector<float> process_frame(const float * frame_spectrum);

    // flush remaining samples at end of stream
    std::vector<float> flush();

  private:
    int n_fft;
    int hop_length;
    int n_fft_bins;

    // Own cache for output processing
    mtmd_audio_cache cache;

    // Streaming state
    std::vector<float> overlap_buffer;
    std::vector<float> window_sum_buffer;
    int                padding_to_remove;

    // Working buffers for IFFT
    std::vector<float> ifft_in;
    std::vector<float> ifft_out;
};
