#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct mtmd_audio_preprocessor {
    const clip_hparams & hparams;

    mtmd_audio_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_whisper(const clip_ctx * ctx);
    ~mtmd_audio_preprocessor_whisper();
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_conformer(const clip_ctx * ctx);
    ~mtmd_audio_preprocessor_conformer();
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct mtmd_audio_streaming_istft {
    mtmd_audio_streaming_istft(int n_fft, int hop_length);
    ~mtmd_audio_streaming_istft();

    // reset streaming state
    void reset();

    // process a single STFT frame (streaming)
    // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
    // returns: up to hop_length samples
    std::vector<float> process_frame(const float * frame_spectrum);

    // flush remaining samples at end of stream
    std::vector<float> flush();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
