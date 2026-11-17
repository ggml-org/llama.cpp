#pragma once

// libllama internal: realtime audio input of the clip engine, moved from mtmd (mtmd_audio_stream_*)

#include "clip-subcontext.h"
#include "clip-audio.h"

#include "llama.h"
#include "llama-ext.h"

#include <memory>
#include <string>

struct llama_context;

// what the streams read from the encoder side (was mtmd_context)
struct clip_stream_ctx {
    std::string aud_beg;
    std::string aud_end;
    std::shared_ptr<clip_audio_preprocessor> audio_preproc;

    clip_subcontext * ctx_a = nullptr; // audio tower

    int n_embd_text = -1;
    int n_threads   = 1;

    const llama_model * text_model = nullptr;
    const llama_vocab * vocab      = nullptr;
};

struct clip_audio_stream;

bool               clip_support_audio_stream(const clip_stream_ctx * ctx);
clip_audio_stream * clip_audio_stream_init(clip_stream_ctx * ctx, llama_audio_stream_params params);
void               clip_audio_stream_free (clip_audio_stream * stream);
int32_t            clip_audio_stream_push (clip_audio_stream * stream, const float * samples, size_t n_samples);
int32_t            clip_audio_stream_flush(clip_audio_stream * stream, bool end);
int32_t            clip_audio_stream_step (clip_audio_stream * stream, llama_token last_token);
const float *      clip_audio_stream_get_output_embd   (const clip_audio_stream * stream);
bool               clip_audio_stream_is_seq_start      (const clip_audio_stream * stream);
int64_t            clip_audio_stream_get_n_rows        (const clip_audio_stream * stream);
int64_t            clip_audio_stream_get_n_rows_flushed(const clip_audio_stream * stream);
