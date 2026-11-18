#pragma once

// libllama internal: the audio generation pipelines of the clip engine, moved from mtmd (mtmd_gen_*)

#include "clip-subcontext.h"
#include "clip-audio.h"
#include "clip-media.h"

#include "llama.h"
#include "llama-ext.h"

#include <memory>
#include <string>

// what the pipelines read from the encoder side (was mtmd_context)
struct clip_gen_ctx {
    clip_subcontext * ctx_gen_a = nullptr; // gen tower
    clip_subcontext * ctx_a     = nullptr; // speaker / reference tower, can be nullptr

    int n_threads = 1;

    // gen-side reference data of the last reference encode (chatterbox), see clip_gen_state_set_ref()
    clip_gen_ref gen_ref;

    // yue2: the NAR layers, a llama_model of their own
    llama_model * yue2_nar = nullptr;
    llama_flash_attn_type yue2_flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;

    // tiny-voca: the frame stacks, run on llama_contexts of their own, see clip_gen_wav_tiny_voca()
    llama_model * tiny_voca_prosody  = nullptr;
    llama_model * tiny_voca_acoustic = nullptr;
    llama_flash_attn_type tiny_voca_flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;

    int64_t n_embd_out() const {
        return ctx_a ? ctx_a->model.n_mmproj_embd() : ctx_gen_a->model.n_mmproj_embd();
    }
};

struct clip_gen_state;

clip_gen_state * clip_gen_state_init (clip_gen_ctx * ctx, uint32_t seed);
void             clip_gen_state_free (clip_gen_state * state);
void             clip_gen_state_reset(clip_gen_state * state);
int32_t          clip_gen_state_set_ref(clip_gen_ctx * ctx, clip_gen_state * state);
int32_t          clip_gen_audio_process(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out);

llama_gen_info   clip_gen_audio_get_info(const clip_gen_ctx * ctx);
llama_gen_inp    clip_gen_inp_default(const clip_gen_ctx * ctx);
size_t           clip_gen_audio_read_tensor(clip_gen_ctx * ctx, const char * name, float * out, size_t n_max);
const char *     clip_gen_audio_voice_name(const clip_gen_ctx * ctx, int32_t idx);

// chatterbox: the reference encode of a speaker clip, sets ctx->gen_ref
int32_t          clip_cbx_ref_encode(clip_gen_ctx * ctx, const clip_audio_tokens * audio_tokens, std::vector<float> & out_embd);

// qwen3tts: the reference codes of a speaker clip (frame-major) and one talker input row per reference frame
int32_t          clip_qwen3tts_ref_codes(clip_gen_ctx * ctx, const clip_image_f32 & wav, std::vector<int32_t> & codes, std::vector<float> & rows);

// pronunciation of a word by the letter-to-sound model of the gen tower, see llama_model_gen_g2p()
int32_t          clip_gen_audio_g2p(const clip_model & model, const char * word, char * buf, size_t buf_size);

// the stages of the mmproj that are libllama text models (yue2 NAR, tiny-voca frame stacks), nullptr on failure
llama_model *    clip_yue2_nar_load(const char * fname, ggml_backend_dev_t device, const clip_hparams & hparams);
llama_model *    clip_tiny_voca_stack_load(const char * fname, ggml_backend_dev_t device, const clip_hparams & hparams, const char * prefix, int n_layer);
