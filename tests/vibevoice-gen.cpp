#include "vibevoice-gen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

llama_memory_t vv_branch::mem() const {
    return llama_get_memory(ctx);
}

bool vv_branch::decode_tokens(const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token   [i]    = tokens[i];
        batch.pos     [i]    = pos + i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i]    = i + 1 == tokens.size();
    }
    batch.n_tokens = tokens.size();

    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        fprintf(stderr, "%s: llama_decode of %zu tokens failed: %d\n", name, tokens.size(), ret);
        return false;
    }
    pos += tokens.size();
    return true;
}

bool vv_branch::decode_embd(const float * embd, int64_t n_embd) {
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    std::copy(embd, embd + n_embd, batch.embd);
    batch.pos     [0]    = pos;
    batch.n_seq_id[0]    = 1;
    batch.seq_id  [0][0] = 0;
    batch.logits  [0]    = true;
    batch.n_tokens = 1;

    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        fprintf(stderr, "%s: llama_decode of an embedding failed: %d\n", name, ret);
        return false;
    }
    pos += 1;
    return true;
}

std::vector<float> vv_branch::hidden(int64_t n_embd) const {
    const float * h = llama_get_embeddings_ith(ctx, -1);
    if (h == nullptr) {
        fprintf(stderr, "%s: no hidden state\n", name);
        return {};
    }
    return std::vector<float>(h, h + n_embd);
}

void vv_branch::reset() {
    llama_memory_seq_rm(mem(), 0, -1, -1);
    pos = 0;
}

int64_t vv_branch::n_cached() const {
    const llama_pos p_min = llama_memory_seq_pos_min(mem(), 0);
    const llama_pos p_max = llama_memory_seq_pos_max(mem(), 0);
    return p_max < 0 ? 0 : p_max - p_min + 1;
}

vv_generator::~vv_generator() {
    if (backend) {
        ggml_backend_free(backend);
    }
    llama_free(pos.ctx);
    llama_free(neg.ctx);
}

bool vv_generator::init(llama_model * lmodel, uint32_t n_ctx) {
    model = dynamic_cast<const llama_model_vibevoice *>(lmodel);
    if (model == nullptr) {
        fprintf(stderr, "not a VibeVoice model\n");
        return false;
    }
    vocab = llama_model_get_vocab(lmodel);

    const auto & vv = model->hparams.vibevoice;
    if (vv.acoustic.vae_dim != vv.diff_latent_size) {
        fprintf(stderr, "latent size %u does not match acoustic vae_dim %u\n", vv.diff_latent_size, vv.acoustic.vae_dim);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (llama_model_vibevoice::audio_bos_token_id >= n_vocab || llama_model_vibevoice::audio_token_id >= n_vocab) {
        fprintf(stderr, "audio token ids are out of the vocab (%d)\n", n_vocab);
        return false;
    }

    n_embd = model->hparams.n_embd;
    hop    = 1;
    for (uint32_t s = 0; s + 1 < vv.acoustic.n_dec_stages; ++s) {
        const uint32_t ratio = vv.acoustic.dec_ratios[s];
        if (ratio == 0 || hop > INT64_MAX / ratio) {
            fprintf(stderr, "vibevoice: invalid tokenizer ratio %u, hop must fit in int64_t\n", ratio);
            return false;
        }
        hop *= ratio;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = n_ctx;
    cparams.n_batch      = n_ctx; // the prompt is one batch
    cparams.n_ubatch     = std::min<uint32_t>(n_ctx, 512);
    cparams.n_seq_max    = 1;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;

    pos.ctx = llama_init_from_model(lmodel, cparams);
    neg.ctx = llama_init_from_model(lmodel, cparams);
    if (pos.ctx == nullptr || neg.ctx == nullptr) {
        fprintf(stderr, "failed to create the contexts\n");
        return false;
    }

    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "failed to init the CPU backend\n");
        return false;
    }

    return true;
}

bool vv_generator::prefill(const std::vector<llama_token> & prompt) {
    if (prompt.empty()) {
        fprintf(stderr, "empty prompt\n");
        return false;
    }
    return pos.decode_tokens(prompt) && neg.decode_tokens({ llama_model_vibevoice::audio_bos_token_id });
}

bool vv_generator::audio_bos() {
    neg.reset();
    return pos.decode_tokens({ llama_model_vibevoice::audio_bos_token_id }) &&
           neg.decode_tokens({ llama_model_vibevoice::audio_bos_token_id });
}

void vv_generator::reset_speech_stream() {
    dec_cache.clear();
    sem_cache.clear();
}

bool vv_generator::audio_frame() {
    const std::vector<float> h_pos = pos.hidden(n_embd);
    const std::vector<float> h_neg = neg.hidden(n_embd);
    if (h_pos.empty() || h_neg.empty()) {
        return false;
    }

    lat = model->sample_speech_latents(backend, h_pos.data(), h_neg.data(), 1, cfg_scale, seed + n_frames);

    std::vector<float> unscaled = lat;
    model->unscale_speech_latents(unscaled.data(), unscaled.size());

    std::vector<float> chunk = model->acoustic_dec_stream(backend, dec_cache, unscaled.data(), 1);
    for (float & x : chunk) {
        if (!std::isfinite(x)) {
            x = 0.0f;
            n_nonfinite++;
        }
    }

    emb = model->speech_feedback_stream(backend, sem_cache, lat.data(), chunk.data(), 1);

    wav.insert(wav.end(), chunk.begin(), chunk.end());
    n_frames++;

    return pos.decode_embd(emb.data(), n_embd) && neg.decode_embd(emb.data(), n_embd);
}

llama_token vv_generator::step(llama_sampler * smpl) {
    const llama_token tok = llama_sampler_sample(smpl, pos.ctx, -1);

    bool ok = true;
    if (tok == llama_model_vibevoice::audio_token_id) {
        ok = audio_frame();
    } else if (tok == llama_model_vibevoice::audio_bos_token_id) {
        ok = audio_bos();
    } else if (!is_eog(tok)) {
        ok = pos.decode_tokens({ tok });
    }

    return ok ? tok : LLAMA_TOKEN_NULL;
}

bool vv_generator::is_eog(llama_token tok) const {
    return llama_vocab_is_eog(vocab, tok);
}
