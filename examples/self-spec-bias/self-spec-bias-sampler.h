#pragma once

#include "llama.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Sampler that steers greedy decoding towards a known token sequence.
//
// For the token at the current position of that sequence, the probability
// becomes p = (1 - beta)*p + beta. beta = 0 is plain greedy, beta = 1 forces
// the sequence. The sampler stops biasing as soon as the model picks a
// different token, so it never fights a model that clearly disagrees.

struct spec_bias_sampler {
    int32_t n_vocab;
    float   beta;

    std::vector<llama_token> seq;

    size_t pos    = 0;
    bool   active = false;
};

static void spec_bias_select_by_logit(llama_token_data_array * cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}

static void spec_bias_select_by_prob(llama_token_data_array * cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].p > cur_p->data[cur_p->selected].p) {
            cur_p->selected = i;
        }
    }
}

// softmax that leaves the candidates unsorted
static void spec_bias_softmax(llama_token_data_array * cur_p) {
    float max_l = cur_p->data[0].logit;
    for (size_t i = 1; i < cur_p->size; ++i) {
        max_l = std::fmax(max_l, cur_p->data[i].logit);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float p = std::exp(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum += p;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= sum;
    }
}

// returns cur_p->size if the token is not a candidate
static size_t spec_bias_find(const llama_token_data_array * cur_p, llama_token token) {
    // fast path: the vocabulary is still in order
    if (token >= 0 && cur_p->size > (size_t) token && cur_p->data[token].id == token) {
        return (size_t) token;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].id == token) {
            return i;
        }
    }

    return cur_p->size;
}

static const char * spec_bias_sampler_name(const struct llama_sampler * /*smpl*/) {
    return "spec-bias";
}

static void spec_bias_sampler_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (spec_bias_sampler *) smpl->ctx;

    GGML_ASSERT(cur_p->size > 0);

    if (!ctx->active || ctx->beta <= 0.0f || ctx->pos >= ctx->seq.size()) {
        spec_bias_select_by_logit(cur_p);
        return;
    }

    const size_t idx = spec_bias_find(cur_p, ctx->seq[ctx->pos]);

    if (idx == cur_p->size) {
        spec_bias_select_by_logit(cur_p);
        return;
    }

    spec_bias_softmax(cur_p);

    const float scale = 1.0f - ctx->beta;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p *= scale;
    }
    cur_p->data[idx].p += ctx->beta;

    spec_bias_select_by_prob(cur_p);
}

static void spec_bias_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (spec_bias_sampler *) smpl->ctx;

    if (!ctx->active) {
        return;
    }

    if (ctx->pos >= ctx->seq.size() || token != ctx->seq[ctx->pos]) {
        ctx->active = false;
        return;
    }

    ctx->pos++;

    if (ctx->pos >= ctx->seq.size()) {
        ctx->active = false;
    }
}

static void spec_bias_sampler_reset(struct llama_sampler * smpl) {
    auto * ctx = (spec_bias_sampler *) smpl->ctx;

    ctx->seq.clear();
    ctx->pos    = 0;
    ctx->active = false;
}

static void spec_bias_sampler_free(struct llama_sampler * smpl) {
    delete (spec_bias_sampler *) smpl->ctx;
}

// CPU only, the backend sampling path is not implemented
static struct llama_sampler_i spec_bias_sampler_i = {
    /* .name              = */ spec_bias_sampler_name,
    /* .accept            = */ spec_bias_sampler_accept,
    /* .apply             = */ spec_bias_sampler_apply,
    /* .reset             = */ spec_bias_sampler_reset,
    /* .clone             = */ nullptr,
    /* .free              = */ spec_bias_sampler_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * spec_bias_sampler_init(int32_t n_vocab, float beta) {
    return llama_sampler_init(&spec_bias_sampler_i, new spec_bias_sampler{ n_vocab, beta, {}, 0, false });
}

// the bias sampler inside a chain that holds it as its only entry
static struct llama_sampler * spec_bias_sampler_of(struct llama_sampler * chain) {
    return llama_sampler_chain_get(chain, 0);
}

// bias the tokens of seq starting at start
static void spec_bias_sampler_set_seq(struct llama_sampler * smpl, const std::vector<llama_token> & seq, size_t start) {
    auto * ctx = (spec_bias_sampler *) smpl->ctx;

    ctx->seq    = seq;
    ctx->pos    = start;
    ctx->active = ctx->beta > 0.0f && start < seq.size();
}
