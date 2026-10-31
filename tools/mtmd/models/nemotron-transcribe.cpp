#include "models.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// NeMo ParallelExpertEncoder: ASR and speaker (Sortformer) transformers, speaker activity fused into the ASR states

// a speaker is active in an 80 ms frame where its prediction exceeds this (speaker_activity_threshold)
static constexpr float SPK_ACTIVITY_THR = 0.5f;

// NeMo streaming Sortformer: PEE online defaults and the checkpoint's sortformer_modules config
static constexpr int   SPK_FIFO_LEN          = 40;
static constexpr int   SPK_CACHE_LEN         = 188;
static constexpr int   SPK_SIL_FRAMES        = 1;
static constexpr float SPK_PRED_SCORE_THR    = 0.25f;
static constexpr float SPK_BOOST_LATEST      = 0.05f;
static constexpr float SPK_STRONG_BOOST_RATE = 0.75f;
static constexpr float SPK_WEAK_BOOST_RATE   = 1.5f;
static constexpr float SPK_MIN_POS_RATE      = 0.5f;

// _compress_spkcache runs here on exp(score): its additive log terms become factors and -inf becomes 0
static constexpr float SPK_SCORE_OFFSET = 2.0f; // -log(0.5) in _get_log_pred_scores
static constexpr float SPK_STRONG_BOOST = 4.0f; // _boost_topk_scores, scale_factor 2
static constexpr float SPK_WEAK_BOOST   = 2.0f; // _boost_topk_scores, scale_factor 1
static constexpr float SPK_SPEECH_THR   = 0.5f; // _disable_low_scores: a speaker's frames at or below it are not cached

// the speaker expert has its own layers, width and head count, which build_vit does not take; full rotary
ggml_tensor * clip_graph_nemotron_transcribe::build_spk_encoder(ggml_tensor * cur, ggml_tensor * pos) {
    const int     n_head = hparams.spk_n_head;
    const int64_t n_embd = cur->ne[0];
    const int64_t n_pos  = cur->ne[1];
    const int64_t d_head = n_embd / n_head;

    cur = build_norm(cur, model.spk_pre_ln_w, model.spk_pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
    for (int il = 0; il < (int) model.spk_layers.size(); il++) {
        const auto & layer = model.spk_layers[il];
        ggml_tensor * x = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        ggml_tensor * qkv = build_mm(layer.qkv_w, x);
        const size_t es = ggml_element_size(qkv);
        ggml_tensor * q = ggml_view_3d(ctx0, qkv, d_head, n_head, n_pos, d_head * es, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_3d(ctx0, qkv, d_head, n_head, n_pos, d_head * es, qkv->nb[1], n_embd * es);
        ggml_tensor * v = ggml_view_3d(ctx0, qkv, d_head, n_head, n_pos, d_head * es, qkv->nb[1], 2 * n_embd * es);
        q = ggml_rope_ext(ctx0, q, pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        x = build_attn(layer.o_w, layer.o_b, q, k, v, nullptr, 1.0f / sqrtf((float) d_head), il);
        cur = ggml_add(ctx0, cur, x);

        x = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        x = build_ffn(x, layer.ff_up_w, layer.ff_up_b, nullptr, nullptr, layer.ff_down_w, layer.ff_down_b, FFN_GELU_ERF, il);
        cur = ggml_add(ctx0, cur, x);
    }

    return build_norm(cur, model.spk_post_ln_w, model.spk_post_ln_b, NORM_TYPE_NORMAL, eps, -1);
}

ggml_cgraph * clip_graph_nemotron_transcribe::build() {
    const int     sub          = hparams.subsampling_factor;
    const int64_t n_mel        = hparams.n_mel_bins;
    const int64_t n_mel_frames = img.nx();
    const int64_t n_pos        = (n_mel_frames + sub - 1) / sub;
    GGML_ASSERT(img.ny() == n_mel);

    // warmup: the largest graph, with a full speaker cache and FIFO
    const int64_t n_prev = state == nullptr ? SPK_CACHE_LEN + SPK_FIFO_LEN
        : (int64_t) state->spk_prev.size() / hparams.spk_n_embd;

    // [n_frames, n_mel] log-mel: raw for the speaker expert, normalized over the recording (encode_windows) for the ASR one
    ggml_tensor * inp = build_inp_raw(1);
    ggml_tensor * inp_norm = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mel_frames, n_mel);
    ggml_set_name(inp_norm, "inp_norm");
    ggml_set_input(inp_norm);
    auto stack_mel = [&](ggml_tensor * mel) {
        return build_stack(ggml_cont(ctx0, ggml_transpose(ctx0, mel)), sub, n_mel);
    };

    // positions over [cache | FIFO | window]; the ASR expert takes the first n_pos
    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prev + n_pos);
    ggml_set_name(pos, "positions");
    ggml_set_input(pos);

    ggml_tensor * spk_prev = nullptr;
    if (n_prev > 0) {
        spk_prev = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.spk_n_embd, n_prev);
        ggml_set_name(spk_prev, "spk_prev");
        ggml_set_input(spk_prev);
    }

    ggml_tensor * asr_pos = ggml_view_1d(ctx0, pos, n_pos, 0);
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_ext(ctx0, cur, asr_pos, nullptr, hparams.audio_rope_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);
    };
    ggml_tensor * asr = build_vit(build_mm(model.inp_proj_w, stack_mel(inp_norm)), n_pos, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr,
                                  add_pos);
    cb(asr, "asr_out", -1);

    // speaker expert over [cache | FIFO | window]; the window's rows and predictions feed the next window's state
    ggml_tensor * chunk = build_mm(model.spk_inp_proj_w, stack_mel(inp));
    ggml_set_name(chunk, "spk_chunk");
    ggml_set_output(chunk);
    ggml_tensor * spk = spk_prev ? ggml_concat(ctx0, spk_prev, chunk, 1) : chunk;
    const int64_t n_spk_pos = n_prev + n_pos;
    spk = build_spk_encoder(spk, pos);
    cb(spk, "spk_out", -1);

    // Sortformer head, 10 ms frames
    ggml_tensor * cur = ggml_add(ctx0, build_mm(model.spk_head_proj_w, spk), model.spk_head_proj_b);
    const int64_t d_tf = cur->ne[0];
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_conv_1d(ctx0, model.spk_head_up_w, cur, 1, 1, 1);                            // [n_spk_pos, sub*d_tf]
    cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, model.spk_head_up_b, 1, sub * d_tf));
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_reshape_2d(ctx0, cur, d_tf, sub * n_spk_pos);
    cur = ggml_add(ctx0, build_mm(model.spk_head_fc1_w, ggml_relu(ctx0, cur)), model.spk_head_fc1_b);
    cur = ggml_add(ctx0, build_mm(model.spk_head_fc2_w, ggml_relu(ctx0, cur)), model.spk_head_fc2_b);
    ggml_tensor * probs = ggml_sigmoid(ctx0, cur);                                          // [n_spk, sub*n_spk_pos]
    cb(probs, "spk_probs", -1);

    // 80 ms predictions
    ggml_tensor * preds = ggml_pool_2d(ctx0, probs, GGML_OP_POOL_AVG, 1, sub, 1, sub, 0, 0);  // [n_spk, n_spk_pos]
    ggml_set_name(preds, "spk_preds");
    ggml_set_output(preds);

    // fusion, on the window's rows
    preds = ggml_view_2d(ctx0, preds, preds->ne[0], n_pos, preds->nb[1], n_prev * preds->nb[1]);
    ggml_tensor * act = ggml_step(ctx0, ggml_scale_bias(ctx0, preds, 1.0f, -SPK_ACTIVITY_THR));
    act = build_norm(act, model.fusion_spk_norm_w, model.fusion_spk_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cur = build_norm(asr, model.fusion_asr_norm_w, model.fusion_asr_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cur = ggml_add(ctx0, cur, build_mm(model.fusion_spk_kernel, act));
    cb(cur, "fused", -1);

    cur = ggml_add(ctx0, build_mm(model.mm_fc_w, cur), model.mm_fc_b);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// indices of the k largest of v[0, n), ties to the lower index
static std::vector<int> spk_topk(const float * v, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    k = std::clamp(k, 0, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return v[a] > v[b] || (v[a] == v[b] && a < b);
    });
    idx.resize(k);
    return idx;
}

// NeMo SortformerModules._compress_spkcache: keep the SPK_CACHE_LEN highest-scoring rows, speaker by speaker
static void spk_compress_cache(std::vector<float> & emb, std::vector<float> & preds, int n, int d, int n_spk,
                               const std::vector<float> & sil_emb) {
    const int   per_spk  = std::max(0, SPK_CACHE_LEN / n_spk - SPK_SIL_FRAMES);
    const int   strong   = (int) std::floor(per_spk * SPK_STRONG_BOOST_RATE);
    const int   weak     = (int) std::floor(per_spk * SPK_WEAK_BOOST_RATE);
    const int   min_pos  = (int) std::floor(per_spk * SPK_MIN_POS_RATE);
    const float disabled = 0.0f;
    const float latest   = std::exp(SPK_BOOST_LATEST);

    // speaker-major: each speaker's n rows, then its silence rows (+inf)
    const int nt = n + SPK_SIL_FRAMES;
    std::vector<float> scores((size_t) n_spk * nt, std::numeric_limits<float>::infinity());
    for (int s = 0; s < n_spk; s++) {
        float * sc = scores.data() + (size_t) s * nt;
        int n_pos = 0;
        for (int t = 0; t < n; t++) {
            const float * pt = preds.data() + (size_t) t * n_spk;
            float prod = 1.0f;
            for (int j = 0; j < n_spk; j++) {
                if (j != s) {
                    prod *= std::max(1.0f - pt[j], SPK_PRED_SCORE_THR);
                }
            }
            prod *= std::max(pt[s], SPK_PRED_SCORE_THR);
            sc[t] = pt[s] > SPK_SPEECH_THR ? prod * SPK_SCORE_OFFSET : disabled;
            n_pos += sc[t] > 1.0f;
        }
        if (n_pos >= min_pos) {
            for (int t = 0; t < n; t++) {
                if (!(sc[t] > 1.0f)) {
                    sc[t] = disabled;
                }
            }
        }
        for (int t = SPK_CACHE_LEN; t < n; t++) {
            sc[t] *= latest;
        }
        for (int t : spk_topk(sc, n, strong)) {
            sc[t] *= SPK_STRONG_BOOST;
        }
        for (int t : spk_topk(sc, n, weak)) {
            sc[t] *= SPK_WEAK_BOOST;
        }
    }

    std::vector<int> pick = spk_topk(scores.data(), (int) scores.size(), SPK_CACHE_LEN);
    const int placeholder = (int) scores.size();
    for (int & i : pick) {
        if (scores[i] == disabled) {
            i = placeholder;
        }
    }
    std::sort(pick.begin(), pick.end());

    std::vector<float> out_emb((size_t) SPK_CACHE_LEN * d);
    std::vector<float> out_preds((size_t) SPK_CACHE_LEN * n_spk, 0.0f);
    for (int i = 0; i < SPK_CACHE_LEN; i++) {
        const int t = pick[i] == placeholder ? -1 : pick[i] % nt;
        float * e = out_emb.data() + (size_t) i * d;
        if (t < 0 || t >= n) {
            std::copy(sil_emb.begin(), sil_emb.end(), e);
        } else {
            std::copy_n(emb.data() + (size_t) t * d, d, e);
            std::copy_n(preds.data() + (size_t) t * n_spk, n_spk, out_preds.data() + (size_t) i * n_spk);
        }
    }
    emb.swap(out_emb);
    preds.swap(out_preds);
}

// speaker cache, with its predictions, and FIFO of one recording
struct spk_stream {
    std::vector<float> cache, cache_preds;
    std::vector<float> fifo;
};

// every update has a full window, which overflows the FIFO and the cache (NeMo's update period of 300 never applies)
static_assert(clip_graph_nemotron_transcribe::window_core - SPK_FIFO_LEN > SPK_CACHE_LEN);

// NeMo SortformerModules.streaming_update: core rows join the FIFO, its overflow moves to the cache, which is compressed
static void update_spk_state(spk_stream & ss, const clip_enc_state & st, const clip_model & model, int64_t core_begin,
                             int64_t n_core) {
    const int64_t d       = model.hparams.spk_n_embd;
    const int64_t n_spk   = model.spk_head_fc2_w->ne[1];
    const int64_t n_cache = (int64_t) ss.cache.size() / d;
    const int64_t n_fifo  = (int64_t) ss.fifo.size()  / d;
    const int64_t n_rows  = (int64_t) st.spk_chunk.size() / d;
    GGML_ASSERT((n_cache + n_fifo + n_rows) * n_spk == (int64_t) st.spk_preds.size() && core_begin + n_core <= n_rows);
    const float * preds      = st.spk_preds.data();
    const float * core_preds = preds + (n_cache + n_fifo + core_begin) * n_spk;

    std::vector<float> fifo_preds(preds + n_cache * n_spk, preds + (n_cache + n_fifo) * n_spk);
    ss.fifo.insert(ss.fifo.end(), st.spk_chunk.begin() + core_begin * d, st.spk_chunk.begin() + (core_begin + n_core) * d);
    fifo_preds.insert(fifo_preds.end(), core_preds, core_preds + n_core * n_spk);

    const int64_t n_pop = n_fifo + n_core - SPK_FIFO_LEN;
    ss.cache.insert(ss.cache.end(), ss.fifo.begin(), ss.fifo.begin() + n_pop * d);
    ss.cache_preds.insert(ss.cache_preds.end(), fifo_preds.begin(), fifo_preds.begin() + n_pop * n_spk);
    ss.fifo.erase(ss.fifo.begin(), ss.fifo.begin() + n_pop * d);
    spk_compress_cache(ss.cache, ss.cache_preds, (int) (ss.cache.size() / d), (int) d, (int) n_spk, model.spk_sil_emb);
}

// NeMo ParallelExpertEncoder._forward_online: windows with left/right context, one graph each, the speaker state carried
bool clip_graph_nemotron_transcribe::encode_windows(clip_ctx * ctx, const clip_model & model, clip_encode_params * params) {
    GGML_ASSERT(params->imgs->entries.size() == 1 && params->out_embd != nullptr);
    const clip_image_f32 & rec = params->imgs->entries[0];
    const auto &  buf      = rec.get_ro_buf();
    const int64_t n_frames = rec.nx();
    const int64_t n_rows   = rec.ny();
    const int64_t n_embd   = clip_n_mmproj_embd(ctx);
    const int     sub      = model.hparams.subsampling_factor;
    const int64_t core     = (int64_t) window_core  * sub;
    const int64_t left     = (int64_t) window_left  * sub;
    const int64_t right    = (int64_t) window_right * sub;
    const int64_t n_win    = std::max<int64_t>(1, (n_frames + core - 1) / core);
    GGML_ASSERT((int64_t) buf.size() == n_frames * n_rows);

    auto & out = *params->out_embd;
    GGML_ASSERT(out.size() == (size_t) n_embd * clip_n_output_tokens(ctx, &rec));

    // the ASR expert's input: per-feature normalization over the whole recording (NeMo normalize_batch, per_feature)
    std::vector<double> mean(n_rows), denom(n_rows);
    for (int64_t r = 0; r < n_rows; r++) {
        const float * row = buf.data() + r * n_frames;
        double sum = 0.0;
        double sq_diff_sum = 0.0;
        for (int64_t t = 0; t < n_frames; t++) {
            sum += (double) row[t];
        }
        mean[r] = sum / n_frames;
        for (int64_t t = 0; t < n_frames; t++) {
            const double diff = (double) row[t] - mean[r];
            sq_diff_sum += diff * diff;
        }
        denom[r] = std::sqrt(sq_diff_sum / (n_frames - 1.0)) + 1e-5;
    }

    spk_stream     ss;
    clip_enc_state st;
    size_t n_out = 0;
    for (int64_t w = 0; w < n_win; w++) {
        const int64_t core_begin = w * core;
        const int64_t core_end   = std::min(core_begin + core, n_frames);
        const int64_t win_begin  = std::max<int64_t>(core_begin - left, 0);
        const int64_t win_end    = std::min(core_end + right, n_frames);
        const int64_t win_len    = win_end - win_begin;

        std::vector<float> data((size_t) win_len * n_rows);
        st.mel_norm.resize(data.size());
        for (int64_t r = 0; r < n_rows; r++) {
            const float * src = buf.data() + r * n_frames + win_begin;
            std::copy_n(src, win_len, data.data() + r * win_len);
            for (int64_t t = 0; t < win_len; t++) {
                st.mel_norm[r * win_len + t] = (float) ((src[t] - mean[r]) / denom[r]);
            }
        }
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.emplace_back();
        batch.entries[0].set_size({(int) win_len, (int) n_rows}, false, true);
        batch.entries[0].cpy_buf(data);

        std::vector<float> embd((size_t) n_embd * clip_n_output_tokens(ctx, &batch.entries[0]));
        clip_encode_params wp = *params;
        wp.imgs      = &batch;
        wp.out_embd  = &embd;
        wp.enc_state = &st;
        if (!clip_encode(ctx, &wp)) {
            return false;
        }

        // output frames of the core, rounded up as NeMo's _asr_output_frame_boundary
        const int64_t n_left = (core_begin - win_begin) / sub;
        const int64_t n_core = (core_end - core_begin + sub - 1) / sub;
        GGML_ASSERT((n_out + n_core) * n_embd <= out.size());
        std::copy_n(embd.data() + n_left * n_embd, n_core * n_embd, out.data() + n_out * n_embd);
        n_out += n_core;

        if (w + 1 < n_win) {
            update_spk_state(ss, st, model, n_left, n_core);
            st.spk_prev = ss.cache;
            st.spk_prev.insert(st.spk_prev.end(), ss.fifo.begin(), ss.fifo.end());
        }
    }
    GGML_ASSERT(n_out * n_embd == out.size());
    return true;
}
