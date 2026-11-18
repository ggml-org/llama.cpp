#include "clip-gen.h"
#include "clip-graph.h"
#include "models/models.h"

#include "llama.h"
#include "llama-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static bool cbx_is_mtl(const clip_model & model) {
    return clip_cbx_is_mtl(model);
}

// a stage of the mmproj that is a libllama text model: its layers are stored under a prefix, with the libllama names
// libllama loads them from a synthetic GGUF: gdst holds the arch hparams, the tensor data is read from the mmproj file
struct clip_owned_llama_source {
    std::ifstream f;
    std::map<std::string, std::pair<size_t, size_t>> tensors; // name in the synthetic model -> offset and size in the mmproj
};

static void clip_owned_llama_set_tensor_data(ggml_tensor * t, void * user_data) {
    auto * src = (clip_owned_llama_source *) user_data;
    std::vector<uint8_t> buf(ggml_nbytes(t), 0);
    const std::string name = t->name;
    auto it = src->tensors.find(name);
    if (it != src->tensors.end()) {
        if (it->second.second != buf.size()) {
            throw std::runtime_error(clip_format("unexpected size of %s", t->name));
        }
        src->f.seekg((std::streamoff) it->second.first, std::ios::beg);
        if (!src->f.read((char *) buf.data(), (std::streamsize) buf.size())) {
            throw std::runtime_error(clip_format("failed to read %s", t->name));
        }
    } else {
        // optional tensors that libllama creates by itself, give them neutral values (scale = 1, dense_2 = identity, others = 0)
        // any other missing tensor is an error: a weight left at 0 gives wrong output without any error
        // (e.g. a model built from metadata has separate q, k, v and no fused qkv)
        auto ends_with = [&](const char * suffix) {
            const size_t n = strlen(suffix);
            return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
        };
        const bool optional = ends_with("scale") || ends_with(".bias") || name == "dense_2.weight" ||
                              name == "token_embd.weight" || name == "output.weight"; // inputs are embedding rows, outputs are hidden states
        if (!optional || t->type != GGML_TYPE_F32) {
            throw std::runtime_error(clip_format("tensor %s is missing from the mmproj", t->name));
        }
        float * w = (float *) buf.data();
        if (name.size() >= 5 && name.compare(name.size() - 5, 5, "scale") == 0) {
            std::fill(w, w + ggml_nelements(t), 1.0f);
        } else if (name == "dense_2.weight") {
            GGML_ASSERT(t->ne[0] == t->ne[1]);
            for (int64_t i = 0; i < t->ne[0]; i++) {
                w[i * t->ne[0] + i] = 1.0f;
            }
        }
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

// the layers are the tensors "<prefix>blk.*" and "<prefix>output_norm.*" of the mmproj, n_tensors receives their count
static llama_model * clip_owned_llama_load(const char * fname, ggml_backend_dev_t device, gguf_context * gdst, const std::string & prefix, size_t & n_tensors) {
    ggml_context * meta = nullptr;
    gguf_init_params gparams = { /*no_alloc =*/ true, /*ctx =*/ &meta };
    gguf_context_ptr gsrc(gguf_init_from_file(fname, gparams));
    ggml_context_ptr meta_ptr(meta);
    if (!gsrc) {
        throw std::runtime_error(clip_format("failed to read %s", fname));
    }

    clip_owned_llama_source src;
    for (int64_t i = 0; i < gguf_get_n_tensors(gsrc.get()); i++) {
        const std::string name = gguf_get_tensor_name(gsrc.get(), i);
        if (name.rfind(prefix + "blk.", 0) != 0 && name.rfind(prefix + "output_norm.", 0) != 0) {
            continue;
        }
        // the synthetic model only takes the type from here, the shapes come from the hparams
        ggml_tensor t = *ggml_get_tensor(meta, name.c_str());
        ggml_set_name(&t, name.substr(prefix.size()).c_str());
        gguf_add_tensor(gdst, &t);
        src.tensors[t.name] = { gguf_get_data_offset(gsrc.get()) + gguf_get_tensor_offset(gsrc.get(), i), ggml_nbytes(&t) };
    }
    n_tensors = src.tensors.size();
    src.f = open_ifstream_binary(fname);
    if (!src.f) {
        throw std::runtime_error(clip_format("failed to open %s", fname));
    }

    llama_model_params mparams = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        devs = { device, nullptr };
        mparams.devices      = devs.data();
        mparams.n_gpu_layers = -1;
    } else {
        mparams.n_gpu_layers = 0;
    }
    llama_model * model = llama_model_init_from_user(gdst, clip_owned_llama_set_tensor_data, &src, mparams);
    if (!model) {
        throw std::runtime_error(clip_format("failed to load the layers under %s", prefix.c_str()));
    }
    return model;
}

// yue2: the NAR layers are a qwen3 text model
llama_model * clip_yue2_nar_load(const char * fname, ggml_backend_dev_t device, const clip_hparams & hparams) {
    const llm_arch arch = LLM_ARCH_QWEN3;
    gguf_context_ptr gdst(gguf_init_empty());
    auto key = [&](llm_kv kv) { return llama_arch_kv_name(arch, kv); };
    gguf_set_val_str(gdst.get(), key(LLM_KV_GENERAL_ARCHITECTURE).c_str(), llama_arch_name(arch));
    gguf_set_val_str(gdst.get(), key(LLM_KV_TOKENIZER_MODEL).c_str(),       "none");
    gguf_set_val_u32(gdst.get(), key(LLM_KV_VOCAB_SIZE).c_str(),                    1);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_CONTEXT_LENGTH).c_str(),                clip_graph_yue2_gen::n_ctx_max);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_EMBEDDING_LENGTH).c_str(),              hparams.n_embd);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_FEED_FORWARD_LENGTH).c_str(),           hparams.n_ff);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_BLOCK_COUNT).c_str(),                   hparams.n_layer);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_ATTENTION_HEAD_COUNT).c_str(),          hparams.n_head);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_ATTENTION_HEAD_COUNT_KV).c_str(),       hparams.n_head_kv);
    gguf_set_val_f32(gdst.get(), key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS).c_str(),   hparams.eps);
    gguf_set_val_f32(gdst.get(), key(LLM_KV_ROPE_FREQ_BASE).c_str(),                clip_graph_yue2_gen::rope_theta);

    size_t n_tensors = 0;
    llama_model * model = clip_owned_llama_load(fname, device, gdst.get(), "a.gen.nar.", n_tensors);
    if (n_tensors != (size_t) hparams.n_layer * 11 + 1) {
        llama_model_free(model);
        throw std::runtime_error("yue2: unexpected NAR tensors in the mmproj");
    }
    return model;
}

// tiny-voca: the prosody and acoustic stacks, same arch as the text model
llama_model * clip_tiny_voca_stack_load(const char * fname, ggml_backend_dev_t device, const clip_hparams & hparams, const char * prefix, int n_layer) {
    const llm_arch arch = LLM_ARCH_TINY_VOCA;
    gguf_context_ptr gdst(gguf_init_empty());
    auto key = [&](llm_kv kv) { return llama_arch_kv_name(arch, kv); };
    gguf_set_val_str(gdst.get(), key(LLM_KV_GENERAL_ARCHITECTURE).c_str(), llama_arch_name(arch));
    gguf_set_val_str(gdst.get(), key(LLM_KV_TOKENIZER_MODEL).c_str(),       "none");
    gguf_set_val_u32(gdst.get(), key(LLM_KV_VOCAB_SIZE).c_str(),                    1);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_CONTEXT_LENGTH).c_str(),                clip_graph_tiny_voca::n_max_frames);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_EMBEDDING_LENGTH).c_str(),              hparams.n_embd);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_FEED_FORWARD_LENGTH).c_str(),           hparams.n_ff);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_BLOCK_COUNT).c_str(),                   n_layer);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_ATTENTION_HEAD_COUNT).c_str(),          hparams.n_head);
    gguf_set_val_u32(gdst.get(), key(LLM_KV_ATTENTION_HEAD_COUNT_KV).c_str(),       hparams.n_head);
    gguf_set_val_f32(gdst.get(), key(LLM_KV_ATTENTION_LAYERNORM_EPS).c_str(),       hparams.eps);
    gguf_set_val_f32(gdst.get(), key(LLM_KV_ROPE_FREQ_BASE).c_str(),                clip_graph_tiny_voca::rope_theta);
    gguf_set_val_bool(gdst.get(), key(LLM_KV_ATTENTION_CAUSAL).c_str(),             false);

    size_t n_tensors = 0;
    llama_model * model = clip_owned_llama_load(fname, device, gdst.get(), prefix, n_tensors);
    if (n_tensors != (size_t) n_layer * 18 + 2) {
        llama_model_free(model);
        throw std::runtime_error(clip_format("tiny_voca: unexpected tensors under %s in the mmproj", prefix));
    }
    return model;
}

// the libllama media engine (llama-ext.h): mmproj as a llama_model, an ENCODER context and a media proc
llama_gen_info clip_gen_audio_get_info(const clip_gen_ctx * ctx) {
    llama_gen_info info{};
    info.model_variant = "";
    info.n_channels    = 1;
    if (!ctx->ctx_gen_a) {
        info.type = LLAMA_GEN_AUDIO_TYPE_NONE;
        return info;
    }
    info.model_variant = ctx->ctx_gen_a->model.hparams.gen_model_variant.c_str();
    info.n_voices      = (int32_t) ctx->ctx_gen_a->model.gen_voices.size();
    switch (ctx->ctx_gen_a->model.proj_type) {
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_QWEN3TTS;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_POCKETTTS;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_SOPRANO:
            info.type = LLAMA_GEN_AUDIO_TYPE_SOPRANO;
            info.sample_rate = 32000;
            break;
        case PROJECTOR_TYPE_LFM3A_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_LFM3A;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_CHATTERBOX:
            info.type = LLAMA_GEN_AUDIO_TYPE_CHATTERBOX;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_DRAMA3_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_DRAMA3;
            info.sample_rate = 44100;
            break;
        case PROJECTOR_TYPE_ZONOS3_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_ZONOS3;
            info.sample_rate = 44100;
            break;
        case PROJECTOR_TYPE_YUE2_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_YUE2;
            info.sample_rate = 48000;
            info.n_channels  = clip_graph_yue2_gen::n_channels;
            break;
        case PROJECTOR_TYPE_MINIMAX_MUSIC3_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_MINIMAX_MUSIC3;
            info.sample_rate = 44100;
            info.n_channels  = clip_graph_music3_gen::n_channels;
            break;
        case PROJECTOR_TYPE_FISH_S2_GEN:
            info.type = LLAMA_GEN_AUDIO_TYPE_FISH_S2;
            info.sample_rate = 44100;
            break;
        case PROJECTOR_TYPE_KOKORO:
            info.type = LLAMA_GEN_AUDIO_TYPE_KOKORO;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_TINY_VOCA:
            info.type = LLAMA_GEN_AUDIO_TYPE_TINY_VOCA;
            info.sample_rate = 24000;
            break;
        default:
            info.type = LLAMA_GEN_AUDIO_TYPE_NONE;
            break;
    }
    return info;
}

int32_t clip_gen_audio_g2p(const clip_model & model, const char * word, char * buf, size_t buf_size) {
    std::string out;
    if (model.proj_type == PROJECTOR_TYPE_TINY_VOCA) {
        out = clip_graph_tiny_voca::g2p(model.tiny_voca, word);
    }
    if (out.size() + 1 > buf_size) {
        return -1;
    }
    memcpy(buf, out.c_str(), out.size() + 1);
    return (int32_t) out.size();
}

size_t clip_gen_audio_read_tensor(clip_gen_ctx * ctx, const char * name, float * out, size_t n_max) {
    if (!ctx->ctx_gen_a) {
        return 0;
    }
    return ctx->ctx_gen_a->model.cbx_read_tensor(name, out, n_max);
}

// raw 192-dim CAMPPlus x-vector of the fbank entry
static bool cbx_ref_xvec(clip_gen_ctx * ctx, const clip_image_f32 & fbank, std::vector<float> & xvec) {
    clip_image_f32 mel_img;
    mel_img.set_size(fbank.get_size(), false, true);
    mel_img.cpy_buf(fbank.get_ro_buf());
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(mel_img));
    xvec.resize(192);
    if (!ctx->ctx_a->encode_batch(ctx->n_threads, &batch, xvec)) {
        LOG_ERR("%s: speaker encoder failed\n", __func__);
        return false;
    }
    return true;
}

// s3 speech tokens of a log-mel entry, rows optionally receives their speech embedding rows
static int32_t cbx_ref_tokenize(clip_gen_ctx * ctx, const clip_image_f32 & mel,
                            std::vector<int32_t> & codes, std::vector<float> * rows) {
    clip_image_f32 mel_img;
    mel_img.set_size(mel.get_size(), false, true);
    mel_img.cpy_buf(mel.get_ro_buf());

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(mel_img));

    std::vector<int32_t> out_codes;
    std::vector<float>   out_embd;

    clip_subctx_encode_params params;
    params.imgs        = &batch;
    params.n_threads   = ctx->n_threads;
    params.gen_process = CLIP_SUBCTX_GEN_PROCESS_TOKENIZE;
    params.out_codes     = &out_codes;
    params.out_code_embd = rows ? &out_embd : nullptr;

    if (!ctx->ctx_gen_a->encode(&params)) {
        LOG_ERR("%s: encode failed (tokenize)\n", __func__);
        return 1;
    }

    codes = std::move(out_codes);
    if (rows) {
        *rows = std::move(out_embd);
    }
    return 0;
}

// talker conditioning rows: the speaker row, then on mtl the perceiver block and the emotion row
static int32_t cbx_ref_cond(clip_gen_ctx * ctx, const clip_image_f32 & ve_mel,
                        const std::vector<float> & pse, std::vector<float> & out_rows) {
    clip_subcontext * sctx = ctx->ctx_a;
    // n_expect 0 accepts any non-empty size
    auto read_t = [&](const char * name, std::vector<float> & v, size_t n_expect) -> bool {
        const size_t n = sctx->model.cbx_read_tensor(name, nullptr, 0);
        if (n == 0 || (n_expect > 0 && n != n_expect)) {
            return false;
        }
        v.resize(n);
        return sctx->model.cbx_read_tensor(name, v.data(), n) == n;
    };

    // voice encoder (embeds_from_wavs): 3-layer lstm per 160-frame partial, the normalized partial embeddings are averaged
    const std::vector<float> & mel = ve_mel.get_ro_buf();
    const int n_mel     = 40;
    const int n_partial = 160;
    const int n_h       = 256;
    const int step      = (int) lround((16000.0 / 1.3) / n_partial); // reference rate 1.3
    if (ve_mel.nx() < n_partial || mel.size() < (size_t) ve_mel.nx() * n_mel) {
        LOG_ERR("%s: voice encoder mel too short\n", __func__);
        return 1;
    }
    const int n_wins = (ve_mel.nx() - n_partial) / step + 1;

    std::vector<float> w_ih[3], w_hh[3], b_ih[3], b_hh[3];
    std::vector<float> w_proj, b_proj;
    for (int l = 0; l < 3; l++) {
        const std::string s = std::to_string(l);
        const size_t n_in = l == 0 ? n_mel : n_h;
        if (!read_t(("a.ve.lstm.weight_ih_l" + s).c_str(), w_ih[l], 4 * n_h * n_in) ||
            !read_t(("a.ve.lstm.weight_hh_l" + s).c_str(), w_hh[l], 4 * n_h * n_h) ||
            !read_t(("a.ve.lstm.bias_ih_l" + s).c_str(), b_ih[l], 4 * n_h) ||
            !read_t(("a.ve.lstm.bias_hh_l" + s).c_str(), b_hh[l], 4 * n_h)) {
            LOG_ERR("%s: model has no voice encoder\n", __func__);
            return 1;
        }
    }
    if (!read_t("a.ve.proj.weight", w_proj, n_h * n_h) || !read_t("a.ve.proj.bias", b_proj, n_h)) {
        LOG_ERR("%s: model has no voice encoder projection\n", __func__);
        return 1;
    }

    std::vector<float> ve(n_h, 0.0f);
    std::vector<float> h((size_t) 3 * n_h), c((size_t) 3 * n_h), x(n_h), g((size_t) 4 * n_h);
    for (int p = 0; p < n_wins; p++) {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        for (int t = 0; t < n_partial; t++) {
            const float * in = mel.data() + (size_t) (p * step + t) * n_mel;
            int n_in = n_mel;
            for (int l = 0; l < 3; l++) {
                float * hl = h.data() + (size_t) l * n_h;
                float * cl = c.data() + (size_t) l * n_h;
                for (int j = 0; j < 4 * n_h; j++) {
                    double acc = b_ih[l][(size_t) j] + b_hh[l][(size_t) j];
                    const float * wi = w_ih[l].data() + (size_t) j * n_in;
                    for (int i = 0; i < n_in; i++) {
                        acc += (double) wi[i] * in[i];
                    }
                    const float * wh = w_hh[l].data() + (size_t) j * n_h;
                    for (int i = 0; i < n_h; i++) {
                        acc += (double) wh[i] * hl[i];
                    }
                    g[(size_t) j] = (float) acc;
                }
                // torch gate order: input, forget, cell, output
                for (int i = 0; i < n_h; i++) {
                    const float gi = 1.0f / (1.0f + expf(-g[(size_t) i]));
                    const float gf = 1.0f / (1.0f + expf(-g[(size_t) i + n_h]));
                    const float gc = tanhf(g[(size_t) i + 2 * n_h]);
                    const float go = 1.0f / (1.0f + expf(-g[(size_t) i + 3 * n_h]));
                    cl[i] = gf * cl[i] + gi * gc;
                    x[(size_t) i] = go * tanhf(cl[i]);
                }
                memcpy(hl, x.data(), (size_t) n_h * sizeof(float));
                in   = hl;
                n_in = n_h;
            }
        }
        // projected, relu'd, normalized partial embedding
        std::vector<float> e(n_h);
        double norm = 0.0;
        for (int o = 0; o < n_h; o++) {
            double acc = b_proj[(size_t) o];
            const float * w = w_proj.data() + (size_t) o * n_h;
            const float * hl = h.data() + (size_t) 2 * n_h;
            for (int i = 0; i < n_h; i++) {
                acc += (double) w[i] * hl[i];
            }
            e[(size_t) o] = (float) std::max(acc, 0.0);
            norm += (double) e[(size_t) o] * e[(size_t) o];
        }
        norm = sqrt(norm);
        for (int o = 0; o < n_h; o++) {
            ve[(size_t) o] += (float) (e[(size_t) o] / norm);
        }
    }
    double norm = 0.0;
    for (float v : ve) {
        norm += (double) v * v;
    }
    norm = sqrt(norm);
    for (float & v : ve) {
        v = (float) (v / norm);
    }

    // speaker projection row
    const int n_e = (int) ctx->n_embd_out();
    std::vector<float> w_spkr, b_spkr;
    if (!read_t("a.cenc.spkr_enc.weight", w_spkr, (size_t) n_e * n_h) || !read_t("a.cenc.spkr_enc.bias", b_spkr, n_e)) {
        LOG_ERR("%s: model has no speaker conditioning projection\n", __func__);
        return 1;
    }
    std::vector<float> rows((size_t) n_e);
    for (int o = 0; o < n_e; o++) {
        double acc = b_spkr[(size_t) o];
        const float * w = w_spkr.data() + (size_t) o * n_h;
        for (int i = 0; i < n_h; i++) {
            acc += (double) w[i] * ve[(size_t) i];
        }
        rows[(size_t) o] = (float) acc;
    }

    // the perceiver runs its shared attention block as cross then self attention
    std::vector<float> query;
    if (read_t("a.cenc.perceiver.pre_attention_query", query, 0)) {
        const int    n_head = 4;
        const size_t n_ee   = (size_t) n_e * n_e;
        if (pse.empty() || pse.size() % n_e != 0 || query.size() % n_e != 0 || n_e % n_head != 0) {
            LOG_ERR("%s: reference speech embeddings required for the perceiver\n", __func__);
            return 1;
        }
        std::vector<float> ln_w, ln_b, wq, bq, wk, bk, wv, bv, wo, bo, emo;
        if (!read_t("a.cenc.perceiver.attn.norm.weight", ln_w, n_e) || !read_t("a.cenc.perceiver.attn.norm.bias", ln_b, n_e) ||
            !read_t("a.cenc.perceiver.attn.to_q.weight", wq, n_ee) || !read_t("a.cenc.perceiver.attn.to_q.bias", bq, n_e) ||
            !read_t("a.cenc.perceiver.attn.to_k.weight", wk, n_ee) || !read_t("a.cenc.perceiver.attn.to_k.bias", bk, n_e) ||
            !read_t("a.cenc.perceiver.attn.to_v.weight", wv, n_ee) || !read_t("a.cenc.perceiver.attn.to_v.bias", bv, n_e) ||
            !read_t("a.cenc.perceiver.attn.proj_out.weight", wo, n_ee) || !read_t("a.cenc.perceiver.attn.proj_out.bias", bo, n_e) ||
            !read_t("a.cenc.emotion_adv_fc.weight", emo, n_e)) {
            LOG_ERR("%s: model has an incomplete perceiver\n", __func__);
            return 1;
        }
        const int d_head = n_e / n_head;

        auto layer_norm = [&](const float * in, float * out) {
            double mean = 0.0, var = 0.0;
            for (int i = 0; i < n_e; i++) {
                mean += in[i];
            }
            mean /= n_e;
            for (int i = 0; i < n_e; i++) {
                var += (in[i] - mean) * (in[i] - mean);
            }
            const double sd = sqrt(var / n_e + 1e-5);
            for (int i = 0; i < n_e; i++) {
                out[i] = (float) ((in[i] - mean) / sd * ln_w[(size_t) i] + ln_b[(size_t) i]);
            }
        };
        auto linear = [&](const std::vector<float> & w, const std::vector<float> & b,
                          const std::vector<float> & in, int n_rows, std::vector<float> & out) {
            out.resize((size_t) n_rows * n_e);
            for (int r = 0; r < n_rows; r++) {
                for (int o = 0; o < n_e; o++) {
                    double acc = b[(size_t) o];
                    const float * wr = w.data() + (size_t) o * n_e;
                    const float * ir = in.data() + (size_t) r * n_e;
                    for (int i = 0; i < n_e; i++) {
                        acc += (double) wr[i] * ir[i];
                    }
                    out[(size_t) r * n_e + o] = (float) acc;
                }
            }
        };
        auto attn_block = [&](const std::vector<float> & x1, int n1,
                              const std::vector<float> & x2, int n2, std::vector<float> & out) {
            std::vector<float> nx1((size_t) n1 * n_e), nx2((size_t) n2 * n_e);
            for (int r = 0; r < n1; r++) {
                layer_norm(x1.data() + (size_t) r * n_e, nx1.data() + (size_t) r * n_e);
            }
            for (int r = 0; r < n2; r++) {
                layer_norm(x2.data() + (size_t) r * n_e, nx2.data() + (size_t) r * n_e);
            }
            std::vector<float> q, k, v;
            linear(wq, bq, nx1, n1, q);
            linear(wk, bk, nx2, n2, k);
            linear(wv, bv, nx2, n2, v);

            std::vector<float> ctxt((size_t) n1 * n_e);
            std::vector<double> sc((size_t) n2);
            for (int hd = 0; hd < n_head; hd++) {
                const int off = hd * d_head;
                for (int t = 0; t < n1; t++) {
                    double mx = -1e30;
                    for (int s = 0; s < n2; s++) {
                        double acc = 0.0;
                        for (int i = 0; i < d_head; i++) {
                            acc += (double) q[(size_t) t * n_e + off + i] * k[(size_t) s * n_e + off + i];
                        }
                        sc[(size_t) s] = acc / sqrt((double) d_head);
                        mx = std::max(mx, sc[(size_t) s]);
                    }
                    double sum = 0.0;
                    for (int s = 0; s < n2; s++) {
                        sc[(size_t) s] = exp(sc[(size_t) s] - mx);
                        sum += sc[(size_t) s];
                    }
                    for (int i = 0; i < d_head; i++) {
                        double acc = 0.0;
                        for (int s = 0; s < n2; s++) {
                            acc += sc[(size_t) s] * v[(size_t) s * n_e + off + i];
                        }
                        ctxt[(size_t) t * n_e + off + i] = (float) (acc / sum);
                    }
                }
            }
            linear(wo, bo, ctxt, n1, out);
            for (size_t i = 0; i < out.size(); i++) {
                out[i] += x1[i];
            }
        };

        const int n_q = (int) (query.size() / n_e);
        std::vector<float> pre, p32;
        attn_block(query, n_q, pse, (int) (pse.size() / (size_t) n_e), pre);
        attn_block(pre, n_q, pre, n_q, p32);

        rows.insert(rows.end(), p32.begin(), p32.end());
        const float exaggeration = 0.5f; // reference default
        for (int i = 0; i < n_e; i++) {
            rows.push_back(emo[(size_t) i] * exaggeration);
        }
    }

    out_rows = std::move(rows);
    return 0;
}

// talker conditioning rows go to out_embd, the flow decoder reference to ctx->gen_ref
// qwen3tts: the reference codes of a speaker clip and one talker input row per reference frame, see the tokenizer
int32_t clip_qwen3tts_ref_codes(clip_gen_ctx * ctx, const clip_image_f32 & wav, std::vector<int32_t> & codes, std::vector<float> & rows) {
    if (!ctx->ctx_gen_a || !ctx->ctx_gen_a->model.has_ref_encoder()) {
        LOG_ERR("%s: the gen tower has no reference encoder\n", __func__);
        return 1;
    }
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(wav);

    clip_subctx_encode_params params;
    params.imgs          = &batch;
    params.n_threads     = ctx->n_threads;
    params.gen_process   = CLIP_SUBCTX_GEN_PROCESS_TOKENIZE;
    params.out_codes     = &codes;
    params.out_code_embd = &rows;
    if (!ctx->ctx_gen_a->encode(&params)) {
        LOG_ERR("%s: encode failed (reference codes)\n", __func__);
        return 1;
    }
    return 0;
}

int32_t clip_cbx_ref_encode(clip_gen_ctx * ctx, const clip_audio_tokens * audio_tokens, std::vector<float> & out_embd) {
    if (!ctx->ctx_gen_a) {
        LOG_ERR("%s: reference encoding needs the audio generation context\n", __func__);
        return 1;
    }
    const auto & entries = audio_tokens->batch_f32.entries;
    GGML_ASSERT(entries.size() == 5);

    std::vector<float> xvec;
    if (!cbx_ref_xvec(ctx, entries[0], xvec)) {
        return 1;
    }

    std::vector<int32_t> flow_codes, t3_codes;
    std::vector<float> t3_rows;
    if (cbx_ref_tokenize(ctx, entries[1], flow_codes, nullptr) != 0 ||
        cbx_ref_tokenize(ctx, entries[2], t3_codes, &t3_rows) != 0) {
        return 1;
    }

    const size_t n_feat = (size_t) entries[3].nx() * 80;
    if ((size_t) entries[3].nx() != 2 * flow_codes.size()) {
        LOG_ERR("%s: reference mel length %d does not match %zu codes\n",
                __func__, entries[3].nx(), flow_codes.size());
        return 1;
    }

    // [spkr, perceiver block] on mtl, [spkr, reference speech rows] on turbo
    const bool mtl = cbx_is_mtl(ctx->ctx_a->model);
    std::vector<float> pse;
    if (mtl) {
        pse = std::move(t3_rows);
    }
    std::vector<float> rows;
    if (cbx_ref_cond(ctx, entries[4], pse, rows) != 0) {
        return 1;
    }
    if (!mtl) {
        rows.insert(rows.end(), t3_rows.begin(), t3_rows.end());
    }
    if (rows.size() != (size_t) audio_tokens->n_tokens * (size_t) ctx->n_embd_out()) {
        LOG_ERR("%s: unexpected number of conditioning rows\n", __func__);
        return 1;
    }

    ctx->gen_ref.tokens = std::move(flow_codes);
    ctx->gen_ref.feat.assign(entries[3].get_ro_buf().begin(), entries[3].get_ro_buf().begin() + n_feat);
    ctx->gen_ref.spk = std::move(xvec);

    out_embd = std::move(rows);
    return 0;
}

const char * clip_gen_audio_voice_name(const clip_gen_ctx * ctx, int32_t idx) {
    if (!ctx->ctx_gen_a || idx < 0) {
        return nullptr;
    }
    const auto & model = ctx->ctx_gen_a->model;
    const auto & voices = model.gen_voices;
    if (idx >= (int32_t) voices.size()) {
        return nullptr;
    }
    // the default voice goes first, the others keep their order
    const auto def = voices.find(model.hparams.gen_default_voice);
    if (def != voices.end() && idx == 0) {
        return def->first.c_str();
    }
    int32_t i = def != voices.end() ? 1 : 0;
    for (auto it = voices.begin(); it != voices.end(); ++it) {
        if (it != def && i++ == idx) {
            return it->first.c_str();
        }
    }
    return nullptr;
}


llama_gen_inp clip_gen_inp_default(const clip_gen_ctx * ctx) {
    llama_gen_inp inp{};
    inp.type  = LLAMA_GEN_STAGE_CODE;
    inp.speed = 1.0f;
    if (!ctx->ctx_gen_a) {
        return inp;
    }

    switch (ctx->ctx_gen_a->model.proj_type) {
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
            // https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/generation_config.json
            inp.top_k = 50;
            inp.top_p = 1.0f;
            inp.temp  = 0.9f; // TODO: handle this on graph
            break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            // https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/default_parameters.py
            inp.top_k = 50;
            inp.top_p = 1.0f;
            inp.temp  = 0.3f;
            break;
        case PROJECTOR_TYPE_SOPRANO:
        case PROJECTOR_TYPE_KOKORO:
        case PROJECTOR_TYPE_TINY_VOCA:
            inp.type = LLAMA_GEN_STAGE_WAV;
            break;
        case PROJECTOR_TYPE_MINIMAX_MUSIC3_GEN:
            // top-k 50 is fixed in the graph, like the reference
            inp.top_k = 50;
            inp.top_p = 1.0f;
            inp.temp  = 1.0f;
            break;
        case PROJECTOR_TYPE_DRAMA3_GEN:
            // codebook heads, same as the reference (sglang-omni fishaudio_drama3)
            inp.top_k = 30;
            inp.top_p = 0.9f;
            inp.temp  = 1.0f;
            break;
        case PROJECTOR_TYPE_ZONOS3_GEN:
            // sampled in the helper, defaults of registries/generation.json
            inp.top_k          = 80;
            inp.top_p          = 0.69f;
            inp.temp           = 0.75f;
            inp.penalty_repeat = 2.4f;
            inp.penalty_last_n = 96;
            break;
        case PROJECTOR_TYPE_LFM3A_GEN:
            {
                // MaskGIT sampling runs in the helper, with the settings of the checkpoint
                const auto * hparams = &ctx->ctx_gen_a->model.hparams;
                inp.top_k          = hparams->gen_top_k;
                inp.top_p          = 1.0f;
                inp.temp           = hparams->gen_temp;
                inp.penalty_repeat = hparams->gen_penalty_repeat;
                inp.penalty_last_n = hparams->gen_penalty_last_n;
            } break;
        default:
            break;
    }
    return inp;
}

struct clip_gen_state {
    clip_subcontext * sctx = nullptr;
    clip_subctx_gen_state * gen_state = nullptr;

    std::vector<int32_t> out_codes; // this frame's 16 sampled codes (GEN_CODE)
    std::vector<float>   out_feats; // this frame's continuous features, if any (GEN_CODE)
    std::vector<float>   out_embd;  // next-step hidden state fed back to backbone (GEN_CODE)
    std::vector<float>   out_audio; // decoded PCM samples for the current frame (GEN_WAV)

    // voice reference for GEN_WAV, from a speaker clip or a predefined voice (chatterbox)
    clip_gen_ref ref;

    // lfm3a: the codec graph yields STFT frames, the ISTFT runs on the host
    std::unique_ptr<clip_audio_streaming_istft> istft;

    // zonos3: the iSTFT head runs on the host
    std::unique_ptr<clip_audio_zonos3_synth> zonos3_synth;

    // minimax_music3: tail of the previous window, the next one starts from it, see clip_gen_wav_music3()
    std::vector<float> flow_prev_latent;
    std::vector<float> flow_prev_cond;


    ~clip_gen_state() {
        delete gen_state;
    }
};


clip_gen_state * clip_gen_state_init(clip_gen_ctx * ctx, uint32_t seed) {
    if (!ctx->ctx_gen_a) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return nullptr;
    }
    auto st = std::make_unique<clip_gen_state>();
    st->sctx   = ctx->ctx_gen_a;
    st->gen_state = ctx->ctx_gen_a->gen_state_init(seed);
    if (!st->gen_state) {
        return nullptr;
    }
    if (ctx->ctx_gen_a->model.proj_type == PROJECTOR_TYPE_LFM3A_GEN) {
        const auto * hparams = &ctx->ctx_gen_a->model.hparams;
        st->istft = std::make_unique<clip_audio_streaming_istft>(hparams->audio_n_fft, hparams->audio_hop_len);
    }
    return st.release();
}

void clip_gen_state_free(clip_gen_state * state) {
    delete state;
}

void clip_gen_state_reset(clip_gen_state * state) {
    state->gen_state->reset();
    if (state->istft) {
        const auto * hparams = &state->sctx->model.hparams;
        state->istft = std::make_unique<clip_audio_streaming_istft>(hparams->audio_n_fft, hparams->audio_hop_len);
    }
    state->zonos3_synth.reset();
    state->flow_prev_latent.clear();
    state->flow_prev_cond.clear();
}

// zonos3 GEN_WAV: n_codes == 0 flushes and resets the state
static int32_t clip_gen_wav_zonos3(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    clip_subcontext * sctx = ctx->ctx_gen_a;
    const auto & z = sctx->model.zonos3;
    using g = clip_graph_zonos3_gen;

    if (!state->zonos3_synth) {
        state->zonos3_synth = std::make_unique<clip_audio_zonos3_synth>(z.mixer, z.resample, g::n_fft[0], g::n_fft[1], g::hop, g::rs_orig, g::rs_new);
    }
    auto & audio = state->out_audio;
    audio.clear();

    if (!inp->codes || inp->n_codes == 0) {
        state->zonos3_synth->flush(audio);
        clip_gen_state_reset(state);
    } else {
        if (inp->n_codes % g::n_codebooks != 0) {
            LOG_ERR("%s: expected whole frames of %d codes\n", __func__, g::n_codebooks);
            return 1;
        }
        std::vector<int32_t> in_codes(inp->codes, inp->codes + inp->n_codes);
        std::vector<float>   spec;

        // no hidden-state input, the batch entry is an unused placeholder
        clip_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        clip_subctx_encode_params params;
        params.imgs        = &batch;
        params.n_threads   = ctx->n_threads;
        params.gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_WAV;
        params.codes       = &in_codes;
        params.out_feats   = &spec;
        params.gen_state   = state->gen_state;
        if (!sctx->encode(&params)) {
            LOG_ERR("%s: encode failed (codec)\n", __func__);
            return 1;
        }
        const int64_t n_rows = (g::n_fft[0] + 2) + (g::n_fft[1] + 2);
        GGML_ASSERT(spec.size() % (size_t) n_rows == 0);
        state->zonos3_synth->push(spec.data(), (int64_t) (spec.size() / n_rows), audio);
    }

    out->audio     = audio.data();
    out->n_samples = audio.size();
    return 0;
}

// lfm3a GEN_WAV: ISTFT of the codec output, n_codes == 0 flushes and resets the state
static int32_t clip_gen_wav_lfm3a(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    clip_subcontext * sctx = ctx->ctx_gen_a;
    const auto & hparams = sctx->model.hparams;

    const int n_bins = hparams.audio_n_fft / 2 + 1;
    const int n_trim = (hparams.audio_n_fft - hparams.audio_hop_len) / 2;

    auto & audio = state->out_audio;
    audio.clear();

    if (!inp->codes || inp->n_codes == 0) {
        // the tail beyond the last frame is padding, like the centered ISTFT of the reference
        audio = state->istft->flush();
        audio.resize(audio.size() - std::min(audio.size(), (size_t) n_trim));
        clip_gen_state_reset(state);
    } else {
        std::vector<int32_t> in_codes(inp->codes, inp->codes + inp->n_codes);
        std::vector<float>   feats;

        // no hidden-state input, the batch entry is an unused placeholder
        clip_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        clip_subctx_encode_params params;
        params.imgs        = &batch;
        params.n_threads   = ctx->n_threads;
        params.gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_WAV;
        params.codes       = &in_codes;
        params.out_feats   = &feats;
        params.gen_state   = state->gen_state;
        if (!sctx->encode(&params)) {
            LOG_ERR("%s: encode failed (codec)\n", __func__);
            return 1;
        }

        // [2 * n_bins, n_stft_frames]: log magnitude, then phase
        GGML_ASSERT(feats.size() % (2 * (size_t) n_bins) == 0);
        const size_t n_frames = feats.size() / (2 * (size_t) n_bins);
        std::vector<float> spec(2 * (size_t) n_bins);
        for (size_t f = 0; f < n_frames; f++) {
            const float * fr = feats.data() + f * 2 * n_bins;
            for (int j = 0; j < n_bins; j++) {
                const float mag = std::exp(fr[j]);
                spec[2 * j + 0] = mag * std::cos(fr[n_bins + j]);
                spec[2 * j + 1] = mag * std::sin(fr[n_bins + j]);
            }
            const std::vector<float> pcm = state->istft->process_frame(spec.data());
            audio.insert(audio.end(), pcm.begin(), pcm.end());
        }
    }

    out->audio     = audio.data();
    out->n_samples = audio.size();
    return 0;
}

// one pass of a graph of the gen tower, rows in -> rows out (PCM for GEN_WAV), used by yue2 and tiny-voca
static bool gen_encode_rows(clip_gen_ctx * ctx, clip_gen_state * state, clip_subctx_gen_process proc, const std::vector<float> & in, std::vector<float> & out) {
    clip_image_f32 dummy;
    dummy.set_size({1, 1}, false, true);
    dummy.cpy_buf(std::vector<float>(1, 0.0f));
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(dummy));

    clip_subctx_encode_params params;
    params.imgs        = &batch;
    params.n_threads   = ctx->n_threads;
    params.gen_process = proc;
    params.feats       = &in;
    if (proc == CLIP_SUBCTX_GEN_PROCESS_GEN_WAV) {
        params.out_audio = &out;
        params.gen_state = state->gen_state;
    } else {
        params.out_feats = &out;
    }
    return ctx->ctx_gen_a->encode(&params);
}

// NAR context: the backbone keys/values, then the NAR rows of each pass
// the state only loads into the same KV layout, try both flash attention settings
static llama_context_ptr yue2_nar_context(clip_gen_ctx * ctx, const llama_gen_inp * inp, int32_t n_rows) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = (uint32_t) (inp->backbone_n_past + n_rows);
    cparams.n_batch        = (uint32_t) n_rows;
    cparams.n_ubatch       = (uint32_t) n_rows; // non-causal, the rows of a pass must be in one ubatch
    cparams.n_seq_max      = 1;
    cparams.n_threads      = ctx->n_threads;
    cparams.n_threads_batch = ctx->n_threads;
    cparams.embeddings     = true;
    cparams.pooling_type   = LLAMA_POOLING_TYPE_NONE;
    cparams.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cparams.no_perf        = true;

    std::vector<llama_flash_attn_type> fa_types = { ctx->yue2_flash_attn };
    if (ctx->yue2_flash_attn == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        fa_types = { LLAMA_FLASH_ATTN_TYPE_ENABLED, LLAMA_FLASH_ATTN_TYPE_DISABLED };
    }
    for (llama_flash_attn_type fa : fa_types) {
        cparams.flash_attn_type = fa;
        llama_context_ptr lctx(llama_init_from_model(ctx->yue2_nar, cparams));
        if (!lctx) {
            continue;
        }
        if (llama_state_seq_set_data(lctx.get(), inp->backbone_state, inp->backbone_state_size, 0) == 0) {
            continue;
        }
        if (llama_memory_seq_pos_max(llama_get_memory(lctx.get()), 0) != inp->backbone_n_past - 1) {
            LOG_ERR("%s: the backbone state does not hold %d positions\n", __func__, inp->backbone_n_past);
            return nullptr;
        }
        return lctx;
    }
    LOG_ERR("%s: cannot load the backbone state, the backbone context must use an F16 KV cache\n", __func__);
    return nullptr;
}

// yue2 GEN_CODE: flow matching from noise to the VAE latents of one chunk, see synthesize() in the reference (nar.py)
static int32_t clip_gen_flow_yue2(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    using g = clip_graph_yue2_gen;
    const auto & y = ctx->ctx_gen_a->model.yue2;
    const int64_t n_embd = ctx->ctx_gen_a->model.hparams.n_embd;

    if (!ctx->yue2_nar || inp->n_codes == 0 || !inp->backbone_state || inp->backbone_n_past <= 0) {
        LOG_ERR("%s: yue2 needs the frame count and the backbone state\n", __func__);
        return 1;
    }
    const int64_t n_frames = (int64_t) inp->n_codes;
    const int64_t n_rows   = n_frames + 2; // latent start, frames, latent end
    const int64_t n_past   = inp->backbone_n_past;
    if (n_past + n_rows > g::n_ctx_max) {
        LOG_ERR("%s: song too long: %lld prompt positions + %lld frames, max %d positions\n", __func__,
                (long long) n_past, (long long) n_frames, g::n_ctx_max);
        return 1;
    }
    llama_context_ptr lctx = yue2_nar_context(ctx, inp, (int32_t) n_rows);
    if (!lctx) {
        return 1;
    }
    llama_memory_t mem = llama_get_memory(lctx.get());

    // sinusoidal position of each NAR row (AudioPositionEmbedding), interleaved sin/cos
    std::vector<float> pos_embd((size_t) (n_rows * n_embd));
    for (int64_t i = 0; i < n_rows; i++) {
        for (int64_t j = 0; j < n_embd / 2; j++) {
            const double a = (double) i * std::exp((double) (2 * j) * -std::log(10000.0) / (double) n_embd);
            pos_embd[i * n_embd + 2 * j]     = (float) std::sin(a);
            pos_embd[i * n_embd + 2 * j + 1] = (float) std::cos(a);
        }
    }

    // TimestepEmbedder: [cos, sin] of t * freqs -> linear -> silu -> linear
    auto time_embd = [&](double t) {
        const double raw = std::clamp(std::log(t / (1.0 - t)), -20.0, 20.0);
        const double ts  = 1.0 / (1.0 + std::exp(-raw)); // timestep_shift = 1
        const int64_t half = g::n_freq / 2;
        std::vector<float> f(g::n_freq);
        for (int64_t j = 0; j < half; j++) {
            const double a = ts * std::exp(-std::log(10000.0) * (double) j / (double) half);
            f[j]        = (float) std::cos(a);
            f[half + j] = (float) std::sin(a);
        }
        std::vector<float> h(y.time0_b), o(y.time2_b);
        for (int64_t r = 0; r < n_embd; r++) {
            for (int64_t c = 0; c < g::n_freq; c++) {
                h[r] += y.time0_w[r * g::n_freq + c] * f[c];
            }
            h[r] = h[r] / (1.0f + std::exp(-h[r]));
        }
        for (int64_t r = 0; r < n_embd; r++) {
            for (int64_t c = 0; c < n_embd; c++) {
                o[r] += y.time2_w[r * n_embd + c] * h[c];
            }
        }
        return o;
    };

    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx.get()));
    std::vector<float> x_rows((size_t) (n_rows * g::n_latent), 0.0f);
    std::vector<float> embd, hidden, vel;
    auto velocity = [&](const std::vector<float> & x, double t, std::vector<float> & v) -> bool {
        std::copy(x.begin(), x.end(), x_rows.begin() + g::n_latent); // the start and end rows stay 0
        if (!gen_encode_rows(ctx, state, CLIP_SUBCTX_GEN_PROCESS_LATENT_IN, x_rows, embd)) {
            return false;
        }
        const std::vector<float> temb = time_embd(t);
        llama_batch_ext_clear(batch.get());
        for (int64_t i = 0; i < n_rows; i++) {
            for (int64_t j = 0; j < n_embd; j++) {
                embd[i * n_embd + j] = embd[i * n_embd + j] + temb[j] + pos_embd[i * n_embd + j];
            }
            const llama_embd row = { embd.data() + i * n_embd, 1, (size_t) n_embd };
            const llama_pos  p   = (llama_pos) (n_past + i);
            const int32_t    idx = llama_batch_ext_add_embd(batch.get(), 0, row);
            if (idx < 0) {
                LOG_ERR("%s: failed to add a row to the batch (error %d)\n", __func__, idx);
                return false;
            }
            llama_batch_ext_set_pos(batch.get(), idx, &p);
            llama_batch_ext_set_output_embd(batch.get(), idx, true);
        }
        if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: NAR decode failed\n", __func__);
            return false;
        }
        const float * h = llama_get_embeddings(lctx.get());
        hidden.assign(h, h + n_rows * n_embd);
        llama_memory_seq_rm(mem, 0, (llama_pos) n_past, -1);
        if (!gen_encode_rows(ctx, state, CLIP_SUBCTX_GEN_PROCESS_LATENT_OUT, hidden, vel)) {
            return false;
        }
        v.assign(vel.begin() + g::n_latent, vel.end() - g::n_latent);
        return true;
    };

    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x((size_t) (n_frames * g::n_latent));
    for (auto & v : x) {
        v = dist(state->gen_state->rng);
    }

    // midpoint solver from t = 1 (noise) to t = 0
    std::vector<float> v1, v2, mid(x.size());
    const int    n_steps = inp->n_steps > 0 ? inp->n_steps : g::ode_steps;
    const double dt = 1.0 / n_steps;
    bool ok = true;
    for (int step = 0; step < n_steps && ok; step++) {
        const double t = 1.0 - step * dt;
        ok = velocity(x, t, v1);
        for (size_t i = 0; ok && i < x.size(); i++) {
            mid[i] = x[i] - v1[i] * (float) (dt / 2);
        }
        ok = ok && velocity(mid, t - dt / 2, v2);
        for (size_t i = 0; ok && i < x.size(); i++) {
            x[i] -= v2[i] * (float) dt;
        }
    }
    if (!ok) {
        return 1;
    }

    state->out_feats = std::move(x);
    out->feats   = state->out_feats.data();
    out->n_feats = state->out_feats.size();
    return 0;
}

// tiny-voca: one pass of a frame stack, embedding rows in -> hidden rows out
static bool tiny_voca_stack(clip_gen_ctx * ctx, llama_model * model, const std::vector<float> & in, std::vector<float> & out) {
    const int64_t n_embd = llama_model_n_embd(model);
    const int32_t n_rows = (int32_t) (in.size() / n_embd);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_rows;
    cparams.n_batch         = (uint32_t) n_rows;
    cparams.n_ubatch        = (uint32_t) n_rows; // non-causal, all rows must be in one ubatch
    cparams.n_seq_max       = 1;
    cparams.n_threads       = ctx->n_threads;
    cparams.n_threads_batch = ctx->n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.attention_type  = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cparams.flash_attn_type = ctx->tiny_voca_flash_attn;
    cparams.no_perf         = true;
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        LOG_ERR("%s: failed to create the context of a frame stack\n", __func__);
        return false;
    }

    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx.get()));
    for (int32_t i = 0; i < n_rows; i++) {
        const llama_embd row = { in.data() + (size_t) i * n_embd, 1, (size_t) n_embd };
        const llama_pos  p   = i;
        const int32_t    idx = llama_batch_ext_add_embd(batch.get(), 0, row);
        if (idx < 0) {
            LOG_ERR("%s: failed to add a row to the batch (error %d)\n", __func__, idx);
            return false;
        }
        llama_batch_ext_set_pos(batch.get(), idx, &p);
        llama_batch_ext_set_output_embd(batch.get(), idx, true);
    }
    if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("%s: frame stack decode failed\n", __func__);
        return false;
    }
    const float * h = llama_get_embeddings(lctx.get());
    out.assign(h, h + (size_t) n_rows * n_embd);
    return true;
}

// tiny-voca GEN_WAV: phoneme tokens + encoder states -> waveform, see TinyVoca.synthesize() in the reference (model.py)
// codes: the tokens; feats: the voice row (encoder half, frame half), then the encoder state of each token
static int32_t clip_gen_wav_tiny_voca(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    using g = clip_graph_tiny_voca;
    const auto &  tv     = ctx->ctx_gen_a->model.tiny_voca;
    const int64_t n_embd = ctx->ctx_gen_a->model.hparams.n_embd;
    const int64_t n_tok  = (int64_t) inp->n_codes;

    if (inp->type != LLAMA_GEN_STAGE_WAV || !inp->codes || n_tok == 0 || !inp->feats ||
            inp->n_feats != (size_t) ((2 + n_tok) * n_embd)) {
        LOG_ERR("%s: tiny_voca requires the phoneme tokens, the voice row and the encoder state of each token\n", __func__);
        return 1;
    }
    if (!std::isfinite(inp->speed) || inp->speed < 0.25f || inp->speed > 4.0f) {
        LOG_ERR("%s: speed must be between 0.25 and 4\n", __func__);
        return 1;
    }
    for (int64_t i = 0; i < n_tok; i++) {
        if (inp->codes[i] < 0 || inp->codes[i] >= (int32_t) tv.token_class.size()) {
            LOG_ERR("%s: token out of range (%d)\n", __func__, inp->codes[i]);
            return 1;
        }
    }
    const float * frame_voice = inp->feats + n_embd;
    const std::vector<float> hidden(inp->feats + 2 * n_embd, inp->feats + inp->n_feats);

    std::vector<float> dur_out;
    if (!gen_encode_rows(ctx, state, CLIP_SUBCTX_GEN_PROCESS_LATENT_IN, hidden, dur_out) || dur_out.size() != (size_t) (2 * n_tok)) {
        LOG_ERR("%s: duration predictor failed\n", __func__);
        return 1;
    }

    // frames per token: a phone lasts at least one frame, a pause token only when its presence logit is > 0
    std::vector<int64_t> dur(n_tok);
    int64_t n_frames = 0;
    for (int64_t i = 0; i < n_tok; i++) {
        const float  o0     = std::clamp(dur_out[2 * i], 0.0f, std::log1p(g::max_dur));
        const double frames = std::max(1.0, std::nearbyint((double) std::expm1(o0) / inp->speed));
        const bool   phone  = tv.token_class[inp->codes[i]] > 0.0f;
        dur[i] = phone || dur_out[2 * i + 1] > 0.0f ? (int64_t) frames : 0;
        n_frames += dur[i];
    }
    if (n_frames < 1 || n_frames > g::n_max_frames) {
        LOG_ERR("%s: the utterance has %lld frames, it must have 1 to %d, split the text\n", __func__, (long long) n_frames, g::n_max_frames);
        return 1;
    }

    // token states repeated by their duration, plus the position inside the token and the frame voice
    std::vector<float>   x((size_t) (n_frames * n_embd));
    std::vector<int64_t> owner(n_frames);
    for (int64_t i = 0, f = 0; i < n_tok; i++) {
        for (int64_t o = 0; o < dur[i]; o++, f++) {
            const float rel  = ((float) o + 0.5f) / (float) dur[i];
            const float feat[3] = { rel, 1.0f - rel, std::log1p((float) dur[i]) / 4.0f };
            float * row = x.data() + f * n_embd;
            for (int64_t c = 0; c < n_embd; c++) {
                const float * w = tv.frame_pos_w.data() + c * 3;
                row[c] = hidden[i * n_embd + c] + tv.frame_pos_b[c] + w[0] * feat[0] + w[1] * feat[1] + w[2] * feat[2] + frame_voice[c];
            }
            owner[f] = i;
        }
    }

    std::vector<float> h_pro;
    if (!tiny_voca_stack(ctx, ctx->tiny_voca_prosody, x, h_pro)) {
        return 1;
    }

    // pitch, voicing and energy; the frames of a vowel are always voiced
    std::vector<float> f0(n_frames);
    for (int64_t f = 0; f < n_frames; f++) {
        float p[3];
        const float * h = h_pro.data() + f * n_embd;
        for (int k = 0; k < 3; k++) {
            const float * w = tv.prosody_head_w.data() + k * n_embd;
            float acc = tv.prosody_head_b[k];
            for (int64_t c = 0; c < n_embd; c++) {
                acc += w[c] * h[c];
            }
            p[k] = acc;
        }
        if (tv.token_class[inp->codes[owner[f]]] == 2.0f) {
            p[1] = std::max(p[1], 2.0f);
        }
        f0[f] = std::clamp(g::f0_ref * std::exp(p[0]), g::f0_min, g::f0_max);

        // the acoustic stack is conditioned on the clamped pitch and energy and the hard voicing decision
        const float cond[3] = { std::clamp(p[0], -2.0f, 1.5f), p[1] > 0.0f ? 1.0f : 0.0f, std::clamp(p[2], -4.5f, 3.0f) };
        float * row = x.data() + f * n_embd;
        for (int64_t c = 0; c < n_embd; c++) {
            const float * w = tv.prosody_proj_w.data() + c * 3;
            row[c] = h[c] + tv.prosody_proj_b[c] + w[0] * cond[0] + w[1] * cond[1] + w[2] * cond[2];
        }
    }

    std::vector<float> h_ac, voc;
    if (!tiny_voca_stack(ctx, ctx->tiny_voca_acoustic, x, h_ac)) {
        return 1;
    }
    if (!gen_encode_rows(ctx, state, CLIP_SUBCTX_GEN_PROCESS_LATENT_OUT, h_ac, voc) || voc.size() != (size_t) (n_frames * g::n_voc)) {
        LOG_ERR("%s: source-filter heads failed\n", __func__);
        return 1;
    }

    g::synth(voc, f0, state->gen_state->rng, state->out_audio);
    out->audio     = state->out_audio.data();
    out->n_samples = state->out_audio.size();
    return 0;
}

// yue2 GEN_WAV: latents -> interleaved stereo PCM in one VAE pass, the caller splits a long song into tiles
static int32_t clip_gen_wav_yue2(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    using g = clip_graph_yue2_gen;
    if (!inp->feats || inp->n_feats == 0 || inp->n_feats % g::n_latent != 0) {
        LOG_ERR("%s: yue2 needs whole latent frames\n", __func__);
        return 1;
    }
    const std::vector<float> feats(inp->feats, inp->feats + inp->n_feats);
    auto & audio = state->out_audio;
    if (!gen_encode_rows(ctx, state, CLIP_SUBCTX_GEN_PROCESS_GEN_WAV, feats, audio)) {
        LOG_ERR("%s: encode failed (vae)\n", __func__);
        return 1;
    }
    for (auto & v : audio) {
        v = std::clamp(v, -1.0f, 1.0f);
    }

    out->audio     = audio.data();
    out->n_samples = audio.size();
    return 0;
}

// minimax_music3: one pass of an mtmd graph on rows of floats
static bool music3_encode(clip_gen_ctx * ctx, clip_gen_state * state, clip_subctx_encode_params & params, const std::vector<float> & in) {
    clip_image_f32 dummy;
    dummy.set_size({1, 1}, false, true);
    dummy.cpy_buf(std::vector<float>(1, 0.0f));
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(dummy));

    params.imgs      = &batch;
    params.n_threads = ctx->n_threads;
    params.feats     = &in;
    params.gen_state = state->gen_state;
    return ctx->ctx_gen_a->encode(&params);
}

// minimax_music3 GEN_CODE: the depth decoder of one frame, see _generate_depth_codes() of the reference (encoders.py)
static int32_t clip_gen_code_music3(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    using g = clip_graph_music3_gen;
    const size_t n_embd = (size_t) ctx->ctx_gen_a->model.n_mmproj_embd();
    if (!inp->feats || (inp->n_feats != n_embd && inp->n_feats != 2 * n_embd)) {
        LOG_ERR("%s: minimax_music3 needs 1 or 2 backbone hidden rows\n", __func__);
        return 1;
    }
    const std::vector<float> rows(inp->feats, inp->feats + inp->n_feats);

    std::vector<float> out_embd(n_embd);
    clip_subctx_encode_params params;
    params.gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_CODE;
    params.out_embd    = &out_embd;
    params.out_codes   = &state->out_codes;
    params.out_feats   = &state->out_feats;
    params.code0       = inp->code0;
    params.cfg_scale   = inp->cfg_scale > 0.0f ? inp->cfg_scale : g::depth_cfg;
    params.temp        = inp->temp;
    if (!music3_encode(ctx, state, params, rows)) {
        LOG_ERR("%s: encode failed (depth decoder)\n", __func__);
        return 1;
    }
    state->out_embd = std::move(out_embd);

    out->embd    = state->out_embd.data();
    out->codes   = state->out_codes.data();
    out->n_codes = state->out_codes.size();
    out->feats   = state->out_feats.data();
    out->n_feats = state->out_feats.size();
    return 0;
}

// minimax_music3 GEN_WAV: condition rows of one window -> flow matching -> vocoder, see denoise.py of the reference
// a window starts from the tail of the previous one, kept in the state
static int32_t clip_gen_wav_music3(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    using g = clip_graph_music3_gen;
    const auto & m = ctx->ctx_gen_a->model.music3;
    const size_t n_embd = (size_t) ctx->ctx_gen_a->model.hparams.n_embd;
    const size_t n_cond = (size_t) m.cond_w->ne[2];
    const size_t n_in   = (size_t) m.pre_w->ne[0]; // [latent, zeros, condition]
    const size_t n_lc   = g::n_latent;
    if (!inp->feats || inp->n_feats == 0 || inp->n_feats % n_embd != 0) {
        LOG_ERR("%s: minimax_music3 needs whole condition rows\n", __func__);
        return 1;
    }
    const std::vector<float> rows(inp->feats, inp->feats + inp->n_feats);

    std::vector<float> cond;
    {
        clip_subctx_encode_params params;
        params.gen_process = CLIP_SUBCTX_GEN_PROCESS_FLOW_COND;
        params.out_feats   = &cond;
        if (!music3_encode(ctx, state, params, rows)) {
            LOG_ERR("%s: encode failed (condition)\n", __func__);
            return 1;
        }
    }
    const size_t n_lat = cond.size() / n_cond;

    // the overlap takes the condition of the previous window, its latents are blended toward the previous ones
    auto & prev_lat  = state->flow_prev_latent;
    auto & prev_cond = state->flow_prev_cond;
    const size_t n_ov = std::min(prev_lat.size() / n_lc, n_lat);
    std::copy(prev_cond.begin(), prev_cond.begin() + n_ov * n_cond, cond.begin());

    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(n_lat * n_lc);
    for (auto & v : x) {
        v = dist(state->gen_state->rng);
    }
    const std::vector<float> noise_ov(x.begin(), x.begin() + n_ov * n_lc);

    // batch 0 is conditioned, batch 1 has a zero condition
    std::vector<float> dit_in(2 * n_lat * n_in, 0.0f);
    for (size_t i = 0; i < n_lat; i++) {
        std::copy(cond.begin() + i * n_cond, cond.begin() + (i + 1) * n_cond, dit_in.begin() + i * n_in + 2 * n_lc);
    }

    // FlowMatchEulerDiscreteScheduler with invert_sigmas: t = 1 - linspace(1, 1 / n, n), then 1
    const int n_steps = inp->n_steps > 0 ? inp->n_steps : g::flow_steps;
    std::vector<float> ts(n_steps + 1, 1.0f);
    for (int i = 0; i < n_steps; i++) {
        const double sigma = n_steps > 1 ? 1.0 + i * (1.0 / n_steps - 1.0) / (n_steps - 1) : 1.0;
        ts[i] = 1.0f - (float) sigma;
    }

    std::vector<float> vel;
    for (int i = 0; i < n_steps; i++) {
        const float t = ts[i];
        for (size_t j = 0; j < n_ov * n_lc; j++) {
            x[j] = (1.0f - (1.0f - 1e-6f) * t) * noise_ov[j] + t * prev_lat[j];
        }
        for (size_t b = 0; b < 2; b++) {
            for (size_t r = 0; r < n_lat; r++) {
                std::copy(x.begin() + r * n_lc, x.begin() + (r + 1) * n_lc, dit_in.begin() + (b * n_lat + r) * n_in);
            }
        }
        clip_subctx_encode_params params;
        params.gen_process = CLIP_SUBCTX_GEN_PROCESS_FLOW_STEP;
        params.flow_t      = t;
        params.cfg_scale   = g::dit_cfg;
        params.out_feats   = &vel;
        if (!music3_encode(ctx, state, params, dit_in) || vel.size() != x.size()) {
            LOG_ERR("%s: encode failed (flow step %d)\n", __func__, i);
            return 1;
        }
        const float dt = ts[i + 1] - ts[i];
        for (size_t j = 0; j < x.size(); j++) {
            x[j] += dt * vel[j];
        }
    }
    std::copy(prev_lat.begin(), prev_lat.begin() + n_ov * n_lc, x.begin());

    // the next window starts from latent frames [L - 2 * n_overlap, L - n_overlap) of this one
    const size_t s0 = n_lat > 2 * (size_t) g::n_overlap ? n_lat - 2 * g::n_overlap : 0;
    const size_t s1 = n_lat > (size_t) g::n_overlap ? n_lat - g::n_overlap : 0;
    prev_lat.assign(x.begin() + s0 * n_lc, x.begin() + s1 * n_lc);
    prev_cond.assign(cond.begin() + s0 * n_cond, cond.begin() + s1 * n_cond);

    clip_subctx_encode_params params;
    params.gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_WAV;
    params.out_audio   = &state->out_audio;
    if (!music3_encode(ctx, state, params, x)) {
        LOG_ERR("%s: encode failed (vocoder)\n", __func__);
        return 1;
    }
    out->audio     = state->out_audio.data();
    out->n_samples = state->out_audio.size();
    return 0;
}

int32_t clip_gen_state_set_ref(clip_gen_ctx * ctx, clip_gen_state * state) {
    if (!state || state->sctx != ctx->ctx_gen_a) {
        LOG_ERR("%s: invalid gen state\n", __func__);
        return 1;
    }
    if (ctx->gen_ref.tokens.empty()) {
        LOG_ERR("%s: the last encode produced no reference data\n", __func__);
        return 1;
    }
    state->ref = ctx->gen_ref;
    return 0;
}

static int32_t clip_gen_audio_process_impl(clip_gen_ctx * ctx, clip_gen_state * state, const llama_gen_inp * inp, llama_gen_out * out) {
    clip_subcontext * sctx = ctx->ctx_gen_a;
    if (!sctx) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return 1;
    }
    if (!state || state->sctx != sctx) {
        LOG_ERR("%s: invalid gen state\n", __func__);
        return 1;
    }

    *out = {};

    if (sctx->model.proj_type == PROJECTOR_TYPE_YUE2_GEN) {
        return inp->type == LLAMA_GEN_STAGE_CODE ? clip_gen_flow_yue2(ctx, state, inp, out)
                                                           : clip_gen_wav_yue2(ctx, state, inp, out);
    }

    if (sctx->model.proj_type == PROJECTOR_TYPE_MINIMAX_MUSIC3_GEN) {
        return inp->type == LLAMA_GEN_STAGE_CODE ? clip_gen_code_music3(ctx, state, inp, out)
                                                 : clip_gen_wav_music3(ctx, state, inp, out);
    }

    if (sctx->model.proj_type == PROJECTOR_TYPE_TINY_VOCA) {
        return clip_gen_wav_tiny_voca(ctx, state, inp, out);
    }

    if (sctx->model.proj_type == PROJECTOR_TYPE_SOPRANO) {
        const size_t n_embd = (size_t) sctx->model.n_mmproj_embd();
        if (inp->type != LLAMA_GEN_STAGE_WAV || !inp->feats || inp->n_feats < 2 * n_embd || inp->n_feats % n_embd != 0 || inp->codes) {
            LOG_ERR("%s: soprano requires at least 2 frames of features, without codes\n", __func__);
            return 1;
        }
    }

    if (sctx->model.proj_type == PROJECTOR_TYPE_KOKORO) {
        if (inp->type != LLAMA_GEN_STAGE_WAV || !inp->codes || inp->n_codes == 0 || !inp->feats || inp->n_feats == 0) {
            LOG_ERR("%s: kokoro requires the phoneme tokens and their features\n", __func__);
            return 1;
        }
        if (!std::isfinite(inp->speed) || inp->speed <= 0.0f) {
            LOG_ERR("%s: speed must be > 0\n", __func__);
            return 1;
        }
        const std::vector<int32_t> in_codes(inp->codes, inp->codes + inp->n_codes);
        const std::vector<float>   in_feats(inp->feats, inp->feats + inp->n_feats);

        // the batch entry is unused, present to satisfy the encode interface
        clip_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        clip_subctx_encode_params params;
        params.imgs        = &batch;
        params.n_threads   = ctx->n_threads;
        params.gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_WAV;
        params.codes       = &in_codes;
        params.feats       = &in_feats;
        params.speed       = inp->speed;
        params.out_audio   = &state->out_audio;
        params.gen_state   = state->gen_state;

        if (!sctx->encode(&params)) {
            LOG_ERR("%s: encode failed (gen_wav)\n", __func__);
            return 1;
        }

        out->audio     = state->out_audio.data();
        out->n_samples = state->out_audio.size();
        return 0;
    }

    if (inp->type == LLAMA_GEN_STAGE_WAV && sctx->model.proj_type == PROJECTOR_TYPE_LFM3A_GEN) {
        return clip_gen_wav_lfm3a(ctx, state, inp, out);
    }
    if (inp->type == LLAMA_GEN_STAGE_WAV && sctx->model.proj_type == PROJECTOR_TYPE_ZONOS3_GEN) {
        return clip_gen_wav_zonos3(ctx, state, inp, out);
    }


    if (inp->type == LLAMA_GEN_STAGE_CODE) {
        const size_t n_embd = (size_t) sctx->model.n_mmproj_embd();

        // the hidden state may be omitted when the graph does not use it
        clip_image_f32 hidden_state;
        hidden_state.set_size({(int) n_embd, 1}, false, true);
        hidden_state.cpy_buf(inp->embd ? std::vector<float>(inp->embd, inp->embd + n_embd) : std::vector<float>(n_embd, 0.0f));

        std::vector<int32_t> in_codes;
        if (inp->codes) {
            in_codes.assign(inp->codes, inp->codes + inp->n_codes);
        }

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(hidden_state));

        std::vector<float>   out_embd(n_embd);
        std::vector<int32_t> out_codes;
        std::vector<float>   out_feats;
        bool is_eos = false;

        clip_subctx_encode_params params;
        params.imgs          = &batch;
        params.n_threads     = ctx->n_threads;
        params.gen_process   = CLIP_SUBCTX_GEN_PROCESS_GEN_CODE;
        params.out_embd      = &out_embd;
        params.out_codes     = &out_codes;
        params.out_feats     = &out_feats;
        params.code0         = inp->code0;
        params.top_k         = inp->top_k;
        params.top_p         = inp->top_p;
        params.temp          = inp->temp;
        params.codes         = inp->codes ? &in_codes : nullptr;
        params.out_is_eos    = &is_eos;
        params.gen_state     = state->gen_state;

        if (!sctx->encode(&params)) {
            LOG_ERR("%s: encode failed (gen_code)\n", __func__);
            return 1;
        }

        state->out_embd  = std::move(out_embd);
        state->out_codes = std::move(out_codes);
        state->out_feats = std::move(out_feats);

        out->embd      = state->out_embd.data();
        out->n_embd    = state->out_embd.size();
        out->codes     = state->out_codes.data();
        out->n_codes   = state->out_codes.size();
        out->feats     = state->out_feats.data();
        out->n_feats   = state->out_feats.size();
        out->is_eos    = is_eos;
        return 0;
    }

    // LLAMA_GEN_STAGE_WAV
    if (sctx->model.proj_type == PROJECTOR_TYPE_CHATTERBOX) {
        if (!inp->codes || inp->n_codes == 0) {
            LOG_ERR("%s: codes required for gen_wav\n", __func__);
            return 1;
        }
        if (state->ref.tokens.empty()) {
            LOG_ERR("%s: gen_wav needs a voice reference, see clip_gen_state_set_ref()\n", __func__);
            return 1;
        }
        std::vector<int32_t> in_codes(inp->codes, inp->codes + inp->n_codes);

        // the batch entry is unused, present to satisfy the encode interface
        clip_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        clip_subctx_encode_params params;
        params.imgs        = &batch;
        params.n_threads   = ctx->n_threads;
        params.gen_process = CLIP_SUBCTX_GEN_PROCESS_TTS;
        params.codes       = &in_codes;
        params.out_audio   = &state->out_audio;
        params.ref         = &state->ref;
        params.gen_state   = state->gen_state;

        if (!sctx->encode(&params)) {
            LOG_ERR("%s: encode failed (gen_wav)\n", __func__);
            return 1;
        }

        out->audio     = state->out_audio.data();
        out->n_samples = state->out_audio.size();
        return 0;
    }

    const bool has_codes = inp->codes && inp->n_codes > 0;
    const bool has_feats = inp->feats && inp->n_feats > 0;
    if (has_codes == has_feats) {
        LOG_ERR("%s: gen_wav requires exactly one of codes or feats\n", __func__);
        return 1;
    }
    std::vector<int32_t> in_codes;
    std::vector<float>   in_feats;
    if (has_codes) {
        in_codes.assign(inp->codes, inp->codes + inp->n_codes);
    } else {
        in_feats.assign(inp->feats, inp->feats + inp->n_feats);
    }

    // gen_wav has no hidden-state input, the batch entry is an unused placeholder
    // TODO @ngxson : some models in the future may require hidden-state input, need to update this code later
    clip_image_f32 dummy;
    dummy.set_size({1, 1}, false, true);
    dummy.cpy_buf(std::vector<float>(1, 0.0f));

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(dummy));

    clip_subctx_encode_params params;
    params.imgs        = &batch;
    params.n_threads   = ctx->n_threads;
    params.gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_WAV;
    params.codes       = has_codes ? &in_codes : nullptr;
    params.feats       = has_feats ? &in_feats : nullptr;
    params.out_audio   = &state->out_audio;
    params.gen_state   = state->gen_state;

    if (!sctx->encode(&params)) {
        LOG_ERR("%s: encode failed (code2wav)\n", __func__);
        return 1;
    }

    out->audio      = state->out_audio.data();
    out->n_samples  = state->out_audio.size();

    return 0;
}

int32_t clip_gen_audio_process(clip_gen_ctx * ctx, clip_gen_state * state, const struct llama_gen_inp * inp, struct llama_gen_out * out) {
    try {
        return clip_gen_audio_process_impl(ctx, state, inp, out);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

