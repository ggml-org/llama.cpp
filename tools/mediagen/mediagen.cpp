#include "mediagen.h"
#include "mediagen-impl.h"
#include "mediagen-ltx.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <stb/stb_image_write.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <nlohmann/json.hpp>

#ifdef MEDIAGEN_FFMPEG
#include <sheredom/subprocess.h>
#endif

#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using json = nlohmann::ordered_json;

struct mediagen_context {
    mediagen_context_params params;
    mediagen_arch    arch = MEDIAGEN_ARCH_UNKNOWN;
    bool             distilled = false;
    mg_backend       be;
    ltx_model        ltx;
    ltx_text_encoder text;
    bool             flash_attn = true;

    // caches of the last enhanced prompt and text conditioning
    std::string   enhanced_from;
    std::string   enhanced;
    bool          enhanced_valid = false;
    std::string   cond_prompt;
    bool          cond_valid = false;
    ltx_text_cond cond;
    std::string   ncond_prompt;
    bool          ncond_valid = false;
    ltx_text_cond ncond;
};

mediagen_context_params mediagen_context_params_default(void) {
    mediagen_context_params p;
    p.model_path      = nullptr;
    p.vae_path        = nullptr;
    p.audio_vae_path  = nullptr;
    p.text_proj_path  = nullptr;
    p.text_model      = nullptr;
    p.use_gpu         = true;
    p.n_threads       = 4;
    p.flash_attn      = true;
    p.verbosity       = GGML_LOG_LEVEL_INFO;
    return p;
}

mediagen_gen_params mediagen_gen_params_default(void) {
    mediagen_gen_params p;
    p.prompt          = "";
    p.negative_prompt = "";
    p.width           = 768;
    p.height          = 512;
    p.n_frames        = 1;
    p.fps             = 24.0f;
    p.n_steps         = 0;
    p.cfg_scale       = 1.0f;
    p.seed            = UINT32_MAX;
    p.gen_audio       = false;
    p.enhance_prompt  = true;
    p.progress        = nullptr;
    p.progress_user_data = nullptr;
    return p;
}

//
// hparams from the gguf `config` json
//

bool ltx_load_hparams(const std::string & config_json, ltx_hparams & hp) {
    if (config_json.empty()) {
        MG_WRN("%s: no config in gguf, using defaults\n", __func__);
        return true;
    }
    json j;
    try {
        j = json::parse(config_json);
    } catch (const std::exception & e) {
        MG_ERR("%s: invalid config json: %s\n", __func__, e.what());
        return false;
    }
    const json & t = j.contains("transformer") ? j["transformer"] : j;
    auto geti = [&](const char * k, int & v) { if (t.contains(k) && t[k].is_number()) v = t[k].get<int>(); };
    auto getf = [&](const char * k, float & v) { if (t.contains(k) && t[k].is_number()) v = t[k].get<float>(); };
    auto getb = [&](const char * k, bool & v) { if (t.contains(k) && t[k].is_boolean()) v = t[k].get<bool>(); };
    geti("num_layers", hp.n_layers);
    geti("num_attention_heads", hp.v_heads);
    geti("attention_head_dim", hp.v_head_dim);
    geti("audio_num_attention_heads", hp.a_heads);
    geti("audio_attention_head_dim", hp.a_head_dim);
    geti("in_channels", hp.in_channels);
    geti("caption_channels", hp.caption_channels);
    geti("connector_num_layers", hp.connector_layers);
    geti("connector_num_learnable_registers", hp.connector_regs);
    getf("positional_embedding_theta", hp.rope_theta);
    getf("timestep_scale_multiplier", hp.timestep_scale);
    getf("av_ca_timestep_scale_multiplier", hp.av_ca_timestep_scale);
    getb("causal_temporal_positioning", hp.causal_temporal_positioning);
    getb("cross_attention_adaln", hp.cross_attention_adaln);
    getb("apply_gated_attention", hp.gated_attention);
    getb("use_audio_video_cross_attention", hp.has_audio);
    if (t.contains("positional_embedding_max_pos") && t["positional_embedding_max_pos"].is_array()) {
        auto v = t["positional_embedding_max_pos"].get<std::vector<float>>();
        for (size_t i = 0; i < std::min<size_t>(3, v.size()); i++) hp.max_pos[i] = v[i];
    }
    if (t.contains("audio_positional_embedding_max_pos") && t["audio_positional_embedding_max_pos"].is_array()) {
        auto v = t["audio_positional_embedding_max_pos"].get<std::vector<float>>();
        if (!v.empty()) hp.audio_max_pos = v[0];
    }
    if (t.contains("connector_positional_embedding_max_pos") && t["connector_positional_embedding_max_pos"].is_array()) {
        auto v = t["connector_positional_embedding_max_pos"].get<std::vector<int>>();
        if (!v.empty()) hp.connector_max_pos = v[0];
    }
    hp.v_dim = hp.v_heads * hp.v_head_dim;
    hp.a_dim = hp.a_heads * hp.a_head_dim;
    hp.text_hidden_size = hp.caption_channels;
    if (j.contains("vae")) {
        const json & v = j["vae"];
        if (v.contains("latent_channels") && v["latent_channels"].is_number()) hp.latent_channels = v["latent_channels"].get<int>();
        if (v.contains("patch_size")      && v["patch_size"].is_number())      hp.vae_patch_size  = v["patch_size"].get<int>();
    }
    return true;
}

//
// init
//

static bool str_contains_ci(const std::string & s, const char * needle) {
    std::string a = s, b = needle;
    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    return a.find(b) != std::string::npos;
}

static bool mediagen_arch_from_str(const std::string & arch, mediagen_arch * out) {
    if (arch == "ltxv" || arch == "ltx-video" || arch == "ltxav") {
        if (out) *out = MEDIAGEN_ARCH_LTXV;
        return true;
    }
    return false;
}

bool mediagen_is_diffusion_model(const char * path) {
    // probing arbitrary files: keep quiet
    const ggml_log_level prev = mediagen_logger::get().verbosity;
    mediagen_logger::get().verbosity = GGML_LOG_LEVEL_CONT;
    std::vector<mg_tensor_desc> descs;
    std::map<std::string, std::string> kv;
    const bool ok = mg_gguf_read_header(path, descs, &kv);
    mediagen_logger::get().verbosity = prev;
    if (!ok) {
        return false;
    }
    auto it = kv.find("general.architecture");
    return it != kv.end() && mediagen_arch_from_str(it->second, nullptr);
}

static mediagen_context * mediagen_init_impl(const mediagen_context_params & params) {
    if (!params.model_path || !params.text_model) {
        MG_ERR("%s: model_path and text_model are required\n", __func__);
        return nullptr;
    }
    std::unique_ptr<mediagen_context> ctx(new mediagen_context());
    ctx->params = params;
    ctx->flash_attn = params.flash_attn;

    // arch
    {
        std::vector<mg_tensor_desc> descs;
        std::map<std::string, std::string> kv;
        if (!mg_gguf_read_header(params.model_path, descs, &kv)) {
            return nullptr;
        }
        const std::string arch = kv.count("general.architecture") ? kv["general.architecture"] : "";
        if (!mediagen_arch_from_str(arch, &ctx->arch)) {
            MG_ERR("%s: unsupported diffusion architecture '%s'\n", __func__, arch.c_str());
            return nullptr;
        }
        if (!ltx_load_hparams(kv.count("config") ? kv["config"] : "", ctx->ltx.hp)) {
            return nullptr;
        }
        ctx->distilled = str_contains_ci(params.model_path, "distilled");
        bool has_audio_tensors = false;
        for (auto & d : descs) {
            if (d.name == "audio_adaln_single.linear.weight") {
                has_audio_tensors = true;
            }
        }
        ctx->ltx.hp.has_audio = has_audio_tensors;
    }

    if (!ctx->be.init(params.use_gpu, params.n_threads, 24 * 1024)) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ctx->be.weight_buft();

    const auto t0 = std::chrono::steady_clock::now();

    // transformer
    {
        mg_load_opts opts;
        if (!mg_weights_load(ctx->ltx.dit, params.model_path, buft, opts)) {
            return nullptr;
        }
        ctx->ltx.has_text_proj = ctx->ltx.dit.has("text_embedding_projection.video_aggregate_embed.weight");
    }
    // text projection
    if (params.text_proj_path && params.text_proj_path[0]) {
        mg_load_opts opts;
        opts.keep = [](const mg_tensor_desc & d) { return d.name.rfind("text_embedding_projection.", 0) == 0; };
        if (!mg_weights_load(ctx->ltx.text_proj, params.text_proj_path, buft, opts)) {
            return nullptr;
        }
        ctx->ltx.has_text_proj = ctx->ltx.text_proj.has("text_embedding_projection.video_aggregate_embed.weight");
    }
    if (!ctx->ltx.has_text_proj) {
        MG_ERR("%s: text embedding projection weights not found (pass the embeddings connectors file)\n", __func__);
        return nullptr;
    }
    // video vae
    if (params.vae_path && params.vae_path[0]) {
        mg_load_opts opts;
        opts.rename = [](const std::string & n) {
            std::string s = n;
            if (s.rfind("vae.", 0) == 0) s = s.substr(4);
            // decoder and statistics only
            if (s.rfind("encoder.", 0) == 0) return std::string();
            return s;
        };
        if (!mg_weights_load(ctx->ltx.vae, params.vae_path, buft, opts)) {
            return nullptr;
        }
        ctx->ltx.has_video_vae = ctx->ltx.vae.has("decoder.conv_in.conv.weight.t0");
    }
    // audio vae + vocoder
    if (params.audio_vae_path && params.audio_vae_path[0]) {
        mg_load_opts opts;
        opts.keep = [](const mg_tensor_desc & d) { return d.name.rfind("audio_vae.encoder.", 0) != 0; };
        opts.transform = ltx_audio_transform;
        if (!mg_weights_load(ctx->ltx.audio_vae, params.audio_vae_path, buft, opts)) {
            return nullptr;
        }
        ctx->ltx.has_audio_vae = ctx->ltx.audio_vae.has("audio_vae.decoder.conv_in.conv.weight");
    }

    // text encoder, with room for prompt enhancement
    if (!ctx->text.init(params.text_model, std::max(2048, ctx->ltx.hp.text_max_tokens), params.n_threads, params.flash_attn)) {
        return nullptr;
    }

    const auto t1 = std::chrono::steady_clock::now();
    MG_INF("%s: %s model loaded in %.1f s (distilled=%d, audio=%d, video_vae=%d, audio_vae=%d)\n", __func__,
           "ltxv", std::chrono::duration<double>(t1 - t0).count(), ctx->distilled, ctx->ltx.hp.has_audio, ctx->ltx.has_video_vae, ctx->ltx.has_audio_vae);
    return ctx.release();
}

// mg_weights::get throws on missing tensors, keep that inside the C API
mediagen_context * mediagen_init(mediagen_context_params params) {
    mediagen_logger::get().verbosity = params.verbosity;
    try {
        return mediagen_init_impl(params);
    } catch (const std::exception & e) {
        MG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mediagen_free(mediagen_context * ctx) {
    if (!ctx) {
        return;
    }
    ctx->text.free();
    delete ctx; // weights are released before the backend (reverse member order)
}

mediagen_arch mediagen_get_arch(const mediagen_context * ctx) { return ctx->arch; }
bool mediagen_supports_image(const mediagen_context * ctx) { return ctx->ltx.has_video_vae; }
bool mediagen_supports_video(const mediagen_context * ctx) { return ctx->ltx.has_video_vae; }
bool mediagen_supports_audio(const mediagen_context * ctx) { return ctx->ltx.hp.has_audio && ctx->ltx.has_audio_vae; }

//
// generation
//

static bool mediagen_get_cond(mediagen_context * ctx, const std::string & prompt, bool negative, const ltx_text_cond ** out) {
    std::string & cached_prompt = negative ? ctx->ncond_prompt : ctx->cond_prompt;
    bool & valid = negative ? ctx->ncond_valid : ctx->cond_valid;
    ltx_text_cond & cond = negative ? ctx->ncond : ctx->cond;
    if (valid && cached_prompt == prompt) {
        *out = &cond;
        return true;
    }
    valid = false;
    int n_tokens = 0, n_hidden = 0, n_states = 0;
    std::vector<float> packed;
    const auto t0 = std::chrono::steady_clock::now();
    if (!ctx->text.encode(prompt, ctx->ltx.hp.text_max_tokens, n_tokens, packed, n_hidden, n_states)) {
        return false;
    }
    if (n_hidden != ctx->ltx.hp.text_hidden_size || n_states != ctx->ltx.hp.text_n_states) {
        MG_ERR("%s: text encoder produced %d states of size %d, expected %d x %d\n", __func__, n_states, n_hidden,
               ctx->ltx.hp.text_n_states, ctx->ltx.hp.text_hidden_size);
        return false;
    }
    if (!ltx_build_text_cond(ctx->ltx, ctx->be, packed, n_tokens, n_hidden, n_states, ctx->flash_attn, cond)) {
        return false;
    }
    mg_dump(negative ? "ntext_packed" : "text_packed", packed);
    mg_dump(negative ? "ncond_video" : "cond_video", cond.video);
    mg_dump(negative ? "ncond_audio" : "cond_audio", cond.audio);
    const auto t1 = std::chrono::steady_clock::now();
    MG_INF("%s: encoded %s prompt (%d tokens) in %.2f s\n", __func__, negative ? "negative" : "positive", n_tokens,
           std::chrono::duration<double>(t1 - t0).count());
    cached_prompt = prompt;
    valid = true;
    *out = &cond;
    return true;
}

int ltx_audio_latent_frames(const ltx_hparams & hp, int n_video_frames, float fps) {
    // cover the whole clip, the decoded audio is trimmed to the video
    const double duration = (double) n_video_frames / fps;
    const double frames_per_sec = (double) hp.audio_sample_rate / hp.audio_hop_length / hp.audio_latent_downsample;
    return std::max(1, (int) std::ceil(duration * frames_per_sec) + 1);
}

static mediagen_result * mediagen_generate_impl(mediagen_context * ctx, const mediagen_gen_params * params) {
    const auto & hp = ctx->ltx.hp;
    if (!ctx->ltx.has_video_vae) {
        MG_ERR("%s: no video VAE loaded\n", __func__);
        return nullptr;
    }
    const int W = params->width, H = params->height;
    if (W <= 0 || H <= 0 || W % hp.vae_scale_s != 0 || H % hp.vae_scale_s != 0) {
        MG_ERR("%s: width and height must be positive multiples of %d\n", __func__, hp.vae_scale_s);
        return nullptr;
    }
    if (!std::isfinite(params->fps) || params->fps <= 0.0f) {
        MG_ERR("%s: fps must be positive\n", __func__);
        return nullptr;
    }
    int n_frames = std::max(1, params->n_frames);
    if ((n_frames - 1) % hp.vae_scale_t != 0) {
        const int fixed = ((n_frames - 1) / hp.vae_scale_t) * hp.vae_scale_t + 1;
        MG_WRN("%s: n_frames must be 8k+1, using %d\n", __func__, fixed);
        n_frames = fixed;
    }
    const bool gen_audio = params->gen_audio && n_frames > 1 && ctx->ltx.hp.has_audio && ctx->ltx.has_audio_vae;
    const float cfg = params->cfg_scale;
    const bool use_cfg = cfg > 1.0f;

    // prompt enhancement
    std::string prompt = params->prompt ? params->prompt : "";
    if (params->enhance_prompt && !prompt.empty()) {
        if (ctx->enhanced_valid && ctx->enhanced_from == prompt) {
            prompt = ctx->enhanced;
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            std::string enhanced;
            // fixed seed like the reference pipelines
            if (ctx->text.enhance(prompt, 10, 512, enhanced)) {
                const auto t1 = std::chrono::steady_clock::now();
                MG_INF("%s: enhanced prompt in %.1f s\n", __func__, std::chrono::duration<double>(t1 - t0).count());
                MG_DBG("%s: enhanced prompt: %s\n", __func__, enhanced.c_str());
                ctx->enhanced_from  = prompt;
                ctx->enhanced       = enhanced;
                ctx->enhanced_valid = true;
                prompt = enhanced;
            } else {
                MG_WRN("%s: prompt enhancement failed, using the prompt as is\n", __func__);
            }
        }
    }

    // text conditioning
    const ltx_text_cond * cond = nullptr, * ncond = nullptr;
    if (!mediagen_get_cond(ctx, prompt, false, &cond)) {
        return nullptr;
    }
    if (use_cfg && !mediagen_get_cond(ctx, params->negative_prompt ? params->negative_prompt : "", true, &ncond)) {
        return nullptr;
    }

    // latents
    ltx_latent_video lat;
    lat.n_frames = (n_frames - 1) / hp.vae_scale_t + 1;
    lat.height   = H / hp.vae_scale_s;
    lat.width    = W / hp.vae_scale_s;
    lat.channels = hp.latent_channels;
    lat.x.resize((size_t) lat.n_tokens() * lat.channels);

    ltx_latent_audio alat;
    if (gen_audio) {
        alat.n_frames = ltx_audio_latent_frames(hp, n_frames, params->fps);
        alat.channels = hp.audio_channels;
        alat.freq     = hp.audio_freq_bins;
        alat.x.resize((size_t) alat.n_frames * alat.channels * alat.freq);
    }

    uint32_t seed = params->seed;
    if (seed == UINT32_MAX) {
        seed = std::random_device{}();
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : lat.x) v = nd(rng);
    for (auto & v : alat.x) v = nd(rng);

    const std::vector<float> sigmas = ltx_get_sigmas(params->n_steps, lat.n_tokens(), ctx->distilled);
    const int n_steps = (int) sigmas.size() - 1;
    MG_INF("%s: generating %dx%d, %d frame(s), %d latent tokens, %d steps, cfg %.1f, seed %u%s\n", __func__, W, H, n_frames,
           lat.n_tokens(), n_steps, cfg, seed, gen_audio ? ", with audio" : "");

    ltx_dit_inputs in;
    in.video = &lat;
    in.audio = gen_audio ? &alat : nullptr;
    in.fps   = params->fps;
    in.flash_attn = ctx->flash_attn;

    mg_dump("noise_video", lat.x);
    mg_dump("noise_audio", alat.x);

    std::vector<float> v_pred, a_pred, v_unc, a_unc;
    for (int i = 0; i < n_steps; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        in.sigma = sigmas[i];
        in.cond  = cond;
        if (!ltx_dit_forward(ctx->ltx, ctx->be, in, v_pred, a_pred)) {
            return nullptr;
        }
        if (i == 0) {
            mg_dump("pred0_video", v_pred);
            mg_dump("pred0_audio", a_pred);
        }
        if (use_cfg) {
            in.cond = ncond;
            if (!ltx_dit_forward(ctx->ltx, ctx->be, in, v_unc, a_unc)) {
                return nullptr;
            }
            for (size_t k = 0; k < v_pred.size(); k++) v_pred[k] = v_unc[k] + cfg * (v_pred[k] - v_unc[k]);
            for (size_t k = 0; k < a_pred.size(); k++) a_pred[k] = a_unc[k] + cfg * (a_pred[k] - a_unc[k]);
        }
        // euler step on the flow: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v
        const float dt = sigmas[i + 1] - sigmas[i];
        for (size_t k = 0; k < lat.x.size(); k++) lat.x[k] += dt * v_pred[k];
        for (size_t k = 0; k < alat.x.size(); k++) alat.x[k] += dt * a_pred[k];
        const auto t1 = std::chrono::steady_clock::now();
        MG_INF("%s: step %d/%d sigma %.4f -> %.4f (%.2f s)\n", __func__, i + 1, n_steps, sigmas[i], sigmas[i + 1],
               std::chrono::duration<double>(t1 - t0).count());
        if (params->progress && !params->progress(i + 1, n_steps, params->progress_user_data)) {
            MG_INF("%s: cancelled\n", __func__);
            return nullptr;
        }
    }

    // decode
    std::unique_ptr<mediagen_result> res(new mediagen_result());
    memset(res.get(), 0, sizeof(*res));
    res->revised_prompt = (char *) malloc(prompt.size() + 1);
    if (res->revised_prompt) {
        memcpy(res->revised_prompt, prompt.c_str(), prompt.size() + 1);
    }
    // the transformer's compute buffers are of no use to the VAE
    ctx->be.release_compute();
    {
        const auto t0 = std::chrono::steady_clock::now();
        int of = 0, oh = 0, ow = 0;
        std::vector<float> rgb;
        const bool ok = ltx_vae_decode(ctx->ltx, ctx->be, lat, of, oh, ow, rgb);
        ctx->be.release_compute();
        if (!ok) {
            return nullptr;
        }
        const auto t1 = std::chrono::steady_clock::now();
        MG_INF("%s: decoded %d frame(s) of %dx%d in %.2f s\n", __func__, of, ow, oh, std::chrono::duration<double>(t1 - t0).count());
        const int nf = std::min(of, n_frames);
        res->width    = ow;
        res->height   = oh;
        res->n_frames = nf;
        res->fps      = params->fps;
        res->rgb      = (uint8_t *) malloc((size_t) ow * oh * 3 * nf);
        if (!res->rgb) {
            mediagen_result_free(res.release());
            return nullptr;
        }
        for (size_t k = 0; k < (size_t) ow * oh * 3 * nf; k++) {
            float v = (rgb[k] + 1.0f) * 0.5f;
            v = std::min(1.0f, std::max(0.0f, v));
            res->rgb[k] = (uint8_t) std::lround(v * 255.0f);
        }
    }
    if (gen_audio) {
        const auto t0 = std::chrono::steady_clock::now();
        int sr = 0, nch = 0;
        std::vector<float> pcm;
        const bool ok = ltx_audio_decode(ctx->ltx, ctx->be, alat, sr, nch, pcm);
        ctx->be.release_compute();
        if (!ok) {
            MG_WRN("%s: audio decoding failed, returning video only\n", __func__);
        } else {
            // trim to the video duration
            int64_t n_samples = pcm.size() / nch;
            const int64_t want = (int64_t) std::llround((double) n_frames / params->fps * sr);
            n_samples = std::min(n_samples, want);
            res->pcm = n_samples > 0 ? (float *) malloc(sizeof(float) * n_samples * nch) : nullptr;
            if (res->pcm) {
                res->sample_rate = sr;
                res->n_channels  = nch;
                res->n_samples   = n_samples;
                memcpy(res->pcm, pcm.data(), sizeof(float) * n_samples * nch);
            }
            const auto t1 = std::chrono::steady_clock::now();
            MG_INF("%s: decoded %.2f s of audio in %.2f s\n", __func__, (double) n_samples / sr, std::chrono::duration<double>(t1 - t0).count());
        }
    }
    return res.release();
}

mediagen_result * mediagen_generate(mediagen_context * ctx, const mediagen_gen_params * params) {
    try {
        return mediagen_generate_impl(ctx, params);
    } catch (const std::exception & e) {
        MG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mediagen_result_free(mediagen_result * res) {
    if (!res) {
        return;
    }
    free(res->rgb);
    free(res->pcm);
    free(res->revised_prompt);
    delete res;
}

//
// encoders
//

// copy to a buffer the caller frees with free()
static uint8_t * mg_malloc_copy(const std::vector<uint8_t> & data, size_t * n_bytes) {
    uint8_t * buf = (uint8_t *) malloc(data.size());
    if (!buf) {
        return nullptr;
    }
    memcpy(buf, data.data(), data.size());
    *n_bytes = data.size();
    return buf;
}

static void png_write_cb(void * context, void * data, int size) {
    std::vector<uint8_t> * out = (std::vector<uint8_t> *) context;
    out->insert(out->end(), (uint8_t *) data, (uint8_t *) data + size);
}

uint8_t * mediagen_encode_png(const uint8_t * rgb, int width, int height, size_t * n_bytes) {
    std::vector<uint8_t> out;
    if (!stbi_write_png_to_func(png_write_cb, &out, width, height, 3, rgb, width * 3)) {
        return nullptr;
    }
    return mg_malloc_copy(out, n_bytes);
}

uint8_t * mediagen_encode_wav(const float * pcm, int64_t n_samples, int n_channels, int sample_rate, size_t * n_bytes) {
    if (n_samples <= 0 || n_channels <= 0 || n_samples * n_channels * 2 > (int64_t) UINT32_MAX - 44) {
        return nullptr;
    }
    const uint32_t data_size = (uint32_t) (n_samples * n_channels * 2);
    std::vector<uint8_t> out;
    out.reserve(44 + data_size);
    auto put32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) out.push_back((v >> (8 * i)) & 0xff); };
    auto put16 = [&](uint16_t v) { for (int i = 0; i < 2; i++) out.push_back((v >> (8 * i)) & 0xff); };
    out.insert(out.end(), { 'R', 'I', 'F', 'F' });
    put32(36 + data_size);
    out.insert(out.end(), { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
    put32(16);
    put16(1);
    put16((uint16_t) n_channels);
    put32((uint32_t) sample_rate);
    put32((uint32_t) (sample_rate * n_channels * 2));
    put16((uint16_t) (n_channels * 2));
    put16(16);
    out.insert(out.end(), { 'd', 'a', 't', 'a' });
    put32(data_size);
    for (int64_t i = 0; i < n_samples * n_channels; i++) {
        float v = std::min(1.0f, std::max(-1.0f, pcm[i]));
        put16((uint16_t) (int16_t) std::lround(v * 32767.0f));
    }
    return mg_malloc_copy(out, n_bytes);
}

//
// video container (ffmpeg)
//

#ifdef MEDIAGEN_FFMPEG
static bool run_ffmpeg(const std::vector<std::string> & args) {
    std::vector<const char *> cmd;
    cmd.push_back("ffmpeg");
    for (auto & a : args) {
        cmd.push_back(a.c_str());
    }
    cmd.push_back(nullptr);
    subprocess_s proc;
    if (subprocess_create(cmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment | subprocess_option_no_window, &proc) != 0) {
        return false;
    }
    int ret = -1;
    subprocess_join(&proc, &ret);
    subprocess_destroy(&proc);
    return ret == 0;
}

// private directory for the intermediate files, "" on failure
static std::string mg_make_temp_dir() {
    std::error_code ec;
    const std::filesystem::path base = std::filesystem::temp_directory_path(ec);
    if (ec) {
        return "";
    }
    std::random_device rd;
    for (int attempt = 0; attempt < 16; attempt++) {
        const std::filesystem::path dir = base / mg_format("mediagen-%08x%08x", rd(), rd());
#ifdef _WIN32
        if (_mkdir(dir.string().c_str()) == 0) {
            return dir.string();
        }
#else
        if (mkdir(dir.c_str(), 0700) == 0) {
            return dir.string();
        }
#endif
    }
    return "";
}

static bool mg_write_file(const std::string & path, const void * data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) data, n);
    f.close();
    return f.good();
}

bool mediagen_has_ffmpeg(void) {
    static const bool has = run_ffmpeg({ "-version", "-loglevel", "quiet" });
    return has;
}

uint8_t * mediagen_encode_mp4(const mediagen_result * res, size_t * n_bytes) {
    if (!res || !res->rgb || res->n_frames <= 0 || !mediagen_has_ffmpeg()) {
        return nullptr;
    }
    const std::string dir = mg_make_temp_dir();
    if (dir.empty()) {
        MG_ERR("%s: failed to create a temporary directory\n", __func__);
        return nullptr;
    }
    const std::string rgb_path = dir + "/frames.rgb";
    const std::string wav_path = dir + "/audio.wav";
    const std::string mp4_path = dir + "/out.mp4";

    uint8_t * out = nullptr;
    bool ok = mg_write_file(rgb_path, res->rgb, (size_t) res->width * res->height * 3 * res->n_frames);
    bool has_audio = false;
    if (ok && res->pcm && res->n_samples > 0) {
        size_t n = 0;
        uint8_t * wav = mediagen_encode_wav(res->pcm, res->n_samples, res->n_channels, res->sample_rate, &n);
        has_audio = wav && mg_write_file(wav_path, wav, n);
        free(wav);
    }
    if (ok) {
        std::vector<std::string> args = {
            "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", mg_format("%dx%d", res->width, res->height),
            "-r", mg_format("%g", res->fps), "-i", rgb_path,
        };
        if (has_audio) {
            args.insert(args.end(), { "-f", "wav", "-i", wav_path });
        }
        args.insert(args.end(), { "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart" });
        if (has_audio) {
            args.insert(args.end(), { "-c:a", "aac", "-b:a", "192k", "-af", "apad" });
        }
        args.insert(args.end(), { "-t", mg_format("%.6f", (double) res->n_frames / res->fps), "-f", "mp4", mp4_path });
        if (run_ffmpeg(args)) {
            std::ifstream f(mp4_path, std::ios::binary | std::ios::ate);
            const size_t n = f ? (size_t) f.tellg() : 0;
            if (n > 0) {
                f.seekg(0);
                out = (uint8_t *) malloc(n);
                if (out && f.read((char *) out, n)) {
                    *n_bytes = n;
                } else {
                    free(out);
                    out = nullptr;
                }
            }
        } else {
            MG_ERR("%s: ffmpeg failed\n", __func__);
        }
    } else {
        MG_ERR("%s: failed to write %s\n", __func__, rgb_path.c_str());
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return out;
}
#else
bool mediagen_has_ffmpeg(void) {
    return false;
}

uint8_t * mediagen_encode_mp4(const mediagen_result * res, size_t * n_bytes) {
    GGML_UNUSED(res);
    GGML_UNUSED(n_bytes);
    return nullptr;
}
#endif
