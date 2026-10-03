#include "mediagen-ltx-graph.h"

#include <algorithm>
#include <cmath>

//
// positions
//

// start and end time in seconds of latent frame f
static void ltx_video_frame_time(const ltx_hparams & hp, int f, float fps, double & t0, double & t1) {
    const double st = hp.vae_scale_t;
    t0 = f * st;
    t1 = (f + 1) * st;
    if (hp.causal_temporal_positioning) {
        t0 = std::max(t0 + 1 - st, 0.0);
        t1 = std::max(t1 + 1 - st, 0.0);
    }
    t0 /= fps;
    t1 /= fps;
}

// fractional (t, h, w) positions of the video tokens (middle of each latent cell)
static void ltx_video_positions(const ltx_hparams & hp, const ltx_latent_video & lat, float fps, std::vector<double> & pos) {
    const int F = lat.n_frames, H = lat.height, W = lat.width;
    pos.resize((size_t) F * H * W * 3);
    const double ss = hp.vae_scale_s;
    size_t i = 0;
    for (int f = 0; f < F; f++) {
        double t0, t1;
        ltx_video_frame_time(hp, f, fps, t0, t1);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                pos[i++] = 0.5 * (t0 + t1) / hp.max_pos[0];
                pos[i++] = 0.5 * (h * ss + (h + 1) * ss) / hp.max_pos[1];
                pos[i++] = 0.5 * (w * ss + (w + 1) * ss) / hp.max_pos[2];
            }
        }
    }
}

// temporal positions of the video tokens, for the audio-video cross attention
static void ltx_video_time_positions(const ltx_hparams & hp, const ltx_latent_video & lat, float fps, double max_pos, std::vector<double> & pos) {
    const int F = lat.n_frames, HW = lat.height * lat.width;
    pos.resize((size_t) F * HW);
    for (int f = 0; f < F; f++) {
        double t0, t1;
        ltx_video_frame_time(hp, f, fps, t0, t1);
        std::fill(pos.begin() + (size_t) f * HW, pos.begin() + (size_t) (f + 1) * HW, 0.5 * (t0 + t1) / max_pos);
    }
}

// time in seconds of audio latent frame boundaries
static double ltx_audio_latent_time(const ltx_hparams & hp, int i) {
    double mel = (double) i * hp.audio_latent_downsample;
    mel = std::max(mel + 1 - hp.audio_latent_downsample, 0.0);
    return mel * hp.audio_hop_length / hp.audio_sample_rate;
}

static void ltx_audio_positions(const ltx_hparams & hp, int n, double max_pos, std::vector<double> & pos) {
    pos.resize(n);
    for (int i = 0; i < n; i++) {
        pos[i] = 0.5 * (ltx_audio_latent_time(hp, i) + ltx_audio_latent_time(hp, i + 1)) / max_pos;
    }
}

//
// blocks
//

// modulation vectors of one transformer block: table rows plus the timestep projections
struct ltx_block_mod {
    ggml_tensor * shift_msa, * scale_msa, * gate_msa;
    ggml_tensor * shift_mlp, * scale_mlp, * gate_mlp;
    ggml_tensor * shift_q,   * scale_q,   * gate_ca;
    ggml_tensor * shift_kv,  * scale_kv;
};

static ltx_block_mod ltx_block_mods(ggml_context * ctx, ggml_tensor * tbl, ggml_tensor * ptbl, ggml_tensor * mod, ggml_tensor * pmod, int64_t dim) {
    ltx_block_mod m;
    m.shift_msa = ltx_ada(ctx, tbl, mod, 0, dim);
    m.scale_msa = ltx_ada(ctx, tbl, mod, 1, dim);
    m.gate_msa  = ltx_ada(ctx, tbl, mod, 2, dim);
    m.shift_mlp = ltx_ada(ctx, tbl, mod, 3, dim);
    m.scale_mlp = ltx_ada(ctx, tbl, mod, 4, dim);
    m.gate_mlp  = ltx_ada(ctx, tbl, mod, 5, dim);
    m.shift_q   = ltx_ada(ctx, tbl, mod, 6, dim);
    m.scale_q   = ltx_ada(ctx, tbl, mod, 7, dim);
    m.gate_ca   = ltx_ada(ctx, tbl, mod, 8, dim);
    m.shift_kv  = ltx_ada(ctx, ptbl, pmod, 0, dim);
    m.scale_kv  = ltx_ada(ctx, ptbl, pmod, 1, dim);
    return m;
}

// self attention with rope, then text cross attention
static ggml_tensor * ltx_stream_attn(ggml_context * ctx, const mg_weights & w, const std::string & bp, const char * attn1, const char * attn2,
                                     ggml_tensor * x, ggml_tensor * text, const ltx_block_mod & m, const ltx_rope_in & pe, int n_heads, bool flash_attn) {
    ltx_attn_weights w1 = ltx_attn_weights::load(w, bp + attn1);
    ggml_tensor * nx = mg_modulate(ctx, ltx_rms(ctx, x, 1e-6f), m.scale_msa, m.shift_msa);
    ggml_tensor * a  = ltx_attention(ctx, w1, nx, nullptr, pe.cos_t, pe.sin_t, pe.cos_t, pe.sin_t, n_heads, flash_attn);
    x = ggml_add(ctx, x, ggml_mul(ctx, a, m.gate_msa));

    ltx_attn_weights w2 = ltx_attn_weights::load(w, bp + attn2);
    ggml_tensor * qx = mg_modulate(ctx, ltx_rms(ctx, x, 1e-6f), m.scale_q, m.shift_q);
    ggml_tensor * cx = mg_modulate(ctx, text, m.scale_kv, m.shift_kv);
    ggml_tensor * c  = ltx_attention(ctx, w2, qx, cx, nullptr, nullptr, nullptr, nullptr, n_heads, flash_attn);
    return ggml_add(ctx, x, ggml_mul(ctx, c, m.gate_ca));
}

static ggml_tensor * ltx_stream_ff(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, const ltx_block_mod & m) {
    ggml_tensor * y = mg_modulate(ctx, ltx_rms(ctx, x, 1e-6f), m.scale_mlp, m.shift_mlp);
    return ggml_add(ctx, x, ggml_mul(ctx, ltx_ff(ctx, w, prefix, y), m.gate_mlp));
}

// final norm modulated by the timestep embedding, then projection
static ggml_tensor * ltx_out_head(ggml_context * ctx, const mg_weights & w, const char * table, const std::string & proj, ggml_tensor * x, ggml_tensor * emb, int64_t dim) {
    ggml_tensor * tbl   = w.get(table); // [dim, 2]
    ggml_tensor * shift = ggml_add(ctx, ltx_row(ctx, tbl, 0, dim), emb);
    ggml_tensor * scale = ggml_add(ctx, ltx_row(ctx, tbl, 1, dim), emb);
    ggml_tensor * y = mg_modulate(ctx, ggml_norm(ctx, x, 1e-6f), scale, shift);
    return mg_linear(ctx, y, w.get(proj + ".weight"), w.get(proj + ".bias"));
}

//
// forward
//

bool ltx_dit_forward(const ltx_model & model, mg_backend & be, const ltx_dit_inputs & in,
                     std::vector<float> & v_out, std::vector<float> & a_out) {
    const auto & hp = model.hp;
    const auto & w  = model.dit;
    const bool has_audio = in.audio != nullptr && in.audio->n_tokens() > 0;
    const int Tv = in.video->n_tokens();
    const int Ta = has_audio ? in.audio->n_tokens() : 0;
    const int Tc = in.cond->n_tokens;
    const int64_t VD = hp.v_dim;
    const int64_t AD = hp.a_dim;

    mg_graph g(be);
    ggml_context * ctx = g.get();

    // inputs
    ggml_tensor * vx = g.new_input_f32({ hp.in_channels, Tv }, in.video->x, "video_latent");
    ggml_tensor * ax = has_audio ? g.new_input_f32({ hp.audio_channels * hp.audio_freq_bins, Ta }, in.audio->x, "audio_latent") : nullptr;
    ggml_tensor * v_ctx = g.new_input_f32({ VD, Tc }, in.cond->video, "v_context");
    ggml_tensor * a_ctx = has_audio ? g.new_input_f32({ AD, Tc }, in.cond->audio, "a_context") : nullptr;

    std::vector<float> t_val = { in.sigma * hp.timestep_scale };
    ggml_tensor * t = g.new_input_f32({ 1 }, t_val, "timestep");
    // timestep of the audio-video cross attention gates
    std::vector<float> t_av_val = { in.sigma * hp.av_ca_timestep_scale };
    ggml_tensor * t_av = has_audio ? g.new_input_f32({ 1 }, t_av_val, "timestep_av") : nullptr;

    std::vector<double> pos;
    ltx_video_positions(hp, *in.video, in.fps, pos);
    ltx_rope_in v_pe = ltx_make_rope(g, pos, Tv, 3, VD, hp.rope_theta, "v_pe");
    ltx_rope_in a_pe, v_cross_pe, a_cross_pe;
    if (has_audio) {
        const double av_max_pos = std::max(hp.max_pos[0], hp.audio_max_pos);
        ltx_audio_positions(hp, Ta, hp.audio_max_pos, pos);
        a_pe = ltx_make_rope(g, pos, Ta, 1, AD, hp.rope_theta, "a_pe");
        ltx_video_time_positions(hp, *in.video, in.fps, av_max_pos, pos);
        v_cross_pe = ltx_make_rope(g, pos, Tv, 1, AD, hp.rope_theta, "v_cross_pe");
        ltx_audio_positions(hp, Ta, av_max_pos, pos);
        a_cross_pe = ltx_make_rope(g, pos, Ta, 1, AD, hp.rope_theta, "a_cross_pe");
    }

    // embeddings
    vx = mg_linear(ctx, vx, w.get("patchify_proj.weight"), w.get("patchify_proj.bias"));
    if (has_audio) {
        ax = mg_linear(ctx, ax, w.get("audio_patchify_proj.weight"), w.get("audio_patchify_proj.bias"));
    }

    // timestep modulation vectors
    ggml_tensor * v_mod = nullptr, * v_emb = nullptr, * vp_mod = nullptr;
    ltx_adaln_single(ctx, w, "adaln_single", t, &v_mod, &v_emb);
    ltx_adaln_single(ctx, w, "prompt_adaln_single", t, &vp_mod, nullptr);

    ggml_tensor * a_mod = nullptr, * a_emb = nullptr, * ap_mod = nullptr;
    ggml_tensor * ca_a_ss = nullptr, * ca_v_ss = nullptr, * ca_a2v_gate = nullptr, * ca_v2a_gate = nullptr;
    if (has_audio) {
        ltx_adaln_single(ctx, w, "audio_adaln_single", t, &a_mod, &a_emb);
        ltx_adaln_single(ctx, w, "audio_prompt_adaln_single", t, &ap_mod, nullptr);
        ltx_adaln_single(ctx, w, "av_ca_audio_scale_shift_adaln_single", t, &ca_a_ss, nullptr);
        ltx_adaln_single(ctx, w, "av_ca_video_scale_shift_adaln_single", t, &ca_v_ss, nullptr);
        ltx_adaln_single(ctx, w, "av_ca_a2v_gate_adaln_single", t_av, &ca_a2v_gate, nullptr);
        ltx_adaln_single(ctx, w, "av_ca_v2a_gate_adaln_single", t_av, &ca_v2a_gate, nullptr);
    }

    for (int il = 0; il < hp.n_layers; il++) {
        const std::string bp = mg_format("transformer_blocks.%d", il);

        const ltx_block_mod vm = ltx_block_mods(ctx, w.get(bp + ".scale_shift_table"), w.get(bp + ".prompt_scale_shift_table"), v_mod, vp_mod, VD);
        vx = ltx_stream_attn(ctx, w, bp, ".attn1", ".attn2", vx, v_ctx, vm, v_pe, hp.v_heads, in.flash_attn);

        if (!has_audio) {
            vx = ltx_stream_ff(ctx, w, bp + ".ff", vx, vm);
            continue;
        }

        const ltx_block_mod am = ltx_block_mods(ctx, w.get(bp + ".audio_scale_shift_table"), w.get(bp + ".audio_prompt_scale_shift_table"), a_mod, ap_mod, AD);
        ax = ltx_stream_attn(ctx, w, bp, ".audio_attn1", ".audio_attn2", ax, a_ctx, am, a_pe, hp.a_heads, in.flash_attn);

        // audio <-> video cross attention
        {
            ggml_tensor * a_tbl = w.get(bp + ".scale_shift_table_a2v_ca_audio"); // [AD, 5]
            ggml_tensor * v_tbl = w.get(bp + ".scale_shift_table_a2v_ca_video"); // [VD, 5]
            // rows 0..3: scale/shift adaln, row 4: gate adaln
            ggml_tensor * a_scale_a2v = ltx_ada(ctx, a_tbl, ca_a_ss, 0, AD);
            ggml_tensor * a_shift_a2v = ltx_ada(ctx, a_tbl, ca_a_ss, 1, AD);
            ggml_tensor * a_scale_v2a = ltx_ada(ctx, a_tbl, ca_a_ss, 2, AD);
            ggml_tensor * a_shift_v2a = ltx_ada(ctx, a_tbl, ca_a_ss, 3, AD);
            ggml_tensor * v_scale_a2v = ltx_ada(ctx, v_tbl, ca_v_ss, 0, VD);
            ggml_tensor * v_shift_a2v = ltx_ada(ctx, v_tbl, ca_v_ss, 1, VD);
            ggml_tensor * v_scale_v2a = ltx_ada(ctx, v_tbl, ca_v_ss, 2, VD);
            ggml_tensor * v_shift_v2a = ltx_ada(ctx, v_tbl, ca_v_ss, 3, VD);
            ggml_tensor * gate_a2v = ggml_add(ctx, ltx_row(ctx, v_tbl, 4, VD), ca_a2v_gate);
            ggml_tensor * gate_v2a = ggml_add(ctx, ltx_row(ctx, a_tbl, 4, AD), ca_v2a_gate);

            ggml_tensor * ax_norm = ltx_rms(ctx, ax, 1e-6f);

            // audio -> video: q = video, kv = audio
            ltx_attn_weights a2v = ltx_attn_weights::load(w, bp + ".audio_to_video_attn");
            ggml_tensor * vq = mg_modulate(ctx, ltx_rms(ctx, vx, 1e-6f), v_scale_a2v, v_shift_a2v);
            ggml_tensor * ak = mg_modulate(ctx, ax_norm, a_scale_a2v, a_shift_a2v);
            ggml_tensor * o  = ltx_attention(ctx, a2v, vq, ak, v_cross_pe.cos_t, v_cross_pe.sin_t, a_cross_pe.cos_t, a_cross_pe.sin_t, hp.a_heads, in.flash_attn);
            vx = ggml_add(ctx, vx, ggml_mul(ctx, o, gate_a2v));

            // video -> audio: q = audio, kv = video
            ltx_attn_weights v2a = ltx_attn_weights::load(w, bp + ".video_to_audio_attn");
            ggml_tensor * aq = mg_modulate(ctx, ax_norm, a_scale_v2a, a_shift_v2a);
            ggml_tensor * vk = mg_modulate(ctx, ltx_rms(ctx, vx, 1e-6f), v_scale_v2a, v_shift_v2a);
            o = ltx_attention(ctx, v2a, aq, vk, a_cross_pe.cos_t, a_cross_pe.sin_t, v_cross_pe.cos_t, v_cross_pe.sin_t, hp.a_heads, in.flash_attn);
            ax = ggml_add(ctx, ax, ggml_mul(ctx, o, gate_v2a));
        }

        vx = ltx_stream_ff(ctx, w, bp + ".ff", vx, vm);
        ax = ltx_stream_ff(ctx, w, bp + ".audio_ff", ax, am);
    }

    vx = ltx_out_head(ctx, w, "scale_shift_table", "proj_out", vx, v_emb, VD);
    g.mark_output(vx);
    if (has_audio) {
        ax = ltx_out_head(ctx, w, "audio_scale_shift_table", "audio_proj_out", ax, a_emb, AD);
        g.mark_output(ax);
    }

    if (!g.compute(be)) {
        return false;
    }
    g.get_output(vx, v_out);
    if (has_audio) {
        g.get_output(ax, a_out);
    } else {
        a_out.clear();
    }
    return true;
}

//
// scheduler
//

std::vector<float> ltx_get_sigmas(int n_steps, int n_tokens, bool distilled) {
    if (distilled) {
        // distilled checkpoints use a fixed schedule
        static const float distilled_sigmas[] = { 1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f };
        std::vector<float> s;
        const int n = (int) (sizeof(distilled_sigmas) / sizeof(distilled_sigmas[0]));
        if (n_steps <= 0 || n_steps >= n) {
            if (n_steps > n) {
                MG_WRN("%s: distilled schedule has %d steps, ignoring n_steps=%d\n", __func__, n, n_steps);
            }
            s.assign(distilled_sigmas, distilled_sigmas + n);
        } else {
            // sub-sample the schedule
            for (int i = 0; i < n_steps; i++) {
                s.push_back(distilled_sigmas[(int) std::lround((double) i * (n - 1) / std::max(n_steps - 1, 1))]);
            }
        }
        s.push_back(0.0f);
        return s;
    }
    if (n_steps <= 0) {
        n_steps = 20;
    }
    n_steps = std::max(n_steps, 2); // the stretch below needs two steps
    // LTX2 scheduler: linear sigmas shifted by the token count, stretched to a terminal value
    const float max_shift = 2.05f, base_shift = 0.95f, terminal = 0.1f;
    const float m = (max_shift - base_shift) / (4096.0f - 1024.0f);
    const float b = base_shift - m * 1024.0f;
    const float shift = std::exp(n_tokens * m + b);
    std::vector<float> s(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        float sigma = 1.0f - (float) i / n_steps;
        if (sigma != 0.0f) {
            sigma = shift / (shift + (1.0f / sigma - 1.0f));
        }
        s[i] = sigma;
    }
    // stretch so the last non-zero sigma equals terminal
    const float one_minus_last = 1.0f - s[n_steps - 1];
    const float scale = one_minus_last / (1.0f - terminal);
    for (int i = 0; i < n_steps; i++) {
        s[i] = 1.0f - (1.0f - s[i]) / scale;
    }
    s[n_steps] = 0.0f;
    return s;
}
