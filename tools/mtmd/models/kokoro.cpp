#include "models.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

// Kokoro: StyleTTS2 prosody predictor + iSTFTNet decoder. Tensors keep the names of the reference modules (a.gen.<module>.*)
// One utterance runs the 3 graphs of stage_type. The LSTMs, the alignment, the harmonic source and the ISTFT run on the host in between
// The graphs keep activations as [T, C], the layout of the conv ops

static const int   KOKORO_N_HIDDEN = 512;
static const int   KOKORO_N_HARM   = 9;     // fundamental + 8 overtones
static const int   KOKORO_UPS      = 300;   // samples per F0 step: upsample rates 10 * 6, times the ISTFT hop
static const int   KOKORO_N_FFT    = 20;
static const int   KOKORO_HOP      = 5;
static const float KOKORO_SR       = 24000.0f;

ggml_tensor * mtmd_graph_kokoro::tensor(const std::string & name) const {
    const auto it = model.cbx_tensors.find(name);
    if (it == model.cbx_tensors.end() || !it->second) {
        throw std::runtime_error("kokoro: missing tensor " + name);
    }
    return it->second;
}

bool mtmd_graph_kokoro::has_tensor(const std::string & name) const {
    return model.cbx_tensors.find(name) != model.cbx_tensors.end();
}

// x [T, IC] -> [T_out, OC], with the bias if the module has one
ggml_tensor * mtmd_graph_kokoro::conv(const std::string & name, ggml_tensor * x, int stride, int pad, int dil) const {
    ggml_tensor * y = ggml_conv_1d(ctx0, tensor(name + ".weight"), x, stride, pad, dil);
    if (has_tensor(name + ".bias")) {
        ggml_tensor * b = tensor(name + ".bias");
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

// AdaIN1d: instance norm over time, then scale and shift from the style
ggml_tensor * mtmd_graph_kokoro::adain(const std::string & name, ggml_tensor * x, ggml_tensor * style) const {
    const int64_t n_ch = x->ne[1];
    ggml_tensor * h = ggml_add(ctx0, ggml_mul_mat(ctx0, tensor(name + ".fc.weight"), style), tensor(name + ".fc.bias"));
    ggml_tensor * gamma = ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, h, n_ch, 0), 1, n_ch);
    ggml_tensor * beta  = ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, h, n_ch, n_ch * ggml_element_size(h)), 1, n_ch);
    ggml_tensor * xn = ggml_norm(ctx0, x, 1e-5f);
    return ggml_add(ctx0, ggml_add(ctx0, xn, ggml_mul(ctx0, xn, gamma)), beta);
}

// x + sin^2(alpha x) / alpha, alpha per channel
ggml_tensor * mtmd_graph_kokoro::snake(const std::string & name, ggml_tensor * x) const {
    ggml_tensor * alpha = tensor(name);
    alpha = ggml_reshape_2d(ctx0, alpha, 1, alpha->ne[0]);
    ggml_tensor * sx = ggml_sin(ctx0, ggml_mul(ctx0, x, alpha));
    sx = ggml_div(ctx0, ggml_mul(ctx0, sx, sx), alpha);
    return ggml_add(ctx0, x, sx);
}

// a [T, C] and b [T, C] -> [2T, C] with a on the even steps
ggml_tensor * mtmd_graph_kokoro::interleave(ggml_tensor * a, ggml_tensor * b) const {
    const int64_t n_t = a->ne[0], n_ch = a->ne[1];
    ggml_tensor * ab = ggml_concat(ctx0, ggml_reshape_3d(ctx0, ggml_cont(ctx0, a), 1, n_t, n_ch),
                                         ggml_reshape_3d(ctx0, ggml_cont(ctx0, b), 1, n_t, n_ch), 0);
    return ggml_reshape_2d(ctx0, ggml_cont(ctx0, ab), 2 * n_t, n_ch);
}

// AdainResBlk1d, upsample doubles the length
ggml_tensor * mtmd_graph_kokoro::adain_resblk(const std::string & name, ggml_tensor * x, ggml_tensor * style, bool upsample) const {
    const int64_t n_t = x->ne[0], n_ch = x->ne[1];

    ggml_tensor * r = adain(name + ".norm1", x, style);
    r = ggml_leaky_relu(ctx0, r, 0.2f, false);
    if (upsample) {
        // depthwise ConvTranspose1d(kernel 3, stride 2, padding 1, output_padding 1), weight stored as [C, 3]:
        // y[2i] = w1 x[i], y[2i+1] = w2 x[i] + w0 x[i+1]
        ggml_tensor * w = tensor(name + ".pool.weight");
        ggml_tensor * b = tensor(name + ".pool.bias");
        auto tap = [&](int k) {
            return ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, w, n_ch, (size_t) k * n_ch * ggml_element_size(w)), 1, n_ch);
        };
        ggml_tensor * next = ggml_cont(ctx0, ggml_view_2d(ctx0, r, n_t - 1, n_ch, r->nb[1], ggml_element_size(r)));
        next = ggml_pad(ctx0, next, 1, 0, 0, 0);
        ggml_tensor * even = ggml_mul(ctx0, r, tap(1));
        ggml_tensor * odd  = ggml_add(ctx0, ggml_mul(ctx0, r, tap(2)), ggml_mul(ctx0, next, tap(0)));
        r = ggml_add(ctx0, interleave(even, odd), ggml_reshape_2d(ctx0, b, 1, n_ch));
    }
    r = conv(name + ".conv1", r, 1, 1, 1);
    r = adain(name + ".norm2", r, style);
    r = ggml_leaky_relu(ctx0, r, 0.2f, false);
    r = conv(name + ".conv2", r, 1, 1, 1);

    ggml_tensor * sc = upsample ? interleave(x, x) : x;
    if (has_tensor(name + ".conv1x1.weight")) {
        sc = conv(name + ".conv1x1", sc, 1, 0, 1);
    }
    return ggml_scale(ctx0, ggml_add(ctx0, r, sc), (float) M_SQRT1_2);
}

// AdaINResBlock1 of the generator: three units with dilations 1/3/5 on convs1
ggml_tensor * mtmd_graph_kokoro::gen_resblock(const std::string & name, ggml_tensor * x, ggml_tensor * style) const {
    static const int dil[3] = {1, 3, 5};
    for (int j = 0; j < 3; j++) {
        const std::string js = std::to_string(j);
        const int k = (int) tensor(name + ".convs1." + js + ".weight")->ne[0];
        ggml_tensor * xt = adain(name + ".adain1." + js, x, style);
        xt = snake(name + ".alpha1." + js, xt);
        xt = conv(name + ".convs1." + js, xt, 1, (k * dil[j] - dil[j]) / 2, dil[j]);
        xt = adain(name + ".adain2." + js, xt, style);
        xt = snake(name + ".alpha2." + js, xt);
        xt = conv(name + ".convs2." + js, xt, 1, (k - 1) / 2, 1);
        x = ggml_add(ctx0, xt, x);
    }
    return x;
}

ggml_tensor * mtmd_graph_kokoro::input(const char * name, ggml_type type, int64_t ne0, int64_t ne1) const {
    ggml_tensor * t = ggml_new_tensor_2d(ctx0, type, ne0, ne1);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_cgraph * mtmd_graph_kokoro::output(const char * name, ggml_tensor * t) const {
    ggml_set_name(t, name);
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
    return gf;
}

// tokens -> text encoder CNN output [C, n_tok]
ggml_cgraph * mtmd_graph_kokoro::build_text() {
    const std::string p = "a.gen.text.";
    ggml_tensor * tokens = input("inp_tokens", GGML_TYPE_I32, n_tok, 1);
    tokens = ggml_reshape_1d(ctx0, tokens, n_tok);

    ggml_tensor * x = ggml_get_rows(ctx0, tensor(p + "embedding.weight"), tokens); // [C, T]
    for (int i = 0; has_tensor(p + "cnn." + std::to_string(i) + ".0.weight"); i++) {
        const std::string q = p + "cnn." + std::to_string(i);
        const int k = (int) tensor(q + ".0.weight")->ne[0];
        x = conv(q + ".0", ggml_cont(ctx0, ggml_transpose(ctx0, x)), 1, (k - 1) / 2, 1);
        // LayerNorm over the channels
        x = ggml_norm(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 1e-5f);
        x = ggml_add(ctx0, ggml_mul(ctx0, x, tensor(q + ".1.gamma")), tensor(q + ".1.beta"));
        x = ggml_leaky_relu(ctx0, x, 0.2f, false);
    }
    return output("out_text", x);
}

// shared LSTM output [C, n_frames] -> F0 and energy curves, 2 steps per frame
ggml_cgraph * mtmd_graph_kokoro::build_f0n() {
    const std::string p = "a.gen.pred.";
    ggml_tensor * inp   = input("inp_x", GGML_TYPE_F32, KOKORO_N_HIDDEN, n_frames);
    ggml_tensor * style = input("inp_style", GGML_TYPE_F32, n_style, 1);
    style = ggml_reshape_1d(ctx0, style, n_style);
    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    for (const char * curve : {"F0", "N"}) {
        ggml_tensor * cur = x;
        for (int i = 0; i < 3; i++) {
            const std::string q = p + curve + "." + std::to_string(i);
            cur = adain_resblk(q, cur, style, has_tensor(q + ".pool.weight"));
        }
        cur = conv(p + curve + "_proj", cur, 1, 0, 1);
        ggml_set_name(cur, curve[0] == 'F' ? "out_f0" : "out_n");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }
    return gf;
}

// aligned text features, F0, energy and the source STFT -> conv_post output [n_fft + 2, n_har]
ggml_cgraph * mtmd_graph_kokoro::build_dec() {
    const std::string p = "a.gen.dec.";
    ggml_tensor * asr   = input("inp_asr", GGML_TYPE_F32, KOKORO_N_HIDDEN, n_frames);
    ggml_tensor * f0    = input("inp_f0", GGML_TYPE_F32, 2 * n_frames, 1);
    ggml_tensor * en    = input("inp_n", GGML_TYPE_F32, 2 * n_frames, 1);
    ggml_tensor * har   = input("inp_har", GGML_TYPE_F32, KOKORO_N_FFT + 2, n_har);
    ggml_tensor * style = input("inp_style", GGML_TYPE_F32, n_style, 1);
    style = ggml_reshape_1d(ctx0, style, n_style);
    asr = ggml_cont(ctx0, ggml_transpose(ctx0, asr));
    har = ggml_cont(ctx0, ggml_transpose(ctx0, har));

    f0 = conv(p + "F0_conv", f0, 2, 1, 1);
    en = conv(p + "N_conv",  en, 2, 1, 1);

    ggml_tensor * x = ggml_concat(ctx0, ggml_concat(ctx0, asr, f0, 1), en, 1);
    x = adain_resblk(p + "encode", x, style, false);

    ggml_tensor * asr_res = conv(p + "asr_res.0", asr, 1, 0, 1);
    for (int i = 0; has_tensor(p + "decode." + std::to_string(i) + ".conv1.weight"); i++) {
        const std::string q = p + "decode." + std::to_string(i);
        x = ggml_concat(ctx0, ggml_concat(ctx0, ggml_concat(ctx0, x, asr_res, 1), f0, 1), en, 1);
        x = adain_resblk(q, x, style, has_tensor(q + ".pool.weight"));
    }

    const std::string g = p + "generator.";
    int n_ups = 0;
    while (has_tensor(g + "ups." + std::to_string(n_ups) + ".weight")) {
        n_ups++;
    }
    for (int i = 0; i < n_ups; i++) {
        const std::string is = std::to_string(i);
        ggml_tensor * up_w = tensor(g + "ups." + is + ".weight");
        ggml_tensor * up_b = tensor(g + "ups." + is + ".bias");
        // the transposed conv has kernel 2 * stride and torch padding stride / 2
        const int stride = (int) up_w->ne[0] / 2;
        const int trim   = stride / 2;

        x = ggml_leaky_relu(ctx0, x, 0.1f, false);

        // the source STFT is brought to the rate of this stage
        ggml_tensor * src_w = tensor(g + "noise_convs." + is + ".weight");
        const int src_k = (int) src_w->ne[0];
        const int src_s = src_k > 1 ? src_k / 2 : 1;
        ggml_tensor * src = conv(g + "noise_convs." + is, har, src_s, src_k > 1 ? (src_s + 1) / 2 : 0, 1);
        src = gen_resblock(g + "noise_res." + is, src, style);

        x = ggml_conv_transpose_1d(ctx0, up_w, x, stride, 0, 1);
        x = ggml_cont(ctx0, ggml_view_2d(ctx0, x, x->ne[0] - 2 * trim, x->ne[1], x->nb[1], (size_t) trim * ggml_element_size(x)));
        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, up_b, 1, up_b->ne[0]));
        if (i == n_ups - 1) {
            x = ggml_pad_reflect_1d(ctx0, x, 1, 0);
        }
        x = ggml_add(ctx0, x, src);

        ggml_tensor * acc = nullptr;
        for (int j = 0; j < 3; j++) {
            ggml_tensor * r = gen_resblock(g + "resblocks." + std::to_string(i * 3 + j), x, style);
            acc = acc ? ggml_add(ctx0, acc, r) : r;
        }
        x = ggml_scale(ctx0, acc, 1.0f / 3.0f);
    }
    x = ggml_leaky_relu(ctx0, x, 0.01f, false);
    x = conv(g + "conv_post", x, 1, 3, 1);
    return output("out_spec", ggml_cont(ctx0, ggml_transpose(ctx0, x)));
}

ggml_cgraph * mtmd_graph_kokoro::build() {
    switch (stage) {
        case STAGE_TEXT: return build_text();
        case STAGE_F0N:  return build_f0n();
        case STAGE_DEC:  return build_dec();
    }
    GGML_ABORT("invalid kokoro stage");
}

//
// host side
//

static std::vector<float> kokoro_read(const mtmd_model & model, const std::string & name, size_t n_expect) {
    std::vector<float> v(n_expect);
    if (model.cbx_read_tensor(name.c_str(), v.data(), v.size()) != n_expect) {
        throw std::runtime_error("kokoro: unexpected tensor " + name);
    }
    return v;
}

static inline float kokoro_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// y = W x, W row-major [n_out, n_in]
static void kokoro_matvec(const float * w, const float * x, int n_out, int n_in, float * y) {
    for (int o = 0; o < n_out; o++) {
        const float * row = w + (size_t) o * n_in;
        float acc[8] = {0.0f};
        int i = 0;
        for (; i + 8 <= n_in; i += 8) {
            for (int k = 0; k < 8; k++) {
                acc[k] += row[i + k] * x[i + k];
            }
        }
        float sum = 0.0f;
        for (; i < n_in; i++) {
            sum += row[i] * x[i];
        }
        for (int k = 0; k < 8; k++) {
            sum += acc[k];
        }
        y[o] = sum;
    }
}

// bidirectional single-layer LSTM (torch layout, gates i f g o): x [n_t, n_in] -> y [n_t, 2 * n_h]
static void kokoro_lstm(const mtmd_model & model, const std::string & name, const std::vector<float> & x, int n_t, int n_in,
                        std::vector<float> & y) {
    const int n_h = KOKORO_N_HIDDEN / 2;
    y.assign((size_t) n_t * 2 * n_h, 0.0f);
    std::vector<float> gi(4 * n_h), gh(4 * n_h), h(n_h), c(n_h);
    for (int dir = 0; dir < 2; dir++) {
        const std::string sfx = dir == 0 ? "" : "_reverse";
        const auto w_ih = kokoro_read(model, name + ".weight_ih_l0" + sfx, (size_t) 4 * n_h * n_in);
        const auto w_hh = kokoro_read(model, name + ".weight_hh_l0" + sfx, (size_t) 4 * n_h * n_h);
        const auto bias = kokoro_read(model, name + ".bias_l0" + sfx, (size_t) 4 * n_h);
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        for (int s = 0; s < n_t; s++) {
            const int t = dir == 0 ? s : n_t - 1 - s;
            kokoro_matvec(w_ih.data(), x.data() + (size_t) t * n_in, 4 * n_h, n_in, gi.data());
            kokoro_matvec(w_hh.data(), h.data(), 4 * n_h, n_h, gh.data());
            float * out = y.data() + (size_t) t * 2 * n_h + (size_t) dir * n_h;
            for (int j = 0; j < n_h; j++) {
                const float ig = kokoro_sigmoid(gi[j] + gh[j] + bias[j]);
                const float fg = kokoro_sigmoid(gi[n_h + j] + gh[n_h + j] + bias[n_h + j]);
                const float gg = tanhf(gi[2 * n_h + j] + gh[2 * n_h + j] + bias[2 * n_h + j]);
                const float og = kokoro_sigmoid(gi[3 * n_h + j] + gh[3 * n_h + j] + bias[3 * n_h + j]);
                c[j] = fg * c[j] + ig * gg;
                h[j] = og * tanhf(c[j]);
                out[j] = h[j];
            }
        }
    }
}

// rows of x repeated by their duration: [n_tok, n_ch] -> [n_frames, n_ch]
static std::vector<float> kokoro_align(const std::vector<float> & x, const std::vector<int32_t> & dur, int n_ch) {
    std::vector<float> y;
    for (size_t t = 0; t < dur.size(); t++) {
        for (int32_t r = 0; r < dur[t]; r++) {
            y.insert(y.end(), x.begin() + t * n_ch, x.begin() + (t + 1) * n_ch);
        }
    }
    return y;
}

int mtmd_graph_kokoro::prosody(const mtmd_model & model, const float * bert, int n_embd, const float * text_cnn, int n_tok,
                               const float * style, float speed, std::vector<float> & shared, std::vector<float> & asr) {
    const int n_h   = KOKORO_N_HIDDEN;
    const int n_cat = n_h + n_style;

    // bert_encoder, then each row is followed by the style
    const auto enc_w = kokoro_read(model, "a.gen.bert_enc.weight", (size_t) n_h * n_embd);
    const auto enc_b = kokoro_read(model, "a.gen.bert_enc.bias", n_h);
    std::vector<float> x((size_t) n_tok * n_cat);
    for (int t = 0; t < n_tok; t++) {
        float * row = x.data() + (size_t) t * n_cat;
        kokoro_matvec(enc_w.data(), bert + (size_t) t * n_embd, n_h, n_embd, row);
        for (int j = 0; j < n_h; j++) {
            row[j] += enc_b[j];
        }
        memcpy(row + n_h, style, n_style * sizeof(float));
    }

    // DurationEncoder: LSTM, AdaLayerNorm, style again
    std::vector<float> y;
    std::vector<float> gamma_beta(2 * n_h);
    for (int il = 0; model.cbx_tensors.count(mtmd_format("a.gen.pred.text_encoder.lstms.%d.weight_ih_l0", 2 * il)); il++) {
        kokoro_lstm(model, mtmd_format("a.gen.pred.text_encoder.lstms.%d", 2 * il), x, n_tok, n_cat, y);
        const std::string norm = mtmd_format("a.gen.pred.text_encoder.lstms.%d.fc", 2 * il + 1);
        const auto fc_w = kokoro_read(model, norm + ".weight", (size_t) 2 * n_h * n_style);
        const auto fc_b = kokoro_read(model, norm + ".bias", (size_t) 2 * n_h);
        kokoro_matvec(fc_w.data(), style, 2 * n_h, n_style, gamma_beta.data());
        for (int t = 0; t < n_tok; t++) {
            const float * src = y.data() + (size_t) t * n_h;
            float * dst = x.data() + (size_t) t * n_cat;
            double mean = 0.0, var = 0.0;
            for (int j = 0; j < n_h; j++) {
                mean += src[j];
            }
            mean /= n_h;
            for (int j = 0; j < n_h; j++) {
                var += (src[j] - mean) * (src[j] - mean);
            }
            const double scale = 1.0 / sqrt(var / n_h + 1e-5);
            for (int j = 0; j < n_h; j++) {
                const float v = (float) ((src[j] - mean) * scale);
                dst[j] = (1.0f + gamma_beta[j] + fc_b[j]) * v + gamma_beta[n_h + j] + fc_b[n_h + j];
            }
        }
    }

    // durations: frames per token
    kokoro_lstm(model, "a.gen.pred.lstm", x, n_tok, n_cat, y);
    const ggml_tensor * proj = model.cbx_tensors.at("a.gen.pred.duration_proj.linear_layer.weight");
    const int n_bins = (int) proj->ne[1];
    const auto proj_w = kokoro_read(model, "a.gen.pred.duration_proj.linear_layer.weight", (size_t) n_bins * n_h);
    const auto proj_b = kokoro_read(model, "a.gen.pred.duration_proj.linear_layer.bias", n_bins);
    std::vector<int32_t> dur(n_tok);
    std::vector<float> logits(n_bins);
    int n_frames = 0;
    for (int t = 0; t < n_tok; t++) {
        kokoro_matvec(proj_w.data(), y.data() + (size_t) t * n_h, n_bins, n_h, logits.data());
        float sum = 0.0f;
        for (int j = 0; j < n_bins; j++) {
            sum += kokoro_sigmoid(logits[j] + proj_b[j]);
        }
        // round half to even, like torch.round
        dur[t] = std::max(1, (int) nearbyintf(sum / speed));
        n_frames += dur[t];
    }

    kokoro_lstm(model, "a.gen.pred.shared", kokoro_align(x, dur, n_cat), n_frames, n_cat, shared);

    // text encoder: its LSTM runs on the CNN output
    kokoro_lstm(model, "a.gen.text.lstm", std::vector<float>(text_cnn, text_cnn + (size_t) n_tok * n_h), n_tok, n_h, y);
    asr = kokoro_align(y, dur, n_h);
    return n_frames;
}

// hn-nsf source: f0 [2 * n_frames] -> harmonic source, then its STFT as [n_har, 22] rows of magnitudes and phases
int mtmd_graph_kokoro::har_source(const mtmd_model & model, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & har) {
    const int n_f0 = (int) f0.size();
    const int64_t n_wav = (int64_t) n_f0 * KOKORO_UPS;
    const auto lin_w = kokoro_read(model, "a.gen.dec.generator.m_source.l_linear.weight", KOKORO_N_HARM);
    const auto lin_b = kokoro_read(model, "a.gen.dec.generator.m_source.l_linear.bias", 1);

    // SineGen: the phase adds up at the F0 rate, then it is interpolated to the sample rate
    // the reference also adds a random start phase, but its own downsampling drops it
    std::vector<float> phase((size_t) n_f0 * KOKORO_N_HARM);
    for (int h = 0; h < KOKORO_N_HARM; h++) {
        float cum = 0.0f;
        for (int i = 0; i < n_f0; i++) {
            float rad = f0[i] * (float) (h + 1) / KOKORO_SR;
            rad -= floorf(rad);
            cum += rad;
            phase[(size_t) i * KOKORO_N_HARM + h] = cum * 2.0f * (float) M_PI * KOKORO_UPS;
        }
    }
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> src((size_t) n_wav);
    for (int64_t t = 0; t < n_wav; t++) {
        const float pos  = std::max(0.0f, ((float) t + 0.5f) / KOKORO_UPS - 0.5f);
        const int   i0   = std::min((int) pos, n_f0 - 1);
        const int   i1   = std::min(i0 + 1, n_f0 - 1);
        const float frac = pos - (float) i0;
        const float uv   = f0[t / KOKORO_UPS] > 10.0f ? 1.0f : 0.0f; // voiced threshold
        const float namp = uv * 0.003f + (1.0f - uv) * 0.1f / 3.0f;
        float merged = lin_b[0];
        for (int h = 0; h < KOKORO_N_HARM; h++) {
            const float ph = (1.0f - frac) * phase[(size_t) i0 * KOKORO_N_HARM + h] + frac * phase[(size_t) i1 * KOKORO_N_HARM + h];
            merged += lin_w[h] * (0.1f * sinf(ph) * uv + namp * gauss(rng));
        }
        src[(size_t) t] = tanhf(merged);
    }

    // STFT: periodic hann window, centered with reflect padding
    const int n_bins = KOKORO_N_FFT / 2 + 1;
    const int n_har  = (int) (n_wav / KOKORO_HOP) + 1;
    float win[KOKORO_N_FFT], tab_cos[KOKORO_N_FFT], tab_sin[KOKORO_N_FFT];
    for (int i = 0; i < KOKORO_N_FFT; i++) {
        win[i]     = 0.5f - 0.5f * cosf(2.0f * (float) M_PI * i / KOKORO_N_FFT);
        tab_cos[i] = cosf(2.0f * (float) M_PI * i / KOKORO_N_FFT);
        tab_sin[i] = sinf(2.0f * (float) M_PI * i / KOKORO_N_FFT);
    }
    har.resize((size_t) n_har * 2 * n_bins);
    for (int fr = 0; fr < n_har; fr++) {
        float frame[KOKORO_N_FFT];
        for (int i = 0; i < KOKORO_N_FFT; i++) {
            int64_t idx = (int64_t) fr * KOKORO_HOP - KOKORO_N_FFT / 2 + i;
            if (idx < 0) {
                idx = -idx;
            }
            if (idx >= n_wav) {
                idx = 2 * (n_wav - 1) - idx;
            }
            frame[i] = src[(size_t) idx] * win[i];
        }
        float * out = har.data() + (size_t) fr * 2 * n_bins;
        for (int k = 0; k < n_bins; k++) {
            float re = 0.0f, im = 0.0f;
            for (int i = 0; i < KOKORO_N_FFT; i++) {
                const int a = (k * i) % KOKORO_N_FFT;
                re += frame[i] * tab_cos[a];
                im -= frame[i] * tab_sin[a];
            }
            out[k]          = sqrtf(re * re + im * im);
            out[n_bins + k] = atan2f(im, re);
        }
    }
    return n_har;
}

// conv_post rows [n_frames, 22] -> waveform: magnitude exp(x), phase sin(x), hann overlap-add
void mtmd_graph_kokoro::istft(const std::vector<float> & spec, std::vector<float> & audio) {
    const int n_bins   = KOKORO_N_FFT / 2 + 1;
    const int n_frames = (int) (spec.size() / (2 * n_bins));
    const int64_t n_out = (int64_t) (n_frames - 1) * KOKORO_HOP;
    float win[KOKORO_N_FFT], tab_cos[KOKORO_N_FFT], tab_sin[KOKORO_N_FFT];
    for (int i = 0; i < KOKORO_N_FFT; i++) {
        win[i]     = 0.5f - 0.5f * cosf(2.0f * (float) M_PI * i / KOKORO_N_FFT);
        tab_cos[i] = cosf(2.0f * (float) M_PI * i / KOKORO_N_FFT);
        tab_sin[i] = sinf(2.0f * (float) M_PI * i / KOKORO_N_FFT);
    }
    std::vector<float> acc((size_t) n_out + KOKORO_N_FFT, 0.0f);
    std::vector<float> wsum((size_t) n_out + KOKORO_N_FFT, 0.0f);
    for (int fr = 0; fr < n_frames; fr++) {
        const float * row = spec.data() + (size_t) fr * 2 * n_bins;
        float re[KOKORO_N_FFT / 2 + 1], im[KOKORO_N_FFT / 2 + 1];
        for (int k = 0; k < n_bins; k++) {
            const float mag = expf(row[k]);
            const float ph  = sinf(row[n_bins + k]);
            re[k] = mag * cosf(ph);
            im[k] = mag * sinf(ph);
        }
        for (int i = 0; i < KOKORO_N_FFT; i++) {
            float v = 0.0f;
            for (int k = 0; k < n_bins; k++) {
                const int   a = (k * i) % KOKORO_N_FFT;
                const float w = (k == 0 || k == KOKORO_N_FFT / 2) ? 1.0f : 2.0f;
                v += w * (re[k] * tab_cos[a] - im[k] * tab_sin[a]);
            }
            const size_t o = (size_t) fr * KOKORO_HOP + i;
            acc[o]  += v / KOKORO_N_FFT * win[i];
            wsum[o] += win[i] * win[i];
        }
    }
    audio.resize((size_t) std::max<int64_t>(n_out, 0));
    for (size_t i = 0; i < audio.size(); i++) {
        const size_t j = i + KOKORO_N_FFT / 2; // drop the centering pad
        audio[i] = wsum[j] > 1e-11f ? acc[j] / wsum[j] : 0.0f;
    }
}
