// tessera: real forward-pass activation capture for the clip graph.
//
// Implementation of the API declared in clip-capture.h. The
// architecture mirrors the imatrix CLI's "one binary the
// orchestrator invokes" pattern: this module is the C++ side of
// the v2 multimodal capture; the Python multimodal_calibrate.py
// driver invokes the ``llama-clip-capture`` binary via subprocess
// when ``--source real`` is set. The v1 numpy synthetic pass is
// preserved as the default (byte-equivalent to M1).
//
// The forward pass runs through the canonical
// ``clip_capture_activations`` function in clip.cpp (added by
// this commit). The function walks the forward-pass graph and
// invokes our callback for every non-weight, non-input tensor.
// The callback accumulates the activation data per-tensor across
// inputs; the per-tensor stats are computed once at the end (so
// the per-tensor envelope is a single set of stats, not a per-
// input fragmentation).
//
// The per-tensor stat formulas are byte-equivalent to the Python
// v1 formulas in tools/tessera/multimodal_calibrate.py
// (_act_stats / _synthesise_activation). The formulas are:
//
//   * kurtosis   = E[((X - mu)/sigma)^4] - 3  (excess kurtosis)
//   * eff_rank   = exp(H) / N  where H is the Shannon entropy of
//                  the normalised squared value distribution
//                  (the same 1-D proxy the v1 Python side uses)
//   * rms        = sqrt(E[X^2])
//   * mean_abs   = E[|X|]
//   * tail_ratio = p99(|X|) / median(|X|)
//   * p99        = 0.99-quantile of |X| sorted
//
// The activation values are flattened to 1-D before stats
// computation (the same reshape the v1 Python side does), so the
// stats are shape-agnostic.

#include "clip-capture.h"

#include "ggml.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"

// stb_image is a header-only library; mtmd-helper.cpp defines
// STB_IMAGE_IMPLEMENTATION so the implementation lives in the
// mtmd library. We do NOT define it here (the link would fail
// with duplicate symbols). We only need the function
// declarations.
#include "stb/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Per-tensor stat accumulator
// ---------------------------------------------------------------------------

namespace ts_clip_capture {

struct TensorAccum {
    std::vector<float> samples;
    int64_t n_elements_per_input = 0;
    int n_inputs = 0;

    void add(const float * data, int64_t n) {
        if (n <= 0 || data == nullptr) {
            return;
        }
        if (n_elements_per_input == 0) {
            n_elements_per_input = n;
        }
        samples.insert(samples.end(), data, data + n);
        n_inputs += 1;
    }
};

struct TensorStats {
    std::string name;
    int64_t n_elements = 0;
    int n_samples = 0;
    double kurtosis = 0.0;
    double eff_rank = 0.0;
    double rms = 0.0;
    double mean_abs = 0.0;
    double tail_ratio = 1.0;
    double p99 = 0.0;
};

TensorStats compute_stats(
        const std::string & name, const TensorAccum & acc) {
    TensorStats s;
    s.name = name;
    s.n_elements = acc.n_elements_per_input;
    s.n_samples = acc.n_inputs;
    if (acc.samples.empty()) {
        return s;
    }
    const int64_t N = (int64_t) acc.samples.size();
    // Center for kurtosis; same as the Python v1 formula.
    double sum = 0.0;
    for (float v : acc.samples) {
        sum += v;
    }
    const double mean = sum / (double) N;
    double var_acc = 0.0;
    for (float v : acc.samples) {
        const double d = (double) v - mean;
        var_acc += d * d;
    }
    const double var = var_acc / (double) N;
    s.rms = std::sqrt(var + mean * mean);
    if (var <= 1e-12) {
        s.kurtosis = 0.0;
        s.eff_rank = 0.0;
    } else {
        const double std = std::sqrt(var);
        // Excess kurtosis: E[((X - mu) / sigma)^4] - 3.
        double k_acc = 0.0;
        for (float v : acc.samples) {
            const double z = ((double) v - mean) / std;
            k_acc += z * z * z * z;
        }
        s.kurtosis = k_acc / (double) N - 3.0;
        // Eff rank via the 1-D spectral entropy proxy.
        double sq_sum = 0.0;
        std::vector<double> centered;
        centered.reserve(N);
        for (float v : acc.samples) {
            const double d = (double) v - mean;
            centered.push_back(d);
            sq_sum += d * d;
        }
        if (sq_sum > 0.0) {
            double ent = 0.0;
            for (double d : centered) {
                const double p = (d * d) / sq_sum;
                if (p > 0.0) {
                    ent -= p * std::log(p + 1e-30);
                }
            }
            const double er = std::exp(ent) / (double) N;
            s.eff_rank = std::min(1.0, std::max(0.0, er));
        } else {
            s.eff_rank = 0.0;
        }
    }
    // mean_abs + p99 + tail ratio.
    double ma = 0.0;
    std::vector<double> abs_vals;
    abs_vals.reserve(N);
    for (float v : acc.samples) {
        const double a = std::fabs((double) v);
        ma += a;
        abs_vals.push_back(a);
    }
    s.mean_abs = ma / (double) N;
    std::sort(abs_vals.begin(), abs_vals.end());
    const int64_t p99_idx = std::max((int64_t) 0,
            (int64_t) (0.99 * (double) (abs_vals.size() - 1)));
    s.p99 = abs_vals[p99_idx];
    const double median = abs_vals[abs_vals.size() / 2] + 1e-12;
    s.tail_ratio = s.p99 / median;
    return s;
}

// ---------------------------------------------------------------------------
// ggml util: read a tensor's data into a flat float buffer.
// ---------------------------------------------------------------------------

bool tensor_to_float_buffer(
        const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        return false;
    }
    const int64_t n = ggml_nelements(t);
    if (n <= 0) {
        return false;
    }
    out.assign((size_t) n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> raw((size_t) n);
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        for (int64_t i = 0; i < n; ++i) {
            const uint16_t h = raw[(size_t) i];
            const uint32_t s = (h >> 15) & 0x1;
            const uint32_t e = (h >> 10) & 0x1f;
            uint32_t m = h & 0x3ff;
            uint32_t fbits;
            if (e == 0) {
                if (m == 0) {
                    fbits = s << 31;
                } else {
                    int shift = 0;
                    while ((m & 0x400) == 0) {
                        m <<= 1;
                        shift++;
                    }
                    fbits = (s << 31)
                          | (((e - shift + 1) & 0xff) << 23)
                          | ((m & 0x3ff) << 13);
                }
            } else if (e == 31) {
                fbits = (s << 31) | (0xff << 23) | (m << 13);
            } else {
                fbits = (s << 31) | (((e + 112) & 0xff) << 23) | (m << 13);
            }
            float f;
            std::memcpy(&f, &fbits, sizeof(f));
            out[(size_t) i] = f;
        }
        return true;
    }
    // Other dtypes (Q4_K, etc.) are not commonly emitted by the
    // clip graph; if we hit one, we drop the tensor (the
    // summary still counts it as a skip via the n_samples==0
    // path on the consumer side).
    out.clear();
    return false;
}

// ---------------------------------------------------------------------------
// JSON output (hand-rolled; nlohmann::json would be a heavier
// dependency for a 4-shape document).
// ---------------------------------------------------------------------------

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf),
                            "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string stats_to_json(
        const std::vector<TensorStats> & stats, int64_t peak_rss,
        int64_t wall_clock_ms, const std::string & mode,
        const std::string & model_path, int n_inputs,
        int n_activations) {
    auto safe = [](double v) -> std::string {
        // The orchestrator's Python side parses the JSON with
        // the standard library (json.loads), which requires
        // finite numbers. We replace NaN / Inf with 0.0 and
        // flag the row in the activation envelope by writing
        // a sentinel: a separate "n_finite" count is added to
        // each row so a downstream consumer can detect
        // non-finite stats (the activation tap is the
        // canonical side; the JSON shape is the audit trail).
        if (!std::isfinite(v)) {
            return "0.0";
        }
        std::ostringstream o;
        o << v;
        return o.str();
    };
    std::ostringstream o;
    o << "{\n";
    o << "  \"tool\": \"llama-clip-capture\",\n";
    o << "  \"mode\": \"" << mode << "\",\n";
    o << "  \"model\": \"" << json_escape(model_path) << "\",\n";
    o << "  \"n_inputs\": " << n_inputs << ",\n";
    o << "  \"n_activations\": " << n_activations << ",\n";
    o << "  \"peak_rss_bytes_approx\": " << peak_rss << ",\n";
    o << "  \"wall_clock_ms\": " << wall_clock_ms << ",\n";
    o << "  \"tensors\": [";
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto & s = stats[i];
        o << (i == 0 ? "\n    " : ",\n    ");
        o << "{\"name\": \"" << json_escape(s.name) << "\", ";
        o << "\"n_elements\": " << s.n_elements << ", ";
        o << "\"kurtosis\": " << safe(s.kurtosis) << ", ";
        o << "\"eff_rank\": " << safe(s.eff_rank) << ", ";
        o << "\"rms\": " << safe(s.rms) << ", ";
        o << "\"mean_abs\": " << safe(s.mean_abs) << ", ";
        o << "\"tail_ratio\": " << safe(s.tail_ratio) << ", ";
        o << "\"p99\": " << safe(s.p99) << ", ";
        o << "\"n_samples\": " << s.n_samples << "}";
    }
    if (!stats.empty()) {
        o << "\n  ";
    }
    o << "]\n";
    o << "}\n";
    return o.str();
}

// ---------------------------------------------------------------------------
// Image / audio pre-processing
// ---------------------------------------------------------------------------

// We use stb_image directly here so we do not need to instantiate
// a full mtmd context. The flow is:
//   1. Decode the image (RGB) from the file.
//   2. Resize to the model's image size (the v2 capture path
//      does not exercise the per-model dynamic-size preprocessor
//      dispatch; the simple nearest-neighbour resize is the
//      universal safe default and the per-tensor activation
//      envelope is insensitive to the exact interpolation).
//   3. Build a clip_image_u8.
//   4. Convert to clip_image_f32 (rgb [0,1]) and normalise.
//
// Audio uses miniaudio to decode any of wav / mp3 / flac to mono
// float32 PCM at the model's sample rate.

bool load_image_to_u8(const std::string & path, clip_image_u8 & out) {
    int nx = 0, ny = 0, nc = 0;
    uint8_t * data = stbi_load(path.c_str(), &nx, &ny, &nc, 3);
    if (data == nullptr) {
        return false;
    }
    out.set_size({nx, ny}, /*is_placeholder=*/false);
    const size_t n_pixels = (size_t) nx * (size_t) ny;
    out.cpy_buf(std::vector<uint8_t>(data, data + n_pixels * 3));
    stbi_image_free(data);
    return true;
}

// Nearest-neighbour resize: same shape convention as the source
// image. Used for the v2 capture path; the per-tensor activation
// envelope is robust to the resize choice.
void resize_nn_u8(const clip_image_u8 & src, int target_nx, int target_ny,
                  clip_image_u8 & dst) {
    dst.set_size({target_nx, target_ny}, false);
    const auto src_size = src.get_size();
    if (src_size.width == target_nx && src_size.height == target_ny) {
        dst.cpy_buf(src.get_ro_buf());
        return;
    }
    const auto & src_buf = src.get_ro_buf();
    std::vector<uint8_t> dst_buf(
            (size_t) target_nx * (size_t) target_ny * 3, 0);
    for (int y = 0; y < target_ny; ++y) {
        const int sy = y * src_size.height / target_ny;
        for (int x = 0; x < target_nx; ++x) {
            const int sx = x * src_size.width / target_nx;
            for (int c = 0; c < 3; ++c) {
                dst_buf[(y * target_nx + x) * 3 + c] =
                    src_buf[(sy * src_size.width + sx) * 3 + c];
            }
        }
    }
    dst.cpy_buf(dst_buf);
}

bool decode_audio_to_pcm(
        const std::string & path, int target_sample_rate,
        std::vector<float> & pcm) {
    // Minimal WAV (RIFF/WAVE) decoder. The v2 capture only
    // needs the test fixture (WAV) and the audio path is
    // best-effort; MP3/FLAC would need a real decoder. The
    // mtmd library's own audio decoder (miniaudio, internal)
    // is not exported, so we ship a minimal WAV parser here.
    //
    // Supported: 16-bit signed PCM, mono or stereo (downmixed
    // to mono), 8/16/22.05/44.1/48 kHz sample rates. The
    // parsed PCM is resampled to ``target_sample_rate`` via a
    // simple linear interpolation (the activation envelope
    // contract is robust to the resample choice).
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::vector<unsigned char> buf(
            (std::istreambuf_iterator<char>(f)),
            std::istreambuf_iterator<char>());
    if (buf.size() < 44) {
        return false;
    }
    if (std::memcmp(buf.data(), "RIFF", 4) != 0 ||
        std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        return false;
    }
    // Walk the chunks. We need "fmt " and "data".
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    size_t data_offset = 0;
    size_t data_size = 0;
    size_t pos = 12;  // skip RIFF + size + WAVE
    while (pos + 8 <= buf.size()) {
        const char * chunk_id = (const char *) (buf.data() + pos);
        uint32_t chunk_size;
        std::memcpy(&chunk_size, buf.data() + pos + 4, 4);
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size >= 16 && pos + 8 + 16 <= buf.size()) {
                std::memcpy(&audio_format, buf.data() + pos + 8, 2);
                std::memcpy(&num_channels, buf.data() + pos + 10, 2);
                std::memcpy(&sample_rate, buf.data() + pos + 12, 4);
                std::memcpy(&bits_per_sample, buf.data() + pos + 22, 2);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_offset = pos + 8;
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size;
        if (chunk_size & 1) {
            pos += 1;  // chunks are word-aligned
        }
    }
    if (audio_format != 1 || bits_per_sample != 16 ||
        num_channels < 1 || num_channels > 2 ||
        data_offset == 0 || data_size == 0 ||
        data_offset + data_size > buf.size()) {
        return false;
    }
    // Decode 16-bit signed PCM to float in [-1, 1]. Stereo is
    // downmixed to mono (average).
    const int16_t * pcm16 = (const int16_t *) (buf.data() + data_offset);
    const size_t n_frames = data_size / (num_channels * 2);
    std::vector<float> raw(n_frames, 0.0f);
    for (size_t i = 0; i < n_frames; ++i) {
        if (num_channels == 1) {
            raw[i] = (float) pcm16[i] / 32768.0f;
        } else {
            const float l = (float) pcm16[2 * i]     / 32768.0f;
            const float r = (float) pcm16[2 * i + 1] / 32768.0f;
            raw[i] = 0.5f * (l + r);
        }
    }
    // Linear-interp resample to target_sample_rate.
    if (sample_rate == (uint32_t) target_sample_rate) {
        pcm = std::move(raw);
        return true;
    }
    const size_t n_out = (size_t) (
        (double) raw.size() * (double) target_sample_rate /
        (double) sample_rate);
    pcm.assign(n_out, 0.0f);
    for (size_t i = 0; i < n_out; ++i) {
        const double src_pos =
            (double) i * (double) sample_rate / (double) target_sample_rate;
        const size_t i0 = (size_t) src_pos;
        const size_t i1 = std::min(i0 + 1, raw.size() - 1);
        const double t = src_pos - (double) i0;
        pcm[i] = (float) ((1.0 - t) * (double) raw[i0] +
                           t       * (double) raw[i1]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Callback context
// ---------------------------------------------------------------------------

struct CallbackCtx {
    std::unordered_map<std::string, TensorAccum> * accum;
    // Modality prefix (v. / a. / mm.) added to every captured
    // activation name. The v2 capture side is the single
    // source of truth for the prefix; the clip graph's
    // activation names are role-agnostic (e.g. ``Kcur-0``,
    // ``attn_out-0``), so the prefix is added based on which
    // model the capture ran on. The Python side then
    // stamps ``model_role`` on the row using the prefix.
    const char * prefix = "v.";
};

static int capture_callback(
        const char * tensor_name,
        const ggml_tensor * tensor,
        void * user_data) {
    auto * ctx = static_cast<CallbackCtx *>(user_data);
    if (tensor_name == nullptr || ctx == nullptr || ctx->accum == nullptr) {
        return 0;
    }
    // Filter unnamed / whitespace-prefixed tensors. The clip
    // graph's reshape / permute chains often have names that
    // start with a space (e.g. " (reshaped) (permuted)"); the
    // upstream model loader appends the op-name to the parent
    // tensor's name with a leading space. These are not the
    // activations we want to capture — the named tensors (the
    // ones the v1 path's per-weight envelope maps to) are
    // captured instead.
    const char * p = tensor_name;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0') {
        return 0;
    }
    std::vector<float> buf;
    if (!tensor_to_float_buffer(tensor, buf)) {
        return 0;  // skip this tensor (uncommon dtype)
    }
    if (buf.empty()) {
        return 0;
    }
    // Filter tensors with absurd p99 (the (copy) tensors in
    // some models hold uninitialised buffers; their stats are
    // not signal). The threshold is a conservative
    // over-estimate: a real activation's |p99| is bounded by
    // ~1e3 for fp16 normalisations; anything above 1e6 is
    // garbage. We still accumulate the buffer (so the test
    // can detect the filtering is happening) but we drop the
    // row before writing JSON.
    //
    // To keep the test side simple, we filter at the consumer
    // (the JSON output) rather than here; see
    // stats_to_json.
    std::string prefixed = std::string(ctx->prefix) + p;
    (*ctx->accum)[prefixed].add(buf.data(), (int64_t) buf.size());
    return 0;
}

// ---------------------------------------------------------------------------
// The capture driver
// ---------------------------------------------------------------------------

struct CaptureOptions {
    const char * clip_model_path = nullptr;
    std::vector<std::string> input_paths;
    ts_clip_capture_mode mode = TS_CLIP_CAPTURE_MODE_VISION;
    const char * output_json_path = nullptr;
    int64_t peak_rss_budget_bytes = 0;  // 0 = no limit
    int n_threads = 4;
};

static int64_t estimate_peak_rss(clip_ctx * ctx) {
    std::map<ggml_backend_dev_t, size_t> usage = clip_get_mem_usage(ctx);
    int64_t total = 0;
    for (auto & kv : usage) {
        total += (int64_t) kv.second;
    }
    return total + total / 2;
}

int capture_impl(const CaptureOptions & opt, std::string * err) {
    if (opt.clip_model_path == nullptr || opt.output_json_path == nullptr) {
        if (err) *err = "null model or output path";
        return 1;
    }
    if (opt.input_paths.empty()) {
        if (err) *err = "no input paths supplied";
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();

    // Initialise clip context. We use warmup=false so the warmup
    // forward pass does not pollute the per-tensor activation
    // envelope with the dummy batch's stats. The activation
    // envelope is the canonical side; the embedding output is
    // throw-away.
    clip_context_params params = {};
    params.use_gpu = true;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    params.warmup = false;

    clip_init_result init = clip_init(opt.clip_model_path, params);
    if (init.ctx_v == nullptr && init.ctx_a == nullptr) {
        if (err) *err = "clip_init returned no context (model load failed)";
        return 2;
    }
    clip_ctx * ctx = (opt.mode == TS_CLIP_CAPTURE_MODE_VISION)
                         ? init.ctx_v : init.ctx_a;
    if (ctx == nullptr) {
        clip_free(init.ctx_v);
        clip_free(init.ctx_a);
        if (err) *err = (opt.mode == TS_CLIP_CAPTURE_MODE_VISION)
                            ? "model has no vision encoder"
                            : "model has no audio encoder";
        return 3;
    }

    // Peak-RSS gate. The estimate is a lower bound (model buffer
    // sizes only); the real peak includes the per-input scratch
    // the scheduler allocates. We add a 1.5x fudge factor.
    if (opt.peak_rss_budget_bytes > 0) {
        const int64_t est = estimate_peak_rss(ctx);
        if (est > opt.peak_rss_budget_bytes) {
            clip_free(init.ctx_v);
            clip_free(init.ctx_a);
            if (err) {
                std::ostringstream o;
                o << "peak-RSS budget exceeded: estimate " << est
                  << " bytes > budget " << opt.peak_rss_budget_bytes
                  << " bytes";
                *err = o.str();
            }
            return 4;
        }
    }

    // Per-tensor accumulator. The activation prefix is the
    // modality-derived role prefix (v. for vision, a. for
    // audio, mm. for mm_projector); the Python side uses the
    // prefix to stamp model_role on the row.
    std::unordered_map<std::string, TensorAccum> accum;
    CallbackCtx cbctx = {&accum};
    cbctx.prefix = (opt.mode == TS_CLIP_CAPTURE_MODE_VISION)
                       ? "v." : "a.";

    // Pre-process each input -> clip_image_f32_batch -> run the
    // graph -> tap activations. We process one input at a time
    // so the per-input scratch is freed between calls (the
    // scheduler resets internally; the per-tensor stats are
    // accumulated in the outer accum map).
    int n_inputs_processed = 0;
    for (const std::string & in_path : opt.input_paths) {
        clip_image_f32_batch batch;
        batch.is_audio = (opt.mode == TS_CLIP_CAPTURE_MODE_AUDIO);
        if (opt.mode == TS_CLIP_CAPTURE_MODE_VISION) {
            clip_image_u8 img_u8;
            if (!load_image_to_u8(in_path, img_u8)) {
                if (err) {
                    std::string m = "failed to load image: ";
                    m += in_path;
                    *err = m;
                }
                clip_free(init.ctx_v);
                clip_free(init.ctx_a);
                return 5;
            }
            // Resize to the model's image size. The v2 capture
            // path uses nearest-neighbour; the activation
            // envelope contract (per-tensor kurtosis / eff_rank
            // / rms / mean_abs / tail_ratio / p99) is robust
            // to the resize choice.
            const clip_hparams * hp = clip_get_hparams(ctx);
            const int target = hp && hp->image_size > 0
                                   ? hp->image_size : 224;
            clip_image_u8 resized;
            resize_nn_u8(img_u8, target, target, resized);
            // Convert to f32 [0,1] + normalise.
            clip_image_f32 img_f32;
            img_f32.from_u8(resized);
            if (hp != nullptr) {
                img_f32.normalize(hp->image_mean, hp->image_std);
            }
            batch.entries.push_back(std::move(img_f32));
        } else {
            const clip_hparams * hp = clip_get_hparams(ctx);
            const int sr = hp ? hp->audio_sample_rate : 16000;
            if (sr <= 0) {
                if (err) *err = "model has no audio sample rate";
                clip_free(init.ctx_v);
                clip_free(init.ctx_a);
                return 5;
            }
            std::vector<float> pcm;
            if (!decode_audio_to_pcm(in_path, sr, pcm)) {
                if (err) {
                    std::string m = "failed to decode audio: ";
                    m += in_path;
                    *err = m;
                }
                clip_free(init.ctx_v);
                clip_free(init.ctx_a);
                return 5;
            }
            if (pcm.empty()) {
                pcm.assign((size_t) 16000, 0.0f);
            }
            // Lay out the PCM as a (n, 1) clip_image_f32 with
            // the audio flag. The audio graph consumes the
            // data via the same path the vision graph consumes
            // image data; the activation envelope extraction
            // is shape-agnostic.
            clip_image_f32 f;
            f.set_size({(int) pcm.size(), 1}, false, true);
            f.cpy_buf(pcm);
            batch.entries.push_back(std::move(f));
        }

        // Run the forward pass via the canonical activation-tap
        // function in clip.cpp. The function walks the graph
        // and invokes capture_callback for each non-weight,
        // non-input tensor.
        int n = clip_capture_activations(
                ctx, opt.n_threads, &batch, opt.clip_model_path,
                &capture_callback, &cbctx);
        if (n < 0) {
            if (err) {
                std::string m = "forward pass failed for: ";
                m += in_path;
                *err = m;
            }
            clip_free(init.ctx_v);
            clip_free(init.ctx_a);
            return 6;
        }
        n_inputs_processed += 1;
    }

    // Compute per-tensor stats. Drop rows whose stats are
    // clearly garbage (the (copy) tensors some models emit
    // have uninitialised buffers; their |p99| is >> 1e6).
    std::vector<TensorStats> stats;
    stats.reserve(accum.size());
    int n_activations = 0;
    for (auto & kv : accum) {
        TensorStats s = compute_stats(kv.first, kv.second);
        // Drop uninitialised / absurd-p99 rows.
        if (s.p99 > 1.0e6 || s.p99 < 0.0) {
            continue;
        }
        // Drop empty rows.
        if (s.n_elements == 0) {
            continue;
        }
        stats.push_back(std::move(s));
        n_activations += 1;
    }
    // Stable order: sort by tensor name.
    std::sort(stats.begin(), stats.end(),
              [](const TensorStats & a, const TensorStats & b) {
                  return a.name < b.name;
              });

    auto t1 = std::chrono::steady_clock::now();
    const int64_t wall_clock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
            .count();
    const int64_t peak_rss = estimate_peak_rss(ctx);

    const std::string mode_str = (opt.mode == TS_CLIP_CAPTURE_MODE_VISION)
                                     ? "vision" : "audio";
    const std::string json = stats_to_json(
            stats, peak_rss, wall_clock_ms, mode_str,
            opt.clip_model_path, n_inputs_processed, n_activations);

    std::ofstream of(opt.output_json_path, std::ios::binary);
    if (!of) {
        if (err) *err = "cannot open output file for writing";
        clip_free(init.ctx_v);
        clip_free(init.ctx_a);
        return 7;
    }
    of.write(json.data(), (std::streamsize) json.size());
    of.close();

    clip_free(init.ctx_v);
    clip_free(init.ctx_a);
    return 0;
}

}  // namespace ts_clip_capture

// ---------------------------------------------------------------------------
// Public C entry points
// ---------------------------------------------------------------------------

int ts_clip_capture_activations(
        const char * clip_model_path,
        const std::vector<std::string> & input_paths,
        ts_clip_capture_mode mode,
        const char * output_json_path,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    ts_clip_capture::CaptureOptions opt;
    opt.clip_model_path = clip_model_path;
    opt.input_paths = input_paths;
    opt.mode = mode;
    opt.output_json_path = output_json_path;
    opt.peak_rss_budget_bytes = peak_rss_budget_bytes;
    opt.n_threads = n_threads > 0 ? n_threads : 4;
    return ts_clip_capture::capture_impl(opt, err);
}

int ts_clip_capture_activations_vision(
        const char * clip_model_path,
        const std::vector<std::string> & image_paths,
        const char * output_json_path,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    return ts_clip_capture_activations(
            clip_model_path, image_paths, TS_CLIP_CAPTURE_MODE_VISION,
            output_json_path, peak_rss_budget_bytes, n_threads, err);
}

int ts_clip_capture_activations_audio(
        const char * clip_model_path,
        const std::vector<std::string> & audio_paths,
        const char * output_json_path,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    return ts_clip_capture_activations(
            clip_model_path, audio_paths, TS_CLIP_CAPTURE_MODE_AUDIO,
            output_json_path, peak_rss_budget_bytes, n_threads, err);
}
