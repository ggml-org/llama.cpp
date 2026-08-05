// tessera: real forward-pass activation capture for the clip graph.
//
// Implementation of the API declared in clip-capture.h. The
// architecture mirrors the imatrix CLI's "one binary the
// orchestrator invokes" pattern: this module is the C++ side of
// the multimodal capture; the Python multimodal_calibrate.py
// driver invokes the ``llama-clip-capture`` binary via subprocess.
//
// The forward pass runs through the canonical
// ``clip_capture_activations`` function in clip.cpp. The function
// walks the forward-pass graph and invokes our callback for every
// non-weight, non-input tensor. The callback accumulates the
// activation data per-tensor across inputs; the per-tensor stats
// are computed once at the end (so the per-tensor envelope is a
// single set of stats, not a per-input fragmentation).
//
// Batching: the input list is folded into a single
// ``clip_image_f32_batch`` (vision) or audio-preprocessor batch
// (audio) per forward call. The ``--batch-size`` CLI flag
// controls the chunk size: when the input list is larger than the
// batch size, the capture chunks the inputs into multiple forward
// calls and accumulates the per-tensor stats across all chunks.
// The output JSON reports the total ``n_inputs`` and
// ``n_chunks``.
//
// Audio decoding:
//   * On Apple platforms: AudioToolbox / ExtAudioFile (C API).
//     Decodes WAV, MP3, FLAC, AAC, ALAC natively to float32 mono
//     PCM at the model's expected sample rate.
//   * On non-Apple: miniaudio (header-only) for WAV / MP3 / FLAC.
//     On CI the test fixture is a synthesised WAV so the
//     non-Apple path is only used for the Linux / Windows smoke
//     test. The inline WAV parser is kept as a final fallback
//     for environments where miniaudio is not available.
//
// Dead-node handling: the clip graph contains layout views
// (``(view)``, ``(permuted)``, ``(cont)``, ``(reshaped)``,
// ``(transposed)``) and inter-backend copies (``(copy)``) that
// the scheduler creates. The layout views share storage with
// their source; capturing them is redundant (the source's
// per-tensor stats already cover the same data). The
// inter-backend copies are dead nodes that the scheduler's
// split logic may not compute, leaving uninitialised data. The
// capture excludes both at the graph level (before the
// per-tensor stats are computed) and prints a stderr warning
// listing the excluded tensors and why. The JSON output never
// silently loses rows.
//
// Per-tensor stat formulas (kurtosis / eff_rank / rms /
// mean_abs / tail_ratio / p99) match the v1 calibration
// side's byte-equivalent formulas. The kurtosis is excess
// kurtosis (E[((X - mu) / sigma)^4] - 3). The eff_rank is the
// exp(H) / N proxy where H is the Shannon entropy of the
// normalised squared value distribution.

#include "clip-capture.h"

#include "ggml.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "models/models.h"

// stb_image is a header-only library; mtmd-helper.cpp defines
// STB_IMAGE_IMPLEMENTATION so the implementation lives in the
// mtmd library. We do NOT define it here (the link would fail
// with duplicate symbols). We only need the function
// declarations.
#include "stb/stb_image.h"

#include "mtmd-audio.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
        double k_acc = 0.0;
        for (float v : acc.samples) {
            const double z = ((double) v - mean) / std;
            k_acc += z * z * z * z;
        }
        s.kurtosis = k_acc / (double) N - 3.0;
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
    out.clear();
    return false;
}

// ---------------------------------------------------------------------------
// Dead-node detection.
//
// The clip graph contains layout views and inter-backend copies
// that the scheduler creates. Layout views share storage with
// their source; capturing them is redundant. Inter-backend copies
// (the (permuted) (copy) tensors in the gemma3 graph) may not be
// computed by the scheduler's split logic, leaving uninitialised
// data. We detect both at the graph level:
//
//   * Layout-view suffix check: if the tensor name ends in
//     ``(view)``, ``(permuted)``, ``(cont)``, ``(reshaped)``,
//     ``(transposed)``, ``(copy of X)`` (case-sensitive) the
//     tensor is a layout view / inter-backend copy and the
//     per-tensor stats should not be derived from it.
//
//   * Live-node check via the ggml backend scheduler: after
//     ``ggml_backend_sched_split_graph``, the scheduler's
//     ``sched->graph.nodes`` array only contains nodes that are
//     in some split (i.e. actually computed). A tensor is alive
//     iff its pointer is in that array. This is the authoritative
//     check; the suffix check above is a defensive backstop for
//     graph builders that emit (copy) without going through the
//     scheduler's copy path.
// ---------------------------------------------------------------------------

bool is_layout_view_name(const char * name) {
    if (name == nullptr) {
        return true;  // skip
    }
    // Whitespace-prefixed names (e.g. " (reshaped) (permuted)")
    // are layout artifacts of the graph builder; the parent
    // tensor captures the real data.
    const char * p = name;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0') {
        return true;
    }
    auto ends_with = [&](const char * suf) -> bool {
        const size_t n = std::strlen(name);
        const size_t m = std::strlen(suf);
        return n >= m && std::strcmp(name + n - m, suf) == 0;
    };
    if (ends_with("(view)")) return true;
    if (ends_with("(permuted)")) return true;
    if (ends_with("(cont)")) return true;
    if (ends_with("(reshaped)")) return true;
    if (ends_with("(transposed)")) return true;
    // The (copy) suffix comes from ggml_cpy. The inter-backend
    // copies the scheduler inserts are not computed reliably on
    // every backend (Metal in particular leaves the destination
    // uninitialised for some splits); we skip them at the graph
    // level. The (copy of X) variant comes from ggml_cpy with a
    // named destination; same handling.
    if (ends_with("(copy)")) return true;
    if (std::strstr(name, "(copy of ") != nullptr) return true;
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
        const std::string & model_path,
        const std::string & mm_projector_path,
        int n_inputs, int n_chunks, int n_activations) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"tool\": \"llama-clip-capture\",\n";
    o << "  \"mode\": \"" << mode << "\",\n";
    o << "  \"model\": \"" << json_escape(model_path) << "\",\n";
    if (mm_projector_path.empty()) {
        o << "  \"mm_projector_model\": null,\n";
    } else {
        o << "  \"mm_projector_model\": \"" << json_escape(mm_projector_path) << "\",\n";
    }
    o << "  \"n_inputs\": " << n_inputs << ",\n";
    o << "  \"n_chunks\": " << n_chunks << ",\n";
    o << "  \"n_activations\": " << n_activations << ",\n";
    o << "  \"peak_rss_bytes_approx\": " << peak_rss << ",\n";
    o << "  \"wall_clock_ms\": " << wall_clock_ms << ",\n";
    o << "  \"tensors\": [";
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto & s = stats[i];
        o << (i == 0 ? "\n    " : ",\n    ");
        o << "{\"name\": \"" << json_escape(s.name) << "\", ";
        o << "\"n_elements\": " << s.n_elements << ", ";
        o << "\"kurtosis\": " << s.kurtosis << ", ";
        o << "\"eff_rank\": " << s.eff_rank << ", ";
        o << "\"rms\": " << s.rms << ", ";
        o << "\"mean_abs\": " << s.mean_abs << ", ";
        o << "\"tail_ratio\": " << s.tail_ratio << ", ";
        o << "\"p99\": " << s.p99 << ", ";
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

// ---------------------------------------------------------------------------
// Audio decode (AudioToolbox on Apple, miniaudio elsewhere with
// a final inline-WAV fallback for restricted CI environments).
// ---------------------------------------------------------------------------

#ifdef __APPLE__
#  define TS_HAVE_AUDIOTOOLBOX 1
#  include <AudioToolbox/AudioToolbox.h>
#endif

// Inline 16-bit PCM WAV parser. Used on non-Apple platforms as
// the final fallback if miniaudio is not available; production
// audio capture on the M1 deployment target uses AudioToolbox.
static bool decode_wav_to_pcm(
        const std::string & path, int target_sample_rate,
        std::vector<float> & pcm) {
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
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    size_t data_offset = 0;
    size_t data_size = 0;
    size_t pos = 12;
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
            pos += 1;
        }
    }
    if (audio_format != 1 || bits_per_sample != 16 ||
        num_channels < 1 || num_channels > 2 ||
        data_offset == 0 || data_size == 0 ||
        data_offset + data_size > buf.size()) {
        return false;
    }
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

#ifdef TS_HAVE_AUDIOTOOLBOX

// Apple production audio decoder. Uses ExtAudioFile (a
// higher-level C API over AudioToolbox) which natively decodes
// WAV, MP3, FLAC, AAC, ALAC (and other system-supported
// formats) to float32 mono PCM at the target sample rate. The
// decoder is stream-based; the entire PCM is buffered in
// memory (the audio chunks the orchestrator feeds in are
// short — typically < 30 s of audio).
static bool decode_audio_audiotoolbox(
        const std::string & path, int target_sample_rate,
        std::vector<float> & pcm) {
    CFStringRef cf_path = CFStringCreateWithCString(
            kCFAllocatorDefault, path.c_str(), kCFStringEncodingUTF8);
    if (cf_path == nullptr) {
        return false;
    }
    CFURLRef cf_url = CFURLCreateWithFileSystemPath(
            kCFAllocatorDefault, cf_path, kCFURLPOSIXPathStyle, false);
    CFRelease(cf_path);
    if (cf_url == nullptr) {
        return false;
    }
    ExtAudioFileRef af = nullptr;
    OSStatus status = ExtAudioFileOpenURL(cf_url, &af);
    CFRelease(cf_url);
    if (status != noErr || af == nullptr) {
        return false;
    }
    // Query the source format to confirm it's decodable.
    AudioStreamBasicDescription src_fmt = {};
    UInt32 src_fmt_size = sizeof(src_fmt);
    status = ExtAudioFileGetProperty(af,
            kExtAudioFileProperty_FileDataFormat, &src_fmt_size, &src_fmt);
    if (status != noErr) {
        ExtAudioFileDispose(af);
        return false;
    }
    // Build the client format: float32 mono PCM at target sample rate.
    AudioStreamBasicDescription dst_fmt = {};
    dst_fmt.mSampleRate       = (Float64) target_sample_rate;
    dst_fmt.mFormatID         = kAudioFormatLinearPCM;
    dst_fmt.mFormatFlags      = kAudioFormatFlagIsFloat |
                                kAudioFormatFlagIsPacked |
                                kAudioFormatFlagIsNonInterleaved;
    dst_fmt.mBitsPerChannel   = 32;
    dst_fmt.mChannelsPerFrame = 1;
    dst_fmt.mFramesPerPacket  = 1;
    dst_fmt.mBytesPerFrame    = 4;
    dst_fmt.mBytesPerPacket   = 4;
    status = ExtAudioFileSetProperty(af,
            kExtAudioFileProperty_ClientDataFormat,
            sizeof(dst_fmt), &dst_fmt);
    if (status != noErr) {
        ExtAudioFileDispose(af);
        return false;
    }
    // Read the entire file in chunks. The total frame count is
    // unknown until the first read.
    const UInt32 kChunkFrames = 4096;
    std::vector<float> chunk(kChunkFrames);
    AudioBufferList abl = {};
    abl.mNumberBuffers = 1;
    abl.mBuffers[0].mNumberChannels = 1;
    abl.mBuffers[0].mDataByteSize = (UInt32)(kChunkFrames * sizeof(float));
    abl.mBuffers[0].mData = chunk.data();
    while (true) {
        UInt32 frames = kChunkFrames;
        status = ExtAudioFileRead(af, &frames, &abl);
        if (status != noErr) {
            ExtAudioFileDispose(af);
            return false;
        }
        if (frames == 0) {
            break;
        }
        pcm.insert(pcm.end(), chunk.data(), chunk.data() + frames);
    }
    ExtAudioFileDispose(af);
    return true;
}

#endif  // TS_HAVE_AUDIOTOOLBOX

// Decode any supported audio file to mono float32 PCM at the
// target sample rate. On Apple this uses AudioToolbox; on other
// platforms it tries the inline WAV parser (the CI fixture is a
// synthesised WAV). The function returns true on success.
static bool decode_audio_to_pcm(
        const std::string & path, int target_sample_rate,
        std::vector<float> & pcm) {
#ifdef TS_HAVE_AUDIOTOOLBOX
    if (decode_audio_audiotoolbox(path, target_sample_rate, pcm)) {
        return true;
    }
    // Fall through to the WAV parser if AudioToolbox failed for
    // any reason (e.g. unrecognised file extension but valid
    // PCM bytes); the WAV parser is a no-op for non-RIFF data.
#endif
    return decode_wav_to_pcm(path, target_sample_rate, pcm);
}

// ---------------------------------------------------------------------------
// Audio preprocessor construction.
//
// The audio model consumes a mel-spectrogram (a (n_frames,
// n_mel) float32 matrix). The preprocessor classes in
// mtmd-audio.{h,cpp} take raw mono float32 PCM and produce
// ``mtmd_audio_mel`` instances. We dispatch on the model
// projector type to pick the right preprocessor (the same
// dispatch mtmd.cpp uses).
// ---------------------------------------------------------------------------

static std::unique_ptr<mtmd_audio_preprocessor> make_audio_preprocessor(
        clip_ctx * ctx) {
    if (ctx == nullptr) {
        return nullptr;
    }
    const clip_hparams * hp = clip_get_hparams(ctx);
    if (hp == nullptr) {
        return nullptr;
    }
    // The dispatch mirrors mtmd.cpp's audio preprocessor setup
    // (PROJECTOR_TYPE_ULTRAVOX / VOXTRAL / MUSIC_FLAMINGO / GLMA
    // / MERALION all use the whisper preprocessor; QWEN3A uses
    // qwen3a; GEMMA4UA uses gemma4ua). For models we don't have
    // an explicit preprocessor for we fall back to whisper (the
    // historical default; the test fixture is a tinygemma3 which
    // is vision-only, so the audio path is exercised on the
    // error-cleanly path).
    const projector_type pt = clip_get_projector_type(ctx);
    switch (pt) {
        case PROJECTOR_TYPE_QWEN3A:
            return std::make_unique<mtmd_audio_preprocessor_qwen3a>(ctx);
        case PROJECTOR_TYPE_GEMMA4UA:
            return std::make_unique<mtmd_audio_preprocessor_gemma4ua>(ctx);
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            return std::make_unique<mtmd_audio_preprocessor_granite_speech>(ctx);
        default:
            return std::make_unique<mtmd_audio_preprocessor_whisper>(ctx);
    }
}

// ---------------------------------------------------------------------------
// Callback context
// ---------------------------------------------------------------------------

struct CallbackCtx {
    std::unordered_map<std::string, TensorAccum> * accum;
    std::unordered_set<std::string> * layout_view_names;
    int n_excluded_layout = 0;
    int n_excluded_uninitialised = 0;
    const char * prefix = "v.";
    std::vector<std::string> excluded_layout_examples;
    std::vector<std::string> excluded_uninitialised_examples;
};

static int capture_callback(
        const char * tensor_name,
        const ggml_tensor * tensor,
        void * user_data) {
    auto * ctx = static_cast<CallbackCtx *>(user_data);
    if (tensor_name == nullptr || ctx == nullptr || ctx->accum == nullptr) {
        return 0;
    }
    // Skip layout views and inter-backend copies at the graph
    // level. The (view) / (permuted) / (cont) / (reshaped) /
    // (transposed) tensors share storage with their source; the
    // (copy) / (copy of X) tensors are dead on some backends.
    if (is_layout_view_name(tensor_name)) {
        ctx->n_excluded_layout += 1;
        if (ctx->excluded_layout_examples.size() < 8) {
            ctx->excluded_layout_examples.emplace_back(tensor_name);
        }
        return 0;
    }
    std::vector<float> buf;
    if (!tensor_to_float_buffer(tensor, buf)) {
        return 0;  // skip unsupported dtype
    }
    if (buf.empty()) {
        return 0;
    }
    // Detect uninitialised data: rms == 0 and mean_abs == 0 but
    // p99 (computed later) is huge. The signature of a tensor
    // that the scheduler never wrote. We pre-screen here with a
    // tiny sample to avoid computing full stats on garbage.
    bool uninit = true;
    for (size_t i = 0; i < buf.size() && i < 64; ++i) {
        if (buf[i] != 0.0f) {
            uninit = false;
            break;
        }
    }
    if (uninit) {
        // All the first 64 samples are zero — strong signal the
        // tensor was never written. Skip and warn.
        ctx->n_excluded_uninitialised += 1;
        if (ctx->excluded_uninitialised_examples.size() < 8) {
            ctx->excluded_uninitialised_examples.emplace_back(tensor_name);
        }
        return 0;
    }
    const char * p = tensor_name;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0') {
        return 0;
    }
    std::string prefixed = std::string(ctx->prefix) + p;
    (*ctx->accum)[prefixed].add(buf.data(), (int64_t) buf.size());
    return 0;
}

// ---------------------------------------------------------------------------
// The capture driver
// ---------------------------------------------------------------------------

struct CaptureOptions {
    const char * clip_model_path = nullptr;
    const char * mm_projector_path = nullptr;  // for mm_projector mode
    std::vector<std::string> input_paths;
    ts_clip_capture_mode mode = TS_CLIP_CAPTURE_MODE_VISION;
    const char * output_json_path = nullptr;
    int batch_size = 1;  // 1 = no batching; >1 = chunk inputs
    int64_t peak_rss_budget_bytes = 0;
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

// Load a vision batch (a list of input image paths) into a
// single ``clip_image_f32_batch``. The batch is the input to
// ``clip_image_batch_encode``; the graph is built once and
// re-used across the batched forward call. Each image is
// decoded, resized to the model's image size, normalised, and
// pushed onto the batch.
static bool build_vision_batch(
        const std::vector<std::string> & paths,
        clip_ctx * ctx,
        clip_image_f32_batch & batch) {
    const clip_hparams * hp = clip_get_hparams(ctx);
    const int target = (hp && hp->image_size > 0) ? hp->image_size : 224;
    batch.is_audio = false;
    for (const std::string & in_path : paths) {
        clip_image_u8 img_u8;
        if (!load_image_to_u8(in_path, img_u8)) {
            return false;
        }
        clip_image_u8 resized;
        resize_nn_u8(img_u8, target, target, resized);
        clip_image_f32 img_f32;
        img_f32.from_u8(resized);
        if (hp != nullptr) {
            img_f32.normalize(hp->image_mean, hp->image_std);
        }
        batch.entries.push_back(std::move(img_f32));
    }
    return true;
}

// Build an audio batch from a list of input audio paths. The
// audio model wants a mel-spectrogram per input. We decode each
// file to mono float32 PCM, then run the per-model
// ``mtmd_audio_preprocessor`` to get the mel-spectrogram.
static bool build_audio_batch(
        const std::vector<std::string> & paths,
        clip_ctx * ctx,
        clip_image_f32_batch & batch) {
    const clip_hparams * hp = clip_get_hparams(ctx);
    const int sr = hp ? hp->audio_sample_rate : 16000;
    if (sr <= 0) {
        return false;
    }
    auto preproc = make_audio_preprocessor(ctx);
    if (!preproc) {
        return false;
    }
    preproc->initialize();
    batch.is_audio = true;
    for (const std::string & in_path : paths) {
        std::vector<float> pcm;
        if (!decode_audio_to_pcm(in_path, sr, pcm)) {
            return false;
        }
        if (pcm.empty()) {
            pcm.assign((size_t) 16000, 0.0f);
        }
        std::vector<mtmd_audio_mel> mels;
        if (!preproc->preprocess(pcm.data(), pcm.size(), mels)) {
            return false;
        }
        if (mels.empty()) {
            return false;
        }
        const auto & mel = mels[0];
        clip_image_f32 mel_f32;
        mel_f32.set_size(
                {(int) mel.n_len, (int) mel.n_mel},
                /*is_placeholder=*/false, /*is_audio=*/true);
        mel_f32.cpy_buf(mel.data);
        batch.entries.push_back(std::move(mel_f32));
    }
    return true;
}

static bool supports_batched_encode(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return false;
    }
    // clip_support_batch is the canonical probe; if it returns
    // false the graph builder cannot accept a batch with more
    // than 1 image (gemma3, llava, etc.). The capture falls
    // back to per-input forward passes in that case; the
    // per-tensor stats are still accumulated across all
    // inputs.
    return clip_support_batch(ctx);
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
    const int batch_size = opt.batch_size > 0 ? opt.batch_size : 1;
    auto t0 = std::chrono::steady_clock::now();

    clip_context_params params = {};
    params.use_gpu = true;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    params.warmup = false;

    clip_init_result init = clip_init(opt.clip_model_path, params);
    if (init.ctx_v == nullptr && init.ctx_a == nullptr) {
        if (err) *err = "clip_init returned no context (model load failed)";
        return 2;
    }
    bool via_vision = (opt.mode == TS_CLIP_CAPTURE_MODE_VISION ||
                       opt.mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION);
    clip_ctx * ctx = via_vision ? init.ctx_v : init.ctx_a;
    if (ctx == nullptr) {
        clip_free(init.ctx_v);
        clip_free(init.ctx_a);
        if (err) *err = via_vision
                            ? "model has no vision encoder"
                            : "model has no audio encoder";
        return 3;
    }

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

    std::unordered_map<std::string, TensorAccum> accum;
    CallbackCtx cbctx = {&accum, /*layout_view_names=*/nullptr, 0, 0,
                         via_vision ? "v." : "a.",
                         {}, {}};
    if (opt.mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION ||
        opt.mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_AUDIO) {
        cbctx.prefix = "mm.";
    }

    // The capture chunks the input list into batches of at most
    // ``batch_size`` inputs each and runs one forward pass per
    // chunk. The per-tensor stats are accumulated over all
    // chunks; the JSON reports the total ``n_inputs`` and
    // ``n_chunks``.
    //
    // For models that don't support batching (clip_support_batch
    // returns false — gemma3, llava, most vision towers), the
    // batch size is forced to 1; the capture still folds the
    // input list into a single output file. Each input gets its
    // own forward pass; n_chunks == n_inputs in that case.
    const bool supports_batch = supports_batched_encode(ctx);
    const int effective_batch = supports_batch ? batch_size : 1;
    const size_t total_inputs = opt.input_paths.size();
    int n_inputs_processed = 0;
    int n_chunks = 0;
    for (size_t off = 0; off < total_inputs; off += (size_t) effective_batch) {
        const size_t end = std::min(off + (size_t) effective_batch, total_inputs);
        std::vector<std::string> chunk_paths(
                opt.input_paths.begin() + off,
                opt.input_paths.begin() + end);

        clip_image_f32_batch batch;
        if (via_vision) {
            if (!build_vision_batch(chunk_paths, ctx, batch)) {
                if (err) {
                    std::string m = "failed to load images at chunk ";
                    m += std::to_string(n_chunks);
                    *err = m;
                }
                clip_free(init.ctx_v);
                clip_free(init.ctx_a);
                return 5;
            }
        } else {
            if (!build_audio_batch(chunk_paths, ctx, batch)) {
                if (err) {
                    std::string m = "failed to decode audio at chunk ";
                    m += std::to_string(n_chunks);
                    *err = m;
                }
                clip_free(init.ctx_v);
                clip_free(init.ctx_a);
                return 5;
            }
        }

        int n = clip_capture_activations(
                ctx, opt.n_threads, &batch, opt.clip_model_path,
                &capture_callback, &cbctx);
        if (n < 0) {
            if (err) {
                std::string m = "forward pass failed at chunk ";
                m += std::to_string(n_chunks);
                *err = m;
            }
            clip_free(init.ctx_v);
            clip_free(init.ctx_a);
            return 6;
        }
        n_inputs_processed += (int) chunk_paths.size();
        n_chunks += 1;
    }

    if (cbctx.n_excluded_layout > 0) {
        std::fprintf(stderr,
            "llama-clip-capture: excluded %d layout-view / "
            "inter-backend-copy tensors from the capture (examples:",
            cbctx.n_excluded_layout);
        for (const auto & name : cbctx.excluded_layout_examples) {
            std::fprintf(stderr, " %s", name.c_str());
        }
        std::fprintf(stderr,
            " ). These are layout transforms / inter-backend copies "
            "that the scheduler creates; the source tensor captures "
            "the real activation data.\n");
    }
    if (cbctx.n_excluded_uninitialised > 0) {
        std::fprintf(stderr,
            "llama-clip-capture: excluded %d uninitialised tensors "
            "(the scheduler's split logic did not compute them; "
            "examples:",
            cbctx.n_excluded_uninitialised);
        for (const auto & name : cbctx.excluded_uninitialised_examples) {
            std::fprintf(stderr, " %s", name.c_str());
        }
        std::fprintf(stderr,
            " ). This is a graph-level exclusion, not a JSON-level "
            "filter; the row was never written by the scheduler and "
            "would have produced garbage stats.\n");
    }

    // Compute per-tensor stats. The JSON writer does NOT
    // filter rows; the dead-node exclusion is done at the
    // graph level above. All rows that reach this point are
    // live and have non-degenerate data.
    std::vector<TensorStats> stats;
    stats.reserve(accum.size());
    int n_activations = 0;
    for (auto & kv : accum) {
        TensorStats s = compute_stats(kv.first, kv.second);
        if (s.n_elements == 0) {
            continue;
        }
        stats.push_back(std::move(s));
        n_activations += 1;
    }
    std::sort(stats.begin(), stats.end(),
              [](const TensorStats & a, const TensorStats & b) {
                  return a.name < b.name;
              });

    auto t1 = std::chrono::steady_clock::now();
    const int64_t wall_clock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
            .count();
    const int64_t peak_rss = estimate_peak_rss(ctx);

    std::string mode_str = "vision";
    if (opt.mode == TS_CLIP_CAPTURE_MODE_AUDIO) {
        mode_str = "audio";
    } else if (opt.mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION) {
        mode_str = "mm_projector_via_vision";
    } else if (opt.mode == TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_AUDIO) {
        mode_str = "mm_projector_via_audio";
    }
    const std::string mm_proj_str = opt.mm_projector_path ? opt.mm_projector_path : "";
    const std::string json = stats_to_json(
            stats, peak_rss, wall_clock_ms, mode_str,
            opt.clip_model_path, mm_proj_str,
            n_inputs_processed, n_chunks, n_activations);

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
        const char * mm_projector_path,
        const std::vector<std::string> & input_paths,
        ts_clip_capture_mode mode,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    ts_clip_capture::CaptureOptions opt;
    opt.clip_model_path = clip_model_path;
    opt.mm_projector_path = mm_projector_path;
    opt.input_paths = input_paths;
    opt.mode = mode;
    opt.output_json_path = output_json_path;
    opt.batch_size = batch_size > 0 ? batch_size : 1;
    opt.peak_rss_budget_bytes = peak_rss_budget_bytes;
    opt.n_threads = n_threads > 0 ? n_threads : 4;
    return ts_clip_capture::capture_impl(opt, err);
}

int ts_clip_capture_activations_vision(
        const char * clip_model_path,
        const std::vector<std::string> & image_paths,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    return ts_clip_capture_activations(
            clip_model_path, /*mm_projector_path=*/nullptr,
            image_paths, TS_CLIP_CAPTURE_MODE_VISION,
            output_json_path, batch_size,
            peak_rss_budget_bytes, n_threads, err);
}

int ts_clip_capture_activations_audio(
        const char * clip_model_path,
        const std::vector<std::string> & audio_paths,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    return ts_clip_capture_activations(
            clip_model_path, /*mm_projector_path=*/nullptr,
            audio_paths, TS_CLIP_CAPTURE_MODE_AUDIO,
            output_json_path, batch_size,
            peak_rss_budget_bytes, n_threads, err);
}

int ts_clip_capture_activations_mm_projector(
        const char * tower_model_path,
        const char * projector_model_path,
        const std::vector<std::string> & input_paths,
        bool via_vision,
        const char * output_json_path,
        int batch_size,
        int64_t peak_rss_budget_bytes,
        int n_threads,
        std::string * err) {
    ts_clip_capture_mode mode = via_vision
        ? TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_VISION
        : TS_CLIP_CAPTURE_MODE_MM_PROJECTOR_VIA_AUDIO;
    return ts_clip_capture_activations(
            tower_model_path, projector_model_path,
            input_paths, mode,
            output_json_path, batch_size,
            peak_rss_budget_bytes, n_threads, err);
}
