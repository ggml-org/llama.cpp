// m5_server.cpp — OpenAI-style HTTP front-end for the native M5-TTS engine.
//
// The engine's stages are Android binaries (HTP), so the server runs on the
// device too, llama-server style:
//
//   adb shell 'cd /data/local/tmp/llama-m5-htp-v73 && \
//       LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD ./m5-server model --port 8080'
//   adb forward tcp:8080 tcp:8080
//   curl localhost:8080/v1/audio/speech \
//        -d '{"input":"今天天气不错","voice":"alice"}' -o out.wav
//
// Endpoints:
//   GET  /health              liveness probe
//   GET  /metrics             synthesis stats (JSON, ?format=prometheus)
//   GET  /v1/models           model list (OpenAI compatibility)
//   GET  /v1/voices           clones cached in voices.json
//   POST /v1/audio/speech     {"input","voice"?,"response_format"?"wav"|"pcm"} -> audio
//   POST /v1/voices/clone     {"name","reference_text","codes":[[16]xT]} -> append cache
//   POST /v1/voices/encode    {"name","reference_text","wav_base64"} -> append cache
//
// Cloning (wav -> codes) runs once — clone_voice.py on the host, the Android
// app, or /v1/voices/encode (the server runs codec_encoder.onnx itself, which
// is what the web console's microphone recording uses). Synthesis consumes the
// same voices.json cache afterwards.
#include "httplib.h"
#include "m5_engine.h"
#include "onnxruntime_c_api.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kBooks = 16;        // codes per frame
constexpr int kVocab = 2051;      // codes are in [0, kVocab)
constexpr int kMaxVoiceFrames = 375;  // 30 s at 12.5 fps
constexpr int kSampleRate = 24000;

// Reads an integer from the environment, falling back when unset/blank. Used
// for the A/B knobs that must be changeable without touching the build.
int env_int(const char * name, int fallback) {
    const char * v = getenv(name);
    if (!v || !*v) return fallback;
    return atoi(v);
}

// Both response formats tune the codec chunking through the same knob pair
// (pcm and wav both default to first=2 / chunk=4), so an A/B env switch
// covers the whole device. Values <= 0 keep the current setting.
void apply_chunking(m5::Engine & engine) {
    engine.set_chunking(env_int("M5_WAV_FIRST", 2), env_int("M5_WAV_CHUNK", 4));
}

// ---------- minimal JSON reading (request bodies, voices.json) ----------

bool parse_json_string(const std::string & s, size_t & at, std::string & out) {
    if (at >= s.size() || s[at] != '"') return false;
    at++;
    out.clear();
    while (at < s.size()) {
        const char c = s[at++];
        if (c == '"') return true;
        if (c != '\\') { out += c; continue; }
        if (at >= s.size()) return false;
        const char e = s[at++];
        switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                if (at + 4 > s.size()) return false;
                unsigned cp = strtoul(s.substr(at, 4).c_str(), nullptr, 16);
                at += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && at + 6 <= s.size() &&
                    s[at] == '\\' && s[at + 1] == 'u') {
                    const unsigned lo = strtoul(s.substr(at + 2, 4).c_str(), nullptr, 16);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        at += 6;
                    }
                }
                if (cp < 0x80) out += char(cp);
                else if (cp < 0x800) {
                    out += char(0xC0 | (cp >> 6));
                    out += char(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    out += char(0xE0 | (cp >> 12));
                    out += char(0x80 | ((cp >> 6) & 0x3F));
                    out += char(0x80 | (cp & 0x3F));
                } else {
                    out += char(0xF0 | (cp >> 18));
                    out += char(0x80 | ((cp >> 12) & 0x3F));
                    out += char(0x80 | ((cp >> 6) & 0x3F));
                    out += char(0x80 | (cp & 0x3F));
                }
                break;
            }
            default: out += e; break;
        }
    }
    return false;
}

// Positions `value_at` at the first byte of the value for `"key":`.
bool find_key(const std::string & body, const char * key, size_t & value_at) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t at = 0;
    while ((at = body.find(pat, at)) != std::string::npos) {
        size_t p = at + pat.size();
        while (p < body.size() && (body[p] == ' ' || body[p] == '\t' ||
                                   body[p] == '\n' || body[p] == '\r')) p++;
        if (p < body.size() && body[p] == ':') {
            p++;
            while (p < body.size() && (body[p] == ' ' || body[p] == '\t' ||
                                       body[p] == '\n' || body[p] == '\r')) p++;
            value_at = p;
            return true;
        }
        at += pat.size();
    }
    return false;
}

bool json_string(const std::string & body, const char * key, std::string & out) {
    size_t at;
    if (!find_key(body, key, at)) return false;
    return parse_json_string(body, at, out);
}

// One past the closing quote/bracket of the string/array/object starting at `at`.
size_t match_value_end(const std::string & s, size_t at) {
    if (at >= s.size()) return std::string::npos;
    if (s[at] == '"') {
        for (size_t p = at + 1; p < s.size(); p++) {
            if (s[p] == '\\') p++;
            else if (s[p] == '"') return p + 1;
        }
        return std::string::npos;
    }
    if (s[at] != '[' && s[at] != '{') {
        size_t p = at;
        while (p < s.size() && strchr(",}] \t\r\n", s[p]) == nullptr) p++;
        return p;
    }
    int depth = 0;
    bool in_string = false;
    for (size_t p = at; p < s.size(); p++) {
        const char c = s[p];
        if (in_string) {
            if (c == '\\') p++;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '[' || c == '{') depth++;
        else if (c == ']' || c == '}') {
            depth--;
            if (depth == 0) return p + 1;
        }
    }
    return std::string::npos;
}

// Captures the raw text of the value for `"key"` (string/array/object/number).
bool json_raw(const std::string & body, const char * key, std::string & out) {
    size_t at;
    if (!find_key(body, key, at)) return false;
    const size_t end = match_value_end(body, at);
    if (end == std::string::npos || end == at) return false;
    out = body.substr(at, end - at);
    return true;
}

// Parses [[int x kBooks] x T] into codes.
bool parse_codes(const std::string & raw, std::vector<std::vector<long>> & codes,
                 std::string & error) {
    codes.clear();
    size_t at = 0;
    while (at < raw.size() && (raw[at] == ' ' || raw[at] == '\t' ||
                               raw[at] == '\n' || raw[at] == '\r')) at++;
    if (at >= raw.size() || raw[at] != '[') { error = "codes must be an array"; return false; }
    at++;
    for (;;) {
        while (at < raw.size() && (raw[at] == ' ' || raw[at] == '\t' ||
                                   raw[at] == '\n' || raw[at] == '\r' || raw[at] == ',')) at++;
        if (at >= raw.size()) { error = "truncated codes"; return false; }
        if (raw[at] == ']') return true;
        if (raw[at] != '[') { error = "codes must be an array of arrays"; return false; }
        at++;
        std::vector<long> row;
        for (;;) {
            while (at < raw.size() && (raw[at] == ' ' || raw[at] == '\t' ||
                                       raw[at] == '\n' || raw[at] == '\r' || raw[at] == ',')) at++;
            if (at >= raw.size()) { error = "truncated codes row"; return false; }
            if (raw[at] == ']') { at++; break; }
            char * end = nullptr;
            const long v = strtol(raw.c_str() + at, &end, 10);
            if (end == raw.c_str() + at) { error = "codes must be integers"; return false; }
            if (v < 0 || v >= kVocab) {
                error = "codes out of range [0, " + std::to_string(kVocab) + ")";
                return false;
            }
            row.push_back(v);
            at = end - raw.c_str();
        }
        if ((int) row.size() != kBooks) {
            error = "each codes row needs " + std::to_string(kBooks) + " books";
            return false;
        }
        codes.push_back(std::move(row));
    }
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else out += c;
        }
    }
    return out;
}

// ---------- voices.json: raw-preserving read / append ----------

struct VoicesFile {
    std::string encoder_sha256;
    std::vector<std::string> entries;  // raw JSON objects, in order
    bool ok = false;
};

VoicesFile read_voices_file(const std::string & text) {
    VoicesFile vf;
    size_t at;
    if (!find_key(text, "encoder_sha256", at)) return vf;
    if (!parse_json_string(text, at, vf.encoder_sha256)) return vf;
    if (!find_key(text, "voices", at)) return vf;
    while (at < text.size() && (text[at] == ' ' || text[at] == '\t' ||
                                text[at] == '\n' || text[at] == '\r')) at++;
    if (at >= text.size() || text[at] != '[') return vf;
    at++;
    for (;;) {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\t' ||
                                    text[at] == '\n' || text[at] == '\r' || text[at] == ',')) at++;
        if (at >= text.size()) return VoicesFile();
        if (text[at] == ']') break;
        if (text[at] != '{') return VoicesFile();
        const size_t end = match_value_end(text, at);
        if (end == std::string::npos) return VoicesFile();
        vf.entries.push_back(text.substr(at, end - at));
        at = end;
    }
    vf.ok = true;
    return vf;
}

std::string write_voices_file(const VoicesFile & vf) {
    std::string out = "{\"encoder_sha256\": \"" + vf.encoder_sha256 + "\", \"voices\": [";
    for (size_t i = 0; i < vf.entries.size(); i++) {
        if (i) out += ", ";
        out += vf.entries[i];
    }
    out += "]}";
    return out;
}

bool save_voices_file(const std::string & path, const VoicesFile & vf, std::string & error) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) { error = "cannot write " + tmp; return false; }
    const std::string text = write_voices_file(vf);
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    fclose(f);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        error = "cannot replace " + path;
        return false;
    }
    return true;
}

std::string file_text(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

std::string read_string_field(const std::string & entry, const char * key) {
    std::string out;
    json_string(entry, key, out);
    return out;
}

int count_frames(const std::string & entry) {
    std::string raw;
    std::vector<std::vector<long>> codes;
    std::string err;
    if (!json_raw(entry, "codes", raw) || !parse_codes(raw, codes, err)) return 0;
    return (int) codes.size();
}

// ---------- PCM streaming state ----------

struct PcmStream {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<short>> chunks;
    bool done = false;
    std::atomic<bool> cancelled{false};
    std::thread worker;

    // The worker must be joined even if the client disconnects before the
    // provider sees the end of the stream (otherwise std::thread dtor aborts).
    ~PcmStream() {
        {
            std::lock_guard<std::mutex> lock(m);
            cancelled = true;
        }
        if (worker.joinable()) worker.join();
    }

    void push(std::vector<short> && chunk) {
        std::lock_guard<std::mutex> lock(m);
        chunks.push_back(std::move(chunk));
        cv.notify_all();
    }
    void finish() {
        std::lock_guard<std::mutex> lock(m);
        done = true;
        cv.notify_all();
    }
};

std::vector<short> floats_to_s16(const float * samples, size_t count) {
    std::vector<short> out(count);
    for (size_t i = 0; i < count; i++) {
        float v = samples[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        out[i] = (short) v;
    }
    return out;
}

// ---------- server state ----------

// ---------- reference audio + on-device voice encoder ----------

bool base64_decode(const std::string & in, std::string & out) {
    static int table[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) table[i] = -1;
        const char * alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) table[(unsigned char) alpha[i]] = i;
        init = true;
    }
    out.clear();
    out.reserve(in.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        if (table[c] < 0) return false;
        val = (val << 6) + table[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return true;
}

// WAV (PCM 8/16/32-bit int or 32-bit float) -> mono float roughly in [-1, 1].
bool wav_to_mono(const std::string & wav, std::vector<float> & mono, int & sample_rate,
                 std::string & error) {
    if (wav.size() < 44 || memcmp(wav.data(), "RIFF", 4) || memcmp(wav.data() + 8, "WAVE", 4)) {
        error = "payload is not a RIFF/WAVE file";
        return false;
    }
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const char * data = nullptr;
    size_t data_len = 0;
    size_t pos = 12;
    while (pos + 8 <= wav.size()) {
        const char * id = wav.data() + pos;
        uint32_t size = 0;
        memcpy(&size, wav.data() + pos + 4, 4);  // little-endian on all our targets
        if (pos + 8 + size > wav.size()) size = (uint32_t) (wav.size() - pos - 8);
        const char * body = wav.data() + pos + 8;
        if (!memcmp(id, "fmt ", 4) && size >= 16) {
            memcpy(&format, body, 2);
            memcpy(&channels, body + 2, 2);
            memcpy(&rate, body + 4, 4);
            memcpy(&bits, body + 14, 2);
        } else if (!memcmp(id, "data", 4)) {
            data = body;
            data_len = size;
        }
        pos += 8 + size + (size & 1);
    }
    if (!data || !data_len) { error = "WAV has no data chunk"; return false; }
    if (!rate || !channels) { error = "WAV fmt chunk is invalid"; return false; }
    const size_t bytes = bits / 8;
    if (!bytes || (format != 1 && !(format == 3 && bits == 32))) {
        error = "unsupported WAV encoding (need PCM or float32)";
        return false;
    }
    const size_t frames = data_len / (bytes * channels);
    if (!frames) { error = "WAV has no samples"; return false; }
    mono.resize(frames);
    for (size_t i = 0; i < frames; i++) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
            const unsigned char * p = (const unsigned char *) data + (i * channels + c) * bytes;
            double v = 0;
            if (format == 3) {
                float f;
                memcpy(&f, p, 4);
                v = f;
            } else if (bits == 16) {
                int16_t s;
                memcpy(&s, p, 2);
                v = s / 32768.0;
            } else if (bits == 32) {
                int32_t s;
                memcpy(&s, p, 4);
                v = s / 2147483648.0;
            } else {  // 8-bit unsigned
                v = ((int) p[0] - 128) / 128.0;
            }
            acc += v;
        }
        mono[i] = (float) (acc / channels);
    }
    sample_rate = (int) rate;
    return true;
}

void linear_resample(std::vector<float> & x, int from, int to) {
    const size_t n_out = (size_t) ((double) x.size() * to / from);
    if (!n_out) return;
    std::vector<float> y(n_out);
    for (size_t i = 0; i < n_out; i++) {
        const double at = (double) i * from / to;
        const size_t j = (size_t) at;
        const double f = at - (double) j;
        const float a = x[j];
        const float b = (j + 1 < x.size()) ? x[j + 1] : a;
        y[i] = (float) (a + (b - a) * f);
    }
    x.swap(y);
}

// Mirrors M5 infer_onnx.read_reference / clone_voice.py: peak normalize, then
// silence-trim (gaps > 0.15 s are cut, 0.05 s of padding is kept around spans).
bool read_reference(std::vector<float> & audio, std::string & error) {
    if (audio.empty()) { error = "reference audio is empty"; return false; }
    for (float v : audio) {
        if (!std::isfinite(v)) { error = "reference audio is invalid"; return false; }
    }
    float peak = 0;
    for (float v : audio) peak = std::max(peak, std::fabs(v));
    for (float & v : audio) v /= (peak + 1e-8f);
    const float threshold = audio.size() > 10 * (size_t) kSampleRate ? 0.02f : 0.015f;
    std::vector<size_t> speech;
    for (size_t i = 0; i < audio.size(); i++) {
        if (std::fabs(audio[i]) > threshold) speech.push_back(i);
    }
    if (speech.empty()) {
        error = "reference audio contains no speech above the silence threshold";
        return false;
    }
    const size_t gap = (size_t) (0.15 * kSampleRate);
    const size_t pad = (size_t) (0.05 * kSampleRate);
    std::vector<std::pair<size_t, size_t>> spans;
    size_t start = speech[0], prev = speech[0];
    for (size_t i = 1; i < speech.size(); i++) {
        if (speech[i] - prev > gap) {
            spans.emplace_back(start, prev + 1);
            start = speech[i];
        }
        prev = speech[i];
    }
    spans.emplace_back(start, audio.size() - prev - 1 >= gap ? prev + 1 : audio.size());
    std::vector<float> out;
    for (const auto & span : spans) {
        const size_t a = span.first > pad ? span.first - pad : 0;
        const size_t b = std::min(audio.size(), span.second + pad);
        out.insert(out.end(), audio.begin() + a, audio.begin() + b);
    }
    audio.swap(out);
    return true;
}

// Compact SHA-256 (FIPS 180-4) for the encoder identity check.
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t total = 0;
    uint8_t buf[64] = {};
    size_t fill = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t) p[i * 4] << 24) | ((uint32_t) p[i * 4 + 1] << 16) |
                   ((uint32_t) p[i * 4 + 2] << 8) | p[i * 4 + 3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        total += n;
        while (n) {
            const size_t take = std::min(n, 64 - fill);
            memcpy(buf + fill, p, take);
            fill += take; p += take; n -= take;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = total * 8;
        const uint8_t pad_byte = 0x80;
        update(&pad_byte, 1);
        const uint8_t zero = 0;
        while (fill != 56) update(&zero, 1);
        uint8_t tail[8];
        for (int i = 0; i < 8; i++) tail[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(tail, 8);
        char out[65];
        for (int i = 0; i < 8; i++) snprintf(out + i * 8, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

std::string file_sha256(const std::string & path, std::string & error) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { error = "cannot open " + path; return {}; }
    Sha256 sha;
    std::vector<char> block(1 << 20);
    while (true) {
        const size_t n = fread(block.data(), 1, block.size(), f);
        if (n) sha.update(block.data(), n);
        if (n < block.size()) break;
    }
    fclose(f);
    return sha.hex();
}

// Runs codec_encoder.onnx once per clone - the same ONNX clone_voice.py uses:
//   input "audio" [1, T] float32 @ 24 kHz
//   output "codes" [1, 16, F], F padded to ceil(T/1920) and sliced like Python.
class VoiceEncoder {
public:
    ~VoiceEncoder() { close(); }

    bool open(const std::string & model_path, std::string & error) {
        api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!api_) { error = "ONNX Runtime api unavailable"; return false; }
        if (!ok(api_->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "m5-voice-encoder", &env_), error)) return false;
        if (!ok(api_->CreateSessionOptions(&so_), error)) return false;
        api_->SetIntraOpNumThreads(so_, 4);
        api_->SetInterOpNumThreads(so_, 1);
        api_->SetSessionGraphOptimizationLevel(so_, ORT_ENABLE_ALL);
        if (!ok(api_->CreateSession(env_, model_path.c_str(), so_, &session_), error)) return false;
        if (!ok(api_->GetAllocatorWithDefaultOptions(&alloc_), error)) return false;
        if (!ok(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_), error)) return false;
        opened_ = true;
        return true;
    }

    bool encode(const std::vector<float> & audio, std::vector<std::vector<long>> & codes,
                std::string & error) {
        if (!opened_) { error = "encoder is not open"; return false; }
        const int64_t dims[2] = {1, (int64_t) audio.size()};
        OrtValue * in = nullptr;
        if (!ok(api_->CreateTensorWithDataAsOrtValue(mem_, (void *) audio.data(),
                                                     audio.size() * sizeof(float), dims, 2,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in), error))
            return false;
        const char * in_names[] = {"audio"};
        const char * out_names[] = {"codes"};
        OrtValue * out = nullptr;
        OrtStatus * status = api_->Run(session_, nullptr, in_names, (const OrtValue * const *) &in, 1,
                                       out_names, 1, &out);
        api_->ReleaseValue(in);
        if (!ok(status, error)) return false;

        OrtTensorTypeAndShapeInfo * info = nullptr;
        if (!ok(api_->GetTensorTypeAndShape(out, &info), error)) { api_->ReleaseValue(out); return false; }
        size_t rank = 0;
        api_->GetDimensionsCount(info, &rank);
        std::vector<int64_t> shape(rank);
        api_->GetDimensions(info, shape.data(), rank);
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        api_->GetTensorElementType(info, &type);
        api_->ReleaseTensorTypeAndShapeInfo(info);

        int64_t books = 0, frames_full = 0;
        if (rank == 3) { books = shape[1]; frames_full = shape[2]; }
        else if (rank == 2) { books = shape[0]; frames_full = shape[1]; }
        else { api_->ReleaseValue(out); error = "unexpected codes rank"; return false; }
        const int64_t want = (int64_t) ((audio.size() + 1919) / 1920);
        if (books != kBooks || frames_full < want || want < 1) {
            api_->ReleaseValue(out);
            error = "unexpected codes shape";
            return false;
        }
        if (want > kMaxVoiceFrames) {
            api_->ReleaseValue(out);
            error = "reference voice is longer than 30 seconds";
            return false;
        }
        void * raw = nullptr;
        if (!ok(api_->GetTensorMutableData(out, &raw), error)) { api_->ReleaseValue(out); return false; }
        codes.assign((size_t) want, std::vector<long>(kBooks, 0));
        for (int64_t k = 0; k < books; k++) {
            for (int64_t t = 0; t < want; t++) {
                long value = 0;
                if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    value = (long) ((const int64_t *) raw)[k * frames_full + t];
                } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                    value = (long) ((const int32_t *) raw)[k * frames_full + t];
                } else {
                    value = (long) ((const float *) raw)[k * frames_full + t];
                }
                if (value < 0 || value > 2047) {
                    api_->ReleaseValue(out);
                    error = "encoder produced out-of-range codes";
                    return false;
                }
                codes[(size_t) t][(size_t) k] = value;
            }
        }
        api_->ReleaseValue(out);
        return true;
    }

    void close() {
        if (api_) {
            if (session_) api_->ReleaseSession(session_);
            if (so_) api_->ReleaseSessionOptions(so_);
            if (mem_) api_->ReleaseMemoryInfo(mem_);
            if (env_) api_->ReleaseEnv(env_);
        }
        session_ = nullptr; so_ = nullptr; mem_ = nullptr; env_ = nullptr;
        opened_ = false;
    }

private:
    bool ok(OrtStatus * status, std::string & error) {
        if (!status) return true;
        error = api_->GetErrorMessage(status);
        api_->ReleaseStatus(status);
        return false;
    }
    const OrtApi * api_ = nullptr;
    OrtEnv * env_ = nullptr;
    OrtSessionOptions * so_ = nullptr;
    OrtSession * session_ = nullptr;
    OrtAllocator * alloc_ = nullptr;
    OrtMemoryInfo * mem_ = nullptr;
    bool opened_ = false;
};

// The stamped identity: codec_identity.json next to the encoder when present
// (must match the file), otherwise the file hash itself.
std::string encoder_identity(const std::string & encoder_path, std::string & error) {
    const std::string digest = file_sha256(encoder_path, error);
    if (digest.empty()) return {};
    const std::string dir = encoder_path.substr(0, encoder_path.find_last_of('/'));
    const std::string identity_text = file_text(dir + "/codec_identity.json");
    if (!identity_text.empty()) {
        std::string want;
        if (json_string(identity_text, "encoder_sha256", want) && !want.empty() && want != digest) {
            error = "codec encoder SHA256 mismatch: " + digest.substr(0, 16) + " != " + want.substr(0, 16);
            return {};
        }
    }
    return digest;
}

// One finished synthesis, kept so /metrics can show what the last requests
// actually cost. The X-M5-* headers only reach non-streaming clients, so this
// is the only place where pcm runs can be observed at all.
struct RequestStat {
    uint64_t id = 0;
    std::string voice;
    std::string format;   // "wav" | "pcm" | "warmup"
    int text_bytes = 0;
    bool ok = true;
    double elapsed = 0, audio = 0, rtf = 0, first_pcm = 0;
    double backbone = 0, depth = 0, codec = 0;
    int frames = 0, segments = 0;
};

// How many finished requests /metrics keeps for percentiles.
constexpr size_t kStatsKeep = 64;

struct ServerState {
    m5::Engine engine;
    std::mutex render_mutex;   // one synthesis at a time (the stages are shared)
    std::string voices_path;   // resolved clone cache
    std::string default_voice;
    std::string workdir;
    std::string encoder_path;  // codec_encoder.onnx for POST /v1/voices/encode
    std::mutex encode_mutex;
    std::unique_ptr<VoiceEncoder> encoder;  // lazy, opened on first encode
    std::string encoder_sha;                // identity stamped by this server

    // /metrics bookkeeping.
    std::mutex stats_mutex;
    std::deque<RequestStat> stats_recent;   // last kStatsKeep, oldest first
    uint64_t stats_total = 0;
    uint64_t stats_failed = 0;
    double stats_render_s = 0;
    double stats_audio_s = 0;
    bool have_warmup = false;
    RequestStat warmup;
};

void record_stat(ServerState & st, RequestStat stat) {
    std::lock_guard<std::mutex> lock(st.stats_mutex);
    st.stats_total++;
    if (stat.id == 0) stat.id = st.stats_total;
    if (!stat.ok) st.stats_failed++;
    st.stats_render_s += stat.elapsed;
    st.stats_audio_s += stat.audio;
    st.stats_recent.push_back(std::move(stat));
    while (st.stats_recent.size() > kStatsKeep) st.stats_recent.pop_front();
}

RequestStat make_stat(const std::string & voice, const std::string & format,
                      const std::string & text, const m5::Stats & stats, bool ok) {
    RequestStat s;
    s.voice = voice;
    s.format = format;
    s.text_bytes = (int) text.size();
    s.ok = ok;
    s.elapsed = stats.elapsed;
    s.audio = stats.audio;
    s.rtf = stats.rtf;
    s.first_pcm = stats.first_pcm;
    s.backbone = stats.backbone;
    s.depth = stats.depth;
    s.codec = stats.codec;
    s.frames = stats.frames;
    s.segments = stats.segments;
    return s;
}

std::string stat_json(const RequestStat & s) {
    char buf[640];
    snprintf(buf, sizeof buf,
             "{\"id\":%llu,\"voice\":\"%s\",\"format\":\"%s\",\"text_bytes\":%d,\"ok\":%s,"
             "\"rtf\":%.3f,\"elapsed_s\":%.3f,\"audio_s\":%.3f,\"first_pcm_s\":%.3f,"
             "\"frames\":%d,\"segments\":%d,\"backbone_s\":%.3f,\"depth_s\":%.3f,\"codec_s\":%.3f}",
             (unsigned long long) s.id, json_escape(s.voice).c_str(), s.format.c_str(),
             s.text_bytes, s.ok ? "true" : "false", s.rtf, s.elapsed, s.audio, s.first_pcm,
             s.frames, s.segments, s.backbone, s.depth, s.codec);
    return buf;
}

// Nearest-rank percentile of an unsorted copy-friendly vector (0 when empty).
double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t at = (size_t) (q * (double) (values.size() - 1) + 0.5);
    if (at >= values.size()) at = values.size() - 1;
    return values[at];
}

void send_json(httplib::Response & res, int status, const std::string & body) {
    res.status = status;
    res.set_content(body, "application/json");
}

void send_error(httplib::Response & res, int status, const std::string & message) {
    send_json(res, status, "{\"error\":{\"message\":\"" + json_escape(message) + "\",\"type\":\"invalid_request_error\"}}");
}

bool apply_voice(ServerState & st, const std::string & selector, std::string & error) {
    const std::string & sel = selector.empty() ? st.default_voice : selector;
    if (!st.engine.set_voice(sel, error)) {
        error = "voice \"" + sel + "\" not found";
        return false;
    }
    return true;
}

// Inline "codes" + "reference_text" win (transient voice, nothing is persisted);
// otherwise the named cache entry is used. Optional "caption" steers style.
bool apply_voice_req(ServerState & st, const std::string & body, const std::string & selector,
                     std::string & error) {
    std::string caption;
    json_string(body, "caption", caption);
    st.engine.set_caption(caption);
    std::string codes_raw;
    if (json_raw(body, "codes", codes_raw)) {
        std::vector<std::vector<long>> codes;
        if (!parse_codes(codes_raw, codes, error)) return false;
        std::string reference_text;
        json_string(body, "reference_text", reference_text);
        return st.engine.set_voice_codes(reference_text, codes, error);
    }
    return apply_voice(st, selector, error);
}

// Re-serializes one entry (integers only, lossless) and upserts it by name.
// The caller holds the voices mutex and has validated the encoder identity.
bool upsert_voice_entry(ServerState & st, VoicesFile & vf, const std::string & name,
                        const std::string & reference_text,
                        const std::vector<std::vector<long>> & codes, bool & replaced,
                        std::string & error) {
    std::string entry = "{\"name\": \"" + json_escape(name) + "\", \"reference_text\": \"" +
                        json_escape(reference_text) + "\", \"codes\": [";
    for (size_t r = 0; r < codes.size(); r++) {
        if (r) entry += ", ";
        entry += "[";
        for (int j = 0; j < kBooks; j++) {
            if (j) entry += ", ";
            entry += std::to_string(codes[r][j]);
        }
        entry += "]";
    }
    entry += "]}";
    replaced = false;
    for (auto & existing : vf.entries) {
        if (read_string_field(existing, "name") == name) {
            existing = entry;
            replaced = true;
            break;
        }
    }
    if (!replaced) vf.entries.push_back(entry);
    return save_voices_file(st.voices_path, vf, error);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s model_dir [--host H] [--port N] [engine options]\n"
                "  --host H           bind address (default 0.0.0.0)\n"
                "  --port N           listen port (default 8080)\n"
                "  --voice N|NAME     default voice (default 0)\n"
                "  --voices FILE      clone cache (default model_dir/voices.json)\n"
                "  --voice-encoder P  codec_encoder.onnx for POST /v1/voices/encode\n"
                "                     (default model_dir/codec_encoder.onnx)\n"
                "  --threads N        ONNX Runtime intra-op threads (default 3)\n"
                "  --cwd DIR          working directory for the stage processes\n"
                "  --seed S           java.util.Random seed (default 1234)\n"
                "\n"
                "POST /v1/audio/speech  {\"input\":\"...\",\"voice\":\"alice\",\"response_format\":\"wav\"}\n"
                "POST /v1/voices/clone  {\"name\":\"alice\",\"reference_text\":\"...\",\"codes\":[[...]]}\n"
                "POST /v1/voices/encode {\"name\":\"alice\",\"reference_text\":\"...\",\"wav_base64\":\"...\"}\n",
                argv[0]);
        return 2;
    }

    m5::Options options;
    options.model_dir = argv[1];
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string encoder_path;

    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--host") host = next("--host");
        else if (a == "--port") port = atoi(next("--port").c_str());
        else if (a == "--voice-encoder") encoder_path = next("--voice-encoder");
        else if (a == "--voice") {
            const std::string v = next("--voice");
            if (v.find_first_not_of("0123456789") == std::string::npos) options.voice = atoi(v.c_str());
            else options.voice_name = v;
        }
        else if (a == "--voices") options.voices_path = next("--voices");
        else if (a == "--seed") options.seed = atol(next("--seed").c_str());
        else if (a == "--backbone") options.backbone_cmd = next("--backbone");
        else if (a == "--depth") options.depth_cmd = next("--depth");
        else if (a == "--codec") options.codec_model = next("--codec");
        else if (a == "--threads") options.threads = atoi(next("--threads").c_str());
        else if (a == "--cwd") options.workdir = next("--cwd");
        else if (a == "--first") options.first_frames = atoi(next("--first").c_str());
        else if (a == "--chunk") options.chunk_frames = atoi(next("--chunk").c_str());
        else if (a == "--ring") options.ring = true;
        else if (a == "--verbose") options.verbose = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    ServerState st;
    st.workdir = options.workdir.empty() ? "." : options.workdir;
    st.default_voice = options.voice_name.empty() ? std::to_string(options.voice) : options.voice_name;
    st.voices_path = options.voices_path.empty() ? options.model_dir + "/voices.json" : options.voices_path;
    st.encoder_path = encoder_path;
    if (st.encoder_path.empty()) {
        auto exists = [](const std::string & p) {
            FILE * f = fopen(p.c_str(), "rb");
            if (f) fclose(f);
            return f != nullptr;
        };
        const std::string candidates[] = {
            options.model_dir + "/codec_encoder.onnx",
            st.workdir + "/codec_encoder.onnx",
        };
        for (const auto & candidate : candidates) {
            if (exists(candidate)) { st.encoder_path = candidate; break; }
        }
    }

    std::string error;
    if (!st.engine.open(options, error)) {
        fprintf(stderr, "open failed: %s\n", error.c_str());
        return 1;
    }
    // Warm every stage (backbone prefill, 15-step depth chain, codec graph, ORT
    // kernels) so the first real request does not pay one-off costs - the first
    // request after a restart measured First-Pcm 0.58 s vs 0.33-0.45 s steady
    // state. Discard mode: nothing is written, the voice selection is untouched.
    {
        m5::Stats warm;
        std::string warm_error;
        st.engine.clear_stop();
        const bool warm_ok = st.engine.render_text("你好，世界。", "", nullptr, warm, warm_error);
        if (warm_ok) {
            fprintf(stderr, "[m5-server] warmup %.2fs (rtf=%.3f frames=%d)\n",
                    warm.elapsed, warm.rtf, warm.frames);
        } else {
            fprintf(stderr, "[m5-server] warmup skipped: %s\n", warm_error.c_str());
        }
        // Kept out of the recent window so it never skews the percentiles.
        st.warmup.format = "warmup";
        st.warmup.voice = st.default_voice;
        st.warmup.text_bytes = (int) std::strlen("你好，世界。");
        st.warmup.ok = warm_ok;
        st.warmup.elapsed = warm.elapsed;
        st.warmup.audio = warm.audio;
        st.warmup.rtf = warm.rtf;
        st.warmup.first_pcm = warm.first_pcm;
        st.warmup.backbone = warm.backbone;
        st.warmup.depth = warm.depth;
        st.warmup.codec = warm.codec;
        st.warmup.frames = warm.frames;
        st.warmup.segments = warm.segments;
        st.have_warmup = true;
    }
    fprintf(stderr, "[m5-server] model=%s voices=%s listening on %s:%d\n",
            options.model_dir.c_str(), st.voices_path.c_str(), host.c_str(), port);

    httplib::Server svr;

    // Expose the X-M5-* stats to browsers (CORS hides custom headers by
    // default), so the web console can show RTF / first-packet / frames.
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Expose-Headers",
                              "X-M5-RTF, X-M5-Elapsed-S, X-M5-First-Pcm-S, X-M5-Frames"}});

    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    // Web console: served from <workdir>/web.html so a browser on this computer
    // (via adb forward) or on the LAN can drive the API directly. CORS is "*",
    // so the same page also works when opened as a local file.
    svr.Get("/", [&st](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store"); // web.html is edited in place; never serve a stale console
        std::string page = file_text(st.workdir + "/web.html");
        if (page.empty()) {
            page = "<!doctype html><meta charset=\"utf-8\"><title>m5-server</title>"
                   "<body style=\"font-family:sans-serif;max-width:40em;margin:3em auto\">"
                   "<h2>m5-server 在线</h2><p>把 <code>web.html</code> 放进工作目录（--cwd）"
                   "即可托管完整控制台。</p><p>API：<code>/health</code> <code>/v1/models</code> "
                   "<code>/v1/voices</code> <code>/v1/audio/speech</code> "
                   "<code>/v1/voices/clone</code></p></body>";
        }
        res.set_content(page, "text/html; charset=utf-8");
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        send_json(res, 200, "{\"status\":\"ok\"}");
    });

    // Everything this process has synthesised. The X-M5-* headers only exist on
    // wav responses (chunked pcm commits its headers before the numbers are
    // known), so streaming runs are observable only here. Default is JSON for
    // the console and regression scripts; ?format=prometheus scrapes cleanly.
    svr.Get("/metrics", [&st](const httplib::Request & req, httplib::Response & res) {
        uint64_t total = 0, failed = 0;
        double render_s = 0, audio_s = 0;
        std::vector<RequestStat> recent;
        RequestStat warm;
        bool have_warm = false;
        {
            std::lock_guard<std::mutex> lock(st.stats_mutex);
            total = st.stats_total;
            failed = st.stats_failed;
            render_s = st.stats_render_s;
            audio_s = st.stats_audio_s;
            recent.assign(st.stats_recent.begin(), st.stats_recent.end());
            have_warm = st.have_warmup;
            warm = st.warmup;
        }
        std::vector<double> rtfs, firsts;
        for (const auto & s : recent) {
            if (!s.ok) continue;
            rtfs.push_back(s.rtf);
            firsts.push_back(s.first_pcm);
        }
        // `last` prefers the most recent successful run, so the m5_last_* gauges
        // never collapse to zero just because the newest request was rejected.
        const RequestStat * last = nullptr;
        for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
            if (it->ok) { last = &*it; break; }
        }
        if (!last && !recent.empty()) last = &recent.back();
        const RequestStat & latest = last ? *last : warm;

        if (req.get_param_value("format") == "prometheus") {
            std::string out;
            auto gauge = [&out](const char * name, double value, const char * help) {
                char line[192];
                snprintf(line, sizeof line, "# HELP %s %s\n# TYPE %s gauge\n%s %.6g\n",
                         name, help, name, name, value);
                out += line;
            };
            auto counter = [&out](const char * name, double value, const char * help) {
                char line[192];
                snprintf(line, sizeof line, "# HELP %s %s\n# TYPE %s counter\n%s %.6g\n",
                         name, help, name, name, value);
                out += line;
            };
            counter("m5_requests_total", (double) total, "Finished syntheses since start.");
            counter("m5_requests_failed_total", (double) failed, "Syntheses that returned an error.");
            counter("m5_render_seconds_total", render_s, "Wall time spent synthesising.");
            counter("m5_audio_seconds_total", audio_s, "Audio seconds produced.");
            gauge("m5_avg_rtf", audio_s > 0 ? render_s / audio_s : 0.0, "Aggregate render/audio ratio.");
            gauge("m5_p50_rtf", percentile(rtfs, 0.5), "Median RTF over the recent window.");
            gauge("m5_p50_first_pcm_seconds", percentile(firsts, 0.5), "Median first-packet latency.");
            gauge("m5_last_rtf", latest.rtf, "RTF of the last request.");
            gauge("m5_last_elapsed_seconds", latest.elapsed, "Wall time of the last request.");
            gauge("m5_last_audio_seconds", latest.audio, "Audio duration of the last request.");
            gauge("m5_last_first_pcm_seconds", latest.first_pcm, "First-packet latency of the last request.");
            gauge("m5_last_frames", (double) latest.frames, "Codec frames of the last request.");
            gauge("m5_last_backbone_seconds", latest.backbone, "Backbone time of the last request.");
            gauge("m5_last_depth_seconds", latest.depth, "Depth time of the last request.");
            gauge("m5_last_codec_seconds", latest.codec, "Codec time of the last request.");
            gauge("m5_warmup_seconds", have_warm ? warm.elapsed : 0.0, "Startup warmup cost.");
            res.set_content(out, "text/plain; version=0.0.4");
            return;
        }

        char head[384];
        snprintf(head, sizeof head,
                 "{\"requests\":%llu,\"failed\":%llu,\"render_s\":%.3f,\"audio_s\":%.3f,"
                 "\"avg_rtf\":%.3f,\"p50_rtf\":%.3f,\"p50_first_pcm_s\":%.3f,\"window\":%zu,",
                 (unsigned long long) total, (unsigned long long) failed, render_s, audio_s,
                 audio_s > 0 ? render_s / audio_s : 0.0, percentile(rtfs, 0.5),
                 percentile(firsts, 0.5), recent.size());
        std::string body = head;
        body += "\"warmup\":";
        body += have_warm ? stat_json(warm) : "null";
        body += ",\"last\":";
        body += last ? stat_json(*last) : "null";
        body += ",\"recent\":[";
        for (size_t i = 0; i < recent.size(); i++) {
            if (i) body += ",";
            body += stat_json(recent[i]);
        }
        body += "]}";
        send_json(res, 200, body);
    });

    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        send_json(res, 200,
                  "{\"object\":\"list\",\"data\":[{\"id\":\"m5-tts\",\"object\":\"model\",\"owned_by\":\"m5\"}]}");
    });

    svr.Get("/v1/voices", [&st](const httplib::Request &, httplib::Response & res) {
        const VoicesFile vf = read_voices_file(file_text(st.voices_path));
        if (!vf.ok) { send_error(res, 500, "cannot parse " + st.voices_path); return; }
        std::string body = "{\"object\":\"list\",\"data\":[";
        for (size_t i = 0; i < vf.entries.size(); i++) {
            if (i) body += ",";
            body += "{\"index\":" + std::to_string(i) +
                    ",\"id\":\"" + json_escape(read_string_field(vf.entries[i], "name")) + "\"" +
                    ",\"object\":\"voice\"" +
                    ",\"name\":\"" + json_escape(read_string_field(vf.entries[i], "name")) + "\"" +
                    ",\"reference_text\":\"" + json_escape(read_string_field(vf.entries[i], "reference_text")) + "\"" +
                    ",\"frames\":" + std::to_string(count_frames(vf.entries[i])) + "}";
        }
        body += "]}";
        send_json(res, 200, body);
    });

    // OpenAI-compatible speech synthesis.
    svr.Post("/v1/audio/speech", [&st](const httplib::Request & req, httplib::Response & res) {
        std::string text, voice, format = "wav";
        if (!json_string(req.body, "input", text) || text.empty()) {
            send_error(res, 400, "missing \"input\"");
            return;
        }
        json_string(req.body, "voice", voice);
        { std::string f; if (json_string(req.body, "response_format", f) && !f.empty()) format = f; }
        if (format != "wav" && format != "pcm") {
            send_error(res, 400, "response_format must be \"wav\" or \"pcm\"");
            return;
        }

        if (format == "pcm") {
            // Fail fast on a bad voice before starting the stream.
            {
                std::lock_guard<std::mutex> lock(st.render_mutex);
                std::string err;
                if (!apply_voice_req(st, req.body, voice, err)) { send_error(res, 404, err); return; }
            }
            auto stream = std::make_shared<PcmStream>();
            const std::string body = req.body;
            // The HTTP provider owns the stream and joins this worker on release.
            stream->worker = std::thread([&st, stream = stream.get(), text, voice, body] {
                {
                    std::lock_guard<std::mutex> lock(st.render_mutex);
                    std::string err;
                    if (!apply_voice_req(st, body, voice, err)) {
                        fprintf(stderr, "[m5-server] pcm voice apply failed: %s\n", err.c_str());
                        record_stat(st, make_stat(voice, "pcm", text, m5::Stats(), false));
                        stream->finish();
                        return;
                    }
                    const auto sink = [&st, stream](const float * samples, size_t count) {
                        if (stream->cancelled) { st.engine.stop(); return; }
                        stream->push(floats_to_s16(samples, count));
                    };
                    // Two frames per first chunk measured best: one frame is
                    // the smallest batch the codec takes but its fixed call
                    // overhead pushes First-Pcm to ~0.51 s versus ~0.29 s.
                    // M5_WAV_FIRST / M5_WAV_CHUNK re-tune without a rebuild.
                    apply_chunking(st.engine);
                    m5::Stats stats;
                    std::string err2;
                    st.engine.clear_stop();
                    const bool ok = st.engine.render_text(text, "", sink, stats, err2);
                    // Chunked responses commit headers before the first chunk, so the
                    // X-M5-* stats cannot ride on this path: they go to the log and to
                    // GET /metrics instead. wav responses carry the headers as well.
                    fprintf(stderr, "[m5-server] pcm rtf=%.3f elapsed=%.3f first=%.3f frames=%d\n",
                            stats.rtf, stats.elapsed, stats.first_pcm, stats.frames);
                    if (!ok) fprintf(stderr, "[m5-server] pcm render failed: %s\n", err2.c_str());
                    record_stat(st, make_stat(voice, "pcm", text, stats, ok));
                }
                stream->finish();
            });
            res.set_chunked_content_provider("audio/pcm", [stream](size_t, httplib::DataSink & sink) {
                std::unique_lock<std::mutex> lock(stream->m);
                while (stream->chunks.empty() && !stream->done) {
                    stream->cv.wait_for(lock, std::chrono::milliseconds(200));
                }
                if (!stream->chunks.empty()) {
                    std::vector<short> chunk = std::move(stream->chunks.front());
                    stream->chunks.pop_front();
                    lock.unlock();
                    const bool more = sink.write((const char *) chunk.data(),
                                                 chunk.size() * sizeof(short));
                    if (!more) stream->cancelled = true;
                    return more;
                }
                lock.unlock();
                sink.done();
                if (stream->worker.joinable()) stream->worker.join();
                return false;
            });
            return;
        }

        // wav: synthesize fully in memory (no temp file; stats ride in headers).
        std::lock_guard<std::mutex> lock(st.render_mutex);
        std::string err;
        const std::string voice_name = voice.empty() ? st.default_voice : voice;
        if (!apply_voice_req(st, req.body, voice, err)) {
            record_stat(st, make_stat(voice_name, "wav", text, m5::Stats(), false));
            send_error(res, 404, err);
            return;
        }
        // 2/4 keeps the codec and depth branches overlapped: the old 4/16 split
        // was tuned when codec was the bottleneck, but after the depth KV32
        // change 16-frame codec bursts leave the pipeline idling (4.00 s vs
        // 3.60 s wall on the 69-frame sentence) and also delay the first
        // packet. 2/4 additionally reproduces the CLI's canonical waveform
        // bit-for-bit.
        apply_chunking(st.engine);
        st.engine.clear_stop();
        m5::Stats stats;
        std::string wav;
        if (!st.engine.render_text(text, "", nullptr, stats, err, &wav)) {
            record_stat(st, make_stat(voice_name, "wav", text, stats, false));
            send_error(res, 500, "render failed: " + err);
            return;
        }
        if (wav.empty()) {
            record_stat(st, make_stat(voice_name, "wav", text, stats, false));
            send_error(res, 500, "render produced no audio");
            return;
        }
        record_stat(st, make_stat(voice_name, "wav", text, stats, true));
        char hdr[128];
        snprintf(hdr, sizeof hdr, "%g", stats.rtf);
        res.set_header("X-M5-RTF", hdr);
        snprintf(hdr, sizeof hdr, "%.3f", stats.elapsed);
        res.set_header("X-M5-Elapsed-S", hdr);
        snprintf(hdr, sizeof hdr, "%.3f", stats.first_pcm);
        res.set_header("X-M5-First-Pcm-S", hdr);
        snprintf(hdr, sizeof hdr, "%d", stats.frames);
        res.set_header("X-M5-Frames", hdr);
        snprintf(hdr, sizeof hdr, "%.3f", stats.backbone);
        res.set_header("X-M5-Backbone-S", hdr);
        snprintf(hdr, sizeof hdr, "%.3f", stats.depth);
        res.set_header("X-M5-Depth-S", hdr);
        snprintf(hdr, sizeof hdr, "%.3f", stats.codec);
        res.set_header("X-M5-Codec-S", hdr);
        res.set_content(wav, "audio/wav");
    });

    // Imports a pre-extracted clone (codes from clone_voice.py or the app).
    svr.Post("/v1/voices/clone", [&st](const httplib::Request & req, httplib::Response & res) {
        std::string name, reference_text, codes_raw, encoder_sha;
        if (!json_string(req.body, "name", name) || name.empty()) {
            send_error(res, 400, "missing \"name\"");
            return;
        }
        if (!json_string(req.body, "reference_text", reference_text) || reference_text.empty()) {
            send_error(res, 400, "missing \"reference_text\"");
            return;
        }
        if (!json_raw(req.body, "codes", codes_raw)) {
            send_error(res, 400, "missing \"codes\"");
            return;
        }
        std::vector<std::vector<long>> codes;
        std::string err;
        if (!parse_codes(codes_raw, codes, err)) { send_error(res, 400, err); return; }
        if (codes.empty()) { send_error(res, 400, "codes is empty"); return; }
        if ((int) codes.size() > kMaxVoiceFrames) {
            send_error(res, 400, "reference voice is longer than 30 seconds");
            return;
        }
        json_string(req.body, "encoder_sha256", encoder_sha);

        std::lock_guard<std::mutex> lock(st.render_mutex);
        VoicesFile vf = read_voices_file(file_text(st.voices_path));
        if (!vf.ok) { send_error(res, 500, "cannot parse " + st.voices_path); return; }
        if (!encoder_sha.empty() && encoder_sha != vf.encoder_sha256) {
            send_error(res, 409, "encoder_sha256 does not match the model's clone cache");
            return;
        }

        // Re-serialize the entry (integers only, lossless) and upsert by name.
        bool replaced = false;
        if (!upsert_voice_entry(st, vf, name, reference_text, codes, replaced, err)) {
            send_error(res, 500, err);
            return;
        }
        fprintf(stderr, "[m5-server] voice %s (%s) %s, %d frames\n",
                name.c_str(), vf.encoder_sha256.substr(0, 12).c_str(),
                replaced ? "updated" : "added", (int) codes.size());
        send_json(res, 200,
                  "{\"name\":\"" + json_escape(name) + "\",\"frames\":" + std::to_string(codes.size()) +
                  ",\"replaced\":" + (replaced ? "true" : "false") + "}");
    });

    // Encodes raw reference audio (browser mic recording or an uploaded file)
    // into codes with codec_encoder.onnx - same preprocessing as
    // clone_voice.py, but a pure compute call: the codes go back to the caller
    // and the caller decides where to keep them (the web console uses browser
    // storage and sends the codes inline per request).
    svr.Post("/v1/voices/encode", [&st](const httplib::Request & req, httplib::Response & res) {
        std::string name, reference_text, wav_b64;
        if (!json_string(req.body, "name", name) || name.empty()) {
            send_error(res, 400, "missing \"name\"");
            return;
        }
        if (!json_string(req.body, "reference_text", reference_text) || reference_text.empty()) {
            send_error(res, 400, "missing \"reference_text\"");
            return;
        }
        if (!json_string(req.body, "wav_base64", wav_b64) || wav_b64.empty()) {
            send_error(res, 400, "missing \"wav_base64\"");
            return;
        }
        if (st.encoder_path.empty()) {
            send_error(res, 501, "voice encoder not configured (start m5-server with --voice-encoder)");
            return;
        }
        std::string wav, err;
        if (!base64_decode(wav_b64, wav)) {
            send_error(res, 400, "wav_base64 is not valid base64");
            return;
        }
        std::vector<float> mono;
        int rate = 0;
        if (!wav_to_mono(wav, mono, rate, err)) { send_error(res, 400, err); return; }
        if (rate != kSampleRate) linear_resample(mono, rate, kSampleRate);
        if (!read_reference(mono, err)) { send_error(res, 400, err); return; }

        std::vector<std::vector<long>> codes;
        {
            std::lock_guard<std::mutex> lock(st.encode_mutex);
            if (!st.encoder) {
                st.encoder.reset(new VoiceEncoder());
                if (!st.encoder->open(st.encoder_path, err)) {
                    st.encoder.reset();
                    send_error(res, 500, "cannot open voice encoder: " + err);
                    return;
                }
                st.encoder_sha = encoder_identity(st.encoder_path, err);
                if (st.encoder_sha.empty()) {
                    st.encoder.reset();
                    send_error(res, 500, err);
                    return;
                }
            }
            if (!st.encoder->encode(mono, codes, err)) {
                send_error(res, 500, "encode failed: " + err);
                return;
            }
        }
        if (codes.empty()) { send_error(res, 500, "encoder returned no codes"); return; }

        // Pure compute: codes go back to the caller and nothing is written to
        // the clone cache (the web console keeps its clones in the browser).
        std::string out = "{\"name\":\"" + json_escape(name) +
                          "\",\"reference_text\":\"" + json_escape(reference_text) +
                          "\",\"encoder_sha256\":\"" + json_escape(st.encoder_sha) +
                          "\",\"frames\":" + std::to_string(codes.size()) +
                          ",\"codes\":[";
        for (size_t r = 0; r < codes.size(); r++) {
            if (r) out += ",";
            out += "[";
            for (int j = 0; j < kBooks; j++) {
                if (j) out += ",";
                out += std::to_string(codes[r][j]);
            }
            out += "]";
        }
        out += "]}";
        fprintf(stderr, "[m5-server] encoded transient voice %s (%s), %d frames\n",
                name.c_str(), st.encoder_sha.substr(0, 12).c_str(), (int) codes.size());
        send_json(res, 200, out);
    });

    if (!svr.listen(host, port)) {
        fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
