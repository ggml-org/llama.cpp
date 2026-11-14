#include "m5_engine.h"
#include "m5_snake_op.h"

// The depth heads stream ~68 MB of f32 weights per audio frame, which is pure
// memory bandwidth. Keeping those two blocks in f16 halves the traffic with a
// relative error around 5e-4 - well under the Gumbel noise the sampler adds on
// top anyway. Requires -march=armv8.2-a+fp16; other builds fall back to f32.
#if defined(__aarch64__)
#include <arm_neon.h>
// Clang 19 (NDK r28) names these __ARM_FEATURE_FP16_VECTOR_ARITHMETIC and
// __ARM_FEATURE_FP16_FML; older GCC-style spellings are accepted too.
#if defined(__ARM_FEATURE_FP16) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define M5_HEADS_FP16 1
#endif
#else
typedef __fp16 float16_t;
#endif


#include "m5_ring.h"
#include "m5_tokenizer.h"
#include "onnxruntime_c_api.h"

#include "m5_depth_heads.h"
namespace heads = m5::heads;

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace m5 {
namespace {

// Per-run diagnostics for the depth heads, printed by the CLI at the end.
double g_heads_s = 0, g_depth_stage_s = 0;

constexpr int kWidth = 512;
constexpr int kVocab = 2051;
constexpr int kBooks = 16;
constexpr int kRate = 24000;
constexpr int kHop = 1920;
constexpr int kTail = 2400;
constexpr int kMaxFrames = 500;
constexpr int kNoiseBooks = 16;
constexpr const char * kConditionPrefix =
    "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
    "<|speaker_start|>None<|speaker_end|><|lang_begin|>Chinese<|lang_end|>";

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- model files

struct Mapping {
    void * data = nullptr;
    size_t size = 0;
    ~Mapping() { if (data) munmap(data, size); }
    Mapping() = default;
    Mapping(const Mapping &) = delete;
    Mapping & operator=(const Mapping &) = delete;
};

bool map_file(const std::string & path, Mapping & out, size_t expect = 0) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
    if (expect && (size_t) st.st_size != expect) {
        fprintf(stderr, "%s: size %lld != %zu\n", path.c_str(), (long long) st.st_size, expect);
        close(fd);
        return false;
    }
    void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { fprintf(stderr, "mmap failed for %s\n", path.c_str()); return false; }
    out.data = p;
    out.size = (size_t) st.st_size;
    return true;
}

struct Npy {
    Mapping map;
    const float * data = nullptr;
    int rows = 0, cols = 0;
};

bool load_npy(const std::string & path, Npy & out) {
    if (!map_file(path, out.map)) return false;
    const unsigned char * p = static_cast<const unsigned char *>(out.map.data);
    if (out.map.size < 128 || memcmp(p, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a npy file\n", path.c_str());
        return false;
    }
    const int version = p[6];
    size_t header_len = 0, offset = 0;
    if (version == 1) { header_len = p[8] | (p[9] << 8); offset = 10 + header_len; }
    else if (version == 2) { memcpy(&header_len, p + 8, 4); offset = 12 + header_len; }
    else { fprintf(stderr, "%s: unsupported npy version\n", path.c_str()); return false; }
    std::string header(reinterpret_cast<const char *>(p + offset - header_len), header_len);
    if (header.find("'<f4'") == std::string::npos) { fprintf(stderr, "need float32 npy\n"); return false; }
    size_t open_paren = header.find("'shape':");
    if (open_paren == std::string::npos) return false;
    open_paren = header.find('(', open_paren);
    int a = 0, b = 0;
    if (sscanf(header.c_str() + open_paren, "(%d,%d)", &a, &b) != 2 &&
        sscanf(header.c_str() + open_paren, "(%d, %d)", &a, &b) != 2) {
        fprintf(stderr, "%s: cannot parse shape\n", path.c_str());
        return false;
    }
    out.rows = a; out.cols = b;
    if (out.map.size != offset + (size_t) a * b * 4) { fprintf(stderr, "%s: bad npy length\n", path.c_str()); return false; }
    out.data = reinterpret_cast<const float *>(p + offset);
    return true;
}

std::string read_text(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    std::string out;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

long json_int(const std::string & text, const char * key, long fallback) {
    size_t at = text.find(std::string("\"") + key + "\"");
    if (at == std::string::npos) return fallback;
    at = text.find(':', at);
    if (at == std::string::npos) return fallback;
    return strtol(text.c_str() + at + 1, nullptr, 10);
}

// Ordered JSON scanners: the packages are written by our own exporter, so a
// full parser would be more surface than value.
bool parse_int_array(const std::string & text, size_t from, std::vector<long> & out, size_t & end_at) {
    out.clear();
    size_t at = text.find('[', from);
    if (at == std::string::npos) return false;
    at++;
    for (;;) {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\r' ||
                                    text[at] == '\t' || text[at] == ',')) at++;
        if (at >= text.size()) return false;
        if (text[at] == ']') { end_at = at + 1; return true; }
        const size_t begin = at;
        if (text[at] == '-') at++;
        while (at < text.size() && text[at] >= '0' && text[at] <= '9') at++;
        if (at == begin) return false;
        out.push_back(strtol(text.c_str() + begin, nullptr, 10));
        if (out.size() > 8192) return false;
    }
}

size_t skip_string(const std::string & text, size_t at) {
    // `at` points at the opening quote; returns the offset past the closing one.
    for (size_t i = at + 1; i < text.size(); i++) {
        if (text[i] == '\\') { i++; continue; }
        if (text[i] == '"') return i + 1;
    }
    return std::string::npos;
}

std::string read_json_string(const std::string & text, size_t quote, size_t & end_at) {
    const size_t stop = skip_string(text, quote);   // one past the closing quote
    if (stop == std::string::npos) return {};
    end_at = stop;
    return text.substr(quote + 1, stop - quote - 2);
}

size_t match_brace(const std::string & text, size_t open) {
    const char target = text[open] == '{' ? '}' : ']';
    int depth = 0;
    for (size_t i = open; i < text.size(); i++) {
        if (text[i] == '"') { i = skip_string(text, i); if (i == std::string::npos) return std::string::npos; i--; continue; }
        if (text[i] == '{' || text[i] == '[') depth++;
        else if (text[i] == '}' || text[i] == ']') {
            depth--;
            if (depth == 0) return text[i] == target ? i : std::string::npos;
        }
    }
    return std::string::npos;
}

struct Voice {
    std::string name, reference;
    std::vector<std::vector<long>> codes;
};

// `selector` is a voice name, or a decimal index into the voices array.
bool load_voice(const std::string & text, const std::string & selector, Voice & out) {
    size_t at = text.find("\"voices\"");
    if (at == std::string::npos) return false;
    at = text.find('[', at);
    if (at == std::string::npos) return false;
    int wanted_index = -1;
    if (!selector.empty() && selector.find_first_not_of("0123456789") == std::string::npos) {
        wanted_index = std::atoi(selector.c_str());
    }
    size_t pos = at + 1;
    for (int seen = 0;; seen++) {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' ||
                                     text[pos] == '\t' || text[pos] == ',')) pos++;
        if (pos >= text.size() || text[pos] == ']') return false;
        if (text[pos] != '{') return false;
        const size_t close = match_brace(text, pos);
        if (close == std::string::npos) return false;
        const std::string body = text.substr(pos, close - pos + 1);
        // Cheap name probe first so name lookup does not parse the code table.
        std::string name;
        size_t name_at = body.find("\"name\"");
        if (name_at != std::string::npos) {
            size_t q = body.find('"', name_at + 6);
            if (q != std::string::npos) { size_t e = 0; name = read_json_string(body, q, e); }
        }
        const bool match = wanted_index >= 0 ? (seen == wanted_index) : (name == selector);
        if (match) {
            out.name = name;
            size_t ref_at = body.find("\"reference_text\"");
            if (ref_at == std::string::npos) return false;
            size_t q = body.find('"', ref_at + 16);
            if (q == std::string::npos) return false;
            size_t e = 0;
            out.reference = read_json_string(body, q, e);

            size_t codes_at = body.find("\"codes\"");
            if (codes_at == std::string::npos) return false;
            size_t pos_codes = body.find('[', codes_at);
            if (pos_codes == std::string::npos) return false;
            pos_codes++;
            out.codes.clear();
            for (;;) {
                while (pos_codes < body.size() && (body[pos_codes] == ' ' || body[pos_codes] == '\n' ||
                                                   body[pos_codes] == '\r' || body[pos_codes] == '\t' ||
                                                   body[pos_codes] == ',')) pos_codes++;
                if (pos_codes >= body.size()) return false;
                if (body[pos_codes] == ']') break;
                std::vector<long> row;
                size_t row_end = 0;
                if (!parse_int_array(body, pos_codes, row, row_end)) return false;
                if (row.size() != (size_t) kBooks) return false;
                out.codes.push_back(std::move(row));
                pos_codes = row_end;
                if (out.codes.size() > 4096) return false;
            }
            return !out.codes.empty() && !out.reference.empty();
        }
        pos = close + 1;
    }
}

// ------------------------------------------------------------------- java rng

class JavaRandom {
public:
    explicit JavaRandom(long seed) : state_((seed ^ 0x5DEECE66DLL) & ((1LL << 48) - 1)) {}
    double next_double() {
        long hi = next(26), lo = next(27);
        return (double) ((hi << 27) + lo) * (1.0 / (double) (1LL << 53));
    }
private:
    int next(int bits) {
        state_ = (state_ * 0x5DEECE66DLL + 0xB) & ((1LL << 48) - 1);
        return (int) ((unsigned long long) state_ >> (48 - bits));
    }
    long state_;
};

// ------------------------------------------------------------------ processes

class PipeProcess {
public:
    PipeProcess() = default;
    ~PipeProcess() { close(); }
    PipeProcess(const PipeProcess &) = delete;
    PipeProcess & operator=(const PipeProcess &) = delete;

    bool start(const std::string & command, const std::string & cwd) {
        int in[2], out[2];
        if (pipe(in) != 0) return false;
        if (pipe(out) != 0) { ::close(in[0]); ::close(in[1]); return false; }
        pid_ = fork();
        if (pid_ < 0) return false;
        if (pid_ == 0) {
            dup2(in[0], STDIN_FILENO);
            dup2(out[1], STDOUT_FILENO);
            ::close(in[0]); ::close(in[1]); ::close(out[0]); ::close(out[1]);
            if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
            // The stage binaries live next to the model package, not on PATH.
            setenv("PATH", (std::string(".:") + getenv("PATH")).c_str(), 1);
            execl("/system/bin/sh", "sh", "-c", command.c_str(), (char *) nullptr);
            _exit(127);
        }
        ::close(in[0]);
        ::close(out[1]);
        to_child_ = in[1];
        from_child_ = out[0];
        return true;
    }

    void reset() {
        int32_t zero = 0;
        write_all(&zero, sizeof(zero));
    }

    bool run(const float * rows_data, int rows, float * out) {
        int32_t count = rows;
        if (!write_all(&count, sizeof(count))) { fprintf(stderr, "[m5] stage count write failed rows=%d\n", rows); return false; }
        if (!write_all(rows_data, sizeof(float) * kWidth * (size_t) rows)) { fprintf(stderr, "[m5] stage rows write failed rows=%d\n", rows); return false; }
        if (!read_all(out, sizeof(float) * kWidth)) { fprintf(stderr, "[m5] stage reply read failed rows=%d\n", rows); return false; }
        return true;
    }

    // Multi-row variant for the --depth-chain protocol: one request of
    // `in_rows` rows goes out and `out_rows` rows come back.
    bool run_rows(const float * rows_data, int in_rows, float * out, int out_rows) {
        int32_t count = in_rows;
        if (!write_all(&count, sizeof(count))) return false;
        if (!write_all(rows_data, sizeof(float) * kWidth * (size_t) in_rows)) return false;
        return read_all(out, sizeof(float) * kWidth * (size_t) out_rows);
    }

    // Prefix-cached prefill: [shared rows][rest rows] with a hash of the shared
    // block. The stage keeps the shared block's KV across requests and answers
    // with the hidden of the last rest row. count == -1 marks the variant.
    bool run_prefill(const float * shared, int shared_rows, uint32_t hash,
                     const float * rest, int rest_rows, float * out) {
        int32_t magic = -1, sh = shared_rows, rs = rest_rows;
        if (!write_all(&magic, sizeof(magic))) return false;
        if (!write_all(&sh, sizeof(sh))) return false;
        if (!write_all(&hash, sizeof(hash))) return false;
        if (!write_all(&rs, sizeof(rs))) return false;
        if (shared_rows > 0 &&
            !write_all(shared, sizeof(float) * kWidth * (size_t) shared_rows)) return false;
        if (rest_rows > 0 &&
            !write_all(rest, sizeof(float) * kWidth * (size_t) rest_rows)) return false;
        return read_all(out, sizeof(float) * kWidth);
    }

    void close() {
        // Closing the pipes first lets the stage exit on EOF; only escalate if
        // it does not, so a stuck reader can never hang the engine.
        if (to_child_ >= 0) { ::close(to_child_); to_child_ = -1; }
        if (from_child_ >= 0) { ::close(from_child_); from_child_ = -1; }
        if (pid_ <= 0) return;
        for (int i = 0; i < 50; i++) {
            int status = 0;
            if (waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; return; }
            usleep(20000);
        }
        kill(pid_, SIGKILL);
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }

    int pid() const { return pid_; }

private:
    bool write_all(const void * data, size_t bytes) {
        const char * p = static_cast<const char *>(data);
        while (bytes) {
            ssize_t n = write(to_child_, p, bytes);
            if (n <= 0) return false;
            p += n;
            bytes -= (size_t) n;
        }
        return true;
    }
    bool read_all(void * data, size_t bytes) {
        char * p = static_cast<char *>(data);
        size_t want = bytes;
        while (want) {
            ssize_t n = read(from_child_, p, want);
            if (n <= 0) return false;
            p += n;
            want -= (size_t) n;
        }
        return true;
    }

    pid_t pid_ = -1;
    int to_child_ = -1;
    int from_child_ = -1;
};

// One stage process, reachable over shared memory when available and over the
// legacy pipe protocol otherwise. The choice is made once, at start.
class Stage {
public:
    bool open(const std::string & command, const std::string & cwd, int capacity, bool want_ring,
              const char * label) {
        command_ = command;
        cwd_ = cwd;
        capacity_ = capacity;
        label_ = label;
        const bool ok = start(want_ring);
        if (ok) fprintf(stderr, "[m5] %s stage: %s transport\n", label_, ring_ ? "ring" : "pipe");
        return ok;
    }

    bool run(const float * data, int rows, float * out) {
        if (ring_) return client_.call(data, rows, out, process_.pid());
        return process_.run(data, rows, out);
    }

    // Multi-row variant used by the --depth-chain protocol (pipe only).
    bool run_rows(const float * data, int in_rows, float * out, int out_rows) {
        return process_.run_rows(data, in_rows, out, out_rows);
    }

    // Prefix-cached prefill (pipe only; the ring protocol carries one row at a
    // time and has no notion of a retained prefix).
    bool prefill(const float * shared, int shared_rows, uint32_t hash,
                 const float * rest, int rest_rows, float * out) {
        if (ring_) return false;
        return process_.run_prefill(shared, shared_rows, hash, rest, rest_rows, out);
    }
    bool supports_prefill() const { return !ring_; }

    bool reset() {
        if (!ring_) { process_.reset(); return true; }
        return client_.reset(process_.pid());
    }

    bool ring() const { return ring_; }
    void close() {
        client_.close();
        process_.close();
    }

private:
    bool start(bool use_ring) {
        ring_ = false;
        std::string command = command_;
        if (use_ring) {
            const int fd = client_.create(capacity_, kWidth);
            if (fd >= 0) command += " --ring-fd " + std::to_string(fd);
            if (fd < 0) use_ring = false;
        }
        if (!process_.start(command, cwd_)) { client_.close(); return false; }
        if (!use_ring) return true;
        // Probe once: a stage binary that predates the ring flag rejects it and
        // exits, so the acknowledgement never arrives and we fall back.
        if (!client_.reset(process_.pid())) {
            client_.close();
            process_.close();
            return start(false);
        }
        ring_ = true;
        return true;
    }

    std::string command_, cwd_;
    const char * label_ = "stage";
    int capacity_ = 0;
    bool ring_ = false;
    ring::Client client_;
    PipeProcess process_;
};

// ---------------------------------------------------------------- depth heads

class DepthHeads {
public:
    // The --depth-chain protocol: kChainRows rows in (hidden + JavaRandom
    // noise + control float), 2 rows out (16 codes + next hidden). The whole
    // sixteen-book chain runs inside the stage process, so the pipe is crossed
    // once per audio frame instead of sixteen times.
    static constexpr int kChainRows = 66;

    bool open(const std::string & weights_path, const std::string & command,
              const std::string & cwd, bool want_ring) {
        std::string cmd = command;
        if (cmd.find("--depth-chain") != std::string::npos) {
            // The child owns the heads in chain mode; the weights path rides
            // on the command line. Probe it: an older binary rejects the
            // command line and the first exchange fails, which falls back to
            // the per-book protocol below. Ring is not used here - the chain
            // reply is two rows and the ring carries one.
            const std::string chain_cmd = cmd + " " + weights_path;
            if (stage_.open(chain_cmd, cwd, 2, false, "depth") && probe_chain()) {
                chain_ = true;
                return true;
            }
            stage_.close();
            const std::string needle = "--depth-chain";
            const size_t at = cmd.find(needle);
            if (at != std::string::npos) cmd.replace(at, needle.size(), "--depth-pipe");
        }
        if (!stage_.open(cmd, cwd, 2, want_ring, "depth")) return false;
        // Legacy path: the heads run here, so map and halve the weights.
        const size_t floats = heads::layout().audio +
                              (size_t) heads::kBooks * heads::kVocab * heads::kWidth;
        if (!map_file(weights_path, weights_, floats * sizeof(float))) return false;
        w_ = static_cast<const float *>(weights_.data);
        w16_.assign(heads::f16_halves(), (float16_t) 0);
        heads::build_f16(w_, w16_.data(), w16_.size());
        // Per-row int8 mirror of the project blocks: half the f16 weight
        // traffic, heads_s 2.5x faster, RTF -13% (2026-11-14 A/B; codes drift
        // like a different seed but ASR stays clean). Set M5_HEADS_INT8=0 for
        // the f16 path.
        const char * env8 = getenv("M5_HEADS_INT8");
        if (!env8 || std::strcmp(env8, "0") != 0) {
            w8_.assign(heads::int8_rows() * heads::kWidth, (int8_t) 0);
            w8s_.assign(heads::int8_rows(), 1.0f);
            heads::build_int8(w_, w8_.data(), w8s_.data(), heads::int8_rows());
        }
        return true;
    }

    bool reset() { return stage_.reset(); }
    void close() { stage_.close(); }
    bool chain() const { return chain_; }

    // Mirrors NativeDepth.run: returns 16 codes and the accumulated next hidden.
    bool run(const float * hidden, const float * noise, bool allow_eos,
             long codes[kBooks], float * next) {
        if (chain_) {
            const size_t noise_f = (size_t) kNoiseBooks * kVocab;
            frame_.assign((size_t) kChainRows * kWidth, 1.0f);
            memcpy(frame_.data(), hidden, sizeof(float) * kWidth);
            memcpy(frame_.data() + kWidth, noise, sizeof(float) * noise_f);
            frame_[kWidth + noise_f] = allow_eos ? 1.0f : 0.0f;
            reply_.assign(2 * (size_t) kWidth, 0.0f);
            const double t0 = now_s();
            const bool ok = stage_.run_rows(frame_.data(), kChainRows, reply_.data(), 2);
            g_depth_stage_s += now_s() - t0;
            if (!ok) return false;
            for (int j = 0; j < kBooks; j++) codes[j] = (long) reply_[j];
            memcpy(next, reply_.data() + kWidth, sizeof(float) * kWidth);
            return true;
        }

        // Legacy path: sixteen book round trips. The maths lives in
        // m5_depth_heads.h so the chain child shares it verbatim.
        if (!stage_.reset()) return false;
        bool failed = false;
        double stage_s = 0;
        auto decode = [&](const float * in, int rows, float * out) {
            const double t0 = now_s();
            const bool ok = stage_.run(in, rows, out);
            stage_s += now_s() - t0;
            if (!ok) failed = true;
            return ok;
        };
        const double t0 = now_s();
        const bool ok = heads::chain(w_, w16_.empty() ? nullptr : w16_.data(),
                                     hidden, noise, allow_eos, codes, next, decode,
                                     w8_.empty() ? nullptr : w8_.data(),
                                     w8_.empty() ? nullptr : w8s_.data());
        g_heads_s += (now_s() - t0) - stage_s;
        g_depth_stage_s += stage_s;
        return ok && !failed;
    }

private:
    // One dummy frame separates chain-capable children from older builds; the
    // reply is discarded and the child clears its state per frame anyway.
    bool probe_chain() {
        const size_t noise_f = (size_t) kNoiseBooks * kVocab;
        frame_.assign((size_t) kChainRows * kWidth, 1.0f);
        frame_[kWidth + noise_f] = 1.0f;
        reply_.assign(2 * (size_t) kWidth, 0.0f);
        return stage_.run_rows(frame_.data(), kChainRows, reply_.data(), 2);
    }

    Mapping weights_;
    const float * w_ = nullptr;
    // Halves of the projection + projected blocks, built once at open().
    // Empty when the build has no f16 path, which keeps the f32 code intact.
    std::vector<float16_t> w16_;
    // Optional int8 mirror of the same blocks (M5_HEADS_INT8), per-row scaled.
    std::vector<int8_t> w8_;
    std::vector<float> w8s_;
    Stage stage_;
    bool chain_ = false;
    std::vector<float> frame_, reply_;
};

// ----------------------------------------------------------------------- codec

class Codec {
public:
    bool open(const std::string & model, int threads) {
        api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!api_) return false;
        check(api_->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "m5-engine", &env_));
        check(api_->CreateSessionOptions(&so_));
        check(api_->SetIntraOpNumThreads(so_, threads));
        check(api_->SetInterOpNumThreads(so_, 1));
        check(api_->SetSessionGraphOptimizationLevel(so_, ORT_ENABLE_ALL));
        // Fused Snake activation (m5_snake_op.h); models that predate the
        // rewrite keep working - unused custom ops are harmless.
        if (!m5::snake::register_ops(api_, so_)) {
            fprintf(stderr, "ort: cannot register m5::Snake\n");
        }
        // M5_CODEC_PROFILE=<prefix> writes an ORT op-level profile on close().
        const char * prof = getenv("M5_CODEC_PROFILE");
        fprintf(stderr, "[m5] codec open: M5_CODEC_PROFILE=%s\n", prof ? prof : "(unset)");
        if (prof && *prof) {
            check(api_->EnableProfiling(so_, prof));
            profiling_ = true;
        }
        check(api_->CreateSession(env_, model.c_str(), so_, &session_));

        size_t n_in = 0, n_out = 0;
        check(api_->SessionGetInputCount(session_, &n_in));
        check(api_->SessionGetOutputCount(session_, &n_out));
        check(api_->GetAllocatorWithDefaultOptions(&alloc_));
        check(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_));

        inputs_.resize(n_in);
        for (size_t i = 0; i < n_in; i++) {
            char * name = nullptr;
            check(api_->SessionGetInputName(session_, i, alloc_, &name));
            Input & in = inputs_[i];
            in.name = name;
            OrtTypeInfo * info = nullptr;
            check(api_->SessionGetInputTypeInfo(session_, i, &info));
            const OrtTensorTypeAndShapeInfo * tsi = nullptr;
            check(api_->CastTypeInfoToTensorInfo(info, &tsi));
            check(api_->GetTensorElementType(tsi, &in.type));
            size_t rank = 0;
            check(api_->GetDimensionsCount(tsi, &rank));
            in.dims.resize(rank);
            check(api_->GetDimensions(tsi, in.dims.data(), rank));
            api_->ReleaseTypeInfo(info);
            for (auto & d : in.dims) if (d < 0) d = 0;
            in.elems = count(in.dims);
            in.initial_dims = in.dims;
            const size_t width = in.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4;
            in.buffer.assign(std::max<size_t>(in.elems, 1) * width, 0);
            if (in.name == "codes") codes_ = (int) i;
            if (in.name == "offset") offset_ = (int) i;
        }
        output_names_.resize(n_out);
        out_to_in_.assign(n_out, -1);
        for (size_t o = 0; o < n_out; o++) {
            char * name = nullptr;
            check(api_->SessionGetOutputName(session_, o, alloc_, &name));
            output_names_[o] = name;
            std::string want = output_names_[o];
            if (want == "present") want = "past";
            else if (want.rfind("next_", 0) == 0) want = want.substr(5);
            for (size_t i = 0; i < n_in; i++) if (inputs_[i].name == want) out_to_in_[o] = (int) i;
        }
        if (codes_ < 0 || offset_ < 0) return false;
        outputs_.assign(n_out, nullptr);
        names_.resize(inputs_.size());
        for (size_t i = 0; i < inputs_.size(); i++) names_[i] = inputs_[i].name.c_str();
        output_cstr_.resize(output_names_.size());
        for (size_t i = 0; i < output_names_.size(); i++) output_cstr_[i] = output_names_[i].c_str();
        const char * cap = getenv("M5_PRIME_CACHE");
        prime_capacity_ = cap ? (size_t) std::max(0, atoi(cap)) : 4;
        prime_stats_ = getenv("M5_PRIME_STATS") != nullptr;
        prime_cache_.clear();
        prime_hits_ = prime_misses_ = 0;
        fprintf(stderr, "[m5] prime cache: capacity=%zu (M5_PRIME_CACHE=1 restores the old single-entry cache)\n",
                prime_capacity_);
        opened_ = true;
        return true;
    }

    // Returns the codec to its initial state so a segment starts from silence.
    void reset_stream() {
        for (size_t i = 0; i < inputs_.size(); i++) {
            Input & in = inputs_[i];
            if ((int) i == codes_ || (int) i == offset_) continue;
            in.dims = in.initial_dims;
            in.elems = count(in.dims);
            const size_t width = in.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4;
            in.buffer.assign(std::max<size_t>(in.elems, 1) * width, 0);
        }
        offset_value_ = 0;
        *reinterpret_cast<int64_t *>(inputs_[offset_].buffer.data()) = 0;
    }

    // The prime state is a pure function of the eight reference frames, so
    // keying the cache by their codes (not by a voice name/index) also covers
    // the browser's inline temporary voices and makes a repeated clone free.
    // The server serialises renders and both calls run on the codec worker,
    // so the cache needs no lock of its own.
    bool restore_prime(const std::vector<long> & codes) {
        auto it = std::find_if(prime_cache_.begin(), prime_cache_.end(),
                               [&](const PrimeEntry & e) { return e.codes == codes; });
        if (it == prime_cache_.end() || it->inputs.size() != inputs_.size()) {
            prime_misses_++;
            if (prime_stats_) fprintf(stderr, "[m5] prime miss entries=%zu\n", prime_cache_.size());
            return false;
        }
        const PrimeEntry entry = *it;  // copy before rotating the LRU
        prime_cache_.erase(it);
        prime_cache_.insert(prime_cache_.begin(), entry);
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if ((int) i == codes_ || (int) i == offset_) continue;
            inputs_[i].dims = entry.inputs[i].dims;
            inputs_[i].elems = entry.inputs[i].elems;
            inputs_[i].buffer = entry.inputs[i].buffer;
        }
        offset_value_ = entry.offset;
        prime_hits_++;
        if (prime_stats_) fprintf(stderr, "[m5] prime hit entries=%zu hits=%llu\n",
                                  prime_cache_.size(), prime_hits_);
        return true;
    }

    void save_prime(const std::vector<long> & codes) {
        if (prime_capacity_ == 0) return;  // M5_PRIME_CACHE=0 disables the cache
        auto it = std::find_if(prime_cache_.begin(), prime_cache_.end(),
                               [&](const PrimeEntry & e) { return e.codes == codes; });
        if (it != prime_cache_.end()) prime_cache_.erase(it);
        PrimeEntry entry;
        entry.codes = codes;
        entry.inputs = inputs_;
        entry.offset = offset_value_;
        prime_cache_.insert(prime_cache_.begin(), std::move(entry));
        if (prime_cache_.size() > prime_capacity_) prime_cache_.resize(prime_capacity_);
        if (prime_stats_) {
            size_t bytes = 0;
            for (const auto & e : prime_cache_)
                for (const auto & in : e.inputs) bytes += in.buffer.size();
            fprintf(stderr, "[m5] prime save entries=%zu bytes=%zu misses=%llu\n",
                    prime_cache_.size(), bytes, prime_misses_);
        }
    }

    // Runs one chunk and fills `pcm` with its samples.
    bool run(const long * frames, int count, std::vector<float> & pcm, std::string & error) {
        auto & codes = inputs_[codes_];
        codes.dims[2] = count;
        codes.elems = (size_t) count * kBooks;
        codes.buffer.assign(codes.elems * 8, 0);
        auto * dst = reinterpret_cast<int64_t *>(codes.buffer.data());
        for (int k = 0; k < kBooks; k++)
            for (int t = 0; t < count; t++) dst[k * count + t] = frames[t * kBooks + k];

        *reinterpret_cast<int64_t *>(inputs_[offset_].buffer.data()) = offset_value_;
        offset_value_ += count;

        std::vector<OrtValue *> in_values(inputs_.size(), nullptr);
        for (size_t i = 0; i < inputs_.size(); i++) {
            const size_t width = inputs_[i].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4;
            if (!create_tensor(&in_values[i], inputs_[i].buffer.data(), inputs_[i].elems * width,
                               inputs_[i].dims, inputs_[i].type, error)) return false;
        }
        OrtStatus * status = api_->Run(session_, nullptr, names_.data(),
                                       (const OrtValue * const *) in_values.data(), in_values.size(),
                                       output_names_c(), outputs_.size(), outputs_.data());
        for (auto * v : in_values) api_->ReleaseValue(v);
        if (status) { error = api_->GetErrorMessage(status); api_->ReleaseStatus(status); return false; }

        size_t elems = 0;
        if (!shape_of(outputs_[0], elems, error)) return false;
        float * data = nullptr;
        if (!mutable_data(outputs_[0], (void **) &data, error)) return false;
        pcm.assign(data, data + elems);

        for (size_t o = 1; o < outputs_.size(); o++) {
            const int target = out_to_in_[o];
            if (target < 0) continue;
            size_t state_elems = 0;
            std::vector<int64_t> shape;
            if (!shape_of(outputs_[o], state_elems, error, &shape)) return false;
            if (state_elems == 0) continue;
            auto & in = inputs_[target];
            in.elems = state_elems;
            in.dims = shape;
            in.buffer.assign(state_elems * sizeof(float), 0);
            float * state = nullptr;
            if (!mutable_data(outputs_[o], (void **) &state, error)) return false;
            memcpy(in.buffer.data(), state, state_elems * sizeof(float));
        }
        for (auto & v : outputs_) { if (v) api_->ReleaseValue(v); v = nullptr; }
        return true;
    }

    bool enable_profiling(const std::string & prefix, std::string & error) {
        OrtStatus * status = api_->EnableProfiling(so_, prefix.c_str());
        if (status) { error = api_->GetErrorMessage(status); api_->ReleaseStatus(status); return false; }
        profiling_ = true;
        return true;
    }

    std::string end_profiling() {
        if (!profiling_) return {};
        char * path = nullptr;
        OrtStatus * status = api_->SessionEndProfiling(session_, alloc_, &path);
        if (status) { api_->ReleaseStatus(status); return {}; }
        std::string out = path ? path : "";
        return out;
    }

    void close() {
        if (!api_) return;
        if (session_ && profiling_) {
            std::string path = end_profiling();
            fprintf(stderr, "[m5] codec profile: %s\n", path.empty() ? "(none)" : path.c_str());
        }
        if (session_) { api_->ReleaseSession(session_); session_ = nullptr; }
        if (so_) { api_->ReleaseSessionOptions(so_); so_ = nullptr; }
        if (mem_) { api_->ReleaseMemoryInfo(mem_); mem_ = nullptr; }
        if (env_) { api_->ReleaseEnv(env_); env_ = nullptr; }
        if (prime_stats_)
            fprintf(stderr, "[m5] prime cache final: hits=%llu misses=%llu entries=%zu\n",
                    prime_hits_, prime_misses_, prime_cache_.size());
        inputs_.clear();
        prime_cache_.clear();
        outputs_.clear();
        opened_ = false;
    }
    bool loaded() const { return opened_; }

private:
    static size_t count(const std::vector<int64_t> & dims) {
        size_t n = 1;
        for (int64_t d : dims) n *= (size_t) (d < 0 ? 1 : d);
        return n;
    }
    void check(OrtStatus * status) {
        if (!status) return;
        fprintf(stderr, "ort: %s\n", api_->GetErrorMessage(status));
        api_->ReleaseStatus(status);
        exit(1);
    }
    bool create_tensor(OrtValue ** out, void * data, size_t bytes, const std::vector<int64_t> & dims,
                       ONNXTensorElementDataType type, std::string & error) {
        OrtStatus * status = api_->CreateTensorWithDataAsOrtValue(mem_, data, bytes, dims.data(),
                                                                  dims.size(), type, out);
        if (!status) return true;
        error = api_->GetErrorMessage(status);
        api_->ReleaseStatus(status);
        return false;
    }
    bool shape_of(OrtValue * value, size_t & elems, std::string & error, std::vector<int64_t> * dims = nullptr) {
        OrtTensorTypeAndShapeInfo * info = nullptr;
        OrtStatus * status = api_->GetTensorTypeAndShape(value, &info);
        if (status) { error = api_->GetErrorMessage(status); api_->ReleaseStatus(status); return false; }
        (void) api_->GetTensorShapeElementCount(info, &elems);
        if (dims) {
            size_t rank = 0;
            (void) api_->GetDimensionsCount(info, &rank);
            dims->assign(rank, 0);
            (void) api_->GetDimensions(info, dims->data(), rank);
        }
        api_->ReleaseTensorTypeAndShapeInfo(info);
        return true;
    }
    bool mutable_data(OrtValue * value, void ** data, std::string & error) {
        OrtStatus * status = api_->GetTensorMutableData(value, data);
        if (!status) return true;
        error = api_->GetErrorMessage(status);
        api_->ReleaseStatus(status);
        return false;
    }
    const char ** output_names_c() { return output_cstr_.data(); }

    struct Input {
        std::string name;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        std::vector<int64_t> dims;
        std::vector<int64_t> initial_dims;
        size_t elems = 0;
        std::vector<char> buffer;
    };

    // One completed prime state. front() is most-recently-used.
    struct PrimeEntry {
        std::vector<long> codes;
        std::vector<Input> inputs;
        int64_t offset = 0;
    };

    const OrtApi * api_ = nullptr;
    OrtEnv * env_ = nullptr;
    OrtSessionOptions * so_ = nullptr;
    OrtSession * session_ = nullptr;
    OrtAllocator * alloc_ = nullptr;
    OrtMemoryInfo * mem_ = nullptr;
    std::vector<Input> inputs_;
    std::vector<PrimeEntry> prime_cache_;
    size_t prime_capacity_ = 4;  // M5_PRIME_CACHE=<n>; 1 reproduces the old single-entry behaviour
    unsigned long long prime_hits_ = 0, prime_misses_ = 0;
    bool prime_stats_ = false;   // M5_PRIME_STATS=1 logs hit/miss/size
    std::vector<std::string> output_names_;
    std::vector<const char *> output_cstr_;
    std::vector<int> out_to_in_;
    std::vector<OrtValue *> outputs_;
    std::vector<const char *> names_;
    int codes_ = -1, offset_ = -1;
    int64_t offset_value_ = 0;
    bool profiling_ = false;
    bool opened_ = false;
};

// ------------------------------------------------------------------------ wav

class WavWriter {
public:
    // File mode (default): writes the WAV container to `path`.
    bool open(const std::string & path) {
        samples_ = 0;
        memory_.clear();
        memory_mode_ = false;
        discard_ = false;
        file_ = fopen(path.c_str(), "wb");
        if (!file_) return false;
        std::vector<char> blank(44, 0);
        fwrite(blank.data(), 1, blank.size(), file_);
        return true;
    }

    // Memory mode: the container is assembled in RAM (no temp file).
    void open_memory() {
        close();
        samples_ = 0;
        memory_.clear();
        memory_mode_ = true;
        discard_ = false;
    }

    // Discard mode: samples are counted for duration but never stored.
    void open_discard() {
        close();
        samples_ = 0;
        memory_.clear();
        memory_mode_ = false;
        discard_ = true;
    }

    void append(const float * pcm, size_t n) {
        std::vector<int16_t> out(n);
        for (size_t i = 0; i < n; i++) {
            float v = pcm[i];
            if (!std::isfinite(v)) { fprintf(stderr, "non-finite pcm\n"); v = 0; }
            v = std::min(32767.0f / 32768.0f, std::max(-1.0f, v));
            out[i] = (int16_t) floorf(v * 32768.0f);
        }
        samples_ += n;
        if (discard_) return;
        if (memory_mode_) {
            memory_.append(reinterpret_cast<const char *>(out.data()), n * 2);
            return;
        }
        fwrite(out.data(), 2, n, file_);
    }

    void close() {
        if (memory_mode_) {
            memory_mode_ = false;
            memory_ = header() + memory_;
            return;
        }
        if (!file_) return;
        const uint32_t data_bytes = (uint32_t) (samples_ * 2);
        uint32_t riff = 36 + data_bytes, fmt = 16, rate = kRate, byte_rate = kRate * 2;
        uint16_t one = 1, bits = 16, block = 2, tag = 1;
        fseek(file_, 0, SEEK_SET);
        fwrite("RIFF", 1, 4, file_); fwrite(&riff, 4, 1, file_);
        fwrite("WAVEfmt ", 1, 8, file_);
        fwrite(&fmt, 4, 1, file_); fwrite(&tag, 2, 1, file_); fwrite(&one, 2, 1, file_);
        fwrite(&rate, 4, 1, file_); fwrite(&byte_rate, 4, 1, file_);
        fwrite(&block, 2, 1, file_); fwrite(&bits, 2, 1, file_);
        fwrite("data", 1, 4, file_); fwrite(&data_bytes, 4, 1, file_);
        fclose(file_);
        file_ = nullptr;
    }

    const std::string & bytes() const { return memory_; }
    double duration() const { return samples_ / (double) kRate; }

private:
    std::string header() const {
        std::string h(44, '\0');
        const uint32_t data_bytes = (uint32_t) (samples_ * 2);
        uint32_t riff = 36 + data_bytes, fmt = 16, rate = kRate, byte_rate = kRate * 2;
        uint16_t one = 1, bits = 16, block = 2, tag = 1;
        char * p = &h[0];
        memcpy(p, "RIFF", 4); memcpy(p + 4, &riff, 4);
        memcpy(p + 8, "WAVEfmt ", 8);
        memcpy(p + 16, &fmt, 4); memcpy(p + 20, &tag, 2); memcpy(p + 22, &one, 2);
        memcpy(p + 24, &rate, 4); memcpy(p + 28, &byte_rate, 4);
        memcpy(p + 32, &block, 2); memcpy(p + 34, &bits, 2);
        memcpy(p + 36, "data", 4); memcpy(p + 40, &data_bytes, 4);
        return h;
    }

    FILE * file_ = nullptr;
    size_t samples_ = 0;
    bool memory_mode_ = false;
    bool discard_ = false;
    std::string memory_;
};

// UTF-8 split into segments, mirroring M5Engine.splitText.
std::vector<std::string> split_text(const std::string & text) {
    std::vector<std::string> out;
    std::string part;
    int count = 0;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = (unsigned char) text[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        if (i + len > text.size()) len = text.size() - i;
        std::string cp = text.substr(i, len);
        i += len;
        part += cp;
        count++;
        const bool break_here = cp == "\xE3\x80\x82" || cp == "\xEF\xBC\x81" || cp == "\xEF\xBC\x9F" ||
                                cp == "!" || cp == "?" || cp == ";" || cp == "\xEF\xBC\x9B" || cp == "\n";
        if (count >= 80 || break_here) {
            size_t b = part.find_first_not_of(" \t\r\n");
            size_t e = part.find_last_not_of(" \t\r\n");
            if (b != std::string::npos) out.push_back(part.substr(b, e - b + 1));
            part.clear();
            count = 0;
        }
    }
    size_t b = part.find_first_not_of(" \t\r\n");
    size_t e = part.find_last_not_of(" \t\r\n");
    if (b != std::string::npos) out.push_back(part.substr(b, e - b + 1));
    return out;
}

int codepoint_count(const std::string & text) {
    int n = 0;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = (unsigned char) text[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        i += std::min(len, text.size() - i);
        n++;
    }
    return n;
}

std::string trim(const std::string & text) {
    size_t b = text.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = text.find_last_not_of(" \t\r\n");
    return text.substr(b, e - b + 1);
}

}  // namespace


// -------------------------------------------------------------------- engine

namespace {

// Render stats live in atomics: the codec worker updates its own share while
// the main thread is still emitting frames.
struct Timings {
    double start = now_s();
    std::atomic<double> backbone{0}, depth{0}, codec{0}, first_pcm{0};
    std::atomic<int> frames{0};

    void reset() {
        start = now_s();
        backbone.store(0);
        depth.store(0);
        codec.store(0);
        first_pcm.store(0);
        frames.store(0);
    }
};

}  // namespace

struct Engine::Impl {
    Options options;
    Voice voice;
    std::string voices_path;
    // mtime-keyed cache of the small clone cache file; set_voice re-reads it
    // only when the file actually changed (the server upserts it at runtime).
    std::string voices_cache_text;
    std::string voices_cache_path;
    long voices_cache_mtime = -1;
    Npy text_emb, audio_emb;
    Tokenizer tokenizer;
    Stage backbone;
    DepthHeads depth;
    Codec codec;
    Timings timings;
    std::unique_ptr<JavaRandom> rng;
    int min_frames = 6;
    int max_sequence = 2048;
    bool opened = false;
    std::string dump_codes_path;

    ~Impl() { close(); }

    void close() {
        codec.close();
        depth.close();
        backbone.close();
        opened = false;
    }

    bool segment(const std::vector<long> & tokens, WavWriter & wav, const PcmSink & sink,
                 std::atomic<bool> & stopped, std::string & error);
};

bool Engine::Impl::segment(const std::vector<long> & tokens, WavWriter & wav, const PcmSink & sink,
                           std::atomic<bool> & stopped, std::string & error) {
    const int length = (int) tokens.size() + (int) voice.codes.size();
    if (length + kMaxFrames > max_sequence) {
        error = "text or reference voice is too long";
        return false;
    }

    // The conditioning prefix (markers + caption + reference text) is identical
    // for every request on this voice. When the byte-level BPE splits cleanly -
    // the shared ids are a prefix of the full sequence - the stage can keep that
    // block's KV and only prefill the new text plus the reference audio codes.
    std::string shared_text = std::string("<|caption_begin|>") + trim(options.caption) +
                              "<|caption_end|><|accent_begin|><|accent_end|>"
                              "<|speaker_start|>None<|speaker_end|><|lang_begin|>Chinese<|lang_end|>" +
                              trim(voice.reference);
    int shared_len = 0;
    if (backbone.supports_prefill()) {
        // Byte-level BPE can merge the reference's last characters with the
        // first character of the text, so the shared boundary is pinned two
        // tokens before the end of the reference block: those ids depend only
        // on the voice (left context), not on the text, so the hash is stable
        // across requests and the warmup primed block keeps hitting.
        const std::vector<long> shared_ids = tokenizer.encode(shared_text);
        const size_t want = shared_ids.size() >= 3 ? shared_ids.size() - 2 : 0;
        size_t n = 0;
        while (n < want && n < tokens.size() && shared_ids[n] == tokens[n]) n++;
        shared_len = (int) n;
        if (getenv("M5_PREFIX_STATS"))
            fprintf(stderr, "[m5] prefix shared=%zu of %zu (full %zu)\n",
                    n, shared_ids.size(), tokens.size());
    }

    // Row layout: [shared text][segment text][reference audio codes].
    const size_t rest_text = tokens.size() - (size_t) shared_len;
    if (getenv("M5_PREFIX_STATS"))
        fprintf(stderr, "[m5] prefix shared=%d text=%zu codes=%zu\n",
                shared_len, rest_text, voice.codes.size());
    std::vector<float> shared_rows((size_t) shared_len * kWidth, 0.0f);
    std::vector<float> rest_rows((rest_text + voice.codes.size()) * kWidth, 0.0f);
    for (size_t i = 0; i < tokens.size(); i++) {
        const long id = tokens[i];
        if (id < 0 || id >= text_emb.rows) { error = "token id out of range"; return false; }
        float * dst = i < (size_t) shared_len
                          ? shared_rows.data() + i * kWidth
                          : rest_rows.data() + (i - (size_t) shared_len) * kWidth;
        memcpy(dst, text_emb.data + (size_t) id * kWidth, sizeof(float) * kWidth);
    }
    for (size_t r = 0; r < voice.codes.size(); r++) {
        float * dst = rest_rows.data() + (rest_text + r) * kWidth;
        for (int j = 0; j < kBooks; j++) {
            const long row = voice.codes[r][j] + 1 + j * kVocab;
            if (row < 0 || row >= audio_emb.rows) { error = "audio token out of range"; return false; }
            const float * src = audio_emb.data + (size_t) row * kWidth;
            for (int k = 0; k < kWidth; k++) dst[k] += src[k];
        }
    }

    // Every segment restarts the codec stream; the backbone runs the
    // prefix-cached prefill below, which doubles as its reset.
    codec.reset_stream();

    std::mutex m;
    std::condition_variable cv;
    bool has_pending = false, stop_worker = false;
    struct Batch { std::vector<std::vector<long>> frames; bool emit; };
    Batch pending;
    std::vector<float> tail;
    bool first_chunk = true;

    std::mutex failure_mutex;
    std::string failure_text;
    std::atomic<int> failure{0};

    auto write_chunk = [&](std::vector<std::vector<long>> batch, bool emit) {
        std::vector<long> flat;
        flat.reserve(batch.size() * kBooks);
        for (auto & frame : batch) flat.insert(flat.end(), frame.begin(), frame.end());
        if (!emit && codec.restore_prime(flat)) return;
        std::vector<float> pcm;
        std::string chunk_error;
        const double t0 = now_s();
        const bool ok = codec.run(flat.data(), (int) batch.size(), pcm, chunk_error);
        timings.codec.store(timings.codec.load() + (now_s() - t0));
        if (!ok) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            failure_text = chunk_error;
            failure.store(1);
            return;
        }
        if (pcm.size() != batch.size() * (size_t) kHop) {
            failure.store(2);
            return;
        }
        if (!emit) {
            codec.save_prime(flat);
            return;
        }
        std::vector<float> joined(tail);
        joined.insert(joined.end(), pcm.begin(), pcm.end());
        const size_t drop = joined.size() > (size_t) kTail ? joined.size() - kTail : 0;
        if (drop > 0) {
            wav.append(joined.data(), drop);
            if (sink) sink(joined.data(), drop);
            if (timings.first_pcm.load() == 0) timings.first_pcm.store(now_s() - timings.start);
        }
        tail.assign(joined.begin() + drop, joined.end());
    };

    const size_t first_reference = voice.codes.size() > 8 ? voice.codes.size() - 8 : 0;
    std::vector<std::vector<long>> prime(voice.codes.begin() + (long) first_reference, voice.codes.end());

    std::thread worker;
    if (options.overlap) {
        worker = std::thread([&] {
            std::unique_lock<std::mutex> lock(m);
            for (;;) {
                cv.wait(lock, [&] { return has_pending || stop_worker; });
                if (stop_worker && !has_pending) break;
                auto batch = std::move(pending);
                pending = {};
                has_pending = false;
                // The producer waits on !has_pending; it must be woken when the
                // batch is taken, not only when a new one is queued.
                cv.notify_all();
                lock.unlock();
                write_chunk(std::move(batch.frames), batch.emit);
                lock.lock();
            }
        });
        std::lock_guard<std::mutex> lock(m);
        pending = { prime, false };  // advance the codec state, discard its audio
        has_pending = true;
        cv.notify_one();
    } else {
        write_chunk(prime, false);
    }

    std::vector<std::vector<long>> frames;
    long codes[kBooks];
    float hidden[kWidth] = {0};
    float next_hidden[kWidth] = {0};
    std::vector<float> noise((size_t) kNoiseBooks * kVocab);

    // Prefix-cached prefill: the stage keeps the shared block's KV when the
    // hash matches, so repeat requests on the same voice only prefill the new
    // text and the reference audio codes.
    {
        const double b0 = now_s();
        bool primed = false;
        const int rest_len = (int) (rest_text + voice.codes.size());
        if (backbone.supports_prefill()) {
            uint32_t hash = 2166136261u;
            const uint8_t * bytes = reinterpret_cast<const uint8_t *>(shared_rows.data());
            for (size_t i = 0; i < shared_rows.size() * sizeof(float); i++) {
                hash = (hash ^ bytes[i]) * 16777619u;
            }
            primed = backbone.prefill(shared_rows.data(), shared_len, hash,
                                      rest_rows.data(), rest_len, hidden);
        } else {
            if (!backbone.reset()) { error = "backbone reset failed"; return false; }
            std::vector<float> all;
            all.reserve(shared_rows.size() + rest_rows.size());
            all.insert(all.end(), shared_rows.begin(), shared_rows.end());
            all.insert(all.end(), rest_rows.begin(), rest_rows.end());
            primed = backbone.run(all.data(), (int) (shared_len + rest_len), hidden);
        }
        timings.backbone.store(timings.backbone.load() + (now_s() - b0));
        if (!primed) { error = "backbone stage failed"; return false; }
    }

    FILE * codes_file = nullptr;
    if (!dump_codes_path.empty()) codes_file = fopen(dump_codes_path.c_str(), "ab");

    bool ok = true;
    for (long step = 0; step < kMaxFrames; step++) {
        if (stopped.load()) { error = "cancelled"; ok = false; break; }
        if (step > 0) {
            const double b0 = now_s();
            if (!backbone.run(next_hidden, 1, hidden)) { error = "backbone stage failed"; ok = false; break; }
            timings.backbone.store(timings.backbone.load() + (now_s() - b0));
        }

        for (auto & n : noise) {
            const double u = 1.0 - rng->next_double();
            n = std::max((float) -log(u), (float) 1.17549435e-38f);
        }
        const double d0 = now_s();
        if (!depth.run(hidden, noise.data(), step > min_frames, codes, next_hidden)) {
            error = "depth stage failed";
            ok = false;
            break;
        }
        timings.depth.store(timings.depth.load() + (now_s() - d0));
        if (failure.load()) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            error = failure.load() == 1 ? failure_text : "codec length mismatch";
            ok = false;
            break;
        }

        std::vector<long> frame(kBooks);
        bool eos = true;
        for (int j = 0; j < kBooks; j++) {
            eos = eos && codes[j] == 0;
            frame[j] = std::max(0L, std::min(2047L, codes[j] - 1));
        }
        if (eos) {
            if (step == 0) { error = "model ended immediately"; ok = false; }
            break;
        }
        timings.frames.store(timings.frames.load() + 1);
        if (options.verbose && step % 10 == 0) {
            fprintf(stderr, "[m5] step %ld t=%.2fs bb=%.2f dp=%.2f cd=%.2f\n", step,
                    now_s() - timings.start, timings.backbone.load(), timings.depth.load(),
                    timings.codec.load());
        }
        if (codes_file) fwrite(codes, sizeof(long), kBooks, codes_file);
        frames.push_back(frame);

        const size_t limit = first_chunk ? (size_t) options.first_frames : (size_t) options.chunk_frames;
        if (frames.size() == limit) {
            if (options.overlap) {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&] { return !has_pending; });
                pending = { std::move(frames), true };
                has_pending = true;
                frames.clear();
                lock.unlock();
                cv.notify_one();
            } else {
                write_chunk(std::move(frames), true);
                frames.clear();
            }
            first_chunk = false;
        }
    }

    if (ok && !frames.empty()) {
        if (options.overlap) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return !has_pending; });
            pending = { std::move(frames), true };
            has_pending = true;
            lock.unlock();
            cv.notify_one();
        } else {
            write_chunk(std::move(frames), true);
        }
    }
    if (options.overlap) {
        {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return !has_pending; });
            stop_worker = true;
        }
        cv.notify_one();
        worker.join();
    }
    if (codes_file) fclose(codes_file);
    if (failure.load() && ok) {
        std::lock_guard<std::mutex> lock(failure_mutex);
        error = failure.load() == 1 ? failure_text : "codec length mismatch";
        ok = false;
    }
    return ok;
}

Engine::Engine() : impl_(new Impl) {}
Engine::~Engine() = default;

bool Engine::open(const Options & options, std::string & error) {
    Impl & impl = *impl_;
    impl.options = options;
    impl.dump_codes_path = options.dump_codes;
    const std::string root = options.model_dir;

    const std::string manifest = read_text(root + "/manifest.json");
    const std::string voices_path = options.voices_path.empty() ? root + "/voices.json" : options.voices_path;
    const std::string voices_text = read_text(voices_path);
    if (manifest.empty() || voices_text.empty()) { error = "model package is incomplete"; return false; }
    impl.voices_path = voices_path;
    impl.min_frames = (int) json_int(manifest, "min_frames", 6);
    impl.max_sequence = (int) json_int(manifest, "max_sequence", 2048);

    const std::string selector = options.voice_name.empty() ? std::to_string(options.voice)
                                                            : options.voice_name;
    if (!load_voice(voices_text, selector, impl.voice)) {
        error = "cannot read voice " + selector;
        return false;
    }
    if (!load_npy(root + "/text_embeddings.npy", impl.text_emb) ||
        !load_npy(root + "/audio_embeddings.npy", impl.audio_emb)) {
        error = "cannot load embeddings";
        return false;
    }
    if (impl.text_emb.cols != kWidth || impl.audio_emb.cols != kWidth) {
        error = "embedding width mismatch";
        return false;
    }
    if (!impl.tokenizer.load(root + "/tokenizer.bin", error)) return false;
    if (!options.stages) { impl.opened = true; return true; }

    // The backbone prefix can be up to max_sequence rows; the depth core never
    // sees more than two, so the ring sizes differ.
    if (!impl.backbone.open(options.backbone_cmd, options.workdir, impl.max_sequence, options.ring, "backbone")) {
        error = "cannot start the backbone stage";
        return false;
    }
    if (!impl.depth.open(root + "/depth-weights.f32", options.depth_cmd, options.workdir, options.ring)) {
        error = "cannot start the depth stage";
        return false;
    }
    if (!impl.codec.open(root + "/" + options.codec_model, options.threads)) {
        error = "cannot load the codec model";
        return false;
    }
    impl.opened = true;
    return true;
}

bool Engine::is_open() const { return impl_->opened; }
void Engine::close() { impl_->close(); }

std::string Engine::condition(const std::string & text) const {
    const std::string combined = trim(impl_->voice.reference) + trim(text);
    const std::string caption = trim(impl_->options.caption);
    static const char * kReserved[] = {"<|caption_begin|>", "<|caption_end|>", "<|accent_begin|>",
                                       "<|accent_end|>", "<|speaker_start|>", "<|speaker_end|>",
                                       "<|lang_begin|>", "<|lang_end|>"};
    for (const char * token : kReserved)
        if (combined.find(token) != std::string::npos) return {};
    for (const char * token : kReserved)
        if (caption.find(token) != std::string::npos) return {};
    return std::string("<|caption_begin|>") + caption +
           "<|caption_end|><|accent_begin|><|accent_end|>"
           "<|speaker_start|>None<|speaker_end|><|lang_begin|>Chinese<|lang_end|>" + combined;
}

void Engine::set_caption(const std::string & caption) {
    impl_->options.caption = caption;
}

void Engine::set_chunking(int first_frames, int chunk_frames) {
    if (first_frames > 0) impl_->options.first_frames = first_frames;
    if (chunk_frames > 0) impl_->options.chunk_frames = chunk_frames;
}

std::vector<std::vector<long>> Engine::encode_segments(const std::string & text) const {
    std::vector<std::vector<long>> out;
    for (const std::string & segment : split_text(text)) {
        const std::string conditioned = condition(segment);
        if (conditioned.empty()) return {};
        out.push_back(impl_->tokenizer.encode(conditioned));
    }
    return out;
}

bool Engine::set_voice(const std::string & selector, std::string & error) {
    Impl & impl = *impl_;
    if (!impl.opened) { error = "engine is not open"; return false; }
    struct stat st {};
    long mtime = -1;
    if (stat(impl.voices_path.c_str(), &st) == 0) mtime = (long) st.st_mtime;
    if (impl.voices_cache_text.empty() || impl.voices_cache_path != impl.voices_path ||
        impl.voices_cache_mtime != mtime) {
        impl.voices_cache_text = read_text(impl.voices_path);
        impl.voices_cache_path = impl.voices_path;
        impl.voices_cache_mtime = mtime;
    }
    const std::string & text = impl.voices_cache_text;
    if (text.empty()) { error = "cannot read " + impl.voices_path; return false; }
    Voice voice;
    if (!load_voice(text, selector, voice)) {
        error = "cannot read voice " + selector;
        return false;
    }
    impl.voice = std::move(voice);
    return true;
}

bool Engine::set_voice_codes(const std::string & reference_text,
                             const std::vector<std::vector<long>> & codes,
                             std::string & error) {
    Impl & impl = *impl_;
    if (!impl.opened) { error = "engine is not open"; return false; }
    if (codes.empty() || codes.size() > 1000) { error = "codes length out of range"; return false; }
    Voice voice;
    voice.name = "(transient)";
    voice.reference = reference_text;
    voice.codes = codes;
    for (const auto & row : voice.codes) {
        if (row.size() != 16) { error = "codes must have 16 books per frame"; return false; }
        for (long v : row) {
            if (v < 0 || v > 2047) { error = "codes out of range"; return false; }
        }
    }
    impl.voice = std::move(voice);
    return true;
}

bool Engine::render_tokens(const std::vector<long> & tokens, const std::string & wav_path,
                           const PcmSink & sink, Stats & stats, std::string & error,
                           std::string * wav_bytes) {
    Impl & impl = *impl_;
    if (!impl.opened) { error = "engine is not open"; return false; }
    if (!impl.options.stages) { error = "engine opened without stages"; return false; }
    WavWriter wav;
    if (wav_bytes) wav.open_memory();
    else if (wav_path.empty()) wav.open_discard();
    else if (!wav.open(wav_path)) { error = "cannot write " + wav_path; return false; }
    impl.timings.reset();
    g_heads_s = 0; g_depth_stage_s = 0;
    impl.rng.reset(new JavaRandom(impl.options.seed));
    const bool ok = impl.segment(tokens, wav, sink, stopped_, error);
    wav.close();
    if (wav_bytes) *wav_bytes = wav.bytes();
    stats = Stats();
    stats.elapsed = now_s() - impl.timings.start;
    stats.audio = wav.duration();
    stats.rtf = stats.audio > 0 ? stats.elapsed / stats.audio : 0.0;
    stats.backbone = impl.timings.backbone.load();
    stats.depth = impl.timings.depth.load();
    stats.codec = impl.timings.codec.load();
    stats.first_pcm = impl.timings.first_pcm.load();
    stats.frames = impl.timings.frames.load();
    stats.segments = 1;
    if (getenv("M5_HEAD_STATS"))
        fprintf(stderr, "[m5] heads_s=%.3f stage_call_s=%.3f\n", g_heads_s, g_depth_stage_s);
    return ok;
}

bool Engine::render_text(const std::string & text, const std::string & wav_path,
                         const PcmSink & sink, Stats & stats, std::string & error,
                         std::string * wav_bytes) {
    Impl & impl = *impl_;
    if (!impl.opened) { error = "engine is not open"; return false; }
    if (!impl.options.stages) { error = "engine opened without stages"; return false; }
    if (codepoint_count(text) > 2000) { error = "text is longer than 2000 characters"; return false; }
    const std::vector<std::vector<long>> segments = encode_segments(text);
    if (segments.empty()) { error = "text is empty or contains a reserved marker"; return false; }

    WavWriter wav;
    if (wav_bytes) wav.open_memory();
    else if (wav_path.empty()) wav.open_discard();
    else if (!wav.open(wav_path)) { error = "cannot write " + wav_path; return false; }
    impl.timings.reset();
    g_heads_s = 0; g_depth_stage_s = 0;
    impl.rng.reset(new JavaRandom(impl.options.seed));
    bool ok = true;
    int done = 0;
    for (const auto & segment : segments) {
        if (stopped_.load()) { error = "cancelled"; ok = false; break; }
        if (!impl.segment(segment, wav, sink, stopped_, error)) { ok = false; break; }
        done++;
    }
    wav.close();
    if (wav_bytes) *wav_bytes = wav.bytes();
    stats = Stats();
    stats.elapsed = now_s() - impl.timings.start;
    stats.audio = wav.duration();
    stats.rtf = stats.audio > 0 ? stats.elapsed / stats.audio : 0.0;
    stats.backbone = impl.timings.backbone.load();
    stats.depth = impl.timings.depth.load();
    stats.codec = impl.timings.codec.load();
    stats.first_pcm = impl.timings.first_pcm.load();
    stats.frames = impl.timings.frames.load();
    stats.segments = done;
    if (getenv("M5_HEAD_STATS"))
        fprintf(stderr, "[m5] heads_s=%.3f stage_call_s=%.3f\n", g_heads_s, g_depth_stage_s);
    if (ok && stats.audio <= 0) { error = "model produced no audio"; return false; }
    return ok;
}

}  // namespace m5
