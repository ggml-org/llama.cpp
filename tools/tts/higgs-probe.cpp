#include "llama.h"

#include "higgs-backend.h"
#include "higgs-codec.h"
#include "higgs-dac-backend.h"
#include "higgs-rvq-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

LLAMA_API void llama_set_embeddings_masked(struct llama_context * ctx, bool value);

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m higgs-qwen3-backbone.gguf --higgs-audio higgs-audio-f16.gguf [-ngl N] [--device DEVICE[,DEVICE...]] [-c N] [--steps N | --duration SEC] [--max-duration SEC] [-o path.wav] [--stream-wav] [--stream-stride N] [--stream-holdback N] [--ref-codes path.json] [--ref-text text] [--temp T] [--top-k K] [--seed N] [--codes-out path.json] [--latents-out path.f32] [--higgs-backend auto|cpu|DEVICE] [--rvq-backend cpu|auto|DEVICE] [--dac-backend cpu|auto|DEVICE] [--flash-attn|--no-flash-attn] [--list-higgs-backends] [--raw-prompt] [--print-codes] [--verbose] [-p text]\n"
            "       omit --seed to use a fresh random seed; pass --seed N for reproducible output\n",
            argv0);
}

static uint32_t make_random_seed() {
    const auto now = (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint32_t seed = (uint32_t) (now ^ (now >> 32));
    try {
        std::random_device rd;
        seed ^= rd();
        seed ^= (uint32_t) (rd() << 1);
    } catch (...) {
        // Some standard-library implementations can fail to open an OS entropy source.
    }
    return seed;
}

struct cached_higgs_model {
    std::mutex mutex;
    std::string key;
    llama_model * model = nullptr;
    std::string ctx_key;
    int ctx_n_batch = 0;
    llama_context * ctx = nullptr;
};

struct cached_higgs_companion {
    std::mutex mutex;
    std::string path;
    std::shared_ptr<const higgs_audio::companion_weights> codebook_weights;
    std::shared_ptr<const higgs_audio::rvq_decoder_weights> rvq_weights;
    std::shared_ptr<const higgs_audio::dac_decoder_weights> dac_weights;
    std::string codebook_backend_key;
    std::unique_ptr<higgs_audio::codebook_backend> codebook_backend;
    std::string dac_backend_key;
    std::unique_ptr<higgs_audio::dac_decoder_backend> dac_backend;
};

static cached_higgs_model & higgs_model_cache() {
    static cached_higgs_model cache;
    return cache;
}

static cached_higgs_companion & higgs_companion_cache() {
    static cached_higgs_companion cache;
    return cache;
}

static bool higgs_model_cache_enabled() {
    const char * value = std::getenv("LLAMA_HIGGS_CACHE_MODEL");
    return value && std::strcmp(value, "0") != 0 && value[0] != '\0';
}

static bool higgs_context_cache_enabled() {
    const char * value = std::getenv("LLAMA_HIGGS_CACHE_CONTEXT");
    return value && std::strcmp(value, "0") != 0 && value[0] != '\0';
}

static bool higgs_companion_cache_enabled() {
    const char * value = std::getenv("LLAMA_HIGGS_CACHE_COMPANION");
    return (value && std::strcmp(value, "0") != 0 && value[0] != '\0') || higgs_context_cache_enabled();
}

static llama_model * load_higgs_model_maybe_cached(
        const std::string & path,
        const llama_model_params & params,
        const std::string & cache_key,
        bool & is_cached_ref) {
    is_cached_ref = false;
    if (!higgs_model_cache_enabled()) {
        return llama_model_load_from_file(path.c_str(), params);
    }

    auto & cache = higgs_model_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.model && cache.key == cache_key) {
        is_cached_ref = true;
        std::fprintf(stderr, "using cached Higgs backbone model\n");
        return cache.model;
    }

    if (cache.model) {
        if (cache.ctx) {
            llama_free(cache.ctx);
            cache.ctx = nullptr;
            cache.ctx_key.clear();
            cache.ctx_n_batch = 0;
        }
        llama_model_free(cache.model);
        cache.model = nullptr;
        cache.key.clear();
    }

    cache.model = llama_model_load_from_file(path.c_str(), params);
    if (cache.model) {
        cache.key = cache_key;
        is_cached_ref = true;
        std::fprintf(stderr, "cached Higgs backbone model for reuse\n");
    }
    return cache.model;
}

static llama_context * init_higgs_context_maybe_cached(
        llama_model * model,
        llama_context_params params,
        const std::string & cache_key,
        const int n_batch_needed,
        bool & is_cached_ref) {
    is_cached_ref = false;
    if (!higgs_context_cache_enabled()) {
        return llama_init_from_model(model, params);
    }

    if ((int) params.n_batch < n_batch_needed) {
        params.n_batch = (uint32_t) n_batch_needed;
    }

    auto & cache = higgs_model_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.ctx && cache.ctx_key == cache_key && cache.ctx_n_batch >= (int) params.n_batch) {
        llama_memory_clear(llama_get_memory(cache.ctx), true);
        is_cached_ref = true;
        std::fprintf(stderr, "using cached Higgs llama context\n");
        return cache.ctx;
    }

    if (cache.ctx) {
        llama_free(cache.ctx);
        cache.ctx = nullptr;
        cache.ctx_key.clear();
        cache.ctx_n_batch = 0;
    }

    cache.ctx = llama_init_from_model(model, params);
    if (cache.ctx) {
        cache.ctx_key = cache_key;
        cache.ctx_n_batch = (int) params.n_batch;
        is_cached_ref = true;
        std::fprintf(stderr, "cached Higgs llama context for reuse\n");
    }
    return cache.ctx;
}

static void reset_higgs_companion_cache_if_needed(cached_higgs_companion & cache, const std::string & path) {
    if (cache.path.empty() || cache.path == path) {
        cache.path = path;
        return;
    }

    cache.codebook_weights.reset();
    cache.rvq_weights.reset();
    cache.dac_weights.reset();
    cache.codebook_backend.reset();
    cache.codebook_backend_key.clear();
    cache.dac_backend.reset();
    cache.dac_backend_key.clear();
    cache.path = path;
}

static std::shared_ptr<const higgs_audio::companion_weights> load_codebook_weights_maybe_cached(
        const higgs_audio::companion_file & file) {
    if (!higgs_companion_cache_enabled()) {
        return std::make_shared<higgs_audio::companion_weights>(file.load_codebook_weights());
    }

    auto & cache = higgs_companion_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    reset_higgs_companion_cache_if_needed(cache, file.file_path());
    if (!cache.codebook_weights) {
        cache.codebook_weights = std::make_shared<higgs_audio::companion_weights>(file.load_codebook_weights());
        std::fprintf(stderr, "cached Higgs companion codebook weights for reuse\n");
    } else {
        std::fprintf(stderr, "using cached Higgs companion codebook weights\n");
    }
    return cache.codebook_weights;
}

static std::shared_ptr<const higgs_audio::rvq_decoder_weights> load_rvq_weights_maybe_cached(
        const higgs_audio::companion_file & file) {
    if (!higgs_companion_cache_enabled()) {
        return std::make_shared<higgs_audio::rvq_decoder_weights>(higgs_audio::load_rvq_decoder_weights(file));
    }

    auto & cache = higgs_companion_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    reset_higgs_companion_cache_if_needed(cache, file.file_path());
    if (!cache.rvq_weights) {
        cache.rvq_weights = std::make_shared<higgs_audio::rvq_decoder_weights>(higgs_audio::load_rvq_decoder_weights(file));
        std::fprintf(stderr, "cached Higgs RVQ weights for reuse\n");
    } else {
        std::fprintf(stderr, "using cached Higgs RVQ weights\n");
    }
    return cache.rvq_weights;
}

static std::shared_ptr<const higgs_audio::dac_decoder_weights> load_dac_weights_maybe_cached(
        const higgs_audio::companion_file & file) {
    if (!higgs_companion_cache_enabled()) {
        return std::make_shared<higgs_audio::dac_decoder_weights>(higgs_audio::load_dac_decoder_weights(file));
    }

    auto & cache = higgs_companion_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    reset_higgs_companion_cache_if_needed(cache, file.file_path());
    if (!cache.dac_weights) {
        cache.dac_weights = std::make_shared<higgs_audio::dac_decoder_weights>(higgs_audio::load_dac_decoder_weights(file));
        std::fprintf(stderr, "cached Higgs DAC weights for reuse\n");
    } else {
        std::fprintf(stderr, "using cached Higgs DAC weights\n");
    }
    return cache.dac_weights;
}

static higgs_audio::codebook_backend * get_codebook_backend_maybe_cached(
        const higgs_audio::companion_weights & weights,
        const std::string & requested_backend) {
    if (!higgs_companion_cache_enabled()) {
        return nullptr;
    }

    auto & cache = higgs_companion_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const std::string key = cache.path + "\n" + requested_backend;
    if (!cache.codebook_backend || cache.codebook_backend_key != key) {
        cache.codebook_backend = std::make_unique<higgs_audio::codebook_backend>(
                weights.layout,
                weights.codebook_head,
                requested_backend);
        cache.codebook_backend_key = key;
        std::fprintf(stderr, "cached Higgs codebook backend for reuse: %s\n", cache.codebook_backend->name().c_str());
    } else {
        std::fprintf(stderr, "using cached Higgs codebook backend: %s\n", cache.codebook_backend->name().c_str());
    }
    return cache.codebook_backend.get();
}

static higgs_audio::dac_decoder_backend * get_dac_backend_maybe_cached(const std::string & requested_backend) {
    if (!higgs_companion_cache_enabled()) {
        return nullptr;
    }

    auto & cache = higgs_companion_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const std::string key = cache.path + "\n" + requested_backend;
    if (!cache.dac_backend || cache.dac_backend_key != key) {
        cache.dac_backend = std::make_unique<higgs_audio::dac_decoder_backend>(requested_backend);
        cache.dac_backend_key = key;
        std::fprintf(stderr, "cached Higgs DAC backend for reuse: %s\n", cache.dac_backend->name().c_str());
    } else {
        std::fprintf(stderr, "using cached Higgs DAC backend: %s\n", cache.dac_backend->name().c_str());
    }
    return cache.dac_backend.get();
}

static bool parse_int(const char * s, int & out) {
    try {
        out = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_float(const char * s, float & out) {
    try {
        out = std::stof(s);
        return true;
    } catch (...) {
        return false;
    }
}

static double elapsed_ms_since(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static void write_u16_le(std::ofstream & out, const uint16_t v) {
    out.put((char) (v & 0xff));
    out.put((char) ((v >> 8) & 0xff));
}

static void write_u32_le(std::ofstream & out, const uint32_t v) {
    out.put((char) (v & 0xff));
    out.put((char) ((v >> 8) & 0xff));
    out.put((char) ((v >> 16) & 0xff));
    out.put((char) ((v >> 24) & 0xff));
}

static bool write_wav_f32(const std::string & path, const std::vector<float> & pcm, const int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 32;
    const uint16_t audio_format = 3; // IEEE float
    const uint32_t data_bytes = (uint32_t) (pcm.size() * sizeof(float));
    const uint32_t byte_rate = (uint32_t) sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;

    out.write("RIFF", 4);
    write_u32_le(out, 36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32_le(out, 16);
    write_u16_le(out, audio_format);
    write_u16_le(out, channels);
    write_u32_le(out, (uint32_t) sample_rate);
    write_u32_le(out, byte_rate);
    write_u16_le(out, block_align);
    write_u16_le(out, bits_per_sample);
    out.write("data", 4);
    write_u32_le(out, data_bytes);
    out.write(reinterpret_cast<const char *>(pcm.data()), (std::streamsize) data_bytes);
    return (bool) out;
}

class wav_f32_stream_writer {
public:
    bool open(const std::string & path, const int sample_rate) {
        out.open(path, std::ios::binary);
        if (!out) {
            return false;
        }
        this->sample_rate = sample_rate;
        write_header(0);
        return (bool) out;
    }

    bool append(const float * pcm, const size_t count) {
        if (!out || count == 0) {
            return (bool) out;
        }
        out.write(reinterpret_cast<const char *>(pcm), (std::streamsize) (count * sizeof(float)));
        data_bytes += (uint64_t) count * sizeof(float);
        return (bool) out;
    }

    bool close() {
        if (!out) {
            return false;
        }
        if (data_bytes > (uint64_t) std::numeric_limits<uint32_t>::max() - 36) {
            return false;
        }
        out.seekp(0, std::ios::beg);
        write_header((uint32_t) data_bytes);
        out.close();
        return (bool) out;
    }

private:
    void write_header(const uint32_t data_size) {
        const uint16_t channels = 1;
        const uint16_t bits_per_sample = 32;
        const uint16_t audio_format = 3; // IEEE float
        const uint32_t byte_rate = (uint32_t) sample_rate * channels * bits_per_sample / 8;
        const uint16_t block_align = channels * bits_per_sample / 8;

        out.write("RIFF", 4);
        write_u32_le(out, 36 + data_size);
        out.write("WAVE", 4);
        out.write("fmt ", 4);
        write_u32_le(out, 16);
        write_u16_le(out, audio_format);
        write_u16_le(out, channels);
        write_u32_le(out, (uint32_t) sample_rate);
        write_u32_le(out, byte_rate);
        write_u16_le(out, block_align);
        write_u16_le(out, bits_per_sample);
        out.write("data", 4);
        write_u32_le(out, data_size);
    }

    std::ofstream out;
    int sample_rate = 24000;
    uint64_t data_bytes = 0;
};

static std::string format_zero_shot_prompt(const std::string & text) {
    return "<|tts|><|text|>" + text + "<|audio|>";
}

static std::string format_voice_clone_prefix(const std::string & ref_text) {
    std::string prompt = "<|tts|>";
    if (!ref_text.empty()) {
        prompt += "<|ref_text|>";
        prompt += ref_text;
    }
    prompt += "<|ref_audio|>";
    return prompt;
}

static std::string format_voice_clone_suffix(const std::string & text) {
    return "<|text|>" + text + "<|audio|>";
}

static bool read_text_file(const std::string & path, std::string & text) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    text = ss.str();
    return (bool) in || in.eof();
}

static bool extract_json_array_after_key(const std::string & text, const std::string & key, std::string & array_text) {
    const std::string quoted_key = "\"" + key + "\"";
    const size_t key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t array_start = text.find('[', key_pos + quoted_key.size());
    if (array_start == std::string::npos) {
        return false;
    }

    int depth = 0;
    for (size_t i = array_start; i < text.size(); ++i) {
        if (text[i] == '[') {
            ++depth;
        } else if (text[i] == ']') {
            --depth;
            if (depth == 0) {
                array_text = text.substr(array_start, i - array_start + 1);
                return true;
            }
        }
    }
    return false;
}

static bool parse_ints_from_text(const std::string & text, std::vector<int> & values) {
    const char * p = text.c_str();
    while (*p) {
        if (*p != '-' && !std::isdigit((unsigned char) *p)) {
            ++p;
            continue;
        }
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
            ++p;
            continue;
        }
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
            return false;
        }
        values.push_back((int) v);
        p = end;
    }
    return true;
}

static bool load_reference_codec_frames(
        const std::string & path,
        const int num_codebooks,
        const int codec_codebook_size,
        std::vector<std::vector<int>> & frames,
        std::string & error) {
    std::string text;
    if (!read_text_file(path, text)) {
        error = "failed to read reference code file";
        return false;
    }

    std::string codes_text;
    if (!extract_json_array_after_key(text, "codec_frames", codes_text)) {
        codes_text = text;
    }

    std::vector<int> values;
    if (!parse_ints_from_text(codes_text, values)) {
        error = "reference code file contains an out-of-range integer";
        return false;
    }
    if (values.empty()) {
        error = "reference code file did not contain any codes";
        return false;
    }
    if ((int) values.size() % num_codebooks != 0) {
        error = "reference code count is not divisible by num_codebooks";
        return false;
    }

    frames.clear();
    frames.reserve(values.size() / (size_t) num_codebooks);
    for (size_t i = 0; i < values.size(); i += (size_t) num_codebooks) {
        std::vector<int> row;
        row.reserve((size_t) num_codebooks);
        for (int cb = 0; cb < num_codebooks; ++cb) {
            const int code = values[i + (size_t) cb];
            if (code < 0 || code >= codec_codebook_size) {
                error = "reference codec frames must contain raw codec IDs in the range [0, 1023]";
                return false;
            }
            row.push_back(code);
        }
        frames.push_back(std::move(row));
    }
    return true;
}

static bool tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        const bool add_special,
        std::vector<llama_token> & tokens) {
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    if (n_tokens <= 0) {
        return false;
    }
    tokens.resize((size_t) n_tokens);
    return llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), add_special, true) >= 0;
}

static bool decode_token_chunk(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int & pos,
        const bool logits_last) {
    if (tokens.empty()) {
        return true;
    }

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        batch.token[i] = tokens[(size_t) i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits_last && i == (int32_t) tokens.size() - 1;
    }
    batch.n_tokens = (int32_t) tokens.size();

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (ok) {
        pos += (int) tokens.size();
    }
    return ok;
}

static bool decode_codebook_embedding_chunk(
        llama_context * ctx,
        const higgs_audio::companion_weights & weights,
        const std::vector<std::vector<int>> & frames,
        int & pos,
        const bool logits_last) {
    if (frames.empty()) {
        return true;
    }

    llama_batch batch = llama_batch_init((int32_t) frames.size(), weights.layout.n_embd, 1);
    for (int32_t i = 0; i < (int32_t) frames.size(); ++i) {
        const auto embd = higgs_audio::embed_codebook_frame(weights.layout, weights.codebook_embedding, frames[(size_t) i]);
        std::memcpy(batch.embd + (size_t) i * (size_t) weights.layout.n_embd,
                embd.data(),
                (size_t) weights.layout.n_embd * sizeof(float));
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits_last && i == (int32_t) frames.size() - 1;
    }
    batch.n_tokens = (int32_t) frames.size();

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (ok) {
        pos += (int) frames.size();
    }
    return ok;
}

static void embed_codebook_frame_into(
        const higgs_audio::codebook_layout & layout,
        const std::vector<float> & embedding_weight,
        const std::vector<int> & codes,
        float * embd) {
    std::fill(embd, embd + layout.n_embd, 0.0f);
    for (int cb = 0; cb < layout.num_codebooks; ++cb) {
        const int row = layout.row_for_code(cb, codes[(size_t) cb]);
        const float * src = embedding_weight.data() + (size_t) row * (size_t) layout.n_embd;
        for (int i = 0; i < layout.n_embd; ++i) {
            embd[i] += src[i];
        }
    }
}

static std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t end = value.find(',', start);
        std::string item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) {
            out.push_back(std::move(item));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

static float estimate_duration_seconds(const std::string & text) {
    int words = 0;
    int punctuation_pauses = 0;
    bool in_word = false;

    for (const unsigned char c : text) {
        if (std::isalnum(c)) {
            if (!in_word) {
                ++words;
                in_word = true;
            }
        } else {
            in_word = false;
            if (c == '.' || c == ',' || c == ';' || c == ':' || c == '?' || c == '!') {
                ++punctuation_pauses;
            }
        }
    }

    const float seconds = (float) std::max(words, 1) / 2.6f + (float) punctuation_pauses * 0.20f + 0.45f;
    return std::min(std::max(seconds, 1.0f), 30.0f);
}

static int duration_to_steps(const float duration_seconds, const int frame_rate, const int num_codebooks) {
    return (int) std::ceil(duration_seconds * (float) frame_rate) + num_codebooks - 1;
}

static void quiet_log_callback(enum ggml_log_level, const char *, void *) {
}

int higgs_tts_main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const std::string argv0 = argc > 0 && argv[0] ? argv[0] : "";
    std::string model_path;
    std::string higgs_audio_path;
    std::string codes_out_path;
    std::string latents_out_path;
    std::string wav_out_path;
    std::string ref_codes_path;
    std::string ref_text;
    std::string higgs_backend = "auto";
    std::string rvq_backend = "cpu";
    std::string dac_backend = "CPU";
    std::string model_device_arg;
    std::string prompt = "Hello from Higgs Audio.";
    bool raw_prompt = false;
    bool verbose = false;
    bool list_higgs_backends = false;
    bool print_codes = argv0.find("probe") != std::string::npos;
    bool stream_wav = false;
    bool flash_attn = true;
    bool steps_were_set = false;
    bool duration_was_set = false;
    float duration_seconds = 0.0f;
    float max_duration_seconds = 0.0f;
    float temperature = 0.0f;
    int n_gpu_layers = 99;
    int n_ctx = 512;
    int n_steps = 0;
    int stream_stride = 75;
    int stream_holdback = 4;
    int top_k = 0;
    int seed = 0;
    bool seed_was_set = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--higgs-audio") == 0 && i + 1 < argc) {
            higgs_audio_path = argv[++i];
        } else if (std::strcmp(argv[i], "--codes-out") == 0 && i + 1 < argc) {
            codes_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--latents-out") == 0 && i + 1 < argc) {
            latents_out_path = argv[++i];
        } else if ((std::strcmp(argv[i], "--wav-out") == 0 || std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--out") == 0) && i + 1 < argc) {
            wav_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--stream-wav") == 0) {
            stream_wav = true;
        } else if (std::strcmp(argv[i], "--stream-stride") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], stream_stride)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--stream-holdback") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], stream_holdback)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--ref-codes") == 0 && i + 1 < argc) {
            ref_codes_path = argv[++i];
        } else if (std::strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) {
            ref_text = argv[++i];
        } else if ((std::strcmp(argv[i], "--temp") == 0 || std::strcmp(argv[i], "--temperature") == 0) && i + 1 < argc) {
            if (!parse_float(argv[++i], temperature)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], top_k)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], seed)) {
                print_usage(argv[0]);
                return 1;
            }
            seed_was_set = true;
        } else if (std::strcmp(argv[i], "--higgs-backend") == 0 && i + 1 < argc) {
            higgs_backend = argv[++i];
        } else if (std::strcmp(argv[i], "--rvq-backend") == 0 && i + 1 < argc) {
            rvq_backend = argv[++i];
        } else if (std::strcmp(argv[i], "--dac-backend") == 0 && i + 1 < argc) {
            dac_backend = argv[++i];
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            model_device_arg = argv[++i];
        } else if (std::strcmp(argv[i], "--flash-attn") == 0) {
            flash_attn = true;
        } else if (std::strcmp(argv[i], "--no-flash-attn") == 0) {
            flash_attn = false;
        } else if (std::strcmp(argv[i], "--list-higgs-backends") == 0) {
            list_higgs_backends = true;
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--raw-prompt") == 0) {
            raw_prompt = true;
        } else if (std::strcmp(argv[i], "--print-codes") == 0) {
            print_codes = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], n_gpu_layers)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], n_ctx)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], n_steps)) {
                print_usage(argv[0]);
                return 1;
            }
            steps_were_set = true;
        } else if ((std::strcmp(argv[i], "--duration") == 0 || std::strcmp(argv[i], "--seconds") == 0) && i + 1 < argc) {
            if (!parse_float(argv[++i], duration_seconds)) {
                print_usage(argv[0]);
                return 1;
            }
            duration_was_set = true;
        } else if (std::strcmp(argv[i], "--max-duration") == 0 && i + 1 < argc) {
            if (!parse_float(argv[++i], max_duration_seconds)) {
                print_usage(argv[0]);
                return 1;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (steps_were_set && duration_was_set) {
        std::fprintf(stderr, "use either --steps or --duration, not both\n");
        return 1;
    }
    if (stream_wav && wav_out_path.empty()) {
        std::fprintf(stderr, "--stream-wav requires -o/--wav-out\n");
        return 1;
    }
    if (stream_stride <= 0) {
        std::fprintf(stderr, "--stream-stride must be positive\n");
        return 1;
    }
    if (stream_holdback < 0) {
        std::fprintf(stderr, "--stream-holdback must be non-negative\n");
        return 1;
    }
    if (!ref_text.empty() && ref_codes_path.empty()) {
        std::fprintf(stderr, "--ref-text requires --ref-codes\n");
        return 1;
    }
    if (temperature < 0.0f) {
        std::fprintf(stderr, "--temp must be non-negative; use 0 for greedy\n");
        return 1;
    }
    if (top_k < 0) {
        std::fprintf(stderr, "--top-k must be non-negative; use 0 for all codes\n");
        return 1;
    }
    if (duration_was_set) {
        if (duration_seconds <= 0.0f) {
            std::fprintf(stderr, "--duration must be positive\n");
            return 1;
        }
        const higgs_audio::codebook_layout default_layout;
        constexpr int default_frame_rate = 25;
        n_steps = duration_to_steps(duration_seconds, default_frame_rate, default_layout.num_codebooks);
    } else if (!steps_were_set) {
        if (wav_out_path.empty() && latents_out_path.empty()) {
            n_steps = 1;
        } else {
            const higgs_audio::codebook_layout default_layout;
            constexpr int default_frame_rate = 25;
            duration_seconds = estimate_duration_seconds(prompt);
            if (max_duration_seconds > 0.0f && duration_seconds > max_duration_seconds) {
                duration_seconds = max_duration_seconds;
            }
            n_steps = duration_to_steps(duration_seconds, default_frame_rate, default_layout.num_codebooks);
            std::fprintf(stderr, "auto Higgs duration: %.2f seconds -> %d steps\n", duration_seconds, n_steps);
        }
    }
    if (n_steps <= 0) {
        std::fprintf(stderr, "--steps must be positive\n");
        return 1;
    }

    if (!raw_prompt && ref_codes_path.empty()) {
        prompt = format_zero_shot_prompt(prompt);
    }
    if (!verbose) {
        llama_log_set(quiet_log_callback, nullptr);
    }

    ggml_backend_load_all();

    if (list_higgs_backends) {
        const auto devices = higgs_audio::codebook_backend::available_devices();
        if (devices.empty()) {
            std::fprintf(stderr, "no ggml backend devices are available\n");
            return 1;
        }
        std::fprintf(stderr, "available Higgs ggml backend devices:\n");
        for (const auto & name : devices) {
            std::fprintf(stderr, "  %s\n", name.c_str());
        }
        return 0;
    }

    if (model_path.empty() || higgs_audio_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    higgs_audio::companion_file higgs_file(higgs_audio_path);
    const auto weights_ref = load_codebook_weights_maybe_cached(higgs_file);
    const auto & weights = *weights_ref;

    std::vector<std::vector<int>> ref_codec_frames;
    std::vector<std::vector<int>> ref_delayed_frames;
    if (!ref_codes_path.empty()) {
        std::string error;
        if (!load_reference_codec_frames(
                    ref_codes_path,
                    weights.layout.num_codebooks,
                    1024,
                    ref_codec_frames,
                    error)) {
            std::fprintf(stderr, "failed to load --ref-codes %s: %s\n", ref_codes_path.c_str(), error.c_str());
            return 1;
        }
        ref_delayed_frames = higgs_audio::apply_delay_pattern(ref_codec_frames, weights.layout.num_codebooks);
        std::fprintf(stderr,
                "loaded Higgs reference codec frames: frames=%zu delayed_frames=%zu path=%s\n",
                ref_codec_frames.size(),
                ref_delayed_frames.size(),
                ref_codes_path.c_str());
    }

    std::unique_ptr<higgs_audio::codebook_backend> owned_codebook_backend;
    higgs_audio::codebook_backend * codebook_backend = nullptr;
    if (higgs_backend != "cpu") {
        try {
            codebook_backend = get_codebook_backend_maybe_cached(weights, higgs_backend);
            if (!codebook_backend) {
                owned_codebook_backend = std::make_unique<higgs_audio::codebook_backend>(
                        weights.layout,
                        weights.codebook_head,
                        higgs_backend);
                codebook_backend = owned_codebook_backend.get();
                std::fprintf(stderr, "using Higgs codebook backend: %s\n", codebook_backend->name().c_str());
            }
        } catch (const std::exception & e) {
            if (higgs_backend != "auto") {
                std::fprintf(stderr, "failed to initialize requested Higgs backend '%s': %s\n", higgs_backend.c_str(), e.what());
                const auto devices = higgs_audio::codebook_backend::available_devices();
                if (!devices.empty()) {
                    std::fprintf(stderr, "available ggml backend devices:");
                    for (const auto & name : devices) {
                        std::fprintf(stderr, " %s", name.c_str());
                    }
                    std::fprintf(stderr, "\n");
                }
                return 1;
            }
            std::fprintf(stderr, "using Higgs codebook backend: cpu\n");
        }
    } else {
        std::fprintf(stderr, "using Higgs codebook backend: cpu\n");
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    std::vector<ggml_backend_dev_t> model_devices;
    if (!model_device_arg.empty()) {
        for (const auto & name : split_csv(model_device_arg)) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) {
                std::fprintf(stderr, "unknown llama backend device '%s'\n", name.c_str());
                const auto devices = higgs_audio::codebook_backend::available_devices();
                if (!devices.empty()) {
                    std::fprintf(stderr, "available ggml backend devices:");
                    for (const auto & dev_name : devices) {
                        std::fprintf(stderr, " %s", dev_name.c_str());
                    }
                    std::fprintf(stderr, "\n");
                }
                return 1;
            }
            model_devices.push_back(dev);
        }
        model_devices.push_back(nullptr);
        model_params.devices = model_devices.data();
    }

    const std::string model_cache_key = model_path + "\n" + std::to_string(n_gpu_layers) + "\n" + model_device_arg;
    bool model_is_cached_ref = false;
    llama_model * model = load_higgs_model_maybe_cached(model_path, model_params, model_cache_key, model_is_cached_ref);
    if (!model) {
        std::fprintf(stderr, "failed to load backbone model: %s\n", model_path.c_str());
        return 1;
    }
    auto release_model = [&]() {
        if (!model_is_cached_ref) {
            llama_model_free(model);
        }
    };

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    std::vector<llama_token> prefix_tokens;
    std::vector<llama_token> suffix_tokens;
    int n_prompt = 0;
    int n_batch_prefill = 1;

    if (ref_codes_path.empty()) {
        if (!tokenize_text(vocab, prompt, true, tokens)) {
            std::fprintf(stderr, "failed to tokenize prompt\n");
            release_model();
            return 1;
        }
        n_prompt = (int) tokens.size();
        n_batch_prefill = std::max(n_batch_prefill, n_prompt);
    } else {
        const std::string prefix = raw_prompt ? std::string("<|tts|><|ref_audio|>") : format_voice_clone_prefix(ref_text);
        const std::string suffix = raw_prompt ? prompt : format_voice_clone_suffix(prompt);
        if (!tokenize_text(vocab, prefix, true, prefix_tokens)) {
            std::fprintf(stderr, "failed to tokenize Higgs reference prefix\n");
            release_model();
            return 1;
        }
        if (!tokenize_text(vocab, suffix, false, suffix_tokens)) {
            std::fprintf(stderr, "failed to tokenize Higgs target suffix\n");
            release_model();
            return 1;
        }
        n_prompt = (int) prefix_tokens.size() + (int) ref_delayed_frames.size() + (int) suffix_tokens.size();
        n_batch_prefill = std::max(n_batch_prefill, (int) prefix_tokens.size());
        n_batch_prefill = std::max(n_batch_prefill, (int) ref_delayed_frames.size());
        n_batch_prefill = std::max(n_batch_prefill, (int) suffix_tokens.size());
    }

    if (n_prompt <= 0) {
        std::fprintf(stderr, "prompt is empty\n");
        release_model();
        return 1;
    }
    if (n_prompt > n_ctx) {
        std::fprintf(stderr, "prompt tokens/embeddings (%d) exceed context (%d)\n", n_prompt, n_ctx);
        release_model();
        return 1;
    }
    if (n_prompt + n_steps > n_ctx) {
        std::fprintf(stderr, "prompt tokens/embeddings plus requested Higgs steps (%d) exceed context (%d)\n", n_prompt + n_steps, n_ctx);
        release_model();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) n_ctx;
    ctx_params.n_batch = (uint32_t) n_batch_prefill;
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.no_perf = true;

    if (higgs_context_cache_enabled()) {
        ctx_params.n_batch = (uint32_t) std::max(n_batch_prefill, n_ctx);
    }
    const std::string ctx_cache_key = model_cache_key + "\nctx=" + std::to_string(n_ctx) + "\nfa=" + (flash_attn ? "1" : "0");
    bool ctx_is_cached_ref = false;
    llama_context * ctx = init_higgs_context_maybe_cached(model, ctx_params, ctx_cache_key, n_batch_prefill, ctx_is_cached_ref);
    if (!ctx) {
        std::fprintf(stderr, "failed to create llama context\n");
        release_model();
        return 1;
    }
    llama_set_embeddings_masked(ctx, true);
    auto release_context = [&]() {
        if (!ctx_is_cached_ref) {
            llama_free(ctx);
        }
    };

    const int n_embd_out = llama_model_n_embd_out(model);
    if (n_embd_out != weights.layout.n_embd) {
        std::fprintf(stderr,
                "backbone hidden size (%d) does not match Higgs companion hidden size (%d)\n",
                n_embd_out,
                weights.layout.n_embd);
        release_context();
        release_model();
        return 1;
    }

    double prompt_decode_ms = 0.0;
    double codebook_ms = 0.0;
    double embed_ms = 0.0;
    double step_decode_ms = 0.0;
    double stream_decode_ms = 0.0;
    double rvq_ms = 0.0;
    double dac_ms = 0.0;

    int prompt_pos = 0;
    if (ref_codes_path.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!decode_token_chunk(ctx, tokens, prompt_pos, true)) {
            std::fprintf(stderr, "failed to decode prompt\n");
            release_context();
            release_model();
            return 1;
        }
        prompt_decode_ms += elapsed_ms_since(t0);
    } else {
        auto t0 = std::chrono::steady_clock::now();
        if (!decode_token_chunk(ctx, prefix_tokens, prompt_pos, false)) {
            std::fprintf(stderr, "failed to decode Higgs reference prefix\n");
            release_context();
            release_model();
            return 1;
        }
        prompt_decode_ms += elapsed_ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        if (!decode_codebook_embedding_chunk(ctx, weights, ref_delayed_frames, prompt_pos, false)) {
            std::fprintf(stderr, "failed to decode Higgs reference code embeddings\n");
            release_context();
            release_model();
            return 1;
        }
        prompt_decode_ms += elapsed_ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        if (!decode_token_chunk(ctx, suffix_tokens, prompt_pos, true)) {
            std::fprintf(stderr, "failed to decode Higgs target suffix\n");
            release_context();
            release_model();
            return 1;
        }
        prompt_decode_ms += elapsed_ms_since(t0);
    }

    if (prompt_pos != n_prompt) {
        std::fprintf(stderr, "internal prompt accounting mismatch\n");
        release_context();
        release_model();
        return 1;
    }

    const float * hidden = llama_get_embeddings_ith(ctx, -1);
    if (!hidden) {
        std::fprintf(stderr, "failed to get final hidden vector from backbone\n");
        release_context();
        release_model();
        return 1;
    }

    higgs_audio::delay_sampler sampler(weights.layout.num_codebooks);
    higgs_audio::codebook_sampling_params sampling_params;
    sampling_params.temperature = temperature;
    sampling_params.top_k = top_k;
    const uint32_t effective_seed = seed_was_set ? (uint32_t) seed : make_random_seed();
    std::mt19937 rng(effective_seed);
    if (sampling_params.temperature > 0.0f) {
        std::fprintf(stderr,
                "using Higgs codebook sampling: temp=%.3f top_k=%d seed=%u%s\n",
                sampling_params.temperature,
                sampling_params.top_k,
                effective_seed,
                seed_was_set ? "" : " (random)");
    }
    std::vector<float> hidden_vec(hidden, hidden + n_embd_out);
    std::vector<std::vector<int>> delayed_frames;
    delayed_frames.reserve((size_t) n_steps);

    std::shared_ptr<const higgs_audio::rvq_decoder_weights> stream_rvq_weights;
    std::shared_ptr<const higgs_audio::dac_decoder_weights> stream_dac_weights;
    std::unique_ptr<higgs_audio::rvq_decoder_backend> stream_rvq_runner;
    std::unique_ptr<higgs_audio::dac_decoder_backend> stream_dac_runner;
    wav_f32_stream_writer stream_writer;
    size_t stream_written_samples = 0;
    if (stream_wav) {
        stream_rvq_weights = load_rvq_weights_maybe_cached(higgs_file);
        stream_dac_weights = load_dac_weights_maybe_cached(higgs_file);
        if (rvq_backend != "cpu") {
            try {
                stream_rvq_runner = std::make_unique<higgs_audio::rvq_decoder_backend>(rvq_backend);
                std::fprintf(stderr, "using Higgs streaming RVQ backend: %s\n", stream_rvq_runner->name().c_str());
            } catch (const std::exception & e) {
                std::fprintf(stderr, "failed to initialize requested Higgs streaming RVQ backend '%s': %s\n", rvq_backend.c_str(), e.what());
                release_context();
                release_model();
                return 1;
            }
        }
        if (dac_backend != "cpu") {
            try {
                stream_dac_runner = std::make_unique<higgs_audio::dac_decoder_backend>(dac_backend);
                std::fprintf(stderr, "using Higgs streaming DAC backend: %s\n", stream_dac_runner->name().c_str());
            } catch (const std::exception & e) {
                std::fprintf(stderr, "failed to initialize requested Higgs streaming DAC backend '%s': %s\n", dac_backend.c_str(), e.what());
                release_context();
                release_model();
                return 1;
            }
        }
        if (!stream_writer.open(wav_out_path, higgs_file.metadata().sample_rate)) {
            std::fprintf(stderr, "failed to open streaming WAV output file: %s\n", wav_out_path.c_str());
            release_context();
            release_model();
            return 1;
        }
    }

    auto flush_stream_wav = [&](const bool final_flush) -> bool {
        if (!stream_wav) {
            return true;
        }
        if (delayed_frames.size() < (size_t) weights.layout.num_codebooks) {
            return true;
        }

        size_t stable_delayed = delayed_frames.size();
        if (!final_flush) {
            if (stable_delayed <= (size_t) stream_holdback) {
                return true;
            }
            stable_delayed -= (size_t) stream_holdback;
        }
        if (stable_delayed < (size_t) weights.layout.num_codebooks) {
            return true;
        }

        const std::vector<std::vector<int>> delayed_prefix(
                delayed_frames.begin(),
                delayed_frames.begin() + (std::ptrdiff_t) stable_delayed);
        const auto stable_codec_frames = higgs_audio::reverse_delay_pattern(delayed_prefix, weights.layout.num_codebooks);
        if (stable_codec_frames.empty()) {
            return true;
        }

        const auto t_stream = std::chrono::steady_clock::now();
        const auto acoustic_latents = stream_rvq_runner
                ? stream_rvq_runner->decode_acoustic_latents(*stream_rvq_weights, stable_codec_frames)
                : higgs_audio::decode_rvq_acoustic_latents(*stream_rvq_weights, stable_codec_frames);
        if (!higgs_audio::all_finite(acoustic_latents)) {
            std::fprintf(stderr, "streamed Higgs acoustic latents contain non-finite values\n");
            return false;
        }
        const auto pcm = stream_dac_runner
                ? stream_dac_runner->decode_pcm(*stream_dac_weights, acoustic_latents, (int) stable_codec_frames.size())
                : higgs_audio::dac_decode_pcm(*stream_dac_weights, acoustic_latents, (int) stable_codec_frames.size());
        stream_decode_ms += elapsed_ms_since(t_stream);
        if (!higgs_audio::all_finite(pcm)) {
            std::fprintf(stderr, "streamed Higgs PCM contains non-finite values\n");
            return false;
        }
        if (pcm.size() > stream_written_samples) {
            if (!stream_writer.append(pcm.data() + stream_written_samples, pcm.size() - stream_written_samples)) {
                std::fprintf(stderr, "failed to append streaming WAV output file: %s\n", wav_out_path.c_str());
                return false;
            }
            stream_written_samples = pcm.size();
            if (verbose) {
                std::fprintf(stderr,
                        "streamed Higgs WAV chunk: stable_frames=%zu samples=%zu path=%s\n",
                        stable_codec_frames.size(),
                        stream_written_samples,
                        wav_out_path.c_str());
            }
        }
        return true;
    };

    llama_batch embd_batch = llama_batch_init(1, n_embd_out, 1);
    for (int step = 0; step < n_steps; ++step) {
        const auto t_codebook = std::chrono::steady_clock::now();
        std::vector<int> raw_codes;
        if (sampling_params.temperature > 0.0f) {
            if (codebook_backend) {
                raw_codes = higgs_audio::sample_codebook_codes(
                        weights.layout,
                        codebook_backend->codebook_logits(hidden_vec),
                        sampling_params,
                        rng);
            } else {
                const auto logits = higgs_audio::project_codebook_logits(weights.layout, weights.codebook_head, hidden_vec);
                raw_codes = higgs_audio::sample_codebook_codes(weights.layout, logits, sampling_params, rng);
            }
        } else {
            raw_codes = codebook_backend
                    ? codebook_backend->greedy_codebook_codes(hidden_vec)
                    : higgs_audio::greedy_codebook_codes(weights.layout, weights.codebook_head, hidden_vec);
        }
        const auto codes = sampler.step_from_codes(raw_codes);
        delayed_frames.push_back(codes);
        codebook_ms += elapsed_ms_since(t_codebook);

        if (print_codes) {
            std::printf("higgs_codebook_frame %d", step);
            for (int code : codes) {
                std::printf(" %d", code);
            }
            std::printf("\n");
        }

        if (sampler.done() || step == n_steps - 1) {
            break;
        }

        if (stream_wav && !delayed_frames.empty() && delayed_frames.size() % (size_t) stream_stride == 0) {
            if (!flush_stream_wav(false)) {
                llama_batch_free(embd_batch);
                release_context();
                release_model();
                return 1;
            }
        }

        auto t_step = std::chrono::steady_clock::now();
        embed_codebook_frame_into(weights.layout, weights.codebook_embedding, codes, embd_batch.embd);
        embed_ms += elapsed_ms_since(t_step);
        embd_batch.pos[0] = n_prompt + step;
        embd_batch.n_seq_id[0] = 1;
        embd_batch.seq_id[0][0] = 0;
        embd_batch.logits[0] = 1;
        embd_batch.n_tokens = 1;

        t_step = std::chrono::steady_clock::now();
        if (llama_decode(ctx, embd_batch) != 0) {
            std::fprintf(stderr, "failed to decode Higgs codebook embedding at step %d\n", step);
            llama_batch_free(embd_batch);
            release_context();
            release_model();
            return 1;
        }
        step_decode_ms += elapsed_ms_since(t_step);

        hidden = llama_get_embeddings_ith(ctx, -1);
        if (!hidden) {
            std::fprintf(stderr, "failed to get hidden vector after Higgs codebook embedding at step %d\n", step);
            llama_batch_free(embd_batch);
            release_context();
            release_model();
            return 1;
        }
        hidden_vec.assign(hidden, hidden + n_embd_out);
    }

    if (stream_wav) {
        if (!flush_stream_wav(true) || !stream_writer.close()) {
            std::fprintf(stderr, "failed to finalize streaming WAV output file: %s\n", wav_out_path.c_str());
            llama_batch_free(embd_batch);
            release_context();
            release_model();
            return 1;
        }
        std::fprintf(stderr,
                "streamed Higgs WAV: samples=%zu sample_rate=%d path=%s\n",
                stream_written_samples,
                higgs_file.metadata().sample_rate,
                wav_out_path.c_str());
    }

    std::vector<std::vector<int>> codec_frames;
    if (delayed_frames.size() >= (size_t) weights.layout.num_codebooks) {
        codec_frames = higgs_audio::reverse_delay_pattern(delayed_frames, weights.layout.num_codebooks);
    }

    if (!codes_out_path.empty()) {
        std::ofstream out(codes_out_path);
        if (!out) {
            std::fprintf(stderr, "failed to open code output file: %s\n", codes_out_path.c_str());
            llama_batch_free(embd_batch);
            release_context();
            release_model();
            return 1;
        }

        out << "{\n";
        out << "  \"format\": \"higgs-audio-v3-codes\",\n";
        out << "  \"sample_rate\": " << higgs_file.metadata().sample_rate << ",\n";
        out << "  \"frame_rate\": " << higgs_file.metadata().frame_rate << ",\n";
        out << "  \"num_codebooks\": " << weights.layout.num_codebooks << ",\n";
        out << "  \"codebook_size\": " << weights.layout.codebook_size << ",\n";
        out << "  \"acoustic_latent_channels\": 256,\n";
        out << "  \"delayed_frames\": [\n";
        for (size_t i = 0; i < delayed_frames.size(); ++i) {
            out << "    [";
            for (size_t j = 0; j < delayed_frames[i].size(); ++j) {
                if (j) {
                    out << ", ";
                }
                out << delayed_frames[i][j];
            }
            out << "]";
            if (i + 1 < delayed_frames.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ],\n";
        out << "  \"codec_frames\": [\n";
        for (size_t i = 0; i < codec_frames.size(); ++i) {
            out << "    [";
            for (size_t j = 0; j < codec_frames[i].size(); ++j) {
                if (j) {
                    out << ", ";
                }
                out << codec_frames[i][j];
            }
            out << "]";
            if (i + 1 < codec_frames.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }

    if (!latents_out_path.empty() || (!wav_out_path.empty() && !stream_wav)) {
        if (codec_frames.empty()) {
            std::fprintf(stderr, "not enough delayed frames to reverse into codec frames for audio output\n");
            llama_batch_free(embd_batch);
            release_context();
            release_model();
            return 1;
        }

        const auto rvq_weights_ref = load_rvq_weights_maybe_cached(higgs_file);
        const auto & rvq_weights = *rvq_weights_ref;
        std::vector<float> acoustic_latents;
        if (rvq_backend == "cpu") {
            const auto t_rvq = std::chrono::steady_clock::now();
            acoustic_latents = higgs_audio::decode_rvq_acoustic_latents(rvq_weights, codec_frames);
            rvq_ms += elapsed_ms_since(t_rvq);
        } else {
            try {
                higgs_audio::rvq_decoder_backend rvq_runner(rvq_backend);
                std::fprintf(stderr, "using Higgs RVQ backend: %s\n", rvq_runner.name().c_str());
                const auto t_rvq = std::chrono::steady_clock::now();
                acoustic_latents = rvq_runner.decode_acoustic_latents(rvq_weights, codec_frames);
                rvq_ms += elapsed_ms_since(t_rvq);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "failed to run requested Higgs RVQ backend '%s': %s\n", rvq_backend.c_str(), e.what());
                llama_batch_free(embd_batch);
                release_context();
                release_model();
                return 1;
            }
        }
        if (!higgs_audio::all_finite(acoustic_latents)) {
            std::fprintf(stderr, "generated Higgs acoustic latents contain non-finite values\n");
            llama_batch_free(embd_batch);
            release_context();
            release_model();
            return 1;
        }

        if (!latents_out_path.empty()) {
            std::ofstream out(latents_out_path, std::ios::binary);
            if (!out) {
                std::fprintf(stderr, "failed to open latent output file: %s\n", latents_out_path.c_str());
                llama_batch_free(embd_batch);
                release_context();
                release_model();
                return 1;
            }
            out.write(reinterpret_cast<const char *>(acoustic_latents.data()),
                    (std::streamsize) (acoustic_latents.size() * sizeof(float)));
            if (!out) {
                std::fprintf(stderr, "failed to write latent output file: %s\n", latents_out_path.c_str());
                llama_batch_free(embd_batch);
                release_context();
                release_model();
                return 1;
            }
            std::fprintf(stderr,
                    "wrote Higgs acoustic latents: frames=%zu channels=%d path=%s\n",
                    codec_frames.size(),
                    rvq_weights.acoustic_size,
                    latents_out_path.c_str());
        }

        if (!wav_out_path.empty() && !stream_wav) {
            const auto dac_weights_ref = load_dac_weights_maybe_cached(higgs_file);
            const auto & dac_weights = *dac_weights_ref;
            std::vector<float> pcm;
            if (dac_backend == "cpu") {
                const auto t_dac = std::chrono::steady_clock::now();
                pcm = higgs_audio::dac_decode_pcm(dac_weights, acoustic_latents, (int) codec_frames.size());
                dac_ms += elapsed_ms_since(t_dac);
            } else {
                std::unique_ptr<higgs_audio::dac_decoder_backend> owned_dac_runner;
                try {
                    higgs_audio::dac_decoder_backend * dac_runner = get_dac_backend_maybe_cached(dac_backend);
                    if (!dac_runner) {
                        owned_dac_runner = std::make_unique<higgs_audio::dac_decoder_backend>(dac_backend);
                        dac_runner = owned_dac_runner.get();
                    }
                    std::fprintf(stderr, "using Higgs DAC backend: %s\n", dac_runner->name().c_str());
                    const auto t_dac = std::chrono::steady_clock::now();
                    pcm = dac_runner->decode_pcm(dac_weights, acoustic_latents, (int) codec_frames.size());
                    dac_ms += elapsed_ms_since(t_dac);
                } catch (const std::exception & e) {
                    std::fprintf(stderr,
                            "failed to run requested Higgs DAC backend '%s': %s\n"
                            "falling back to ggml Higgs DAC CPU backend\n",
                            dac_backend.c_str(),
                            e.what());
                    const auto t_dac = std::chrono::steady_clock::now();
                    try {
                        higgs_audio::dac_decoder_backend * cpu_dac_runner = get_dac_backend_maybe_cached("CPU");
                        if (!cpu_dac_runner) {
                            owned_dac_runner = std::make_unique<higgs_audio::dac_decoder_backend>("CPU");
                            cpu_dac_runner = owned_dac_runner.get();
                        }
                        std::fprintf(stderr, "using fallback Higgs DAC backend: %s\n", cpu_dac_runner->name().c_str());
                        pcm = cpu_dac_runner->decode_pcm(dac_weights, acoustic_latents, (int) codec_frames.size());
                    } catch (const std::exception & cpu_e) {
                        std::fprintf(stderr,
                                "failed to run fallback ggml Higgs DAC CPU backend: %s\n"
                                "falling back to scalar Higgs DAC CPU decoder\n",
                                cpu_e.what());
                        pcm = higgs_audio::dac_decode_pcm(dac_weights, acoustic_latents, (int) codec_frames.size());
                    }
                    dac_ms += elapsed_ms_since(t_dac);
                }
            }
            if (!higgs_audio::all_finite(pcm)) {
                std::fprintf(stderr, "generated Higgs PCM contains non-finite values\n");
                llama_batch_free(embd_batch);
                release_context();
                release_model();
                return 1;
            }
            if (!write_wav_f32(wav_out_path, pcm, higgs_file.metadata().sample_rate)) {
                std::fprintf(stderr, "failed to write WAV output file: %s\n", wav_out_path.c_str());
                llama_batch_free(embd_batch);
                release_context();
                release_model();
                return 1;
            }
            std::fprintf(stderr,
                    "wrote Higgs WAV: samples=%zu sample_rate=%d path=%s\n",
                    pcm.size(),
                    higgs_file.metadata().sample_rate,
                    wav_out_path.c_str());
        }
    }

    if (verbose) {
        const int generated_steps = (int) delayed_frames.size();
        std::fprintf(stderr,
                "Higgs timing: prompt_decode=%.3f ms codebook=%.3f ms embed=%.3f ms step_decode=%.3f ms rvq=%.3f ms dac=%.3f ms stream_decode=%.3f ms steps=%d avg_step_decode=%.3f ms\n",
                prompt_decode_ms,
                codebook_ms,
                embed_ms,
                step_decode_ms,
                rvq_ms,
                dac_ms,
                stream_decode_ms,
                generated_steps,
                generated_steps > 1 ? step_decode_ms / (double) (generated_steps - 1) : 0.0);
    }

    llama_batch_free(embd_batch);
    release_context();
    release_model();
    return 0;
}

#ifndef HIGGS_TTS_NO_MAIN
int main(int argc, char ** argv) {
    return higgs_tts_main(argc, argv);
}
#endif
