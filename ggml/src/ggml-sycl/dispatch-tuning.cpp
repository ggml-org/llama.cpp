#include "dispatch-tuning.hpp"

#include "ggml-impl.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <mutex>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {
namespace dispatch_tuning {

namespace {

struct ParsedEntry {
    ggml_type quant_type = GGML_TYPE_F32;
    int64_t   dim_m = 0;
    int64_t   dim_n = 0;
    int64_t   dim_k = 0;
    int64_t   instances = 0;
    std::string winner;
};

std::string to_upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

bool parse_ggml_type(const std::string & value, ggml_type & out) {
    const std::string upper = to_upper(value);
    if (upper == "Q4_0") { out = GGML_TYPE_Q4_0; return true; }
    if (upper == "Q5_0") { out = GGML_TYPE_Q5_0; return true; }
    if (upper == "Q5_1") { out = GGML_TYPE_Q5_1; return true; }
    if (upper == "Q8_0") { out = GGML_TYPE_Q8_0; return true; }
    if (upper == "Q2_K") { out = GGML_TYPE_Q2_K; return true; }
    if (upper == "Q3_K") { out = GGML_TYPE_Q3_K; return true; }
    if (upper == "Q4_K") { out = GGML_TYPE_Q4_K; return true; }
    if (upper == "Q5_K") { out = GGML_TYPE_Q5_K; return true; }
    if (upper == "Q6_K") { out = GGML_TYPE_Q6_K; return true; }
    if (upper == "MXFP4") { out = GGML_TYPE_MXFP4; return true; }
    if (upper == "F16") { out = GGML_TYPE_F16; return true; }
    if (upper == "BF16") { out = GGML_TYPE_BF16; return true; }
    if (upper == "F32") { out = GGML_TYPE_F32; return true; }
    return false;
}

size_t skip_ws(const std::string & text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        pos++;
    }
    return pos;
}

std::optional<int64_t> parse_int_field(const std::string & text, const char * key) {
    const std::string needle = std::string("\"") + key + "\":";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = skip_ws(text, pos + needle.size());
    bool neg = false;
    if (pos < text.size() && text[pos] == '-') {
        neg = true;
        pos++;
    }
    int64_t value = 0;
    bool found = false;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        value = value * 10 + (text[pos] - '0');
        pos++;
        found = true;
    }
    if (!found) {
        return std::nullopt;
    }
    return neg ? -value : value;
}

std::optional<std::string> parse_string_field(const std::string & text, const char * key) {
    const std::string needle = std::string("\"") + key + "\":";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = skip_ws(text, pos + needle.size());
    if (pos >= text.size()) {
        return std::nullopt;
    }
    if (text.compare(pos, 4, "null") == 0) {
        return std::string();
    }
    if (text[pos] != '"') {
        return std::nullopt;
    }
    pos++;
    std::string value;
    while (pos < text.size()) {
        char c = text[pos++];
        if (c == '"') {
            break;
        }
        if (c == '\\' && pos < text.size()) {
            value += text[pos++];
            continue;
        }
        value += c;
    }
    return value;
}

bool parse_summary_entries(const std::string & content, std::vector<ParsedEntry> & entries, std::string * error) {
    size_t pos = content.find("\"results\"");
    if (pos == std::string::npos) {
        if (error) {
            *error = "missing results array";
        }
        return false;
    }
    pos = content.find('[', pos);
    if (pos == std::string::npos) {
        if (error) {
            *error = "missing results list";
        }
        return false;
    }

    size_t cursor = pos + 1;
    while (cursor < content.size()) {
        cursor = content.find('{', cursor);
        if (cursor == std::string::npos) {
            break;
        }
        size_t start = cursor;
        int depth = 0;
        bool in_string = false;
        bool escape = false;
        for (; cursor < content.size(); ++cursor) {
            char c = content[cursor];
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\' && in_string) {
                escape = true;
                continue;
            }
            if (c == '"') {
                in_string = !in_string;
                continue;
            }
            if (in_string) {
                continue;
            }
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    cursor++;
                    break;
                }
            }
        }
        if (depth != 0) {
            if (error) {
                *error = "unterminated object";
            }
            return false;
        }
        std::string obj = content.substr(start, cursor - start);
        ParsedEntry entry;
        auto quant = parse_string_field(obj, "quant");
        auto winner = parse_string_field(obj, "winner");
        auto dim_m = parse_int_field(obj, "dim_m");
        auto dim_n = parse_int_field(obj, "dim_n");
        auto dim_k = parse_int_field(obj, "dim_k");
        auto instances = parse_int_field(obj, "tensor_instances");
        if (!quant || !winner || !dim_m || !dim_n || !dim_k) {
            continue;
        }
        if (!parse_ggml_type(*quant, entry.quant_type)) {
            continue;
        }
        entry.dim_m = *dim_m;
        entry.dim_n = *dim_n;
        entry.dim_k = *dim_k;
        entry.instances = instances.value_or(0);
        entry.winner = *winner;
        entries.push_back(entry);
    }
    return !entries.empty();
}

int32_t bucket_dim(int64_t dim) {
    if (dim <= 0) {
        return 0;
    }
    const int32_t buckets[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    for (int32_t bucket : buckets) {
        if (dim <= bucket) {
            return bucket;
        }
    }
    const int32_t step = 4096;
    return static_cast<int32_t>(((dim + step - 1) / step) * step);
}

DispatchTuningKey make_key(ggml_type type, int64_t M, int64_t N, int64_t K) {
    DispatchTuningKey key;
    key.quant_type = static_cast<int32_t>(type);
    key.batch_bucket = ggml_sycl_tuning::bucket_for_batch(static_cast<int>(M));
    key.n_bucket = bucket_dim(N);
    key.k_bucket = bucket_dim(K);
    return key;
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::optional<ggml_sycl_mul_mat_kernel> map_winner_to_kernel(const std::string & winner_raw) {
    if (winner_raw.empty()) {
        return std::nullopt;
    }
    const std::string winner = to_lower(winner_raw);
    if (winner.find("onednn_") == 0) {
        return ggml_sycl_mul_mat_kernel::ONEDNN_AOS;
    }
    if (winner.find("unified_matmul") == 0) {
        return ggml_sycl_mul_mat_kernel::UNIFIED_MATMUL;
    }
    if (winner.find("mmvq_") == 0 || winner == "mmvq") {
        if (winner.find("coalesced") != std::string::npos) {
            return ggml_sycl_mul_mat_kernel::MMVQ_COALESCED;
        }
        if (winner.find("soa") != std::string::npos) {
            return ggml_sycl_mul_mat_kernel::MMVQ_SOA;
        }
        return ggml_sycl_mul_mat_kernel::MMVQ_AOS;
    }
    if (winner.find("mmq_") == 0 || winner == "mmq") {
        if (winner.find("coalesced") != std::string::npos) {
            return ggml_sycl_mul_mat_kernel::MMQ_COALESCED;
        }
        if (winner.find("soa") != std::string::npos) {
            return ggml_sycl_mul_mat_kernel::MMQ_SOA;
        }
        return ggml_sycl_mul_mat_kernel::MMQ_AOS;
    }
    return std::nullopt;
}

bool tuning_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_DISPATCH_TUNING");
        enabled = (env == nullptr || std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

std::string tuning_path() {
    const char * env = std::getenv("GGML_SYCL_DISPATCH_TUNING_JSON");
    if (env && env[0]) {
        return std::string(env);
    }
    return "/tmp/onednn_unified_bench.json";
}

struct ModelCache {
    std::unique_ptr<DispatchTuningCache> cache;
    bool loaded = false;
};

std::unordered_map<uint64_t, ModelCache> g_model_cache;
std::shared_mutex g_model_cache_mutex;

}  // namespace

std::optional<DispatchTuningEntry> DispatchTuningCache::lookup(const DispatchTuningKey & key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void DispatchTuningCache::insert(const DispatchTuningEntry & entry) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = cache_.find(entry.key);
    if (it == cache_.end() || entry.instances >= it->second.instances) {
        cache_[entry.key] = entry;
    }
}

bool DispatchTuningCache::empty() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cache_.empty();
}

size_t DispatchTuningCache::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cache_.size();
}

void DispatchTuningCache::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cache_.clear();
}

bool load_dispatch_tuning_from_file(const std::string & path, DispatchTuningCache & cache, std::string * error) {
    std::ifstream in(path);
    if (!in.is_open()) {
        if (error) {
            *error = "unable to open file";
        }
        return false;
    }
    cache.clear();
    std::ostringstream buffer;
    buffer << in.rdbuf();
    std::string content = buffer.str();

    std::vector<ParsedEntry> parsed;
    if (!parse_summary_entries(content, parsed, error)) {
        return false;
    }

    for (const auto & entry : parsed) {
        auto kernel = map_winner_to_kernel(entry.winner);
        if (!kernel.has_value()) {
            continue;
        }
        DispatchTuningEntry tuned;
        tuned.key = make_key(entry.quant_type, entry.dim_m, entry.dim_n, entry.dim_k);
        tuned.kernel = kernel.value();
        tuned.instances = entry.instances;
        cache.insert(tuned);
    }
    return !cache.empty();
}

DispatchTuningKey make_dispatch_tuning_key(ggml_type type, int64_t M, int64_t N, int64_t K) {
    return make_key(type, M, N, K);
}

void ensure_model_loaded(uint64_t model_id) {
    if (!tuning_enabled() || model_id == 0) {
        return;
    }

    {
        std::shared_lock<std::shared_mutex> lock(g_model_cache_mutex);
        auto it = g_model_cache.find(model_id);
        if (it != g_model_cache.end() && it->second.loaded) {
            return;
        }
    }

    std::unique_lock<std::shared_mutex> lock(g_model_cache_mutex);
    auto & entry = g_model_cache[model_id];
    if (entry.loaded) {
        return;
    }
    entry.loaded = true;
    if (!entry.cache) {
        entry.cache = std::make_unique<DispatchTuningCache>();
    }

    std::string error;
    const std::string path = tuning_path();
    if (!load_dispatch_tuning_from_file(path, *entry.cache, &error)) {
        if (error.empty()) {
            GGML_LOG_WARN("[SYCL] dispatch tuning: no entries loaded from %s\n", path.c_str());
        } else {
            GGML_LOG_WARN("[SYCL] dispatch tuning: failed to load %s (%s)\n", path.c_str(), error.c_str());
        }
        return;
    }
    GGML_LOG_INFO("[SYCL] dispatch tuning: loaded %zu entries from %s for model=%llu\n",
                  entry.cache->size(),
                  path.c_str(),
                  static_cast<unsigned long long>(model_id));
}

std::optional<ggml_sycl_mul_mat_kernel> lookup_kernel(uint64_t model_id,
                                                      ggml_type type,
                                                      int64_t M,
                                                      int64_t N,
                                                      int64_t K) {
    if (!tuning_enabled() || model_id == 0) {
        return std::nullopt;
    }
    ensure_model_loaded(model_id);
    std::shared_lock<std::shared_mutex> lock(g_model_cache_mutex);
    auto it = g_model_cache.find(model_id);
    if (it == g_model_cache.end()) {
        return std::nullopt;
    }
    if (!it->second.cache) {
        return std::nullopt;
    }
    const DispatchTuningKey key = make_key(type, M, N, K);
    auto entry = it->second.cache->lookup(key);
    if (!entry.has_value()) {
        return std::nullopt;
    }
    return entry->kernel;
}

}  // namespace dispatch_tuning
}  // namespace ggml_sycl
