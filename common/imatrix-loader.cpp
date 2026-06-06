#include "imatrix-loader.h"
#include "common.h"
#include "log.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>

static bool common_imatrix_load_legacy(const std::string & fname, common_imatrix & imatrix) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        LOG_ERR("%s: failed to open %s\n", __func__, fname.c_str());
        return false;
    }

    int n_entries;
    in.read((char *) &n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, fname.c_str());
        return false;
    }

    for (int i = 0; i < n_entries; ++i) {
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        std::vector<char> name_as_vec(len + 1);
        in.read((char *) name_as_vec.data(), len);
        if (in.fail()) {
            LOG_ERR("%s: failed reading name for entry %d from %s\n", __func__, i + 1, fname.c_str());
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{ name_as_vec.data() };

        int32_t ncall = 0;
        in.read((char *) &ncall, sizeof(ncall));
        int32_t nval = 0;
        in.read((char *) &nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            LOG_ERR("%s: failed reading number of values for entry %d\n", __func__, i);
            return false;
        }

        auto & e = imatrix.entries[std::move(name)];
        e.sums.resize(nval);
        in.read((char *) e.sums.data(), nval * sizeof(float));
        if (in.fail()) {
            LOG_ERR("%s: failed reading data for entry %d\n", __func__, i);
            return false;
        }

        e.counts.resize(1);
        e.counts[0] = ncall;
    }

    // the trailing data (chunk count + dataset name) is optional
    if (in.peek() != EOF) {
        int32_t n_calls = 0;
        in.read((char *) &n_calls, sizeof(n_calls));
        imatrix.chunk_count = n_calls;

        if (!in.fail()) {
            int32_t len = 0;
            in.read((char *) &len, sizeof(len));
            if (!in.fail() && len > 0) {
                std::vector<char> dataset(len + 1, 0);
                in.read(dataset.data(), len);
                if (!in.fail()) {
                    imatrix.datasets.push_back(dataset.data());
                }
            }
        }
    }

    imatrix.chunk_size = 0;
    imatrix.is_legacy  = true;

    return true;
}

bool common_imatrix_load(const std::string & fname, common_imatrix & imatrix) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        return common_imatrix_load_legacy(fname, imatrix);
    }

    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        LOG_ERR("%s: no data in file %s\n", __func__, fname.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return false;
    }

    const int64_t datasets_key   = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_DATASETS);
    const int64_t chunk_count_key = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT);
    const int64_t chunk_size_key  = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE);

    if (datasets_key != -1 && gguf_get_arr_type(ctx_gguf, datasets_key) == GGUF_TYPE_STRING) {
        const int64_t n = gguf_get_arr_n(ctx_gguf, datasets_key);
        imatrix.datasets.reserve(imatrix.datasets.size() + n);
        for (int64_t i = 0; i < n; ++i) {
            imatrix.datasets.push_back(gguf_get_arr_str(ctx_gguf, datasets_key, i));
        }
    }

    imatrix.has_metadata = datasets_key != -1 && chunk_count_key != -1 && chunk_size_key != -1;
    imatrix.chunk_count  = chunk_count_key != -1 ? gguf_get_val_u32(ctx_gguf, chunk_count_key) : 0;
    imatrix.chunk_size   = chunk_size_key  != -1 ? gguf_get_val_u32(ctx_gguf, chunk_size_key)  : 0;

    // stats schema: maps file-order positions to canonical metric indices
    const int64_t schema_idx = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_STATS_SCHEMA);
    const std::unordered_map<std::string, int> default_schema_map = {
        {"sum_sq", 0}, {"mean", 1}, {"elements", 2}, {"std_deviation", 3}, {"skewness", 4},
        {"kurtosis", 5}, {"gain", 6}, {"h_norm", 7}, {"l2_dist", 8}, {"cossim", 9}, {"pearson", 10}, {"covariance", 11}
    };
    std::vector<int> stats_indices;
    if (schema_idx >= 0) {
        const int64_t n_schema = gguf_get_arr_n(ctx_gguf, schema_idx);
        for (int64_t i = 0; i < n_schema; ++i) {
            const std::string key = gguf_get_arr_str(ctx_gguf, schema_idx, i);
            auto it = default_schema_map.find(key);
            stats_indices.push_back(it != default_schema_map.end() ? it->second : -1);
        }
    } else {
        for (size_t i = 0; i < default_schema_map.size(); ++i) {
            stats_indices.push_back((int) i);
        }
    }

    // store canonical schema names in order
    imatrix.stats_schema.resize(default_schema_map.size());
    for (const auto & [name, idx] : default_schema_map) {
        imatrix.stats_schema[idx] = name;
    }

    const std::string in_sum_suffix{ ".in_sum" };
    const std::string in_sum2_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };
    const std::string stats_suffix{ ".stats" };

    struct sum_tensors {
        struct ggml_tensor * in_sum  = nullptr;
        struct ggml_tensor * in_sum2 = nullptr;
        struct ggml_tensor * counts  = nullptr;
        struct ggml_tensor * stats  = nullptr;
    };

    std::map<std::string, sum_tensors> sums_counts_for;
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;
        if (name.empty()) { continue; }

        if (string_remove_suffix(name, in_sum_suffix)) {
            sums_counts_for[std::move(name)].in_sum = cur;
        } else if (string_remove_suffix(name, in_sum2_suffix)) {
            sums_counts_for[std::move(name)].in_sum2 = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            sums_counts_for[std::move(name)].counts = cur;
        } else if (string_remove_suffix(name, stats_suffix)) {
            sums_counts_for[std::move(name)].stats = cur;
        }
    }

    for (const auto & sc : sums_counts_for) {
        const std::string &        name    = sc.first;
        const struct ggml_tensor * in_sum  = sc.second.in_sum;
        const struct ggml_tensor * in_sum2 = sc.second.in_sum2;
        const struct ggml_tensor * counts  = sc.second.counts;
        const struct ggml_tensor * stats  = sc.second.stats;

        if (!in_sum2 || !counts || (in_sum != nullptr && ggml_nelements(in_sum) != ggml_nelements(in_sum2))) {
            LOG_ERR("%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return false;
        }

        auto & e = imatrix.entries[name];

        const int64_t nval    = ggml_nelements(in_sum2);
        const int64_t ncounts = ggml_nelements(counts);

        e.sums.resize(nval);
        for (int64_t j = 0; j < nval; ++j) {
            e.sums[j] = ((const float *) in_sum2->data)[j];
        }

        e.counts.resize(ncounts);
        for (int64_t j = 0; j < ncounts; ++j) {
            e.counts[j] = std::lround(((const float *) counts->data)[j]);
        }

        if (in_sum && ggml_nelements(in_sum) == nval) {
            e.activations.resize(nval);
            for (int64_t j = 0; j < nval; ++j) {
                e.activations[j] = ((const float *) in_sum->data)[j];
            }
        }

        if (stats && stats->type == GGML_TYPE_F32) {
            e.stats.resize(default_schema_map.size(), 0.0f);
            const auto * stats_data = (const float *) stats->data;
            const int64_t n_stats = ggml_nelements(stats);
            for (int64_t j = 0; j < (int64_t) stats_indices.size() && j < n_stats; ++j) {
                if (stats_indices[j] >= 0 && stats_indices[j] < (int) e.stats.size()) {
                    e.stats[stats_indices[j]] = stats_data[j];
                }
            }
        }
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return true;
}
