#include "common.h"
#include "llama.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <fstream>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#define getpid GetCurrentProcessId
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "MXFP4_MOE",LLAMA_FTYPE_MOSTLY_MXFP4_MOE," MXFP4 MoE",  },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "TQ1_0",    LLAMA_FTYPE_MOSTLY_TQ1_0,    " 1.69 bpw ternarization",           },
    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.96G, +3.1836 ppl @ Llama-3-8B",  },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M"                   },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization",             },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M",                  },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 4.37G, +0.2689 ppl @ Llama-3-8B",  },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M",                  },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 5.21G, +0.1049 ppl @ Llama-3-8B",  },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, +0.0020 ppl @ Mistral-7B",  },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B",  },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B",          },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};

// Quantization types. Changes to this struct must be replicated in llama-quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";

// TODO: share with imatrix.cpp
static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

// Check if two paths refer to the same physical file
// Returns true if they are the same file (including via hardlinks or symlinks)
static bool same_file(const std::string & path_a, const std::string & path_b) {
    std::error_code ec_a, ec_b;

    // First try using std::filesystem to resolve canonical paths
    auto canonical_a = std::filesystem::weakly_canonical(path_a, ec_a);
    auto canonical_b = std::filesystem::weakly_canonical(path_b, ec_b);

    if (!ec_a && !ec_b) {
        // If both paths were successfully canonicalized, compare them
        if (canonical_a == canonical_b) {
            return true;
        }
    }

#ifndef _WIN32
    // On Unix-like systems, also check using stat() to handle hardlinks
    struct stat stat_a, stat_b;
    if (stat(path_a.c_str(), &stat_a) == 0 && stat(path_b.c_str(), &stat_b) == 0) {
        // Same file if device and inode match
        if (stat_a.st_dev == stat_b.st_dev && stat_a.st_ino == stat_b.st_ino) {
            return true;
        }
    }
#endif

    return false;
}

static bool striequals(const char * a, const char * b) {
    while (*a && *b) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
        a++; b++;
    }
    return *a == *b;
}

static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (const auto & it : QUANT_OPTIONS) {
        if (striequals(it.name.c_str(), ftype_str.c_str())) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (const auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]\n", executable);
    printf("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--prune-layers] [--keep-split] [--override-kv]\n");
    printf("       [--inplace] [--overwrite] model-f32.gguf [model-quant.gguf] type [nthreads]\n\n");
    printf("  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit\n");
    printf("  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure: Disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --imatrix file_name: use data in file_name as importance matrix for quant optimizations\n");
    printf("  --include-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor\n");
    printf("  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor\n");
    printf("  --tensor-type TENSOR=TYPE: quantize this tensor to this ggml_type. example: --tensor-type attn_q=q8_0\n");
    printf("      Advanced option to selectively quantize tensors. May be specified multiple times.\n");
    printf("  --prune-layers L0,L1,L2...comma-separated list of layer numbers to prune from the model\n");
    printf("      Advanced option to remove all tensors from the given layers\n");
    printf("  --keep-split: will generate quantized model in the same shards as input\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("      Advanced option to override model metadata by key in the quantized model. May be specified multiple times.\n");
    printf("  --inplace: Allow in-place quantization (input == output). Uses temp file + atomic rename for safety.\n");
    printf("  --overwrite: Allow overwriting existing output file (still uses temp file for safety).\n");
    printf("Note: --include-weights and --exclude-weights cannot be used together\n");
    printf("\nAllowed quantization types:\n");
    for (const auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-7s : %s\n", it.name.c_str(), it.desc.c_str());
    }
    exit(1);
}

static int load_legacy_imatrix(const std::string & imatrix_file, std::vector<std::string> & imatrix_datasets, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        printf("%s: failed to open %s\n",__func__, imatrix_file.c_str());
        exit(1);
    }
    int n_entries;
    in.read((char *)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        printf("%s: no data in file %s\n", __func__, imatrix_file.c_str());
        exit(1);
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            printf("%s: failed reading name for entry %d from %s\n", __func__, i+1, imatrix_file.c_str());
            exit(1);
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto & e = imatrix_data[name];
        int ncall;
        in.read((char *)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            printf("%s: failed reading number of values for entry %d\n", __func__, i);
            imatrix_data = {};
            exit(1);
        }
        e.resize(nval);
        in.read((char *)e.data(), nval*sizeof(float));
        if (in.fail()) {
            printf("%s: failed reading data for entry %d\n", __func__, i);
            imatrix_data = {};
            exit(1);
        }
        if (ncall > 0) {
            for (auto & v : e) {
                v /= ncall;
            }
        }

        if (getenv("LLAMA_TRACE")) {
            printf("%s: loaded data (size = %6d, ncall = %6d) for '%s'\n", __func__, int(e.size()), ncall, name.c_str());
        }
    }

    // latest legacy imatrix version contains the dataset filename at the end of the file
    int m_last_call = 0;
    if (in.peek() != EOF) {
        in.read((char *)&m_last_call, sizeof(m_last_call));
        int dataset_len;
        in.read((char *)&dataset_len, sizeof(dataset_len));
        std::vector<char> dataset_as_vec(dataset_len);
        in.read(dataset_as_vec.data(), dataset_len);
        imatrix_datasets.resize(1);
        imatrix_datasets[0].assign(dataset_as_vec.begin(), dataset_as_vec.end());
        printf("%s: imatrix dataset='%s'\n", __func__, imatrix_datasets[0].c_str());
    }
    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_call);
    return m_last_call;
}

static int load_imatrix(const std::string & imatrix_file, std::vector<std::string> & imatrix_datasets, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false, // the data is needed
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(imatrix_file.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: imatrix file '%s' is using old format\n", __func__, imatrix_file.c_str());
        return load_legacy_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
    }
    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        exit(1);
    }

    const int dataset_idx     = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_DATASETS);
    const int chunk_count_idx = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT);
    const int chunk_size_idx  = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE);
    if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        exit(1);
    }

    const uint32_t chunk_size = gguf_get_val_u32(ctx_gguf, chunk_size_idx);

    const std::string sums_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };

    // Using an ordered map to get a deterministic iteration order.
    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) { continue; }

        if (string_remove_suffix(name, sums_suffix)) {
            // in_sum2
            sums_counts_for[std::move(name)].first = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            // counts
            sums_counts_for[std::move(name)].second = cur;
        } else {
            // ignore other tensors
        }
    }

    for (const auto & sc : sums_counts_for) {
        const        std::string & name   = sc.first;
        const struct ggml_tensor * sums   = sc.second.first;
        const struct ggml_tensor * counts = sc.second.second;

        if (!sums || !counts) {
            fprintf(stderr, "%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            exit(1);
        }

        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];

        auto & e = imatrix_data[name];
        e.resize(ggml_nelements(sums));
        float max_count = 0.0f;
        for (int64_t j = 0; j < ne1; ++j) {
            const float count = ((const float *) counts->data)[j];
            if (count > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j*ne0 + i] = ((const float *) sums->data)[j*ne0 + i] / count;
                }
            } else {
                // Partial imatrix data, this tensor never got any input during calibration
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j*ne0 + i] = 1;
                }
            }
            if (count > max_count) {
                max_count = count;
            }
        }
        if (getenv("LLAMA_TRACE")) {
            printf("%s: loaded data (size = %6d, n_tokens = %6d, n_chunks = %6d) for '%s'\n", __func__, int(e.size()), int(max_count), int(max_count / chunk_size), name.c_str());
        }
    }

    int m_last_chunk = gguf_get_val_u32(ctx_gguf, chunk_count_idx);

    int64_t n_datasets = gguf_get_arr_n(ctx_gguf, dataset_idx);
    imatrix_datasets.reserve(n_datasets);
    for (int64_t i = 0; i < n_datasets; ++i) {
        imatrix_datasets.push_back(gguf_get_arr_str(ctx_gguf, dataset_idx, i));
    }
    printf("%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
    for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
        printf(", '%s'", imatrix_datasets[i].c_str());
    }
    printf("]\n");

    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_chunk);

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return m_last_chunk;
}

static int prepare_imatrix(const std::string & imatrix_file,
        std::vector<std::string> & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    int m_last_call = -1;
    if (!imatrix_file.empty()) {
        m_last_call = load_imatrix(imatrix_file, imatrix_dataset, imatrix_data);
    }
    if (imatrix_data.empty()) {
        return m_last_call;
    }
    if (!excluded_weights.empty()) {
        for (const auto & name : excluded_weights) {
            for (auto it = imatrix_data.begin(); it != imatrix_data.end();) {
                auto pos = it->first.find(name);
                if (pos != std::string::npos) {
                    it = imatrix_data.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp;
        for (const auto & name : included_weights) {
            for (auto & e : imatrix_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp.emplace(std::move(e));
                }
            }
        }
        imatrix_data = std::move(tmp);
    }
    if (!imatrix_data.empty()) {
        printf("%s: have %d importance matrix entries\n", __func__, int(imatrix_data.size()));
    }
    return m_last_call;
}

static ggml_type parse_ggml_type(const char * arg) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        auto type = (ggml_type)i;
        const auto * name = ggml_type_name(type);
        if (name && striequals(name, arg)) {
            return type;
        }
    }
    fprintf(stderr, "\n%s: invalid ggml_type '%s'\n\n", __func__, arg);
    return GGML_TYPE_COUNT;
}

static bool parse_tensor_type(const char * data, std::vector<tensor_quantization> & tensor_type) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr) {
        printf("\n%s: malformed tensor type '%s'\n\n", __func__, data);
        return false;
    }

    const size_t tn_len = sep - data;
    if (tn_len == 0) {
        printf("\n%s: missing tensor name\n\n", __func__);
        return false;
    }
    if (const size_t qt_len = strlen(sep); qt_len == 1) {
        printf("\n%s: missing quantization type\n\n", __func__);
        return false;
    }

    std::string tn(data, tn_len);
    std::transform(tn.begin(), tn.end(), tn.begin(), tolower);
    sep++;
    tensor_quantization tqz;
    tqz.name = tn;
    tqz.quant = parse_ggml_type(sep);
    tensor_type.emplace_back(std::move(tqz));
    if (tqz.quant == GGML_TYPE_COUNT) {
        printf("\n%s: invalid quantization type '%s'\n\n", __func__, sep);
        return false;
    }

    return true;
}

static bool parse_layer_prune(const char * data, std::vector<int> & prune_layers) {
    if (!data) {
        printf("\n%s: no layer pruning ids provided\n\n", __func__);
        return false;
    }

    const auto block_ids = string_split<std::string>(data, ',');
    for (const auto & block_id : block_ids) {
        int id;
        try {
            id = std::stoi(block_id);
        } catch (...) {
            id = -1;
        }
        if (id < 0) {
            printf("\n%s: invalid layer id '%s'\n\n", __func__, block_id.c_str());
            return false;
        }
        prune_layers.emplace_back(id);
    }

    sort(prune_layers.begin(), prune_layers.end());
    prune_layers.erase(std::unique(prune_layers.begin(), prune_layers.end()), prune_layers.end());
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<tensor_quantization> tensor_types;
    std::vector<int> prune_layers;
    bool allow_inplace = false;
    bool allow_overwrite = false;

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.output_tensor_type = parse_ggml_type(argv[++arg_idx]);
                if (params.output_tensor_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
            if (arg_idx < argc-1) {
                params.token_embedding_type = parse_ggml_type(argv[++arg_idx]);
                if (params.token_embedding_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--tensor-type") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type(argv[++arg_idx], tensor_types)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--prune-layers") == 0) {
            if (arg_idx == argc-1 || !parse_layer_prune(argv[++arg_idx], prune_layers)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--keep-split") == 0) {
            params.keep_split = true;
        } else if (strcmp(argv[arg_idx], "--inplace") == 0) {
            allow_inplace = true;
        } else if (strcmp(argv[arg_idx], "--overwrite") == 0) {
            allow_overwrite = true;
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_datasets, included_weights, excluded_weights, imatrix_data);
    if (!imatrix_data.empty()) {
        params.imatrix = &imatrix_data;
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, imatrix_file.c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (!imatrix_datasets.empty()) {
            llama_model_kv_override kvo;
            // TODO: list multiple datasets when there are more than one
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_DATASET);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, imatrix_datasets[0].c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }

        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = imatrix_data.size();
            kv_overrides.emplace_back(std::move(kvo));
        }

        if (m_last_call > 0) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = m_last_call;
            kv_overrides.emplace_back(std::move(kvo));
        }
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = &kv_overrides;
    }
    if (!tensor_types.empty()) {
        params.tensor_types = &tensor_types;
    }
    if (!prune_layers.empty()) {
        params.prune_layers = &prune_layers;
    }

    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        std::string fpath;
        const size_t pos = fname_inp.find_last_of("/\\");
        if (pos != std::string::npos) {
            fpath = fname_inp.substr(0, pos + 1);
        }

        // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
        fname_out = fpath + "ggml-model-" + ftype_str;
        if (!params.keep_split) {
            fname_out += suffix;
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[arg_idx]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if ((params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  ||
         params.ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_S  ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) && imatrix_data.empty()) {
        fprintf(stderr, "\n==========================================================================================================\n");
        fprintf(stderr, "Please do not use IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS or Q2_K_S quantization without an importance matrix\n");
        fprintf(stderr, "==========================================================================================================\n\n\n");
        return 1;
    }

    print_build_info();

    // Check if input and output refer to the same physical file
    // This prevents catastrophic data loss from truncating the input while reading it
    if (same_file(fname_inp, fname_out)) {
        if (!allow_inplace) {
            fprintf(stderr, "\n==========================================================================================================\n");
            fprintf(stderr, "ERROR: Input and output files are the same: '%s'\n", fname_inp.c_str());
            fprintf(stderr, "This would truncate the input file while reading it, causing data corruption and SIGBUS errors.\n");
            fprintf(stderr, "\n");
            fprintf(stderr, "Solutions:\n");
            fprintf(stderr, "  1. Specify a different output filename\n");
            fprintf(stderr, "  2. Use --inplace flag to enable safe in-place quantization (uses temp file + atomic rename)\n");
            fprintf(stderr, "==========================================================================================================\n\n");
            llama_backend_free();
            return 1;
        }
        fprintf(stderr, "%s: WARNING: in-place quantization detected, using temporary file for safety\n", __func__);
    }

    // Check if output file already exists (unless overwrite is allowed)
    if (!allow_overwrite && !allow_inplace) {
        std::error_code ec;
        if (std::filesystem::exists(fname_out, ec)) {
            fprintf(stderr, "\n==========================================================================================================\n");
            fprintf(stderr, "ERROR: Output file already exists: '%s'\n", fname_out.c_str());
            fprintf(stderr, "Use --overwrite flag to allow overwriting existing files.\n");
            fprintf(stderr, "==========================================================================================================\n\n");
            llama_backend_free();
            return 1;
        }
    }

    // Prepare actual output path (may be temporary for in-place or overwrite mode)
    std::string fname_out_actual = fname_out;
    std::string fname_out_temp;
    bool use_temp_file = allow_inplace && same_file(fname_inp, fname_out);

    if (use_temp_file || allow_overwrite) {
        // Create temp file in the same directory as output for atomic rename
        std::filesystem::path out_path(fname_out);
        std::filesystem::path out_dir = out_path.parent_path();
        if (out_dir.empty()) {
            out_dir = ".";
        }

        // Generate temp filename
        fname_out_temp = (out_dir / ("." + out_path.filename().string() + ".tmp.XXXXXX")).string();

        // Create the temp file safely
        // Note: mkstemp would be safer but requires char* and creates the file
        // For simplicity, we'll use a simpler approach with PID
        fname_out_temp = (out_dir / ("." + out_path.filename().string() + ".tmp." + std::to_string(getpid()))).string();
        fname_out_actual = fname_out_temp;

        fprintf(stderr, "%s: using temporary file: '%s'\n", __func__, fname_out_actual.c_str());
    }

    fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out_actual.c_str(), &params)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());

            // Clean up temp file on failure
            if (!fname_out_temp.empty()) {
                std::error_code ec;
                std::filesystem::remove(fname_out_temp, ec);
            }

            llama_backend_free();
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    // If we used a temp file, atomically rename it to the final output
    if (!fname_out_temp.empty()) {
        fprintf(stderr, "%s: atomically moving temp file to final output\n", __func__);
        std::error_code ec;

        // On POSIX systems, rename() is atomic when both paths are on the same filesystem
        std::filesystem::rename(fname_out_temp, fname_out, ec);

        if (ec) {
            fprintf(stderr, "%s: failed to rename temp file '%s' to '%s': %s\n",
                    __func__, fname_out_temp.c_str(), fname_out.c_str(), ec.message().c_str());

            // Try to clean up temp file
            std::filesystem::remove(fname_out_temp, ec);

            llama_backend_free();
            return 1;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    llama_backend_free();

    return 0;
}
