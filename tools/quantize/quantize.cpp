#include "llama.h"

#include "build-info.h"
#include "common.h"
#include "imatrix-loader.h"

#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <filesystem>

// changes to this struct must also be reflected in src/llama-quant.cpp
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q1_0",     LLAMA_FTYPE_MOSTLY_Q1_0,     " 1.125 bpw quantization",           },
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

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";


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
    printf("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file] [--prune-layers]\n");
    printf("       [--keep-split] [--override-kv] [--dry-run] [--target-bpw] [--target-size] [--save-state] [--state-file]\n");
    printf("       model-f32.gguf [model-quant.gguf] type [nthreads]\n\n");
    printf("  --allow-requantize\n");
    printf("                                      allow requantizing tensors that have already been quantized\n");
    printf("                                      WARNING: this can severely reduce quality compared to quantizing\n");
    printf("                                               from 16bit or 32bit!\n");
    printf("  --leave-output-tensor\n");
    printf("                                      leave output.weight un(re)quantized\n");
    printf("                                      increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure\n");
    printf("                                      disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --imatrix file_name\n");
    printf("                                      use data in file_name as importance matrix for quant optimizations\n");
    printf("  --include-weights tensor_name\n");
    printf("                                      use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name\n");
    printf("                                      do not use importance matrix for this/these tensor(s)\n");
    printf("  --output-tensor-type ggml_type\n");
    printf("                                      use this ggml_type for the output.weight tensor\n");
    printf("  --token-embedding-type ggml_type\n");
    printf("                                      use this ggml_type for the token embeddings tensor\n");
    printf("  --tensor-type tensor_name=ggml_type\n");
    printf("                                      quantize this tensor to this ggml_type\n");
    printf("                                      this is an advanced option to selectively quantize tensors; may be specified multiple times.\n");
    printf("                                      example: --tensor-type attn_q=q8_0\n");
    printf("  --tensor-type-file tensor_types.txt\n");
    printf("                                      list of tensors to quantize to a specific ggml_type\n");
    printf("                                      this is an advanced option to selectively quantize a long list of tensors.\n");
    printf("                                      the file should use the same format as above, separated by spaces or newlines.\n");
    printf("  --prune-layers L0,L1,L2...\n");
    printf("                                      comma-separated list of layer numbers to prune from the model\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --keep-split\n");
    printf("                                      generate quantized model in the same shards as input\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("                                      override model metadata by key in the quantized model; may be specified multiple times.\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --dry-run\n");
    printf("                                      calculate and show the final quantization size without performing quantization\n");
    printf("                                      example: llama-quantize --dry-run model-f32.gguf Q4_K\n\n");
    printf("  --target-bpw n\n");
    printf("                                      target a bits per weight (bpw); must be a positive number between 0.0 and 16.0\n");
    printf("                                      advanced option to automatically select quantization types to achieve a total bits per weight (bpw) target\n\n");
    printf("  --target-size n\n");
    printf("                                      target a file size; must be a positive number\n");
    printf("                                      advanced option to automatically select quantization types to achieve a total file size\n");
    printf("                                      allowed units: b, k|kib, m|mib, g|gib, t|tib; defaults to b (bytes) if none is provided\n\n");
    printf("  --state-file [filename]\n");
    printf("                                      file name to use/save; if none is provided, the default name will be used\n\n");
    printf("note: --include-weights and --exclude-weights cannot be used together\n");
    printf("      --target-bpw and --target-size cannot be used together\n\n");
    printf("-----------------------------------------------------------------------------\n");
    printf(" allowed quantization types\n");
    printf("-----------------------------------------------------------------------------\n\n");
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

static int load_imatrix(const std::string & imatrix_file,
    std::vector<std::string> & imatrix_datasets,
    std::unordered_map<std::string, std::vector<float>> & values_data,
    std::unordered_map<std::string, std::vector<float>> & activations_data,
    std::unordered_map<std::string, std::vector<float>> & statistics_data)
{
    common_imatrix loaded;
    if (!common_imatrix_load(imatrix_file, loaded)) {
        fprintf(stderr, "%s: failed to load imatrix from '%s'\n", __func__, imatrix_file.c_str());
        exit(1);
    }

    if (!loaded.is_legacy && !loaded.has_metadata) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        exit(1);
    }

    for (const auto & [name, entry] : loaded.entries) {
        const int64_t ncounts = (int64_t) entry.counts.size();
        if (ncounts == 0) {
            fprintf(stderr, "%s: no counts for entry '%s', skipping\n", __func__, name.c_str());
            continue;
        }

        const int64_t nsums = (int64_t) entry.sums.size();
        if (nsums == 0) {
            continue;
        }
        const int64_t ne0 = nsums / ncounts;

        auto & values = values_data[name];
        values.resize(nsums);

        if (!loaded.is_legacy) {
            for (int64_t j = 0; j < ncounts; ++j) {
                const float count = (float) entry.counts[j];
                if (count > 0.0f) {
                    for (int64_t i = 0; i < ne0; ++i) {
                        values[j*ne0 + i] = entry.sums[j*ne0 + i] / count;
                    }
                } else {
                    for (int64_t i = 0; i < ne0; ++i) {
                        values[j*ne0 + i] = 1;
                    }
                }
            }
        } else {
            const int64_t ncall = entry.counts[0];
            if (ncall > 0) {
                for (int64_t i = 0; i < nsums; ++i) {
                    values[i] = entry.sums[i] / ncall;
                }
            } else {
                for (int64_t i = 0; i < nsums; ++i) {
                    values[i] = entry.sums[i];
                }
            }
        }

        if (!entry.activations.empty()) {
            auto & activations = activations_data[name];
            activations.resize(entry.activations.size());
            if (!loaded.is_legacy) {
                for (int64_t j = 0; j < ncounts; ++j) {
                    const float count = (float) entry.counts[j];
                    if (count > 0.0f) {
                        for (int64_t i = 0; i < ne0; ++i) {
                            activations[j*ne0 + i] = entry.activations[j*ne0 + i] / count;
                        }
                    } else {
                        for (int64_t i = 0; i < ne0; ++i) {
                            activations[j*ne0 + i] = 0;
                        }
                    }
                }
            } else {
                const int64_t ncall = entry.counts[0];
                if (ncall > 0) {
                    for (size_t i = 0; i < entry.activations.size(); ++i) {
                        activations[i] = entry.activations[i] / ncall;
                    }
                } else {
                    for (size_t i = 0; i < entry.activations.size(); ++i) {
                        activations[i] = entry.activations[i];
                    }
                }
            }
        }

        if (!entry.stats.empty()) {
            statistics_data[name] = entry.stats;
        }
    }

    imatrix_datasets = std::move(loaded.datasets);

    if (!imatrix_datasets.empty()) {
        printf("%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
        for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
            printf(", '%s'", imatrix_datasets[i].c_str());
        }
        printf("]\n");
    }

    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(values_data.size()), imatrix_file.c_str(), loaded.chunk_count);

    return loaded.chunk_count;
}

static int prepare_imatrix(const std::string & imatrix_file,
        std::vector<std::string> & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & values_data,
        std::unordered_map<std::string, std::vector<float>> & activations_data,
        std::unordered_map<std::string, std::vector<float>> & statistics_data) {
    int m_last_call = -1;
    if (!imatrix_file.empty()) {
        m_last_call = load_imatrix(imatrix_file, imatrix_dataset, values_data, activations_data, statistics_data);
    }
    if (values_data.empty()) {
        return m_last_call;
    }
    if (!excluded_weights.empty()) {
        for (const auto & name : excluded_weights) {
            for (auto vt = values_data.begin(); vt != values_data.end();) {
                auto pos = vt->first.find(name);
                if (pos != std::string::npos) {
                    vt = values_data.erase(vt);
                } else {
                    ++vt;
                }
            }
            for (auto at = activations_data.begin(); at != activations_data.end();) {
                auto pos = at->first.find(name);
                if (pos != std::string::npos) {
                    at = activations_data.erase(at);
                } else {
                    ++at;
                }
            }
            for (auto st = statistics_data.begin(); st != statistics_data.end();) {
                auto pos = st->first.find(name);
                if (pos != std::string::npos) {
                    st = statistics_data.erase(st);
                } else {
                    ++st;
                }
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp_values;
        std::unordered_map<std::string, std::vector<float>> tmp_activations;
        std::unordered_map<std::string, std::vector<float>> tmp_statistics;
        for (const auto & name : included_weights) {
            for (auto & e : values_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp_values.emplace(std::move(e));
                }
            }
            for (auto & a : activations_data) {
                auto pos = a.first.find(name);
                if (pos != std::string::npos) {
                    tmp_activations.emplace(std::move(a));
                }
            }
            for (auto & s : statistics_data) {
                auto pos = s.first.find(name);
                if (pos != std::string::npos) {
                    tmp_statistics.emplace(std::move(s));
                }
            }
        }
        values_data = std::move(tmp_values);
        activations_data = std::move(tmp_activations);
        statistics_data = std::move(tmp_statistics);
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

static bool parse_tensor_type(const char * data, std::vector<tensor_type_option> & tensor_type) {
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
    tensor_type_option tensor_type_opt;
    tensor_type_opt.name = tn;
    tensor_type_opt.type = parse_ggml_type(sep);
    tensor_type.emplace_back(std::move(tensor_type_opt));
    if (tensor_type_opt.type == GGML_TYPE_COUNT) {
        printf("\n%s: invalid quantization type '%s'\n\n", __func__, sep);
        return false;
    }

    return true;
}

static bool parse_tensor_type_file(const char * filename, std::vector<tensor_type_option> & tensor_type) {
    std::ifstream file(filename);
    if (!file) {
        printf("\n%s: failed to open file '%s': %s\n\n", __func__, filename, std::strerror(errno));
        return false;
    }

    std::string arg;
    while (file >> arg) {
        if (!parse_tensor_type(arg.c_str(), tensor_type)) {
            return false;
        }
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

static bool parse_target_bpw(const char * data, float & target_bpw) {
    if (!data) {
        printf("\n%s: no target bits per weight (bpw) provided\n\n", __func__);
        return false;
    }

    try {
        target_bpw = std::stof(data);
        if (target_bpw < 0.0f || target_bpw > 16.0f) {
            printf("\n%s: target bits per weight (bpw) must be a positive number between 0.0 and 16.0\n\n", __func__);
            return false;
        }
    }
    catch (const std::exception & e) {
        printf("\n%s: '%s' is not valid. Target bits per weight (bpw) must be a positive number between 0.0 and 16.0\n\n", __func__, data);
        return false;
    }

    return true;
}

static bool parse_target_size(const char * data, int64_t & target_size) {
    if (!data) {
        printf("\n%s: no target file size provided\n\n", __func__);
        return false;
    }

    char * end = nullptr;
    const double val = std::strtod(data, &end);
    if (end == data || val < 0) {
        printf("\n%s: invalid target file size '%s'\n\n", __func__, data);
        return false;
    }

    std::string suffix(end);
    for (auto & c : suffix) { c = std::tolower(c); }

    int64_t mul = 0;
    if (suffix.empty() || suffix == "b") {
        mul = 1;
    } else if (suffix == "k" || suffix == "kib") {
        mul = 1024;
    } else if (suffix == "m" || suffix == "mib") {
        mul = 1024 * 1024;
    } else if (suffix == "g" || suffix == "gib") {
        mul = 1024 * 1024 * 1024;
    } else if (suffix == "t" || suffix == "tib") {
        mul = 1024LL * 1024 * 1024 * 1024;
    } else {
        printf("\n%s: invalid unit '%s' in '%s'. Allowed: b, k or kib, m or mib, g or gib, t or tib\n\n", __func__, suffix.c_str(), data);
        return false;
    }

    target_size = (int64_t)val * mul;
    return true;
}

static const char * get_ftype(const float bpw) {
    const std::map<float, const char *> quant_bpw = {
        {1.5625, "IQ1_S"},
        {1.7500, "IQ1_M"},
        {2.0625, "IQ2_XXS"},
        {2.3125, "IQ2_XS"},
        {2.5625, "IQ2_S"},
        {2.6250, "Q2_K"},
        {3.0625, "IQ3_XXS"},
        {3.4375, "Q3_K"},
        {4.2500, "IQ4_XS"},
        {4.5000, "Q4_K"},
        {5.5000, "Q5_K"},
        {6.5625, "Q6_K"},
        {8.5000, "Q8_0"},
#ifdef GGML_USE_METAL
        {16.0000, "F16"}
#else
        {16.0000, "BF16"}
#endif
    };

    return quant_bpw.lower_bound(bpw)->second;
}

// satisfies -Wmissing-declarations
int llama_quantize(int argc, char ** argv);

int llama_quantize(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::vector<std::string> included_weights;
    std::vector<std::string> excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<tensor_type_option> tensor_type_opts;
    std::vector<int> prune_layers;
    float target_bpw = -1.0f;
    int64_t target_size = -1;

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
            if (arg_idx == argc-1 || !parse_tensor_type(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--tensor-type-file") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type_file(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--target-bpw") == 0) {
            if (arg_idx == argc-1 || !parse_target_bpw(argv[++arg_idx], target_bpw)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--target-size") == 0) {
            if (arg_idx == argc-1 || !parse_target_size(argv[++arg_idx], target_size)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--state-file") == 0) {
            if (arg_idx + 1 < argc && argv[arg_idx + 1][0] != '-') {
                params.state_file = argv[++arg_idx];
            } else {
                static char empty[] = "";
                params.state_file = empty;
            }
        } else if (strcmp(argv[arg_idx], "--prune-layers") == 0) {
            if (arg_idx == argc-1 || !parse_layer_prune(argv[++arg_idx], prune_layers)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--dry-run") == 0) {
            params.dry_run = true;
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
    if (target_bpw != -1.0f && target_size != -1) {
        usage(argv[0]);
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> values_data;
    std::unordered_map<std::string, std::vector<float>> activations_data;
    std::unordered_map<std::string, std::vector<float>> statistics_data;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_datasets, included_weights, excluded_weights, values_data, activations_data, statistics_data);

    std::vector<llama_model_imatrix_data> vdata;
    std::vector<llama_model_imatrix_data> adata;
    std::vector<llama_model_imatrix_data> sdata;
    std::vector<llama_model_tensor_override> tto;

    if (!values_data.empty()) {
        vdata.reserve(values_data.size() + 1);
        for (const auto & kv : values_data) { vdata.push_back({kv.first.c_str(), kv.second.data(), kv.second.size()}); }
        vdata.push_back({nullptr, nullptr, 0});  // array terminator
        params.imatrix = vdata.data();
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
            kvo.val_i64 = values_data.size();
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
    if (!activations_data.empty()) {
        adata.reserve(activations_data.size() + 1);
        for (const auto & kv : activations_data) { adata.push_back({kv.first.c_str(), kv.second.data(), kv.second.size()}); }
        adata.push_back({nullptr, nullptr, 0});  // array terminator
        params.activations = adata.data();
    }
    if (!statistics_data.empty()) {
        sdata.reserve(statistics_data.size() + 1);
        for (const auto & kv : statistics_data) { sdata.push_back({kv.first.c_str(), kv.second.data(), kv.second.size()}); }
        sdata.push_back({nullptr, nullptr, 0});  // array terminator
        params.statistics = sdata.data();
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = kv_overrides.data();
    }
    if (!tensor_type_opts.empty()) {
        tto.reserve(tensor_type_opts.size() + 1);
        for (const auto & tt : tensor_type_opts) { tto.push_back({tt.name.c_str(), tt.type}); }
        tto.push_back({nullptr, GGML_TYPE_COUNT});  // array terminator
        params.tt_overrides = tto.data();
    }
    if (!prune_layers.empty()) {
        prune_layers.push_back(-1);  // array terminator
        params.prune_layers = prune_layers.data();
    }
    if (target_bpw != -1.0f) {
        params.target_bpw = target_bpw;
    }
    if (target_size != -1) {
        params.target_size = target_size;
    }

    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    std::vector<const char *> tmp_argv(argv, argv + argc);
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        // argv[arg_idx] is the ftype directly: <input> <ftype>
        if (!params.dry_run) {
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
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        // argv[arg_idx] is not a valid ftype, so treat it as output path: <input> <output> <ftype>
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        // If --target-bpw or --target-size are set, select a quantization type and ignore any user specified type
        if (params.target_bpw != -1.0f || params.target_size != -1) {
            auto * ftype = params.target_bpw != -1.0f ? const_cast<char *>(get_ftype(params.target_bpw)) : const_cast<char *>("F16");
            bool added_type = false;
            if (argc == arg_idx) {
                tmp_argv.push_back(ftype);
                added_type = true;
            } else {
                tmp_argv[arg_idx] = ftype;
            }
            tmp_argv.push_back(nullptr);
            argv = const_cast<char **>(tmp_argv.data());
            if (added_type) {
                argc++;
            }
        } else if (argc <= arg_idx) {
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

    if (!params.dry_run) {
        if (std::error_code ec; std::filesystem::equivalent(fname_inp, fname_out, ec)) {
            fprintf(stderr, "%s: error: input and output files are the same: '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }
    }

    llama_print_build_info();

    if (params.dry_run) {
        fprintf(stderr, "%s: calculating quantization size for '%s' as %s", __func__, fname_inp.c_str(), ftype_str.c_str());
    } else {
        fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    }

    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
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
