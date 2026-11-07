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

// result of parsing --tensor-type option
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
    { "Q2_0",     LLAMA_FTYPE_MOSTLY_Q2_0,     " 2.25 bpw quantization (group 64)",  },
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
    printf("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file]\n");
    printf("       [--prune-layers] [--keep-split] [--override-kv] [--dry-run] [--max-buffer-size] [--verify]\n");
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
    printf("                                      this is an advanced option to selectively quantize tensors. may be specified multiple times.\n");
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
    printf("                                      override model metadata by key in the quantized model. may be specified multiple times.\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --dry-run\n");
    printf("                                      calculate and show the final quantization size without performing quantization\n");
    printf("                                      example: llama-quantize --dry-run model-f32.gguf Q4_K\n");
    printf("  --max-buffer-size MiB\n");
    printf("                                      max amount of tensor rows kept in memory while quantizing one tensor (default: 8192)\n");
    printf("                                      lower it to quantize models with very large tensors on a machine with little RAM\n");
    printf("  --verify\n");
    printf("                                      dequantize the output model and report per-tensor quantization error as JSON\n");
    printf("                                      prints one JSON object with max_abs_error, rms_error and error histogram per tensor\n");
    printf("                                      exit 0 on success, exit 1 on structural failure (missing tensor, bad read),\n");
    printf("                                      exit 2 when a tensor max_abs_error exceeds --verify-max-error (default 1.0)\n");
    printf("  --verify-max-error float\n");
    printf("                                      fail --verify with exit 2 when any tensor max_abs_error exceeds this value\n");
    printf("                                      (default 1.0; must be > 0)\n\n");
    printf("note: --include-weights and --exclude-weights cannot be used together\n\n");
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

static int load_imatrix(const std::string & imatrix_file, std::vector<std::string> & imatrix_datasets, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
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
        auto & e = imatrix_data[name];
        e.resize(entry.sums.size());

        if (!loaded.is_legacy) {
            // GGUF format: normalize by per-expert counts
            const int64_t ncounts = entry.counts.size();
            const int64_t ne0     = (int64_t) entry.sums.size() / ncounts;

            for (int64_t j = 0; j < ncounts; ++j) {
                const float count = (float) entry.counts[j];
                if (count > 0.0f) {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = entry.sums[j*ne0 + i] / count;
                    }
                } else {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = 1;
                    }
                }
            }

            if (getenv("LLAMA_TRACE")) {
                float max_count = 0.0f;
                for (int64_t j = 0; j < ncounts; ++j) {
                    const float count = (float) entry.counts[j];
                    if (count > max_count) {
                        max_count = count;
                    }
                }
                printf("%s: loaded data (size = %6d, n_tokens = %6d, n_chunks = %6d) for '%s'\n",
                       __func__, int(e.size()), int(max_count), int(max_count / loaded.chunk_size), name.c_str());
            }
        } else {
            // Legacy format: sums contain (raw/count)*ncall, divide by ncall
            const int64_t ncall = entry.counts.empty() ? 0 : entry.counts[0];
            if (ncall > 0) {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i] / ncall;
                }
            } else {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i];
                }
            }

            if (getenv("LLAMA_TRACE")) {
                printf("%s: loaded data (size = %6d, ncall = %6d) for '%s'\n",
                       __func__, int(e.size()), int(ncall), name.c_str());
            }
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

    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), loaded.chunk_count);

    return loaded.chunk_count;
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

constexpr size_t VERIFY_HISTOGRAM_BUCKETS = 150;
constexpr double VERIFY_HISTOGRAM_RANGE = 0.03;
// threshold for the --verify error gate. measured healthy Q8_0 max errors are ~0.004 and Q4_K_M ~0.03,
// while a 1 KB corruption produced 710996. 1.0 is far above any sane quantization error for typical
// weight magnitudes and far below real corruption. weights with much larger magnitudes can exceed it,
// so the flag --verify-max-error exists to raise it.
constexpr double VERIFY_DEFAULT_MAX_ERROR = 1.0;

// JSON schema printed by --verify (one object on stdout):
// { "input": "<path>", "output": "<path>", "tensors": [
//   { "name": "<tensor>", "type": "<ggml type>", "nelements": N,
//     "max_abs_error": E, "rms_error": R, "histogram": [c0 .. c149] } ] }
// histogram[i] counts |diff| in [i*RANGE/BUCKETS, (i+1)*RANGE/BUCKETS); the last bucket clamps.
// output tensors that were not quantized report zero error. a tensor missing from the output is a hard error.
// after the JSON is printed, a tensor with max_abs_error above the gate (default 1.0, see --verify-max-error)
// fails with exit code 2.

static void json_escape_append(std::string & out, const char * s) {
    for (; *s; ++s) {
        switch (*s) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) *s < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char) *s);
                    out += buf;
                } else {
                    out += *s;
                }
        }
    }
}

static void json_append_double(std::string & out, double v) {
    if (!std::isfinite(v)) {
        // corrupt tensors can dequantize to inf/nan; never emit invalid JSON
        out += "null";
        return;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.10g", v);
    out += buf;
}

// convert raw tensor bytes to float; handles f32, f16, bf16 directly and any quantized type via to_float
static bool tensor_to_float(const uint8_t * data, ggml_type type, int64_t nelements, std::vector<float> & out) {
    out.resize(nelements);
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(out.data(), data, nelements * sizeof(float));
            return true;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, out.data(), nelements);
            return true;
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) data, out.data(), nelements);
            return true;
        default:
            break;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        return false;
    }
    traits->to_float(data, out.data(), nelements);
    return true;
}

// dequantize every output tensor and compare against the input; print one JSON object on stdout
// returns 0 on success, 1 on structural failure (missing tensor, bad read), 2 when a tensor
// max_abs_error exceeds max_error. the JSON is always printed before a gate failure so the data is kept.
// non-finite max_abs_error (inf/nan from corrupt scales) also fails the gate.
static int verify_quantization(const char * fname_inp, const char * fname_out, double max_error) {
    struct ggml_context * meta_in = NULL;
    struct ggml_context * meta_out = NULL;

    struct gguf_init_params gparams = { true, &meta_in };
    struct gguf_context * ctx_in = gguf_init_from_file(fname_inp, gparams);
    if (!ctx_in) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname_inp);
        return 1;
    }
    gparams.ctx = &meta_out;
    struct gguf_context * ctx_out = gguf_init_from_file(fname_out, gparams);
    if (!ctx_out) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname_out);
        gguf_free(ctx_in);
        return 1;
    }

    std::ifstream fin_in(fname_inp, std::ios::binary);
    std::ifstream fin_out(fname_out, std::ios::binary);
    if (!fin_in.is_open() || !fin_out.is_open()) {
        fprintf(stderr, "%s: failed to reopen model files for reading\n", __func__);
        gguf_free(ctx_in);
        gguf_free(ctx_out);
        return 1;
    }

    const size_t data_off_in  = gguf_get_data_offset(ctx_in);
    const size_t data_off_out = gguf_get_data_offset(ctx_out);

    std::vector<uint8_t> buf_in, buf_out;
    std::vector<float> f_in, f_out;
    std::string json = "{ \"input\": \"";
    json_escape_append(json, fname_inp);
    json += "\", \"output\": \"";
    json_escape_append(json, fname_out);
    json += "\", \"tensors\": [";

    int rc = 0;
    const int64_t n_tensors = gguf_get_n_tensors(ctx_in);
    std::string worst_tensor;
    double worst_error = 0.0;
    for (int64_t i = 0; i < n_tensors && rc == 0; ++i) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        const int64_t j = gguf_find_tensor(ctx_out, name);
        if (j < 0) {
            fprintf(stderr, "%s: tensor '%s' missing from output\n", __func__, name);
            rc = 1;
            break;
        }

        struct ggml_tensor * t_in  = ggml_get_tensor(meta_in, name);
        struct ggml_tensor * t_out = ggml_get_tensor(meta_out, name);
        const int64_t nelements = ggml_nelements(t_in);
        if (nelements != ggml_nelements(t_out)) {
            fprintf(stderr, "%s: tensor '%s' changed shape\n", __func__, name);
            rc = 1;
            break;
        }

        const ggml_type type_in  = gguf_get_tensor_type(ctx_in, i);
        const ggml_type type_out = gguf_get_tensor_type(ctx_out, j);

        buf_in.resize(ggml_nbytes(t_in));
        fin_in.seekg(data_off_in + gguf_get_tensor_offset(ctx_in, i));
        fin_in.read((char *) buf_in.data(), buf_in.size());
        buf_out.resize(ggml_nbytes(t_out));
        fin_out.seekg(data_off_out + gguf_get_tensor_offset(ctx_out, j));
        fin_out.read((char *) buf_out.data(), buf_out.size());
        if (!fin_in || !fin_out) {
            fprintf(stderr, "%s: failed reading tensor '%s'\n", __func__, name);
            rc = 1;
            break;
        }

        if (!tensor_to_float(buf_in.data(), type_in, nelements, f_in)) {
            fprintf(stderr, "%s: cannot decode input tensor '%s' of type %s\n", __func__, name, ggml_type_name(type_in));
            rc = 1;
            break;
        }

        double max_abs = 0.0;
        double sum_sq = 0.0;
        uint64_t hist[VERIFY_HISTOGRAM_BUCKETS] = {};

        if (ggml_is_quantized(type_out)) {
            if (nelements % ggml_blck_size(type_out) != 0) {
                fprintf(stderr, "%s: tensor '%s' size is not a multiple of block size\n", __func__, name);
                rc = 1;
                break;
            }
            if (!tensor_to_float(buf_out.data(), type_out, nelements, f_out)) {
                fprintf(stderr, "%s: cannot decode output tensor '%s' of type %s\n", __func__, name, ggml_type_name(type_out));
                rc = 1;
                break;
            }
            for (int64_t k = 0; k < nelements; ++k) {
                const double diff = (double) f_in[k] - (double) f_out[k];
                const double adiff = fabs(diff);
                sum_sq += diff * diff;
                if (adiff > max_abs) {
                    max_abs = adiff;
                }
                size_t b = (size_t) (adiff / VERIFY_HISTOGRAM_RANGE * VERIFY_HISTOGRAM_BUCKETS);
                if (b >= VERIFY_HISTOGRAM_BUCKETS) {
                    b = VERIFY_HISTOGRAM_BUCKETS - 1;
                }
                hist[b]++;
            }
        }

        if (i > 0) {
            json += ",";
        }
        json += "\n    { \"name\": \"";
        json_escape_append(json, name);
        json += "\", \"type\": \"";
        json += ggml_type_name(type_out);
        json += "\", \"nelements\": ";
        json += std::to_string(nelements);
        json += ", \"max_abs_error\": ";
        json_append_double(json, max_abs);
        json += ", \"rms_error\": ";
        json_append_double(json, nelements > 0 ? sqrt(sum_sq / (double) nelements) : 0.0);
        json += ", \"histogram\": [";
        for (size_t b = 0; b < VERIFY_HISTOGRAM_BUCKETS; ++b) {
            if (b > 0) {
                json += ",";
            }
            json += std::to_string(hist[b]);
        }
        json += "] }";

        // NaN max_abs_error (corrupt scales dequantize to inf/nan) must also fail the gate;
        // !(x <= limit) is false for NaN, so this catches it.
        if (!(max_abs <= max_error)) {
            if (worst_tensor.empty() || !(max_abs <= worst_error)) {
                worst_tensor = name;
                worst_error = max_abs;
            }
        }
    }

    gguf_free(ctx_in);
    gguf_free(ctx_out);

    if (rc != 0) {
        return rc;
    }

    json += "\n] }\n";
    printf("%s", json.c_str());

    if (!worst_tensor.empty()) {
        fprintf(stderr, "%s: tensor '%s' max_abs_error %.10g exceeds limit %.10g\n",
                __func__, worst_tensor.c_str(), worst_error, max_error);
        return 2;
    }

    return 0;
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
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<tensor_type_option> tensor_type_opts;
    std::vector<int> prune_layers;
    bool verify = false;
    double verify_max_error = VERIFY_DEFAULT_MAX_ERROR;

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
        } else if (strcmp(argv[arg_idx], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[arg_idx], "--verify-max-error") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            verify_max_error = atof(argv[++arg_idx]);
            if (!(verify_max_error > 0.0)) {
                fprintf(stderr, "%s: invalid --verify-max-error '%s'\n", __func__, argv[arg_idx]);
                return 1;
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
        } else if (strcmp(argv[arg_idx], "--max-buffer-size") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            const int mib = atoi(argv[++arg_idx]);
            if (mib <= 0) {
                fprintf(stderr, "%s: invalid --max-buffer-size '%s'\n", __func__, argv[arg_idx]);
                return 1;
            }
            params.max_buf_size = (size_t) mib * 1024 * 1024;
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
    if (verify) {
        if (params.dry_run) {
            fprintf(stderr, "%s: --verify requires an actual quantization, not --dry-run\n", __func__);
            return 1;
        }
        if (params.keep_split) {
            fprintf(stderr, "%s: --verify is not supported with --keep-split\n", __func__);
            return 1;
        }
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_datasets, included_weights, excluded_weights, imatrix_data);

    std::vector<llama_model_imatrix_data> i_data;
    std::vector<llama_model_tensor_override> t_override;
    if (!imatrix_data.empty()) {
        i_data.reserve(imatrix_data.size() + 1);
        for (const auto & kv : imatrix_data) {
            i_data.push_back({kv.first.c_str(), kv.second.data(), kv.second.size()});
        }
        i_data.push_back({nullptr, nullptr, 0});  // array terminator
        params.imatrix = i_data.data();
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
        params.kv_overrides = kv_overrides.data();
    }
    if (!tensor_type_opts.empty()) {
        t_override.reserve(tensor_type_opts.size() + 1);
        for (const auto & tt : tensor_type_opts) {
            t_override.push_back({tt.name.c_str(), tt.type});
        }
        t_override.push_back({nullptr, GGML_TYPE_COUNT});  // array terminator
        params.tt_overrides = t_override.data();
    }
    if (!prune_layers.empty()) {
        prune_layers.push_back(-1);  // array terminator
        params.prune_layers = prune_layers.data();
    }

    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
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

    if (!params.dry_run) {
        if (std::error_code ec; std::filesystem::equivalent(fname_inp, fname_out, ec)) {
            fprintf(stderr, "%s: error: input and output files are the same: '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }
    }

    llama_print_build_info(llama_version());

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

    if (verify) {
        const int vrc = verify_quantization(fname_inp.c_str(), fname_out.c_str(), verify_max_error);
        if (vrc) {
            fprintf(stderr, "%s: verification failed\n", __func__);
            return vrc;
        }
    }

    llama_backend_free();

    return 0;
}
