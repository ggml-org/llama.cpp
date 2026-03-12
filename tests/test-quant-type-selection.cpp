#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-quant.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf-model-data.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// ftype name <-> enum mapping
// ---------------------------------------------------------------------------

struct ftype_name_entry {
    const char * name;
    llama_ftype  ftype;
};

static const ftype_name_entry ftype_name_table[] = {
    { "F32",        LLAMA_FTYPE_ALL_F32 },
    { "F16",        LLAMA_FTYPE_MOSTLY_F16 },
    { "BF16",       LLAMA_FTYPE_MOSTLY_BF16 },
    { "Q4_0",       LLAMA_FTYPE_MOSTLY_Q4_0 },
    { "Q4_1",       LLAMA_FTYPE_MOSTLY_Q4_1 },
    { "Q5_0",       LLAMA_FTYPE_MOSTLY_Q5_0 },
    { "Q5_1",       LLAMA_FTYPE_MOSTLY_Q5_1 },
    { "Q8_0",       LLAMA_FTYPE_MOSTLY_Q8_0 },
    { "Q2_K",       LLAMA_FTYPE_MOSTLY_Q2_K },
    { "Q2_K_S",     LLAMA_FTYPE_MOSTLY_Q2_K_S },
    { "Q3_K_S",     LLAMA_FTYPE_MOSTLY_Q3_K_S },
    { "Q3_K_M",     LLAMA_FTYPE_MOSTLY_Q3_K_M },
    { "Q3_K_L",     LLAMA_FTYPE_MOSTLY_Q3_K_L },
    { "Q4_K_S",     LLAMA_FTYPE_MOSTLY_Q4_K_S },
    { "Q4_K_M",     LLAMA_FTYPE_MOSTLY_Q4_K_M },
    { "Q5_K_S",     LLAMA_FTYPE_MOSTLY_Q5_K_S },
    { "Q5_K_M",     LLAMA_FTYPE_MOSTLY_Q5_K_M },
    { "Q6_K",       LLAMA_FTYPE_MOSTLY_Q6_K },
    { "IQ1_S",      LLAMA_FTYPE_MOSTLY_IQ1_S },
    { "IQ1_M",      LLAMA_FTYPE_MOSTLY_IQ1_M },
    { "IQ2_XXS",    LLAMA_FTYPE_MOSTLY_IQ2_XXS },
    { "IQ2_XS",     LLAMA_FTYPE_MOSTLY_IQ2_XS },
    { "IQ2_S",      LLAMA_FTYPE_MOSTLY_IQ2_S },
    { "IQ2_M",      LLAMA_FTYPE_MOSTLY_IQ2_M },
    { "IQ3_XXS",    LLAMA_FTYPE_MOSTLY_IQ3_XXS },
    { "IQ3_XS",     LLAMA_FTYPE_MOSTLY_IQ3_XS },
    { "IQ3_S",      LLAMA_FTYPE_MOSTLY_IQ3_S },
    { "IQ3_M",      LLAMA_FTYPE_MOSTLY_IQ3_M },
    { "IQ4_NL",     LLAMA_FTYPE_MOSTLY_IQ4_NL },
    { "IQ4_XS",     LLAMA_FTYPE_MOSTLY_IQ4_XS },
    { "TQ1_0",      LLAMA_FTYPE_MOSTLY_TQ1_0 },
    { "TQ2_0",      LLAMA_FTYPE_MOSTLY_TQ2_0 },
    { "MXFP4_MOE",  LLAMA_FTYPE_MOSTLY_MXFP4_MOE },
};

static llama_ftype llama_ftype_from_name(const char * name) {
    for (const auto & e : ftype_name_table) {
        if (strcmp(name, e.name) == 0) {
            return e.ftype;
        }
    }
    return (llama_ftype)-1;
}

static const char * llama_ftype_to_name(llama_ftype ftype) {
    for (const auto & e : ftype_name_table) {
        if (e.ftype == ftype) {
            return e.name;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Mock tensor construction - may be better to extract this in the future
// ---------------------------------------------------------------------------

struct mock_tensor {
    ggml_context_ptr ctx;
    ggml_tensor *    tensor;
};

static mock_tensor make_mock_tensor(const std::string & name,
                                    int64_t             ne0,
                                    int64_t             ne1,
                                    int64_t             ne2 = 1,
                                    int64_t             ne3 = 1) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * t;
    if (ne3 > 1) {
        t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    } else if (ne2 > 1) {
        t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    } else {
        t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    }
    ggml_set_name(t, name.c_str());
    return { std::move(ctx), t };
}

static ggml_type ggml_type_from_name(const std::string & name) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * tname = ggml_type_name((ggml_type) i);
        if (tname && name == tname) {
            return (ggml_type) i;
        }
    }
    return GGML_TYPE_COUNT;
}

// ---------------------------------------------------------------------------
// File parser for snapshot files (quant type schemas)
// ---------------------------------------------------------------------------

struct snapshot_section {
    llama_ftype                                    ftype;
    ggml_type                                      default_type;
    std::vector<std::pair<std::string, ggml_type>> overrides;
};

// This function is pretty ugly, but it's a trade-off of readable snapshot files
// versus readable parsing code
static bool parse_snapshot_file(const std::string & path, std::vector<snapshot_section> & sections) {
    std::ifstream f(path);
    if (!f.good()) {
        return false;
    }

    snapshot_section * cur = nullptr;
    std::string        line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // section header: [FTYPE_NAME] default_type
        if (line[0] == '[') {
            auto close = line.find(']');
            if (close == std::string::npos) {
                fprintf(stderr, "parse error: missing ] in '%s'\n", line.c_str());
                return false;
            }
            std::string ftype_str = line.substr(1, close - 1);
            std::string default_str;
            size_t      pos = close + 1;
            while (pos < line.size() && line[pos] == ' ') {
                pos++;
            }
            default_str = line.substr(pos);

            llama_ftype ftype = llama_ftype_from_name(ftype_str.c_str());
            if ((int) ftype < 0) {
                fprintf(stderr, "parse error: unknown ftype '%s'\n", ftype_str.c_str());
                return false;
            }

            ggml_type dtype = ggml_type_from_name(default_str);
            if (dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "parse error: unknown default type '%s'\n", default_str.c_str());
                return false;
            }

            sections.push_back({ ftype, dtype, {} });
            cur = &sections.back();
            continue;
        }

        if (!cur) {
            fprintf(stderr, "parse error: tensor line before any section: '%s'\n", line.c_str());
            return false;
        }

        auto sp = line.rfind(' ');
        if (sp == std::string::npos) {
            fprintf(stderr, "parse error: no space in tensor line: '%s'\n", line.c_str());
            return false;
        }

        std::string tname = line.substr(0, sp);
        std::string ttype = line.substr(sp + 1);

        ggml_type gt = ggml_type_from_name(ttype);
        if (gt == GGML_TYPE_COUNT) {
            fprintf(stderr, "parse error: unknown type '%s' for tensor '%s'\n", ttype.c_str(), tname.c_str());
            return false;
        }

        cur->overrides.push_back({ tname, gt });
    }

    return true;
}

// ---------------------------------------------------------------------------
// Remote model support using gguf-model-data.cpp
// ---------------------------------------------------------------------------

struct remote_model_spec {
    const char * repo;
    const char * quant;
};

// Get model name from repo: strip org prefix, strip -GGUF suffix,
// and strip anything up to and including first '_' (e.g. "deepseek-ai_DeepSeek-V3.1").
static std::string model_name_from_repo(const char * repo) {
    std::string s(repo);

    auto slash = s.find('/');
    if (slash != std::string::npos) {
        s = s.substr(slash + 1);
    }

    const std::string suffix = "-GGUF";
    if (s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0) {
        s = s.substr(0, s.size() - suffix.size());
    }

    auto underscore = s.find('_');
    if (underscore != std::string::npos) {
        s = s.substr(underscore + 1);
    }

    return s;
}

static std::string snapshot_file_from_name(const std::string & name) {
    std::string lower = name;
    for (auto & c : lower) {
        c = std::tolower(c);
    }
    return lower;
}

static const remote_model_spec model_specs[] = {
    { "ggml-org/Qwen3-0.6B-GGUF",                   "Q8_0"   },
    { "ggml-org/GLM-4.6V-GGUF",                     "Q8_0"   },
    { "ggml-org/Step-3.5-Flash-GGUF",               "Q4_K"   },
    { "ggml-org/Qwen3-Coder-Next-GGUF",             "Q8_0"   },
    { "ggml-org/Qwen3-14B-GGUF",                    "Q8_0"   },
    { "ggml-org/Nemotron-Nano-3-30B-A3B-GGUF",      "Q8_0"   },
    { "ggml-org/gpt-oss-120b-GGUF",                 "mxfp4"  },
    { "ggml-org/gemma-3-4b-it-GGUF",                "Q8_0"   },
    { "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF", "Q4_K_M" },
    { "bartowski/deepseek-ai_DeepSeek-V3.1-GGUF",   "IQ1_M"  },
    { "bartowski/Qwen_Qwen3.5-397B-A17B-GGUF",      "IQ1_S"  }, // TODO: swap with ggml-org if/when it's released
    { "bartowski/Qwen_Qwen3.5-27B-GGUF",            "Q8_0"   }, // TODO: swap with ggml-org if/when it's released
};

static const int n_model_specs = (int) (sizeof(model_specs) / sizeof(model_specs[0]));

// Determine llm_type from metadata.
// Only LLM_TYPE_70B matters -> probably can/should be dropped in the future
static llm_type infer_llm_type(llm_arch arch, const gguf_remote_model & remote) {
    if (arch == LLM_ARCH_LLAMA && remote.n_layer == 80 && remote.n_head != remote.n_head_kv) {
        return LLM_TYPE_70B;
    }
    return LLM_TYPE_UNKNOWN;
}

static std::unique_ptr<llama_model> build_mock_model_from_remote(const gguf_remote_model & remote) {
    struct llama_model_params mparams = llama_model_default_params();
    auto                      model   = std::make_unique<llama_model>(mparams);

    model->arch = llm_arch_from_string(remote.architecture);
    model->type = infer_llm_type(model->arch, remote);

    model->hparams.n_embd        = remote.n_embd;
    model->hparams.n_embd_head_k_full = remote.n_embd_head_k;
    model->hparams.n_embd_head_v_full = remote.n_embd_head_v;
    model->hparams.n_layer       = remote.n_layer;
    model->hparams.n_expert      = remote.n_expert;

    for (uint32_t i = 0; i < remote.n_layer; i++) {
        model->hparams.n_head_arr[i]    = remote.n_head;
        model->hparams.n_head_kv_arr[i] = remote.n_head_kv;
        model->hparams.n_ff_arr[i]      = remote.n_ff;
    }

    return model;
}

static std::vector<mock_tensor> build_mock_tensors(const gguf_remote_model &           remote,
                                                   llm_arch                            arch,
                                                   const llama_model_quantize_params & qparams) {
    std::vector<mock_tensor> result;

    for (const auto & t : remote.tensors) {
        auto mt = make_mock_tensor(t.name, t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
        if (tensor_allows_quantization(&qparams, arch, mt.tensor)) {
            result.push_back(std::move(mt));
        }
    }

    return result;
}

static std::string read_file_contents(const std::string & path) {
    std::ifstream f(path);
    if (!f.good()) {
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// Compute quantization type assignments per target ftype
// ---------------------------------------------------------------------------

// Returns {tensor_name, assigned_type} for each tensor, in order.
static std::vector<std::pair<std::string, ggml_type>> compute_quant_types(llama_model &                    mdl,
                                                                          const std::vector<mock_tensor> & tensors,
                                                                          llama_ftype                      ftype) {
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.ftype                       = ftype;

    quantize_state_impl qs(mdl, &qparams);

    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        metadata[i].name = tensors[i].tensor->name;
    }
    init_quantize_state_counters(qs, metadata);

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    std::vector<std::pair<std::string, ggml_type>> result;
    result.reserve(tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_type got = llama_tensor_get_type(qs, &qparams, tensors[i].tensor, default_type, metadata[i]);
        result.push_back({ metadata[i].name, got });
    }

    return result;
}

// ---------------------------------------------------------------------------
// Generate mode: regenerate all snapshot files
// Use this when either adding new models or modifying quants
// ---------------------------------------------------------------------------

static std::string generate_snapshot(const std::string &              name,
                                     const gguf_remote_model &        remote,
                                     llama_model &                    mdl,
                                     const std::vector<mock_tensor> & tensors) {
    std::ostringstream out;

    out << "# Model: " << name << "\n";
    out << "# n_embd=" << remote.n_embd << ", n_ff=" << remote.n_ff << ", n_vocab=" << remote.n_vocab
        << ", n_layer=" << remote.n_layer << ", n_head=" << remote.n_head << ", n_head_kv=" << remote.n_head_kv;
    if (remote.n_expert > 0) {
        out << ", n_expert=" << remote.n_expert;
    }
    out << "\n";

    for (int i = 0; i < LLAMA_FTYPE_GUESSED; i++) {
        llama_ftype ft           = (llama_ftype) i;
        ggml_type   default_type = llama_ftype_get_default_type(ft);
        if (default_type == GGML_TYPE_COUNT) {
            continue;
        }
        const char * fname = llama_ftype_to_name(ft);
        if (!fname) {
            continue;
        }

        auto types = compute_quant_types(mdl, tensors, ft);

        out << "\n[" << fname << "] " << ggml_type_name(default_type) << "\n";
        for (const auto & [name, type] : types) {
            if (type != default_type) {
                out << name << " " << ggml_type_name(type) << "\n";
            }
        }
    }

    return out.str();
}

static int run_generate(const std::string & snapshot_dir) {
    fprintf(stderr, "This will overwrite all snapshot files in:\n  %s\n", snapshot_dir.c_str());
    fprintf(stderr, "Continue? [y/N] ");
    int ch = fgetc(stdin);
    if (ch != 'y' && ch != 'Y') {
        fprintf(stderr, "Aborted.\n");
        return 1;
    }

    fprintf(stderr, "\n");

    int n_written = 0;

    for (int m = 0; m < n_model_specs; m++) {
        const auto & spec = model_specs[m];
        std::string  name = model_name_from_repo(spec.repo);

        fprintf(stderr, "Fetching model metadata for %s from %s...\n", name.c_str(), spec.repo);
        auto result = gguf_fetch_model_meta(spec.repo, spec.quant);
        if (!result.has_value()) {
            fprintf(stderr, "ERROR: could not fetch model metadata for %s\n", name.c_str());
            return 1;
        }

        const auto &                remote  = result.value();
        auto                        model   = build_mock_model_from_remote(remote);
        llama_model_quantize_params qparams = llama_model_quantize_default_params();
        auto                        tensors = build_mock_tensors(remote, model->arch, qparams);

        std::string content = generate_snapshot(name, remote, *model, tensors);
        std::string path    = snapshot_dir + "/" + snapshot_file_from_name(name) + ".schema";

        std::ofstream f(path);
        if (!f.good()) {
            fprintf(stderr, "ERROR: could not write %s\n", path.c_str());
            return 1;
        }
        f << content;
        n_written++;
        fprintf(stderr, "  wrote %s\n", path.c_str());
    }

    fprintf(stderr, "%d files written\n", n_written);
    return 0;
}

// ---------------------------------------------------------------------------
// Test mode: compare against snapshot files
// ---------------------------------------------------------------------------

static bool run_test_section(llama_model &                    mdl,
                             const std::vector<mock_tensor> & tensors,
                             const snapshot_section &         section) {
    // verify default_type matches what llama_ftype_get_default_type returns
    ggml_type computed_default = llama_ftype_get_default_type(section.ftype);
    if (computed_default != section.default_type) {
        printf("  FAIL  [%s] default type mismatch: file says %s, code says %s\n", llama_ftype_to_name(section.ftype),
               ggml_type_name(section.default_type), ggml_type_name(computed_default));
        return false;
    }

    auto types = compute_quant_types(mdl, tensors, section.ftype);

    std::map<std::string, ggml_type> override_map(section.overrides.begin(), section.overrides.end());

    bool all_pass         = true;
    int  n_override_found = 0;

    for (const auto & [name, got] : types) {
        ggml_type expected = section.default_type;
        auto      it       = override_map.find(name);
        if (it != override_map.end()) {
            expected = it->second;
            n_override_found++;
        }

        if (got != expected) {
            printf("  FAIL  %-50s %-10s expected %s, got %s\n", name.c_str(), llama_ftype_to_name(section.ftype), ggml_type_name(expected), ggml_type_name(got));
            all_pass = false;
        }
    }

    if (n_override_found != (int) section.overrides.size()) {
        printf("  FAIL  [%s] override count mismatch: listed %d, matched %d\n", llama_ftype_to_name(section.ftype),
               (int) section.overrides.size(), n_override_found);
        all_pass = false;
    }

    return all_pass;
}

static int run_remote_tests(const std::string & snapshot_dir, const char * argv0) {
    int total_pass = 0;
    int total_fail = 0;
    int total_skip = 0;

    for (int m = 0; m < n_model_specs; m++) {
        const auto & spec = model_specs[m];
        std::string  name = model_name_from_repo(spec.repo);
        printf("=== %s ===\n", name.c_str());

        auto result = gguf_fetch_model_meta(spec.repo, spec.quant, "", false);
        if (!result.has_value()) {
            printf("  SKIP  (could not fetch model metadata)\n\n");
            total_skip++;
            continue;
        }

        const auto &                remote  = result.value();
        auto                        model   = build_mock_model_from_remote(remote);
        llama_model_quantize_params qparams = llama_model_quantize_default_params();
        auto                        tensors = build_mock_tensors(remote, model->arch, qparams);

        std::string                   snapshot_path = snapshot_dir + "/" + snapshot_file_from_name(name) + ".schema";
        std::vector<snapshot_section> sections;
        if (!parse_snapshot_file(snapshot_path, sections)) {
            printf("  SKIP  (could not read snapshot file: %s)\n\n", snapshot_path.c_str());
            total_skip++;
            continue;
        }

        int model_pass = 0;
        int model_fail = 0;

        for (const auto & section : sections) {
            bool pass = run_test_section(*model, tensors, section);
            if (pass) {
                model_pass++;
            } else {
                model_fail++;
            }
        }

        printf("  %s  %s: %d/%d ftype sections passed (%d tensors)\n", model_fail == 0 ? "PASS" : "FAIL", name.c_str(),
               model_pass, model_pass + model_fail, (int) tensors.size());
        printf("\n");

        if (model_fail == 0) {
            total_pass++;
        } else {
            total_fail++;
        }
    }

    printf("%d/%d models passed", total_pass, total_pass + total_fail);
    if (total_skip > 0) {
        printf(", %d skipped", total_skip);
    }
    printf("\n");

    if (total_fail > 0) {
        printf("\nIf these changes are intentional, regenerate snapshot files with:\n");
        printf("  %s --generate\n", argv0);
    }

    return total_fail > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    std::string snapshot_dir = SNAPSHOT_DIR;
    bool        generate     = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--snapshot-dir") == 0 && i + 1 < argc) {
            snapshot_dir = argv[++i];
        } else if (strcmp(argv[i], "--generate") == 0) {
            generate = true;
        }
    }

    if (generate) {
        return run_generate(snapshot_dir);
    }

    // suppress llama log warnings during test (e.g. tensor type fallback messages)
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    return run_remote_tests(snapshot_dir, argv[0]);
}
