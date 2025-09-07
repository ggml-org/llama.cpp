#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "sampling.h"
#include <algorithm>

#include <cstdio>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include "ggml-backend.h"

#include <fstream>

#include <unordered_map>
#include <memory>
#include <filesystem>
#include <unordered_set>

std::ofstream prompt_output_file;
std::ofstream tensor_output_file;


// sanitize names like "blk.0.output" -> "blk_0_output"
static std::string sanitize(const std::string &s) {
    std::string out = s;
    for (char &c : out) {
        if (c == '/' || c == '\\' || c == ' ' || c == ':' || c == '.' ) c = '_';
    }
    return out;
}
struct callback_data {
    std::vector<uint8_t> data;

    std::unordered_set<std::string> exact_targets;
    std::vector<std::string>        prefix_targets;

    int  current_token_index = -1;
    bool list_mode = false;

    // NEW: per-tensor streams + base directory
    std::string base_dir; // e.g., "<output_prefix>/tensors"
    std::unordered_map<std::string, std::unique_ptr<std::ofstream>> streams;
};

struct sampling_cfg {
    int   top_k   = -1;    // <1 = disabled
    float top_p   = 1.0f;  // >=1 = disabled
    float temp    = 1.0f;  // we will always apply temperature (min clamp)
};

static bool matches_target(const std::string &name, const callback_data *cb) {
    if (cb->exact_targets.find(name) != cb->exact_targets.end()) return true;
    for (const auto &pref : cb->prefix_targets) {
        if (name.rfind(pref, 0) == 0) return true; // starts_with
    }
    return false;
}


static std::ostream & get_stream_for(const std::string &name, callback_data *cb) {
    auto it = cb->streams.find(name);
    if (it != cb->streams.end()) return *it->second;

    const std::string fname = cb->base_dir + "/" + sanitize(name) + ".txt";
    auto ofs = std::make_unique<std::ofstream>(fname, std::ios::app);
    if (!ofs->is_open()) {
        // fall back to global file if something goes wrong
        return tensor_output_file;
    }
    std::ostream &ref = *ofs;
    cb->streams.emplace(name, std::move(ofs));
    return ref;
}




static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor_block(std::ostream &os,
                                    const std::string& tensor_name,
                                    uint8_t * data, ggml_type type,
                                    const int64_t * ne, const size_t * nb,
                                    int64_t token_idx) {
    const int64_t dim = ne[0];

    os << "=== TOKEN " << token_idx << " ===\n";
    os << "--- TENSOR: " << tensor_name << " ---\n";
    os << "SHAPE: [" << dim << "]\n";
    os << "DATA:\n";

    for (int64_t i = 0; i < dim; ++i) {
        size_t offset = i * nb[0];
        float v;

        switch (type) {
            case GGML_TYPE_F16: v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[offset]); break;
            case GGML_TYPE_F32: v = *(float *) &data[offset]; break;
            default: GGML_ABORT("Unsupported tensor type");
        }

        os << v;
        if (i < dim - 1) os << ", ";
    }
    os << "\n\n";
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (callback_data *) user_data;
    const std::string name = t->name;

    if (ask) {
        if (cb->list_mode) {
            // print once per tensor name, return false so we don't hook/copy data
            static std::unordered_set<std::string> printed;
            if (printed.insert(name).second) {
                tensor_output_file << name << "\n";
            }
            return false;
        }
        // normal (non-list) mode: only hook matches
        return matches_target(name, cb);
    }

    if (cb->list_mode) {
        // we already printed in the ask branch
        return false;
    }

    if (!matches_target(name, cb)) return false;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb->data.data(), 0, n_bytes);
    }
    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb->data.data();
        std::ostream &os = get_stream_for(name, cb);
        ggml_print_tensor_block(os, name, data, t->type, t->ne, t->nb, cb->current_token_index);
        os.flush();
    }
    return true;
}


static bool run(llama_context * ctx,
                const common_params & params,
                const sampling_cfg & samp,
                callback_data & cb_data){
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    auto chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);

    // Always apply provided temperature (clamped to >0 above)
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(samp.temp));

    // Optional: top-k
    if (samp.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(samp.top_k));
    }

    // Optional: top-p
    if (samp.top_p < 1.0f) {
        // min_keep = 1 is sane
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(samp.top_p, 1));
    }

    // Add RNG distribution so temp/top-k/top-p actually randomize
    uint32_t seed = (uint32_t) ggml_time_us();
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));


    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    cb_data.current_token_index = -1;
    if (llama_decode(ctx, batch)) {
        LOG_ERR("Failed to evaluate prompt\n");
        llama_sampler_free(sampler);
        return false;
    }

    std::string result;
    llama_token token;

    for (int i = 0; i < params.n_predict; ++i) {
        token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            LOG_ERR("Failed to convert token to string\n");
            llama_sampler_free(sampler);
            return false;
        }
        result += std::string(buf, n);  // <-- store instead of printing

        llama_batch new_batch = llama_batch_get_one(&token, 1);
        cb_data.current_token_index = i;
        if (llama_decode(ctx, new_batch)) {
            LOG_ERR("Failed to decode sampled token\n");
            llama_sampler_free(sampler);
            return false;
        }
    }

    llama_sampler_free(sampler);

    // Output final result
    prompt_output_file << "\n\nFull output:\n" << result << "\n";

    return true;
}


int main(int argc, char **argv) {
    std::string output_prefix = "default";

    callback_data cb_data;
    sampling_cfg samp;    // <-- add this

    common_params params;
    bool list_layers = false;
    std::string list_layers_filter = "";
    std::vector<std::string> parse_layer_values; // multi or comma-separated
    std::vector<char*> filtered_argv;
    std::vector<std::string> prompts;

    filtered_argv.push_back(argv[0]);
    params.n_gpu_layers = 20;

// --------- ARG PARSING ---------
for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.compare(0, 2, "--") == 0) {
        std::replace(arg.begin(), arg.end(), '_', '-');
    }

    // --parse-layer <a,b,c>
    if (arg == "--parse-layer") {
        if (i + 1 < argc) {
            std::string raw = argv[++i];
            size_t start = 0;
            while (true) {
                size_t pos  = raw.find(',', start);
                std::string item = raw.substr(start, pos - start);
                if (!item.empty()) parse_layer_values.push_back(item);
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        } else {
            fprintf(stderr, "error: --parse-layer requires an argument\n");
            return 1;
        }
        continue;
    }

    // --prompt "..."
    if (arg == "--prompt") {
        if (i + 1 < argc) {
            prompts.emplace_back(argv[++i]);
        } else {
            fprintf(stderr, "error: --prompt requires an argument\n");
            return 1;
        }
        continue;
    }

    // --top-k N
    if (arg == "--top-k") {
        if (i + 1 < argc) {
            samp.top_k = std::stoi(argv[++i]);
            if (samp.top_k < 1) samp.top_k = -1;   // disable if <1
        } else {
            fprintf(stderr, "error: --top-k requires an int\n");
            return 1;
        }
        continue;
    }

    // --top-p F
    if (arg == "--top-p") {
        if (i + 1 < argc) {
            samp.top_p = std::stof(argv[++i]);
            if (samp.top_p <= 0.0f) samp.top_p = 1.0f;
            if (samp.top_p > 1.0f)  samp.top_p = 1.0f; // clamp
        } else {
            fprintf(stderr, "error: --top-p requires a float\n");
            return 1;
        }
        continue;
    }

    // --temp F   (or --temperature F)
    if (arg == "--temp" || arg == "--temperature") {
        if (i + 1 < argc) {
            samp.temp = std::stof(argv[++i]);
            if (samp.temp <= 0.0f) samp.temp = 1e-6f; // avoid greedy (force >0)
        } else {
            fprintf(stderr, "error: --temperature requires a float\n");
            return 1;
        }
        continue;
    }

    // --output-prefix STR
    if (arg == "--output-prefix") {
        if (i + 1 < argc) {
            output_prefix = argv[++i];
        } else {
            fprintf(stderr, "error: --output-prefix requires a string argument\n");
            return 1;
        }
        continue;
    }

    // --n-gpu-layers N
    if (arg == "--n-gpu-layers") {
        if (i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "error: --n-gpu-layers requires an integer argument\n");
            return 1;
        }
        continue;
    }

    // --list-layers [optional_filter]
    if (arg == "--list-layers") {
        list_layers = true;
        if (i + 1 < argc && argv[i + 1][0] != '-') {
            list_layers_filter = argv[++i];  // optional, currently unused
        }
        continue;
    }

    // Unrecognized flag/arg: pass through to common_params_parse
    filtered_argv.push_back(argv[i]);
}


    // open standard outputs
    prompt_output_file.open(output_prefix + "_prompt_output.txt");
    tensor_output_file.open(output_prefix + "_tensor_output.txt");
    if (!prompt_output_file || !tensor_output_file) {
        std::cerr << "❌ Failed to open output files.\n";
        return 1;
    }

    // create tensors dir AFTER we know output_prefix
    try {
        std::filesystem::create_directories(output_prefix + "/tensors");
    } catch (const std::exception &e) {
        std::cerr << "❌ Failed to create tensors directory: " << e.what() << "\n";
        return 1;
    }
    cb_data.base_dir = output_prefix + "/tensors";

    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // configure selector sets
    if (list_layers) {
        cb_data.list_mode = true;
    } else {
        if (parse_layer_values.empty()) {
            // sensible default (keeps legacy behavior)
            cb_data.exact_targets.insert("l_out-31");
        } else {
            for (auto s : parse_layer_values) {
                if (s == "__LIST__") { cb_data.list_mode = true; continue; }
                if (!s.empty() && s.back() == '*') {
                    s.pop_back(); // treat trailing * as prefix
                    if (!s.empty()) cb_data.prefix_targets.push_back(s);
                } else {
                    cb_data.exact_targets.insert(s);
                }
            }
        }
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    if (prompts.empty()) {
        prompts.emplace_back("What is the capital of France?");  // fallback
    }

    if (cb_data.list_mode) {
        params.n_predict = 1;
        params.prompt = "dummy";  // any valid prompt to trigger eval

        if (!run(ctx, params, samp, cb_data)) {
            LOG_ERR("Failed during layer listing run\n");
            return 1;
        }
        prompt_output_file.close();
        tensor_output_file.close();
        // close any opened per-tensor streams
        for (auto &kv : cb_data.streams) {
            if (kv.second && kv.second->is_open()) kv.second->close();
        }
        return 0;
    }

    for (const auto& prompt : prompts) {
        prompt_output_file << "Running prompt: " << prompt << "\n";
        params.prompt = prompt;
        if (!run(ctx, params, samp, cb_data)) {
            LOG_ERR("Failed on prompt: %s\n", prompt.c_str());
            return 1;
        }
    }

    LOG_INF("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();
    prompt_output_file.close();
    tensor_output_file.close();
    for (auto &kv : cb_data.streams) {
        if (kv.second && kv.second->is_open()) kv.second->close();
    }
    return 0;
}
