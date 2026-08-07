#include "dataset.h"
#include "hard-prune.h"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "moe-prune.h"

#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct options {
    std::string command;
    std::string model;
    std::string dataset;
    std::string profile;
    std::string output;
    std::string output_dir;
    std::string metric = "router-output";
    aikar_ppl_mask mask = aikar_ppl_mask::ASSISTANT;
    std::vector<double> ratios;
    double max_layer_ratio = 0.25;
    int32_t seed = 42;
    int32_t n_ctx = 4096;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_threads = -1;
    int32_t n_gpu_layers = -1;
};

struct route_layer_state {
    std::vector<int32_t> ids;
    std::vector<float> probabilities;
    int64_t n_used = 0;
    int64_t n_tokens = 0;
};

struct route_collector {
    int32_t n_expert = 0;
    bool collect_output_norm = false;
    common_moe_prune_stats stats;
    std::map<int32_t, route_layer_state> pending;
    uint64_t invalid_routing = 0;
    double entropy_sum = 0.0;
    uint64_t entropy_tokens = 0;
};

struct evaluation_result {
    std::array<double, 4> nll = {};
    std::array<int64_t, 4> evaluated = {};
    int64_t total_tokens = 0;
    int64_t processed_tokens = 0;
    double elapsed_seconds = 0.0;
    double throughput = 0.0;
    double router_load_imbalance = 0.0;
    double router_entropy = 0.0;
    uint64_t invalid_routing = 0;

    double ppl(aikar_ppl_mask mask) const {
        const size_t i = (size_t) mask;
        return evaluated[i] == 0 ? INFINITY : std::exp(nll[i] / evaluated[i]);
    }
};

void usage() {
    std::cout <<
        "usage:\n"
        "  aikar-prune analyze --model MODEL --dataset DATA --ratios RATIO,... --output-dir DIR [options]\n"
        "  aikar-prune inspect --model MODEL --profile PROFILE\n"
        "  aikar-prune hard --model MODEL --profile PROFILE --output MODEL [--dataset DATA]\n\n"
        "options:\n"
        "  --metric router-output\n"
        "  --ppl-mask all|assistant|reasoning|content\n"
        "  --max-layer-ratio RATIO\n"
        "  baseline-checkpoint.json in --output-dir is reused when model, dataset, mask, and context match\n"
        "  --seed N\n"
        "  --ctx-size N --batch-size N --ubatch-size N\n"
        "  --threads N --n-gpu-layers N\n";
}

std::vector<double> parse_ratios(const std::string & value) {
    std::vector<double> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::runtime_error("empty pruning ratio");
        size_t used = 0;
        double ratio = std::stod(item, &used);
        if (used != item.size()) throw std::runtime_error("invalid pruning ratio: " + item);
        result.push_back(ratio);
    }
    return result;
}

options parse_options(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("missing subcommand");
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        usage();
        std::exit(0);
    }
    options result;
    result.command = argv[1];
    if (result.command != "analyze" && result.command != "inspect" && result.command != "hard") {
        throw std::runtime_error("unknown subcommand: " + result.command);
    }
    auto value = [&](int & i) -> std::string {
        if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i - 1]);
        return argv[i];
    };
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); std::exit(0); }
        else if (arg == "--model" || arg == "-m") result.model = value(i);
        else if (arg == "--dataset") result.dataset = value(i);
        else if (arg == "--profile") result.profile = value(i);
        else if (arg == "--output") result.output = value(i);
        else if (arg == "--output-dir") result.output_dir = value(i);
        else if (arg == "--ratios") result.ratios = parse_ratios(value(i));
        else if (arg == "--metric") result.metric = value(i);
        else if (arg == "--ppl-mask") result.mask = aikar_ppl_mask_parse(value(i));
        else if (arg == "--max-layer-ratio") result.max_layer_ratio = std::stod(value(i));
        else if (arg == "--seed") result.seed = std::stoi(value(i));
        else if (arg == "--ctx-size") result.n_ctx = std::stoi(value(i));
        else if (arg == "--batch-size") result.n_batch = std::stoi(value(i));
        else if (arg == "--ubatch-size") result.n_ubatch = std::stoi(value(i));
        else if (arg == "--threads") result.n_threads = std::stoi(value(i));
        else if (arg == "--n-gpu-layers" || arg == "-ngl") result.n_gpu_layers = std::stoi(value(i));
        else if (arg == "--validate") {}
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (result.model.empty()) throw std::runtime_error("--model is required");
    if (result.command == "analyze" && (result.dataset.empty() || result.ratios.empty() || result.output_dir.empty())) {
        throw std::runtime_error("analyze requires --dataset, --ratios, and --output-dir");
    }
    if ((result.command == "inspect" || result.command == "hard") && result.profile.empty()) throw std::runtime_error("--profile is required");
    if (result.command == "hard" && result.output.empty()) throw std::runtime_error("hard requires --output");
    if (result.metric != "router-output") throw std::runtime_error("unsupported importance metric: " + result.metric);
    if (result.n_ctx < 2 || result.n_batch < 1 || result.n_ubatch < 1) throw std::runtime_error("invalid context or batch size");
    return result;
}

common_params make_common_params(const options & opts) {
    common_params params;
    params.model.path = opts.model;
    params.n_ctx = opts.n_ctx;
    params.n_batch = opts.n_batch;
    params.n_ubatch = opts.n_ubatch;
    params.n_gpu_layers = opts.n_gpu_layers;
    params.cpuparams.n_threads = opts.n_threads;
    params.cpuparams_batch.n_threads = opts.n_threads;
    params.sampling.seed = opts.seed;
    params.warmup = false;
    return params;
}

int32_t tensor_layer(const char * name, const char * prefix) {
    int32_t layer = -1;
    std::string pattern = std::string(prefix) + "-%d";
    return sscanf(name, pattern.c_str(), &layer) == 1 ? layer : -1;
}

std::vector<uint8_t> tensor_bytes(ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        memcpy(result.data(), tensor->data, result.size());
    } else {
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    }
    return result;
}

float tensor_float(const std::vector<uint8_t> & data, ggml_type type, size_t index) {
    if (type == GGML_TYPE_F32) return reinterpret_cast<const float *>(data.data())[index];
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(data.data())[index]);
    if (type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(reinterpret_cast<const ggml_bf16_t *>(data.data())[index]);
    throw std::runtime_error(std::string("unsupported calibration tensor type: ") + ggml_type_name(type));
}

bool route_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    route_collector & collector = *static_cast<route_collector *>(user_data);
    const std::string name = tensor->name;
    const bool wanted = name.rfind("ffn_moe_topk-", 0) == 0 || name.rfind("ffn_moe_weights_norm-", 0) == 0 ||
                        (collector.collect_output_norm && name.rfind("ffn_moe_down-", 0) == 0);
    if (ask) return wanted;
    if (!wanted) return true;

    if (name.rfind("ffn_moe_topk-", 0) == 0) {
        const int32_t layer = tensor_layer(name.c_str(), "ffn_moe_topk");
        route_layer_state & state = collector.pending[layer];
        state.n_used = tensor->ne[0];
        state.n_tokens = tensor->ne[1];
        const std::vector<uint8_t> data = tensor_bytes(tensor);
        const int32_t * ids = reinterpret_cast<const int32_t *>(data.data());
        state.ids.assign(ids, ids + state.n_used * state.n_tokens);
        return true;
    }
    if (name.rfind("ffn_moe_weights_norm-", 0) == 0) {
        const int32_t layer = tensor_layer(name.c_str(), "ffn_moe_weights_norm");
        route_layer_state & state = collector.pending[layer];
        if (state.ids.empty()) return true;
        const std::vector<uint8_t> data = tensor_bytes(tensor);
        state.probabilities.resize(state.n_used * state.n_tokens);
        for (size_t i = 0; i < state.probabilities.size(); ++i) state.probabilities[i] = tensor_float(data, tensor->type, i);
        auto & layer_stats = collector.stats[layer];
        if (layer_stats.empty()) layer_stats.resize(collector.n_expert);
        for (int64_t token = 0; token < state.n_tokens; ++token) {
            double sum = 0.0;
            for (int64_t slot = 0; slot < state.n_used; ++slot) {
                const size_t index = token * state.n_used + slot;
                const int32_t expert = state.ids[index];
                const float probability = state.probabilities[index];
                if (expert < 0 || expert >= collector.n_expert || !std::isfinite(probability)) {
                    ++collector.invalid_routing;
                    continue;
                }
                ++layer_stats[expert].selection_count;
                layer_stats[expert].probability_sum += probability;
                sum += std::max(0.0f, probability);
            }
            if (sum > 0.0) {
                double entropy = 0.0;
                for (int64_t slot = 0; slot < state.n_used; ++slot) {
                    const double p = std::max(0.0f, state.probabilities[token * state.n_used + slot]) / sum;
                    if (p > 0.0) entropy -= p * std::log(p);
                }
                collector.entropy_sum += entropy;
                ++collector.entropy_tokens;
            }
        }
        return true;
    }

    const int32_t layer = tensor_layer(name.c_str(), "ffn_moe_down");
    route_layer_state & state = collector.pending[layer];
    if (state.ids.empty() || state.probabilities.empty() || tensor->ne[1] != state.n_used || tensor->ne[2] != state.n_tokens) return true;
    const std::vector<uint8_t> data = tensor_bytes(tensor);
    auto & layer_stats = collector.stats[layer];
    for (int64_t token = 0; token < state.n_tokens; ++token) {
        for (int64_t slot = 0; slot < state.n_used; ++slot) {
            const size_t route_index = token * state.n_used + slot;
            const int32_t expert = state.ids[route_index];
            if (expert < 0 || expert >= collector.n_expert) continue;
            double sum_sq = 0.0;
            const size_t base = (token * state.n_used + slot) * tensor->ne[0];
            for (int64_t i = 0; i < tensor->ne[0]; ++i) {
                const double value = tensor_float(data, tensor->type, base + i);
                sum_sq += value * value;
            }
            const double norm = std::sqrt(sum_sq);
            layer_stats[expert].output_norm_sum += norm;
            layer_stats[expert].weighted_output_sum += norm * state.probabilities[route_index];
        }
    }
    return true;
}

struct loaded_model {
    common_init_result_ptr init;
    llama_context_ptr context;
};

loaded_model load_model(const options & opts, route_collector * collector, const common_moe_prune_profile * profile) {
    common_params params = make_common_params(opts);
    if (collector != nullptr) {
        params.cb_eval = route_callback;
        params.cb_eval_user_data = collector;
    }
    loaded_model result;
    result.init = common_init_from_params(params, true);
    if (!result.init || result.init->model() == nullptr) throw std::runtime_error("failed to load model");
    if (profile != nullptr) common_moe_prune_profile_apply(result.init->model(), *profile);
    llama_context_params cparams = common_context_params_to_llama(params);
    result.context.reset(llama_init_from_model(result.init->model(), cparams));
    if (!result.context) throw std::runtime_error("failed to create model context");
    return result;
}

double token_nll(const float * logits, int32_t n_vocab, llama_token target) {
    float max_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - max_logit);
    return -((double) logits[target] - max_logit - std::log(sum));
}

evaluation_result evaluate(
        llama_context * context,
        const aikar_dataset & dataset,
        route_collector & collector,
        const options & opts,
        const std::string & label) {
    evaluation_result result;
    result.total_tokens = dataset.total_tokens;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(context)));
    const auto started = std::chrono::steady_clock::now();
    auto last_progress = started;
    llama_batch batch = llama_batch_init(opts.n_batch, 0, 1);
    for (size_t record_index = 0; record_index < dataset.records.size(); ++record_index) {
        const aikar_dataset_record & record = dataset.records[record_index];
        for (size_t window_start = 0; window_start + 1 < record.tokens.size(); window_start += opts.n_ctx) {
            const size_t window_end = std::min(record.tokens.size(), window_start + (size_t) opts.n_ctx);
            llama_memory_clear(llama_get_memory(context), true);
            for (size_t batch_start = window_start; batch_start + 1 < window_end; batch_start += opts.n_batch) {
                const size_t batch_end = std::min(window_end - 1, batch_start + (size_t) opts.n_batch);
                common_batch_clear(batch);
                std::vector<size_t> targets;
                for (size_t i = batch_start; i < batch_end; ++i) {
                    bool need_logits = false;
                    for (size_t mask = 0; mask < 4; ++mask) need_logits |= aikar_token_is_evaluated(record, i + 1, (aikar_ppl_mask) mask);
                    common_batch_add(batch, record.tokens[i], (llama_pos) (i - window_start), { 0 }, need_logits);
                    if (need_logits) targets.push_back(i + 1);
                }
                if (llama_decode(context, batch) != 0) {
                    llama_batch_free(batch);
                    throw std::runtime_error("model evaluation failed at JSONL line " + std::to_string(record.line));
                }
                const float * logits = llama_get_logits(context);
                for (size_t output = 0; output < targets.size(); ++output) {
                    const size_t target_index = targets[output];
                    const double nll = token_nll(logits + output * n_vocab, n_vocab, record.tokens[target_index]);
                    for (size_t mask = 0; mask < 4; ++mask) {
                        if (aikar_token_is_evaluated(record, target_index, (aikar_ppl_mask) mask)) {
                            result.nll[mask] += nll;
                            ++result.evaluated[mask];
                        }
                    }
                }
                result.processed_tokens += batch.n_tokens;
                const auto now = std::chrono::steady_clock::now();
                if (now - last_progress >= std::chrono::seconds(5)) {
                    std::cerr << "aikar-prune: " << label << ": record " << record_index + 1 << '/' << dataset.records.size()
                              << ", processed " << result.processed_tokens << " tokens\n";
                    last_progress = now;
                }
            }
        }
    }
    llama_batch_free(batch);
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.throughput = result.elapsed_seconds == 0.0 ? 0.0 : result.processed_tokens / result.elapsed_seconds;
    result.invalid_routing = collector.invalid_routing;
    result.router_entropy = collector.entropy_tokens == 0 ? 0.0 : collector.entropy_sum / collector.entropy_tokens;
    double imbalance_sum = 0.0;
    size_t imbalance_layers = 0;
    for (const auto & layer : collector.stats) {
        uint64_t total = 0;
        uint64_t maximum = 0;
        for (const auto & expert : layer.second) {
            total += expert.selection_count;
            maximum = std::max(maximum, expert.selection_count);
        }
        if (total > 0 && !layer.second.empty()) {
            imbalance_sum += maximum / ((double) total / layer.second.size());
            ++imbalance_layers;
        }
    }
    result.router_load_imbalance = imbalance_layers == 0 ? 0.0 : imbalance_sum / imbalance_layers;
    std::cerr << "aikar-prune: " << label << " complete: " << result.processed_tokens << " tokens in "
              << result.elapsed_seconds << " seconds (" << result.throughput << " tokens/s)\n";
    return result;
}

json result_json(const evaluation_result & result, aikar_ppl_mask primary) {
    return {
        { "ppl", result.ppl(primary) },
        { "ppl_all", result.ppl(aikar_ppl_mask::ALL) },
        { "ppl_assistant", result.ppl(aikar_ppl_mask::ASSISTANT) },
        { "ppl_reasoning", result.ppl(aikar_ppl_mask::REASONING) },
        { "ppl_content", result.ppl(aikar_ppl_mask::CONTENT) },
        { "evaluated_token_count", result.evaluated[(size_t) primary] },
        { "total_token_count", result.total_tokens },
        { "elapsed_seconds", result.elapsed_seconds },
        { "prompt_tokens_per_second", result.throughput },
        { "router_load_imbalance", result.router_load_imbalance },
        { "router_entropy", result.router_entropy },
        { "invalid_routing_count", result.invalid_routing },
    };
}

std::string baseline_checkpoint_path(const options & opts) {
    return opts.output_dir + "/baseline-checkpoint.json";
}

json baseline_checkpoint_json(
        const evaluation_result & result,
        const route_collector & collector,
        const common_moe_prune_model_info & model,
        const std::string & dataset_hash,
        const options & opts) {
    json stats = json::object();
    for (const auto & layer : collector.stats) {
        json experts = json::array();
        for (const auto & expert : layer.second) {
            experts.push_back({
                { "selection_count", expert.selection_count },
                { "probability_sum", expert.probability_sum },
                { "output_norm_sum", expert.output_norm_sum },
                { "weighted_output_sum", expert.weighted_output_sum },
            });
        }
        stats[std::to_string(layer.first)] = experts;
    }
    return {
        { "format", "aikar-moe-prune-baseline-checkpoint" },
        { "version", 1 },
        { "model_hash", model.model_hash },
        { "expert_tensor_hash", model.expert_tensor_hash },
        { "dataset_hash", dataset_hash },
        { "ppl_mask", aikar_ppl_mask_name(opts.mask) },
        { "ctx_size", opts.n_ctx },
        { "nll", result.nll },
        { "evaluated", result.evaluated },
        { "total_tokens", result.total_tokens },
        { "processed_tokens", result.processed_tokens },
        { "elapsed_seconds", result.elapsed_seconds },
        { "throughput", result.throughput },
        { "router_load_imbalance", result.router_load_imbalance },
        { "router_entropy", result.router_entropy },
        { "invalid_routing", result.invalid_routing },
        { "stats", stats },
    };
}

bool load_baseline_checkpoint(
        const std::string & path,
        evaluation_result & result,
        route_collector & collector,
        const common_moe_prune_model_info & model,
        const std::string & dataset_hash,
        const options & opts) {
    try {
        std::ifstream in(path);
        json root;
        in >> root;
        if (root.value("format", "") != "aikar-moe-prune-baseline-checkpoint" || root.value("version", 0) != 1 ||
            root.value("model_hash", "") != model.model_hash || root.value("expert_tensor_hash", "") != model.expert_tensor_hash ||
            root.value("dataset_hash", "") != dataset_hash || root.value("ppl_mask", "") != aikar_ppl_mask_name(opts.mask) ||
            root.value("ctx_size", 0) != opts.n_ctx) return false;
        result.nll = root.at("nll").get<std::array<double, 4>>();
        result.evaluated = root.at("evaluated").get<std::array<int64_t, 4>>();
        result.total_tokens = root.at("total_tokens").get<int64_t>();
        result.processed_tokens = root.at("processed_tokens").get<int64_t>();
        result.elapsed_seconds = root.at("elapsed_seconds").get<double>();
        result.throughput = root.at("throughput").get<double>();
        result.router_load_imbalance = root.at("router_load_imbalance").get<double>();
        result.router_entropy = root.at("router_entropy").get<double>();
        result.invalid_routing = root.at("invalid_routing").get<uint64_t>();
        collector.stats.clear();
        for (auto it = root.at("stats").begin(); it != root.at("stats").end(); ++it) {
            auto & experts = collector.stats[std::stoi(it.key())];
            for (const auto & item : it.value()) {
                experts.push_back({
                    item.at("selection_count").get<uint64_t>(), item.at("probability_sum").get<double>(),
                    item.at("output_norm_sum").get<double>(), item.at("weighted_output_sum").get<double>(),
                });
            }
        }
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

void write_baseline_checkpoint(const std::string & path, const json & value) {
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) throw std::runtime_error("failed to write baseline checkpoint");
    out << value.dump(2) << '\n';
    out.close();
    if (!out || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to replace baseline checkpoint");
    }
}

std::string profile_name(double ratio) {
    const int value = (int) std::lround(ratio * 100.0);
    std::ostringstream out;
    out << "profile-" << std::setw(3) << std::setfill('0') << value << ".json";
    return out.str();
}

void run_inspect(const options & opts) {
    const common_moe_prune_model_info model = common_moe_prune_inspect_model(opts.model);
    const common_moe_prune_profile profile = common_moe_prune_profile_load(opts.profile);
    common_moe_prune_profile_validate(profile, model);
    const size_t disabled = profile.layers.begin()->second.disabled_experts.size();
    const int32_t surviving = model.expert_count - disabled;
    std::cout << "compatible: yes\narchitecture: " << model.architecture << "\nMoE layers: " << model.moe_layers.size()
              << "\ndisabled experts per layer: " << disabled << "\nsurviving experts per layer: " << surviving
              << "\neffective ratio: " << (double) disabled / model.expert_count
              << "\nexpected expert/router savings: " << (uint64_t) (model.expert_bytes * ((double) disabled / model.expert_count)) << " bytes\n";
    for (const auto & layer : profile.layers) {
        std::cout << "layer " << layer.first << " disabled:";
        for (int32_t expert : layer.second.disabled_experts) std::cout << ' ' << expert;
        std::cout << '\n';
    }
}

void run_analyze(const options & opts) {
    std::filesystem::create_directories(opts.output_dir);
    std::cerr << "aikar-prune: inspecting model and hashing GGUF\n";
    const common_moe_prune_model_info model_info = common_moe_prune_inspect_model(opts.model);
    route_collector baseline_collector;
    baseline_collector.n_expert = model_info.expert_count;
    baseline_collector.collect_output_norm = true;
    aikar_dataset dataset;
    evaluation_result baseline;
    const std::string dataset_hash = common_moe_prune_sha256_file(opts.dataset);
    const std::string checkpoint = baseline_checkpoint_path(opts);
    const bool checkpoint_loaded = load_baseline_checkpoint(checkpoint, baseline, baseline_collector, model_info, dataset_hash, opts);
    if (checkpoint_loaded) {
        std::cerr << "aikar-prune: loaded baseline checkpoint " << checkpoint << '\n';
    } else {
        std::cerr << "aikar-prune: loading baseline model\n";
        loaded_model baseline_model = load_model(opts, &baseline_collector, nullptr);
        common_chat_templates_ptr templates = common_chat_templates_init(baseline_model.init->model(), "");
        std::cerr << "aikar-prune: loading and tokenizing dataset\n";
        dataset = aikar_dataset_load(opts.dataset, baseline_model.init->model(), templates.get());
        std::cerr << "aikar-prune: dataset contains " << dataset.records.size() << " records and " << dataset.total_tokens << " tokens\n";
        baseline = evaluate(baseline_model.context.get(), dataset, baseline_collector, opts, "baseline");
        write_baseline_checkpoint(checkpoint, baseline_checkpoint_json(baseline, baseline_collector, model_info, dataset_hash, opts));
        std::cerr << "aikar-prune: saved baseline checkpoint " << checkpoint << '\n';
    }
    std::cerr << "aikar-prune: released baseline model before pruned evaluations\n";
    const int64_t evaluated_tokens = baseline.evaluated[(size_t) opts.mask];
    if (evaluated_tokens == 0) throw std::runtime_error("the selected perplexity mask evaluates zero tokens");

    std::vector<common_moe_prune_profile> profiles = common_moe_prune_make_profiles(
        model_info, baseline_collector.stats, opts.ratios, opts.max_layer_ratio, dataset_hash,
        aikar_ppl_mask_name(opts.mask), opts.metric, evaluated_tokens);

    json analysis = {
        { "format", "aikar-moe-prune-analysis" },
        { "version", 1 },
        { "model", {
            { "architecture", model_info.architecture },
            { "model_hash", model_info.model_hash },
            { "expert_tensor_hash", model_info.expert_tensor_hash },
            { "expert_count", model_info.expert_count },
            { "experts_used", model_info.experts_used },
        } },
        { "baseline", result_json(baseline, opts.mask) },
        { "ratios", json::array() },
        { "importance", json::object() },
    };
    for (const auto & layer : baseline_collector.stats) {
        json experts = json::array();
        uint64_t layer_total = 0;
        for (const auto & stat : layer.second) layer_total += stat.selection_count;
        for (size_t expert = 0; expert < layer.second.size(); ++expert) {
            const auto & stat = layer.second[expert];
            experts.push_back({
                { "expert", expert },
                { "selection_count", stat.selection_count },
                { "selection_frequency", layer_total == 0 ? 0.0 : (double) stat.selection_count / layer_total },
                { "router_probability_sum", stat.probability_sum },
                { "mean_router_probability", stat.mean_probability() },
                { "mean_output_activation_norm", stat.mean_output_norm() },
                { "weighted_output_importance", stat.importance() },
            });
        }
        analysis["importance"][std::to_string(layer.first)] = experts;
    }

    std::ofstream csv(opts.output_dir + "/analysis.csv", std::ios::trunc);
    csv << "requested_ratio,actual_ratio,baseline_ppl,pruned_ppl,absolute_delta,relative_delta_percent,evaluated_tokens,total_tokens,elapsed_seconds,tokens_per_second,pruned_experts,remaining_experts,router_load_imbalance,router_entropy,invalid_routing_count\n";
    for (common_moe_prune_profile & profile : profiles) {
        const std::string path = opts.output_dir + "/" + profile_name(profile.requested_ratio);
        common_moe_prune_profile_write(profile, path);
        route_collector collector;
        collector.n_expert = model_info.expert_count;
        std::cerr << "aikar-prune: loading model for ratio " << profile.requested_ratio << '\n';
        loaded_model pruned_model = load_model(opts, &collector, &profile);
        if (dataset.records.empty()) {
            std::cerr << "aikar-prune: loading and tokenizing dataset\n";
            common_chat_templates_ptr templates = common_chat_templates_init(pruned_model.init->model(), "");
            dataset = aikar_dataset_load(opts.dataset, pruned_model.init->model(), templates.get());
            std::cerr << "aikar-prune: dataset contains " << dataset.records.size() << " records and " << dataset.total_tokens << " tokens\n";
        }
        const evaluation_result pruned = evaluate(
            pruned_model.context.get(), dataset, collector, opts, "ratio " + std::to_string(profile.requested_ratio));
        const double baseline_ppl = baseline.ppl(opts.mask);
        const double pruned_ppl = pruned.ppl(opts.mask);
        const double delta = pruned_ppl - baseline_ppl;
        const double relative = baseline_ppl == 0.0 ? 0.0 : delta * 100.0 / baseline_ppl;
        const int32_t per_layer_pruned = profile.layers.begin()->second.disabled_experts.size();
        json per_layer = json::object();
        for (const auto & layer : profile.layers) per_layer[std::to_string(layer.first)] = layer.second.disabled_experts.size();
        json row = result_json(pruned, opts.mask);
        row["requested_ratio"] = profile.requested_ratio;
        row["actual_ratio"] = profile.actual_ratio;
        row["absolute_perplexity_delta"] = delta;
        row["relative_perplexity_delta_percent"] = relative;
        row["number_of_pruned_experts"] = per_layer_pruned * profile.layers.size();
        row["number_of_remaining_experts"] = (model_info.expert_count - per_layer_pruned) * profile.layers.size();
        row["per_layer_pruned_expert_count"] = per_layer;
        row["profile"] = std::filesystem::path(path).filename().string();
        analysis["ratios"].push_back(row);
        csv << profile.requested_ratio << ',' << profile.actual_ratio << ',' << baseline_ppl << ',' << pruned_ppl << ',' << delta << ',' << relative << ','
            << pruned.evaluated[(size_t) opts.mask] << ',' << pruned.total_tokens << ',' << pruned.elapsed_seconds << ',' << pruned.throughput << ','
            << per_layer_pruned * profile.layers.size() << ',' << (model_info.expert_count - per_layer_pruned) * profile.layers.size() << ','
            << pruned.router_load_imbalance << ',' << pruned.router_entropy << ',' << pruned.invalid_routing << '\n';
        if (model_info.expert_count - per_layer_pruned < model_info.experts_used * 2) {
            std::cerr << "warning: ratio " << profile.requested_ratio << " leaves fewer than 2x router Top-K experts\n";
        }
    }
    std::ofstream out(opts.output_dir + "/analysis.json", std::ios::trunc);
    out << analysis.dump(2) << '\n';
    std::ofstream summary(opts.output_dir + "/README.txt", std::ios::trunc);
    summary << "Gemma 4 26B A4B static MoE pruning analysis\nBaseline perplexity (" << aikar_ppl_mask_name(opts.mask) << "): " << baseline.ppl(opts.mask) << "\n";
    for (const auto & row : analysis["ratios"]) summary << "ratio " << row["requested_ratio"] << ": ppl " << row["ppl"] << ", delta " << row["absolute_perplexity_delta"] << "\n";
}

void run_hard(const options & opts) {
    const common_moe_prune_model_info model = common_moe_prune_inspect_model(opts.model);
    const common_moe_prune_profile profile = common_moe_prune_profile_load(opts.profile);
    common_moe_prune_profile_validate(profile, model);
    const std::string staging_output = opts.output + ".validation.tmp";
    struct staging_guard {
        std::string model;
        bool committed = false;
        ~staging_guard() {
            if (!committed) {
                std::remove(model.c_str());
                std::remove((model + ".report.json").c_str());
            }
        }
    } guard { staging_output };
    const aikar_hard_prune_report report = aikar_hard_prune_gemma4_q4_0(opts.model, profile, model, staging_output);

    options validation_opts = opts;
    validation_opts.model = staging_output;
    const common_moe_prune_model_info pruned_info = common_moe_prune_inspect_model(staging_output);
    if (pruned_info.expert_count != model.expert_count - (int32_t) profile.layers.begin()->second.disabled_experts.size()) {
        throw std::runtime_error("hard-pruned model metadata validation failed");
    }
    std::optional<evaluation_result> hard_evaluation;
    {
        route_collector collector;
        collector.n_expert = pruned_info.expert_count;
        loaded_model validation = load_model(validation_opts, &collector, nullptr);
        const llama_vocab * vocab = llama_model_get_vocab(validation.init->model());
        std::vector<llama_token> tokens = common_tokenize(vocab, "Hello", true, true);
        if (tokens.empty() || llama_decode(validation.context.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            throw std::runtime_error("hard-pruned model inference smoke test failed");
        }
        if (!opts.dataset.empty()) {
            common_chat_templates_ptr templates = common_chat_templates_init(validation.init->model(), "");
            const aikar_dataset dataset = aikar_dataset_load(opts.dataset, validation.init->model(), templates.get());
            hard_evaluation = evaluate(validation.context.get(), dataset, collector, opts, "hard validation");
        }
    }
    if (hard_evaluation) {
        route_collector soft_collector;
        soft_collector.n_expert = model.expert_count;
        loaded_model soft_model = load_model(opts, &soft_collector, &profile);
        common_chat_templates_ptr templates = common_chat_templates_init(soft_model.init->model(), "");
        const aikar_dataset dataset = aikar_dataset_load(opts.dataset, soft_model.init->model(), templates.get());
        const evaluation_result soft_evaluation = evaluate(soft_model.context.get(), dataset, soft_collector, opts, "soft validation");
        if (soft_evaluation.evaluated[(size_t) opts.mask] != hard_evaluation->evaluated[(size_t) opts.mask]) {
            throw std::runtime_error("soft and hard evaluations used different token counts");
        }
        const double soft_ppl = soft_evaluation.ppl(opts.mask);
        const double hard_ppl = hard_evaluation->ppl(opts.mask);
        const double difference = hard_ppl - soft_ppl;
        const std::string report_path = staging_output + ".report.json";
        std::ifstream report_in(report_path);
        json report_json_value;
        report_in >> report_json_value;
        report_json_value["validation"] = {
            { "dataset_hash", common_moe_prune_sha256_file(opts.dataset) },
            { "ppl_mask", aikar_ppl_mask_name(opts.mask) },
            { "soft_perplexity", soft_ppl },
            { "hard_perplexity", hard_ppl },
            { "absolute_difference", difference },
            { "evaluated_token_count", hard_evaluation->evaluated[(size_t) opts.mask] },
        };
        const std::string tmp_report = report_path + ".tmp";
        std::ofstream report_out(tmp_report, std::ios::trunc);
        report_out << report_json_value.dump(2) << '\n';
        report_out.close();
        if (!report_out || std::rename(tmp_report.c_str(), report_path.c_str()) != 0) {
            std::remove(tmp_report.c_str());
            throw std::runtime_error("failed to update hard-pruning validation report");
        }
        std::cout << "soft perplexity: " << soft_ppl << "\nhard perplexity: " << hard_ppl << "\nabsolute difference: " << difference << '\n';
    }
    if (std::rename(staging_output.c_str(), opts.output.c_str()) != 0) {
        throw std::runtime_error("failed to atomically replace hard-pruned GGUF output");
    }
    if (std::rename((staging_output + ".report.json").c_str(), (opts.output + ".report.json").c_str()) != 0) {
        throw std::runtime_error("failed to replace hard-pruning report");
    }
    guard.committed = true;
    std::cout << "hard-pruned model validated\nsource bytes: " << report.source_bytes << "\noutput bytes: " << report.output_bytes
              << "\nexpert bytes removed: " << report.expert_bytes_removed << "\nreport: " << opts.output << ".report.json\n";
}

}

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        common_init();
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        if (opts.command == "inspect") run_inspect(opts);
        else if (opts.command == "analyze") run_analyze(opts);
        else run_hard(opts);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        usage();
        return 1;
    }
}
