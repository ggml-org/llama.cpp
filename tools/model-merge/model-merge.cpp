#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "llama.h"
#include "jsonl.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct merge_params {
    std::string base;
    std::vector<std::string> models;
    std::string output;
    std::string method = "ties";
    float density = 0.5f;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    size_t memory_budget = 2ull * 1024 * 1024 * 1024;
    std::string calibration;
    std::string target_type;
    int population = 8;
    int generations = 10;
    int elite_count = 2;
    int gpu_layers = 0;
    std::string device;
    bool merge_gpu = false;
    int context_size = 512;
    float mutation = 0.10f;
    uint32_t seed = 0;
};

static void usage(const char * executable) {
    printf("usage: %s --base BASE.gguf --model MODEL.gguf [--model MODEL.gguf ...] --output OUT.gguf [options]\n", executable);
    printf("       %s --config merge.ini\n\n", executable);
    printf("options:\n");
    printf("  --method {ties,evo}           merge method (default: ties)\n");
    printf("  --density N                   TIES task-vector density in (0, 1] (default: 0.5)\n");
    printf("  -m, --model MODEL.gguf        input model; may be specified multiple times\n");
    printf("  -o, --output OUT.gguf         output GGUF\n");
    printf("  -t, --threads N               merge worker count (default: CPU core count)\n");
    printf("  --memory-budget SIZE          worker memory budget, e.g. 8G or 16384M (default: 2G)\n");
    printf("  --calibration FILE            Evo calibration .txt or .jsonl\n");
    printf("  --target-type TYPE            Evo target: q4_0, q3_k, q4_k, or mxfp4\n");
    printf("  --population N                Evo population size (default: 8)\n");
    printf("  --generations N               Evo generation count (default: 10)\n");
    printf("  --elite-count N               Evo elites retained per generation (default: 2)\n");
    printf("  --sigma0 N                    Evo CMA-ES initial sigma (default: 0.10)\n");
    printf("  --seed N                      Evo random seed (default: random)\n");
    printf("  --gpu-layers N                layers offloaded for fitness; -1 means all (default: 0)\n");
    printf("  --device NAME                 fitness backend device, e.g. CUDA0 or Vulkan0\n");
    printf("  --merge-gpu                  run Evo weighted merge math on the selected GPU\n");
    printf("  --ctx-size N                  fitness context size (default: 512)\n");
    printf("  --config FILE                 INI file: base=, models=comma,separated, output=, method=, density=, threads=, memory_budget=\n");
}

static size_t parse_memory_size(const std::string & value) {
    if (value.empty()) {
        throw std::runtime_error("empty memory budget");
    }
    size_t suffix_at = value.find_first_not_of("0123456789");
    const std::string number = value.substr(0, suffix_at);
    std::string suffix = suffix_at == std::string::npos ? "" : value.substr(suffix_at);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);
    size_t multiplier = 1;
    if (suffix == "K" || suffix == "KB") {
        multiplier = 1024;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1024 * 1024;
    } else if (suffix == "G" || suffix == "GB") {
        multiplier = 1024 * 1024 * 1024;
    } else if (!suffix.empty() && suffix != "B") {
        throw std::runtime_error("invalid memory budget suffix '" + suffix + "'");
    }
    const size_t amount = std::stoull(number);
    if (amount == 0 || amount > SIZE_MAX / multiplier) {
        throw std::runtime_error("invalid memory budget '" + value + "'");
    }
    return amount * multiplier;
}

static bool parse_bool(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "true" || value == "1" || value == "yes" || value == "on") return true;
    if (value == "false" || value == "0" || value == "no" || value == "off") return false;
    throw std::runtime_error("invalid boolean value '" + value + "'");
}

static std::string trim(const std::string & value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::vector<std::string> split(const std::string & value, char separator) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(separator, begin);
        const std::string item = trim(value.substr(begin, end - begin));
        if (!item.empty()) {
            result.push_back(item);
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

static void load_config(const std::string & path, merge_params & params) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config " + path);
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') {
            continue;
        }
        const size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("invalid config line " + std::to_string(line_number));
        }
        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (key == "base") {
            params.base = value;
        } else if (key == "models") {
            params.models = split(value, ',');
        } else if (key == "output") {
            params.output = value;
        } else if (key == "method") {
            params.method = value;
        } else if (key == "density") {
            params.density = std::stof(value);
        } else if (key == "threads") {
            params.n_threads = std::stoi(value);
        } else if (key == "memory_budget") {
            params.memory_budget = parse_memory_size(value);
        } else if (key == "calibration") {
            params.calibration = value;
        } else if (key == "target_type") {
            params.target_type = value;
        } else if (key == "population") {
            params.population = std::stoi(value);
        } else if (key == "generations") {
            params.generations = std::stoi(value);
        } else if (key == "elite_count") {
            params.elite_count = std::stoi(value);
        } else if (key == "mutation") {
            params.mutation = std::stof(value);
        } else if (key == "sigma0") {
            params.mutation = std::stof(value);
        } else if (key == "seed") {
            params.seed = std::stoul(value);
        } else if (key == "gpu_layers") {
            params.gpu_layers = std::stoi(value);
        } else if (key == "device") {
            params.device = value;
        } else if (key == "merge_gpu") {
            params.merge_gpu = parse_bool(value);
        } else if (key == "ctx_size") {
            params.context_size = std::stoi(value);
        } else {
            throw std::runtime_error("unknown config key '" + key + "'");
        }
    }
}

static merge_params parse_args(int argc, char ** argv) {
    merge_params params;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            exit(0);
        }
        if (arg == "--config" && i + 1 < argc) {
            load_config(argv[++i], params);
        } else if (arg == "--base" && i + 1 < argc) {
            params.base = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            params.models.push_back(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            params.method = argv[++i];
        } else if (arg == "--density" && i + 1 < argc) {
            params.density = std::stof(argv[++i]);
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--memory-budget" && i + 1 < argc) {
            params.memory_budget = parse_memory_size(argv[++i]);
        } else if (arg == "--calibration" && i + 1 < argc) {
            params.calibration = argv[++i];
        } else if (arg == "--target-type" && i + 1 < argc) {
            params.target_type = argv[++i];
        } else if (arg == "--population" && i + 1 < argc) {
            params.population = std::stoi(argv[++i]);
        } else if (arg == "--generations" && i + 1 < argc) {
            params.generations = std::stoi(argv[++i]);
        } else if (arg == "--elite-count" && i + 1 < argc) {
            params.elite_count = std::stoi(argv[++i]);
        } else if (arg == "--mutation" && i + 1 < argc) {
            params.mutation = std::stof(argv[++i]);
        } else if (arg == "--sigma0" && i + 1 < argc) {
            params.mutation = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            params.seed = std::stoul(argv[++i]);
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            params.gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            params.device = argv[++i];
        } else if (arg == "--merge-gpu") {
            params.merge_gpu = true;
        } else if (arg == "--ctx-size" && i + 1 < argc) {
            params.context_size = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("unknown or incomplete option '" + arg + "'");
        }
    }
    if (params.base.empty() || params.models.empty() || params.output.empty()) {
        throw std::runtime_error("--base, at least one --model, and --output are required");
    }
    if (params.method != "ties" && params.method != "evo") {
        throw std::runtime_error("--method must be ties or evo");
    }
    if (params.density <= 0.0f || params.density > 1.0f) {
        throw std::runtime_error("--density must be in (0, 1]");
    }
    if (params.n_threads < 1) {
        throw std::runtime_error("--threads must be positive");
    }
    if (params.method == "evo") {
        if (params.calibration.empty() || params.target_type.empty()) {
            throw std::runtime_error("evo requires --calibration and --target-type");
        }
        if (params.population < 2 || params.generations < 1 || params.elite_count < 1 ||
                params.elite_count >= params.population || params.mutation <= 0.0f || params.context_size < 2) {
            throw std::runtime_error("invalid Evo population, generation, elite, sigma, or context setting");
        }
    }
    if (params.merge_gpu && params.method != "evo") {
        throw std::runtime_error("--merge-gpu currently requires --method evo");
    }
    return params;
}

static void write_zeros(std::ofstream & output, size_t count) {
    static const char zero = 0;
    for (size_t i = 0; i < count; ++i) {
        output.write(&zero, 1);
    }
}

static std::string get_kv_string(const gguf_context * gguf, const char * key) {
    const int64_t idx = gguf_find_key(gguf, key);
    return idx < 0 ? "" : gguf_get_val_str(gguf, idx);
}

static uint32_t get_file_type(const gguf_context * gguf) {
    const int64_t idx = gguf_find_key(gguf, "general.file_type");
    if (idx < 0 || gguf_get_kv_type(gguf, idx) != GGUF_TYPE_UINT32) {
        return LLAMA_FTYPE_GUESSED;
    }
    return gguf_get_val_u32(gguf, idx);
}

struct gguf_input {
    struct tensor_ref {
        ggml_tensor * tensor;
    };

    std::string path;
    mutable std::ifstream file;
    mutable std::mutex mutex;
    ggml_context * ctx = nullptr;
    gguf_context * gguf = nullptr;
    std::map<std::string, tensor_ref> tensors;
    uint32_t file_type = LLAMA_FTYPE_GUESSED;

    explicit gguf_input(const std::string & path) : path(path), file(path, std::ios::binary) {
        if (!file) {
            throw std::runtime_error("failed to open input GGUF " + path);
        }
        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };
        gguf = gguf_init_from_file(path.c_str(), params);
        if (!gguf) {
            throw std::runtime_error("failed to parse input GGUF " + path);
        }
        for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            if (!tensors.emplace(tensor->name, tensor_ref { tensor }).second) {
                throw std::runtime_error("duplicate tensor '" + std::string(tensor->name) + "' in " + path);
            }
        }
        file_type = get_file_type(gguf);
    }

    ~gguf_input() {
        gguf_free(gguf);
        ggml_free(ctx);
    }

    void read_tensor(const std::string & name, std::vector<uint8_t> & data) const {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("missing tensor '" + name + "' in " + path);
        }
        const int tensor_index = gguf_find_tensor(gguf, name.c_str());
        const size_t size = ggml_nbytes(it->second.tensor);
        const size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_index);
        data.resize(size);
        file.clear();
        file.seekg(offset);
        file.read((char *) data.data(), size);
        if (!file) {
            throw std::runtime_error("failed to read tensor '" + name + "' from " + path);
        }
    }
};

static bool same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return true;
}

static ggml_type select_precision_type(const std::vector<std::unique_ptr<gguf_input>> & inputs) {
    bool has_f32 = false;
    bool has_bf16 = false;
    bool all_same = true;
    const uint32_t first = inputs[0]->file_type;
    for (const auto & input : inputs) {
        has_f32 |= input->file_type == LLAMA_FTYPE_ALL_F32;
        has_bf16 |= input->file_type == LLAMA_FTYPE_MOSTLY_BF16;
        all_same &= input->file_type == first;
    }
    if (all_same && first != LLAMA_FTYPE_ALL_F32 && first != LLAMA_FTYPE_MOSTLY_BF16 && first != LLAMA_FTYPE_MOSTLY_F16) {
        return GGML_TYPE_COUNT;
    }
    if (has_f32) {
        return GGML_TYPE_F32;
    }
    if (has_bf16) {
        return GGML_TYPE_BF16;
    }
    return GGML_TYPE_F16;
}

static uint32_t type_to_ftype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return LLAMA_FTYPE_ALL_F32;
        case GGML_TYPE_BF16: return LLAMA_FTYPE_MOSTLY_BF16;
        case GGML_TYPE_F16:  return LLAMA_FTYPE_MOSTLY_F16;
        case GGML_TYPE_Q4_0: return LLAMA_FTYPE_MOSTLY_Q4_0;
        case GGML_TYPE_Q3_K: return LLAMA_FTYPE_MOSTLY_Q3_K_M;
        case GGML_TYPE_Q4_K: return LLAMA_FTYPE_MOSTLY_Q4_K_M;
        case GGML_TYPE_MXFP4:return LLAMA_FTYPE_MOSTLY_MXFP4_MOE;
        default:             return LLAMA_FTYPE_GUESSED;
    }
}

static ggml_type parse_target_type(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "q4_0")  return GGML_TYPE_Q4_0;
    if (value == "q3_k")  return GGML_TYPE_Q3_K;
    if (value == "q4_k")  return GGML_TYPE_Q4_K;
    if (value == "mxfp4") return GGML_TYPE_MXFP4;
    throw std::runtime_error("Evo target type must be q4_0, q3_k, q4_k, or mxfp4");
}

static int select_worker_count(const merge_params & params, const gguf_input & base, size_t n_inputs) {
    // TIES retains one F32 task vector per input model plus the base, result,
    // and a temporary magnitude vector. Bound concurrent tensors so a large
    // embedding or output tensor cannot multiply that allocation by core count.
    size_t largest_tensor = 0;
    for (const auto & entry : base.tensors) {
        largest_tensor = std::max(largest_tensor, (size_t) ggml_nelements(entry.second.tensor));
    }
    if (largest_tensor == 0) {
        throw std::runtime_error("input model has no tensors");
    }

    const size_t vectors_per_worker = n_inputs + 2;
    const size_t bytes_per_worker = largest_tensor > SIZE_MAX / (vectors_per_worker * sizeof(float))
        ? SIZE_MAX
        : largest_tensor * vectors_per_worker * sizeof(float);
    const size_t memory_workers = bytes_per_worker == 0 ? 1 : std::max<size_t>(1, params.memory_budget / bytes_per_worker);
    return std::min<int>(params.n_threads, std::min<size_t>(base.tensors.size(), memory_workers));
}

static void decode_tensor(const gguf_input & input, const std::string & name, std::vector<uint8_t> & bytes, std::vector<float> & result) {
    const ggml_tensor * tensor = input.tensors.at(name).tensor;
    input.read_tensor(name, bytes);
    const int64_t count = ggml_nelements(tensor);
    result.resize(count);
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(result.data(), bytes.data(), count * sizeof(float));
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error("tensor '" + name + "' has unsupported type " + ggml_type_name(tensor->type));
    }
    traits->to_float(bytes.data(), result.data(), count);
}

static void trim_task_vector(std::vector<float> & values, float density) {
    if (density >= 1.0f || values.empty()) {
        return;
    }
    const size_t keep = std::max<size_t>(1, (size_t) std::ceil(values.size() * density));
    std::vector<float> magnitudes(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        magnitudes[i] = std::fabs(values[i]);
    }
    std::nth_element(magnitudes.begin(), magnitudes.end() - keep, magnitudes.end());
    const float threshold = magnitudes[magnitudes.size() - keep];
    for (float & value : values) {
        if (std::fabs(value) < threshold) {
            value = 0.0f;
        }
    }
}

static void ties_merge(const std::vector<float> & base, std::vector<std::vector<float>> & tasks, float density, std::vector<float> & result) {
    for (std::vector<float> & task : tasks) {
        trim_task_vector(task, density);
    }
    result.resize(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        float sign_sum = 0.0f;
        for (const std::vector<float> & task : tasks) {
            sign_sum += task[i];
        }
        if (sign_sum == 0.0f) {
            result[i] = base[i];
            continue;
        }
        const bool positive = sign_sum > 0.0f;
        float sum = 0.0f;
        int count = 0;
        for (const std::vector<float> & task : tasks) {
            if ((positive && task[i] > 0.0f) || (!positive && task[i] < 0.0f)) {
                sum += task[i];
                ++count;
            }
        }
        result[i] = base[i] + sum / count;
    }
}

static void write_tensor(
        std::ofstream & output,
        ggml_type type,
        const ggml_tensor * shape,
        const std::vector<float> & values,
        size_t alignment) {
    if (type == GGML_TYPE_F32) {
        const size_t size = values.size() * sizeof(float);
        output.write((const char *) values.data(), size);
        write_zeros(output, GGML_PAD(size, alignment) - size);
        return;
    }
    if (ggml_is_quantized(type)) {
        const int64_t n_per_row = shape->ne[0];
        const int64_t n_rows = values.size() / n_per_row;
        const size_t size = ggml_row_size(type, n_per_row) * n_rows;
        std::vector<uint8_t> quantized(size);
        ggml_quantize_chunk(type, values.data(), quantized.data(), 0, n_rows, n_per_row, nullptr);
        output.write((const char *) quantized.data(), size);
        write_zeros(output, GGML_PAD(size, alignment) - size);
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->from_float_ref) {
        throw std::runtime_error("cannot encode output type " + std::string(ggml_type_name(type)));
    }
    const size_t size = ggml_row_size(type, shape->ne[0]) * (values.size() / shape->ne[0]);
    std::vector<uint8_t> encoded(size);
    traits->from_float_ref(values.data(), encoded.data(), values.size());
    output.write((const char *) encoded.data(), size);
    write_zeros(output, GGML_PAD(size, alignment) - size);
}

struct evo_candidate {
    std::vector<float> genes;
    double fitness = INFINITY;
};

static void normalize_genes(evo_candidate & candidate, size_t n_tensors, size_t n_inputs) {
    for (size_t tensor = 0; tensor < n_tensors; ++tensor) {
        float sum = 0.0f;
        for (size_t input = 0; input < n_inputs; ++input) {
            float & gene = candidate.genes[tensor*n_inputs + input];
            gene = std::max(0.0f, gene);
            sum += gene;
        }
        if (sum == 0.0f) {
            for (size_t input = 0; input < n_inputs; ++input) {
                candidate.genes[tensor*n_inputs + input] = 1.0f / n_inputs;
            }
        } else {
            for (size_t input = 0; input < n_inputs; ++input) {
                candidate.genes[tensor*n_inputs + input] /= sum;
            }
        }
    }
}

static void merge_weighted_gpu(
        ggml_backend_t backend,
        const std::vector<std::unique_ptr<gguf_input>> & inputs,
        const std::string & name,
        const float * weights,
        std::vector<float> & result) {
    const ggml_tensor * shape = inputs[0]->tensors.at(name).tensor;
    const size_t tensor_count = inputs.size() * 2 + inputs.size();
    ggml_init_params init_params = {
        /*.mem_size   = */ tensor_count * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        throw std::runtime_error("failed to create GPU merge graph context");
    }
    std::vector<ggml_tensor *> source(inputs.size());
    ggml_tensor * merged = nullptr;
    for (size_t input = 0; input < inputs.size(); ++input) {
        source[input] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, shape->ne);
        ggml_tensor * weighted = ggml_scale(ctx, source[input], weights[input]);
        merged = merged ? ggml_add(ctx, merged, weighted) : weighted;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, merged);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate GPU merge tensor buffer");
    }

    try {
        std::vector<uint8_t> bytes;
        std::vector<float> values;
        for (size_t input = 0; input < inputs.size(); ++input) {
            decode_tensor(*inputs[input], name, bytes, values);
            ggml_backend_tensor_set(source[input], values.data(), 0, values.size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GPU merge graph computation failed for tensor '" + name + "'");
        }
        result.resize(ggml_nelements(merged));
        ggml_backend_tensor_get(merged, result.data(), 0, result.size() * sizeof(float));
    } catch (...) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void write_evo_candidate(
        const std::string & path,
        const merge_params & params,
        const std::vector<std::unique_ptr<gguf_input>> & inputs,
        const std::vector<std::string> & tensor_names,
        const evo_candidate & candidate,
        ggml_type target_type,
        ggml_backend_t merge_backend) {
    const gguf_input & base = *inputs[0];
    gguf_context * output_gguf = gguf_init_empty();
    gguf_set_kv(output_gguf, base.gguf);
    gguf_remove_key(output_gguf, GGUF_KEY_GENERAL_ALIGNMENT);
    gguf_set_val_u32(output_gguf, "general.file_type", type_to_ftype(target_type));
    gguf_remove_key(output_gguf, "split.no");
    gguf_remove_key(output_gguf, "split.count");
    gguf_remove_key(output_gguf, "split.tensors.count");

    ggml_init_params init_params = {
        /*.mem_size   = */ base.tensors.size() * ggml_tensor_overhead(),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * output_ctx = ggml_init(init_params);
    std::map<std::string, ggml_type> output_types;
    for (const std::string & name : tensor_names) {
        const ggml_tensor * source = base.tensors.at(name).tensor;
        const ggml_type output_type = ggml_is_quantized(source->type) ? target_type : source->type;
        ggml_tensor * tensor = ggml_new_tensor(output_ctx, output_type, GGML_MAX_DIMS, source->ne);
        ggml_set_name(tensor, source->name);
        gguf_add_tensor(output_gguf, tensor);
        output_types[name] = output_type;
    }

    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    write_zeros(output, gguf_get_meta_size(output_gguf));
    std::atomic<size_t> next_job { 0 };
    size_t next_to_write = 0;
    bool stop = false;
    std::exception_ptr worker_error;
    std::mutex write_mutex;
    std::condition_variable write_condition;
    const int n_workers = merge_backend ? 1 : select_worker_count(params, base, inputs.size());
    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for (int worker = 0; worker < n_workers; ++worker) {
        workers.emplace_back([&] {
            try {
                std::vector<uint8_t> bytes;
                std::vector<float> values;
                std::vector<float> merged;
                for (;;) {
                    const size_t tensor_index = next_job.fetch_add(1);
                    if (tensor_index >= tensor_names.size()) return;
                    const std::string & name = tensor_names[tensor_index];
                    const float * weights = candidate.genes.data() + tensor_index*inputs.size();
                    if (merge_backend) {
                        merge_weighted_gpu(merge_backend, inputs, name, weights, merged);
                    } else {
                        merged.assign(ggml_nelements(base.tensors.at(name).tensor), 0.0f);
                        for (size_t input = 0; input < inputs.size(); ++input) {
                            decode_tensor(*inputs[input], name, bytes, values);
                            for (size_t element = 0; element < merged.size(); ++element) {
                                merged[element] += weights[input] * values[element];
                            }
                        }
                    }
                    std::unique_lock<std::mutex> lock(write_mutex);
                    write_condition.wait(lock, [&] { return stop || tensor_index == next_to_write; });
                    if (stop) return;
                    write_tensor(output, output_types.at(name), base.tensors.at(name).tensor, merged, GGUF_DEFAULT_ALIGNMENT);
                    ++next_to_write;
                    lock.unlock();
                    write_condition.notify_all();
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(write_mutex);
                if (!worker_error) worker_error = std::current_exception();
                stop = true;
                write_condition.notify_all();
            }
        });
    }
    for (std::thread & worker : workers) worker.join();
    if (worker_error) std::rethrow_exception(worker_error);
    std::vector<uint8_t> metadata(gguf_get_meta_size(output_gguf));
    gguf_get_meta_data(output_gguf, metadata.data());
    output.seekp(0);
    output.write((const char *) metadata.data(), metadata.size());
    output.close();
    ggml_free(output_ctx);
    gguf_free(output_gguf);
}

static std::vector<std::string> load_calibration(const std::string & path, int32_t n_threads) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open calibration file " + path);
    }
    std::vector<std::string> samples;
    const bool jsonl = path.size() >= 6 && path.substr(path.size() - 6) == ".jsonl";
    if (!jsonl) {
        std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (!text.empty()) samples.push_back(std::move(text));
        return samples;
    }

    std::vector<common_jsonl_line> lines = common_jsonl_read_lines(path, COMMON_JSONL_EMPTY_LINE_SKIP);
    std::vector<std::string> parsed(lines.size());
    std::vector<std::string> errors(lines.size());
    common_jsonl_worker_pool pool(common_jsonl_worker_count(n_threads, lines.size()));
    pool.parallel_for(lines.size(), [&](size_t i, size_t) {
        if (trim(lines[i].text).empty()) return;
        try {
            const nlohmann::json item = nlohmann::json::parse(lines[i].text);
            std::string text;
            if (item.contains("messages") && item["messages"].is_array()) {
                for (const auto & data : item["messages"]) {
                    const common_chat_msg message = common_jsonl_parse_chat_message(data, COMMON_JSONL_CHAT_PARSE_OPTIONAL_TEXT);
                    if (!message.content_parts.empty()) {
                        text += message.role + ": " + message.content_parts[0].text + "\n";
                    }
                }
            } else if (item.contains("prompt") && item.contains("response")) {
                text = item["prompt"].get<std::string>() + item["response"].get<std::string>();
            } else if (item.contains("text")) {
                text = item["text"].get<std::string>();
            }
            parsed[i] = std::move(text);
        } catch (const std::exception & error) {
            errors[i] = error.what();
        }
    });
    for (common_jsonl_line & line : lines) std::string().swap(line.text);
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!errors[i].empty()) throw std::runtime_error("invalid calibration JSONL line " + std::to_string(lines[i].number) + ": " + errors[i]);
        if (!parsed[i].empty()) samples.push_back(std::move(parsed[i]));
    }
    if (samples.empty()) {
        throw std::runtime_error("calibration file contains no usable samples");
    }
    return samples;
}

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    if (count < 0) {
        tokens.resize(-count);
        count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    }
    if (count < 0) {
        throw std::runtime_error("failed to tokenize calibration text");
    }
    tokens.resize(count);
    return tokens;
}

static double evaluate_candidate(const std::string & path, const merge_params & params, const std::vector<std::string> & samples) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.gpu_layers;
    std::vector<ggml_backend_dev_t> devices;
    if (!params.device.empty()) {
        ggml_backend_dev_t device = ggml_backend_dev_by_name(params.device.c_str());
        if (!device) {
            throw std::runtime_error("backend device not found: " + params.device);
        }
        devices = { device, nullptr };
        model_params.devices = devices.data();
    }
    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load Evo candidate " + path);
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = params.context_size;
    context_params.n_batch = params.context_size;
    context_params.n_ubatch = params.context_size;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        throw std::runtime_error("failed to create Evo fitness context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    double total_nll = 0.0;
    size_t total_tokens = 0;
    for (const std::string & sample : samples) {
        const std::vector<llama_token> tokens = tokenize_text(vocab, sample);
        for (size_t begin = 0; begin + 1 < tokens.size();) {
            const size_t count = std::min<size_t>(params.context_size, tokens.size() - begin);
            llama_batch batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (size_t i = 0; i < count; ++i) {
                batch.token[i] = tokens[begin + i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i + 1 < count;
            }
            llama_memory_clear(llama_get_memory(context), true);
            if (llama_decode(context, batch) != 0) {
                llama_batch_free(batch);
                llama_free(context);
                llama_model_free(model);
                throw std::runtime_error("llama_decode failed during Evo fitness evaluation");
            }
            for (size_t i = 0; i + 1 < count; ++i) {
                const float * logits = llama_get_logits_ith(context, i);
                const float max_logit = *std::max_element(logits, logits + n_vocab);
                double sum = 0.0;
                for (int32_t token = 0; token < n_vocab; ++token) {
                    sum += std::exp(logits[token] - max_logit);
                }
                total_nll += max_logit + std::log(sum) - logits[tokens[begin + i + 1]];
                ++total_tokens;
            }
            llama_batch_free(batch);
            begin += count > 1 ? count - 1 : count;
        }
    }
    llama_free(context);
    llama_model_free(model);
    if (total_tokens == 0) {
        throw std::runtime_error("calibration data produced fewer than two tokens");
    }
    return total_nll / total_tokens;
}

static void run_evo(const merge_params & params, const std::vector<std::unique_ptr<gguf_input>> & inputs) {
    const ggml_type target_type = parse_target_type(params.target_type);
    std::vector<std::string> tensor_names;
    tensor_names.reserve(inputs[0]->tensors.size());
    for (const auto & entry : inputs[0]->tensors) {
        tensor_names.push_back(entry.first);
    }
    const std::vector<std::string> calibration = load_calibration(params.calibration, params.n_threads);
    const uint32_t seed = params.seed ? params.seed : std::random_device{}();
    std::mt19937 rng(seed);
    const size_t gene_count = tensor_names.size() * inputs.size();
    if (gene_count == 0) {
        throw std::runtime_error("input model has no tensors");
    }
    std::vector<float> mean(gene_count, 1.0f / inputs.size());
    std::vector<float> variance(gene_count, 1.0f);
    std::vector<float> recombination(params.elite_count);
    float recombination_sum = 0.0f;
    for (int i = 0; i < params.elite_count; ++i) {
        recombination[i] = std::log(params.elite_count + 0.5f) - std::log(i + 1.0f);
        recombination_sum += recombination[i];
    }
    for (float & weight : recombination) weight /= recombination_sum;
    float sigma = params.mutation;
    const float covariance_rate = std::min(0.25f, 2.0f / (std::sqrt((float) gene_count) + 2.0f));
    std::normal_distribution<float> normal(0.0f, 1.0f);

    evo_candidate best;
    const std::string temporary = params.output + ".evo-candidate.tmp.gguf";
    llama_backend_init();
    ggml_backend_t merge_backend = nullptr;
    try {
        if (!params.device.empty() && !ggml_backend_dev_by_name(params.device.c_str())) {
            throw std::runtime_error("backend device not found: " + params.device);
        }
        if (params.merge_gpu) {
            ggml_backend_dev_t device = params.device.empty()
                ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)
                : ggml_backend_dev_by_name(params.device.c_str());
            if (!device || (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU &&
                            ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_IGPU &&
                            ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_ACCEL)) {
                throw std::runtime_error("--merge-gpu requires a GPU or accelerator device");
            }
            merge_backend = ggml_backend_dev_init(device, nullptr);
            if (!merge_backend) {
                throw std::runtime_error("failed to initialize merge device " + std::string(ggml_backend_dev_name(device)));
            }
            printf("evo: GPU merge device %s (%s)\n", ggml_backend_dev_name(device), ggml_backend_dev_description(device));
        }
        for (int generation = 0; generation < params.generations; ++generation) {
            std::vector<evo_candidate> population(params.population);
            for (int candidate = 0; candidate < params.population; ++candidate) {
                population[candidate].genes.resize(gene_count);
                for (size_t gene = 0; gene < gene_count; ++gene) {
                    const float sample = candidate == 0 ? 0.0f : normal(rng);
                    population[candidate].genes[gene] = mean[gene] + sigma * std::sqrt(variance[gene]) * sample;
                }
                normalize_genes(population[candidate], tensor_names.size(), inputs.size());
                printf("evo generation %d/%d candidate %d/%d: merging\n",
                        generation + 1, params.generations, candidate + 1, params.population);
                write_evo_candidate(temporary, params, inputs, tensor_names, population[candidate], target_type, merge_backend);
                population[candidate].fitness = evaluate_candidate(temporary, params, calibration);
                if (!std::isfinite(population[candidate].fitness)) {
                    population[candidate].fitness = INFINITY;
                }
                printf("evo generation %d candidate %d: nll %.6f ppl %.6f\n",
                        generation + 1, candidate + 1, population[candidate].fitness,
                        std::exp(population[candidate].fitness));
            }
            std::sort(population.begin(), population.end(), [](const evo_candidate & a, const evo_candidate & b) {
                return a.fitness < b.fitness;
            });
            if (!std::isfinite(population[0].fitness)) {
                throw std::runtime_error("all Evo candidates produced non-finite fitness");
            }
            if (best.genes.empty() || population[0].fitness < best.fitness) {
                best = population[0];
            }
            printf("evo generation %d best: nll %.6f ppl %.6f\n",
                    generation + 1, population[0].fitness, std::exp(population[0].fitness));
            if (generation + 1 == params.generations) break;

            const std::vector<float> previous_mean = mean;
            std::fill(mean.begin(), mean.end(), 0.0f);
            for (int elite = 0; elite < params.elite_count; ++elite) {
                for (size_t gene = 0; gene < gene_count; ++gene) {
                    mean[gene] += recombination[elite] * population[elite].genes[gene];
                }
            }
            double normalized_step2 = 0.0;
            for (size_t gene = 0; gene < gene_count; ++gene) {
                float selected_step2 = 0.0f;
                for (int elite = 0; elite < params.elite_count; ++elite) {
                    const float step = (population[elite].genes[gene] - previous_mean[gene]) / std::max(sigma, 1e-6f);
                    selected_step2 += recombination[elite] * step * step;
                }
                variance[gene] = std::max(1e-4f, (1.0f - covariance_rate) * variance[gene] + covariance_rate * selected_step2);
                const float mean_step = (mean[gene] - previous_mean[gene]) /
                    (std::max(sigma, 1e-6f) * std::sqrt(variance[gene]));
                normalized_step2 += mean_step * mean_step;
            }
            const double normalized_step = std::sqrt(normalized_step2 / gene_count);
            sigma = std::max(0.005f, std::min(1.0f, (float) (sigma * std::exp(0.2 * (normalized_step - 1.0)))));
            printf("evo generation %d CMA-ES sigma %.6f\n", generation + 1, sigma);
        }
        printf("evo final: writing %s (nll %.6f, ppl %.6f, seed %u)\n",
                params.output.c_str(), best.fitness, std::exp(best.fitness), seed);
        write_evo_candidate(params.output, params, inputs, tensor_names, best, target_type, merge_backend);
        std::remove(temporary.c_str());
        if (merge_backend) ggml_backend_free(merge_backend);
        llama_backend_free();
    } catch (...) {
        std::remove(temporary.c_str());
        if (merge_backend) ggml_backend_free(merge_backend);
        llama_backend_free();
        throw;
    }
}

int main(int argc, char ** argv) {
    try {
        const merge_params params = parse_args(argc, argv);
        std::vector<std::unique_ptr<gguf_input>> inputs;
        inputs.emplace_back(new gguf_input(params.base));
        for (const std::string & path : params.models) {
            inputs.emplace_back(new gguf_input(path));
        }

        const gguf_input & base = *inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            if (get_kv_string(base.gguf, "general.architecture") != get_kv_string(inputs[i]->gguf, "general.architecture")) {
                throw std::runtime_error("input models have different general.architecture values");
            }
            if (base.tensors.size() != inputs[i]->tensors.size()) {
                throw std::runtime_error("input models have different tensor counts");
            }
            for (const auto & entry : base.tensors) {
                const auto it = inputs[i]->tensors.find(entry.first);
                if (it == inputs[i]->tensors.end() || !same_shape(entry.second.tensor, it->second.tensor)) {
                    throw std::runtime_error("tensor schema differs for '" + entry.first + "'");
                }
            }
        }

        if (params.method == "evo") {
            run_evo(params, inputs);
            return 0;
        }

        const ggml_type precision_output = select_precision_type(inputs);
        const bool preserve_quant = precision_output == GGML_TYPE_COUNT;
        const uint32_t output_ftype = preserve_quant ? base.file_type : type_to_ftype(precision_output);
        if (preserve_quant) {
            for (const auto & entry : base.tensors) {
                for (size_t i = 1; i < inputs.size(); ++i) {
                    if (entry.second.tensor->type != inputs[i]->tensors.at(entry.first).tensor->type) {
                        throw std::runtime_error("same quant model type has different tensor types for '" + entry.first + "'");
                    }
                }
            }
        }

        gguf_context * output_gguf = gguf_init_empty();
        gguf_set_kv(output_gguf, base.gguf);
        gguf_remove_key(output_gguf, GGUF_KEY_GENERAL_ALIGNMENT);
        gguf_set_val_u32(output_gguf, "general.file_type", output_ftype);
        gguf_remove_key(output_gguf, "split.no");
        gguf_remove_key(output_gguf, "split.count");
        gguf_remove_key(output_gguf, "split.tensors.count");

        ggml_init_params init_params = {
            /*.mem_size   = */ base.tensors.size() * ggml_tensor_overhead(),
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ true,
        };
        ggml_context * output_ctx = ggml_init(init_params);
        std::map<std::string, ggml_type> output_types;
        for (const auto & entry : base.tensors) {
            const ggml_tensor * input_tensor = entry.second.tensor;
            const ggml_type output_type = preserve_quant ? input_tensor->type : precision_output;
            ggml_tensor * output_tensor = ggml_new_tensor(output_ctx, output_type, GGML_MAX_DIMS, input_tensor->ne);
            ggml_set_name(output_tensor, input_tensor->name);
            gguf_add_tensor(output_gguf, output_tensor);
            output_types[entry.first] = output_type;
        }

        std::ofstream output(params.output, std::ios::binary);
        output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        write_zeros(output, gguf_get_meta_size(output_gguf));

        std::vector<std::string> tensor_names;
        tensor_names.reserve(base.tensors.size());
        for (const auto & entry : base.tensors) {
            tensor_names.push_back(entry.first);
        }

        std::atomic<size_t> next_job { 0 };
        size_t next_to_write = 0;
        bool stop = false;
        std::exception_ptr worker_error;
        std::mutex write_mutex;
        std::condition_variable write_condition;
        const int n_workers = select_worker_count(params, base, inputs.size());
        if (n_workers < params.n_threads) {
            printf("llama-merge: limiting workers from %d to %d to bound TIES working memory\n", params.n_threads, n_workers);
        }
        std::vector<std::thread> workers;
        workers.reserve(n_workers);

        for (int worker = 0; worker < n_workers; ++worker) {
            workers.emplace_back([&] {
                try {
                    std::vector<uint8_t> bytes;
                    std::vector<float> base_values;
                    std::vector<float> model_values;
                    std::vector<float> merged;

                    for (;;) {
                        const size_t tensor_index = next_job.fetch_add(1);
                        if (tensor_index >= tensor_names.size()) {
                            return;
                        }
                        const std::string & name = tensor_names[tensor_index];
                        decode_tensor(base, name, bytes, base_values);
                        std::vector<std::vector<float>> tasks;
                        tasks.reserve(inputs.size() - 1);
                        for (size_t i = 1; i < inputs.size(); ++i) {
                            decode_tensor(*inputs[i], name, bytes, model_values);
                            for (size_t j = 0; j < model_values.size(); ++j) {
                                model_values[j] -= base_values[j];
                            }
                            tasks.push_back(std::move(model_values));
                        }
                        ties_merge(base_values, tasks, params.density, merged);

                        std::unique_lock<std::mutex> lock(write_mutex);
                        write_condition.wait(lock, [&] { return stop || tensor_index == next_to_write; });
                        if (stop) {
                            return;
                        }
                        write_tensor(output, output_types.at(name), base.tensors.at(name).tensor, merged, GGUF_DEFAULT_ALIGNMENT);
                        printf("merged %s\n", name.c_str());
                        ++next_to_write;
                        lock.unlock();
                        write_condition.notify_all();
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(write_mutex);
                    if (!worker_error) {
                        worker_error = std::current_exception();
                    }
                    stop = true;
                    write_condition.notify_all();
                }
            });
        }
        for (std::thread & worker : workers) {
            worker.join();
        }
        if (worker_error) {
            std::rethrow_exception(worker_error);
        }

        std::vector<uint8_t> metadata(gguf_get_meta_size(output_gguf));
        gguf_get_meta_data(output_gguf, metadata.data());
        output.seekp(0);
        output.write((const char *) metadata.data(), metadata.size());
        ggml_free(output_ctx);
        gguf_free(output_gguf);
        printf("wrote %s using TIES (%zu models, density %.3f, %d threads)\n", params.output.c_str(), inputs.size(), params.density, n_workers);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "llama-merge: %s\n", error.what());
        return 1;
    }
}
