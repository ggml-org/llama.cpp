#include "tiered-memory.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;

struct tensor_location {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct options {
    std::string model;
    std::string manifest = "tiered-memory.manifest";
    uint64_t vram_mib = 0;
    uint64_t dram_mib = 0;
    bool print_command = false;
};

[[noreturn]] void usage(const char * argv0, const std::string & error = {}) {
    if (!error.empty()) {
        std::cerr << "error: " << error << "\n\n";
    }
    std::cerr
        << "usage: " << argv0 << " MODEL.gguf --vram-mib N --dram-mib N [options]\n"
        << "\n"
        << "options:\n"
        << "  --manifest PATH   output runtime registration manifest\n"
        << "  --print-command   print a llama-cli command using the generated plan\n";
    std::exit(error.empty() ? 0 : 1);
}

options parse_options(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0], "missing model path");
    }

    options result;
    result.model = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (++i >= argc) {
                usage(argv[0], std::string("missing value for ") + name);
            }
            return argv[i];
        };

        if (arg == "--vram-mib") {
            result.vram_mib = std::stoull(require_value("--vram-mib"));
        } else if (arg == "--dram-mib") {
            result.dram_mib = std::stoull(require_value("--dram-mib"));
        } else if (arg == "--manifest") {
            result.manifest = require_value("--manifest");
        } else if (arg == "--print-command") {
            result.print_command = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        } else {
            usage(argv[0], "unknown option: " + arg);
        }
    }

    if (result.vram_mib == 0 && result.dram_mib == 0) {
        usage(argv[0], "at least one tier budget must be non-zero");
    }
    return result;
}

uint64_t get_unsigned(const gguf_context * ctx, const std::string & key, uint64_t fallback = 0) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return fallback;
    }

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT8:   return std::max<int64_t>(0, gguf_get_val_i8(ctx, id));
        case GGUF_TYPE_INT16:  return std::max<int64_t>(0, gguf_get_val_i16(ctx, id));
        case GGUF_TYPE_INT32:  return std::max<int64_t>(0, gguf_get_val_i32(ctx, id));
        case GGUF_TYPE_INT64:  return std::max<int64_t>(0, gguf_get_val_i64(ctx, id));
        default:               return fallback;
    }
}

std::string get_string(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    return gguf_get_val_str(ctx, id);
}

std::string regex_escape(const std::string & value) {
    static const std::string special = R"(\.^$|()[]{}*+?)";
    std::string result;
    result.reserve(value.size() * 2);
    for (const char ch : value) {
        if (special.find(ch) != std::string::npos) {
            result.push_back('\\');
        }
        result.push_back(ch);
    }
    return result;
}

std::string shell_quote(const std::string & value) {
    std::string result = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            result += "'\\''";
        } else {
            result.push_back(ch);
        }
    }
    result.push_back('\'');
    return result;
}

std::string canonical_path(const std::string & path) {
    return std::filesystem::canonical(std::filesystem::path(path)).string();
}

void print_summary(const common_tiered_memory_plan & plan) {
    const auto mib = [](double bytes) { return bytes / 1024.0 / 1024.0; };
    std::cout << std::fixed << std::setprecision(2)
              << "VRAM: " << mib(plan.vram_bytes) << " MiB, active " << mib(plan.active_vram_bytes) << " MiB/token\n"
              << "DRAM: " << mib(plan.dram_bytes) << " MiB, active " << mib(plan.active_dram_bytes) << " MiB/token\n"
              << "SSD : " << mib(plan.ssd_bytes)  << " MiB, active " << mib(plan.active_ssd_bytes)  << " MiB/token\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        const std::string model_path = canonical_path(opts.model);

        gguf_init_params init_params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ nullptr,
        };
        gguf_context * raw_ctx = gguf_init_from_file(model_path.c_str(), init_params);
        if (!raw_ctx) {
            throw std::runtime_error("failed to parse GGUF metadata");
        }
        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(raw_ctx, gguf_free);

        const std::string arch = get_string(ctx.get(), "general.architecture");
        if (arch.empty()) {
            throw std::runtime_error("GGUF is missing general.architecture");
        }

        const uint32_t n_expert = static_cast<uint32_t>(get_unsigned(ctx.get(), arch + ".expert_count"));
        const uint32_t n_expert_used = static_cast<uint32_t>(get_unsigned(ctx.get(), arch + ".expert_used_count"));
        const uint64_t data_offset = gguf_get_data_offset(ctx.get());

        std::vector<common_tiered_memory_item> items;
        std::unordered_map<std::string, tensor_location> locations;

        const int64_t n_tensors = gguf_get_n_tensors(ctx.get());
        items.reserve(static_cast<size_t>(n_tensors));
        locations.reserve(static_cast<size_t>(n_tensors));

        for (int64_t i = 0; i < n_tensors; ++i) {
            const std::string name = gguf_get_tensor_name(ctx.get(), i);
            const uint64_t size = gguf_get_tensor_size(ctx.get(), i);
            const uint64_t offset = data_offset + gguf_get_tensor_offset(ctx.get(), i);
            const double active_fraction = common_tiered_memory_active_fraction(name, n_expert, n_expert_used);

            items.push_back({name, static_cast<size_t>(size), active_fraction});
            locations.emplace(name, tensor_location{offset, size});
        }

        const auto plan = common_tiered_memory_make_plan(items, {
            static_cast<size_t>(opts.vram_mib * MiB),
            static_cast<size_t>(opts.dram_mib * MiB),
        });

        std::ofstream manifest(opts.manifest);
        if (!manifest) {
            throw std::runtime_error("failed to create manifest: " + opts.manifest);
        }
        manifest << "# llama-tiered-memory-v1\n";
        manifest << "model\t" << model_path << "\n";

        std::vector<std::string> cpu_tensors;
        for (const auto & placement : plan.placements) {
            const auto location = locations.at(placement.name);
            const char * tier = common_tiered_memory_tier_name(placement.tier);
            std::string tier_lower = tier;
            std::transform(tier_lower.begin(), tier_lower.end(), tier_lower.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            manifest << tier_lower << '\t' << location.offset << '\t' << location.size << '\t' << placement.name << '\n';

            if (placement.tier != common_tiered_memory_tier::VRAM) {
                cpu_tensors.push_back(placement.name);
            }
        }
        manifest.close();

        print_summary(plan);
        std::cout << "manifest: " << opts.manifest << "\n";

        if (!cpu_tensors.empty()) {
            std::string override_pattern = "^(?:";
            for (size_t i = 0; i < cpu_tensors.size(); ++i) {
                if (i != 0) {
                    override_pattern.push_back('|');
                }
                override_pattern += regex_escape(cpu_tensors[i]);
            }
            override_pattern += ")$=CPU";

            std::cout << "override: " << override_pattern << "\n";
            if (opts.print_command) {
                std::cout
                    << "LLAMA_TIERED_MANIFEST=" << shell_quote(canonical_path(opts.manifest)) << ' '
                    << "LD_PRELOAD=libllama-tiered-preload.so${LD_PRELOAD:+:$LD_PRELOAD} "
                    << "llama-cli -m " << shell_quote(model_path)
                    << " -ngl 999 --override-tensor " << shell_quote(override_pattern) << '\n';
            }
        }

        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
