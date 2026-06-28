#include "preset.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string arg_value(const std::vector<std::string> & args, const std::string & key) {
    auto it = std::find(args.begin(), args.end(), key);
    if (it == args.end()) {
        throw std::runtime_error("missing arg: " + key);
    }
    ++it;
    if (it == args.end()) {
        throw std::runtime_error("missing value for arg: " + key);
    }
    return *it;
}

static void require_eq(const std::string & actual, const std::string & expected) {
    if (actual != expected) {
        throw std::runtime_error("expected '" + expected + "', got '" + actual + "'");
    }
}

static void test_ini_semicolon_values() {
    const auto path = std::filesystem::temp_directory_path() / "llama-test-preset.ini";

    {
        std::ofstream file(path);
        file << "; comment line\n"
             << "[model]\n"
             << "model = test.gguf\n"
             << "samplers = top_k;top_p;temperature ; inline comment\n"
             << "temp = 0.7 ; inline comment\n";
    }

    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset global;
    const common_presets presets = ctx.load_from_ini(path.string(), global);
    const auto args = presets.at("model").to_args();

    std::filesystem::remove(path);

    require_eq(arg_value(args, "--model"), "test.gguf");
    require_eq(arg_value(args, "--samplers"), "top_k;top_p;temperature");
    require_eq(arg_value(args, "--temperature"), "0.7");
}

int main(void) {
    try {
        test_ini_semicolon_values();
    } catch (std::exception & e) {
        fprintf(stderr, "test-preset: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
