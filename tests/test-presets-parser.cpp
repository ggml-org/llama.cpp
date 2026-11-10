#include "preset.h"
#include "configuration-options.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

static void test_options(const std::filesystem::path & path) {
    for (const auto & test_case : configuration_options) {
        {
            std::ofstream file(path, std::ios::binary);
            file << "[model]\n" << test_case.ini << '\n';
        }

        std::printf("test-presets-parser: %s\n", test_case.ini.c_str());
        common_preset_context ctx(test_case.example);
        common_preset global;
        common_presets presets;
        common_params params;
        params.model.path = "model_file.gguf";
        bool valid = true;
        try {
            presets = ctx.load_from_ini(path, global);
            auto args = presets.at("model").to_args("binary_name");
            std::vector<char *> argv;
            for (auto & arg : args) {
                argv.push_back(arg.data());
            }
            // Validate preset output as the server's child process does.
            valid = common_params_parse(argv.size(), argv.data(), params, test_case.example);
        } catch (const std::exception & e) {
            std::printf("test-presets-parser: %s\n", e.what());
            valid = false;
        }
        assert(valid == test_case.valid);
        if (!valid) {
            continue;
        }
        test_case.check(params);

        {
            std::ofstream file(path, std::ios::binary);
            file << presets.at("model").to_ini();
        }
        const auto reloaded = ctx.load_from_ini(path, global);
        assert(reloaded.at("model").to_args() == presets.at("model").to_args());
    }
}

static void test_comments(const std::filesystem::path & path) {
    common_preset_context ctx(LLAMA_EXAMPLE_COMMON);
    common_preset global;
    for (const std::string newline : {"\n", "\r\n", "\r"}) {
        for (const char marker : {';', '#'}) {
            for (bool final_newline : {false, true}) {
                {
                    std::ofstream file(path, std::ios::binary);
                    file << marker << " comment" << newline
                         << " \t" << marker << " indented comment" << newline
                         << newline
                         << "[model]" << newline
                         << "model = test.gguf \t";
                    if (final_newline) {
                        file << newline << marker << " final comment" << newline;
                    }
                }
                const auto presets = ctx.load_from_ini(path, global);
                assert(presets.size() == 1);
                common_params params;
                presets.at("model").apply_to_params(params);
                assert(params.model.path == "test.gguf");
            }
        }
    }
    for (const std::string invalid : {"[model] ; comment\n", "[model] # comment\n"}) {
        {
            std::ofstream file(path, std::ios::binary);
            file << invalid;
        }
        try {
            ctx.load_from_ini(path, global);
        } catch (const std::runtime_error &) {
            continue;
        }
        throw std::runtime_error("accepted invalid comment: " + invalid);
    }
    for (const char marker : {';', '#'}) {
        {
            std::ofstream file(path, std::ios::binary);
            file << " \t" << marker << " comment at EOF";
        }
        assert(ctx.load_from_ini(path, global).empty());
    }
}

int main() {
    const auto path = std::filesystem::temp_directory_path() / ("llama-test-presets-parser-" + std::to_string(std::random_device{}()) + ".ini");
    int result = 0;
    try {
        test_options(path);
        test_comments(path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-presets-parser: %s\n", e.what());
        result = 1;
    }
    std::filesystem::remove(path);
    return result;
}
