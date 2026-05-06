#include "preset.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#undef NDEBUG
#include <cassert>

static std::filesystem::path write_ini(const std::string & contents) {
    static int counter = 0;
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count() + counter++;
    const auto path = std::filesystem::temp_directory_path() /
        ("llama-test-preset-" + std::to_string(id) + ".ini");

    std::ofstream file(path);
    assert(file.good());
    file << contents;
    file.close();

    return path;
}

static common_presets load_ini(
        const common_preset_context & ctx,
        const std::string & contents,
        common_preset & global) {
    const auto path = write_ini(contents);
    auto presets = ctx.load_from_ini(path.string(), global);
    std::filesystem::remove(path);

    return presets;
}

static void test_metadata_only_default_is_skipped() {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset global;

    auto presets = load_ini(ctx,
        "version = 1\n"
        "\n"
        "[first]\n"
        "model = /tmp/first.gguf\n"
        "\n"
        "[second]\n"
        "model = /tmp/second.gguf\n",
        global);

    assert(presets.size() == 2);
    assert(presets.find(COMMON_PRESET_DEFAULT_NAME) == presets.end());
    assert(presets.find("first") != presets.end());
    assert(presets.find("second") != presets.end());

    std::string value;
    assert(presets.at("first").get_option("LLAMA_ARG_MODEL", value));
    assert(value == "/tmp/first.gguf");
    assert(presets.at("second").get_option("LLAMA_ARG_MODEL", value));
    assert(value == "/tmp/second.gguf");
}

static void test_default_with_options_is_kept() {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset global;

    auto presets = load_ini(ctx,
        "version = 1\n"
        "model = /tmp/default.gguf\n"
        "\n"
        "[named]\n"
        "model = /tmp/named.gguf\n",
        global);

    assert(presets.size() == 2);
    assert(presets.find(COMMON_PRESET_DEFAULT_NAME) != presets.end());
    assert(presets.find("named") != presets.end());

    std::string value;
    assert(presets.at(COMMON_PRESET_DEFAULT_NAME).get_option("LLAMA_ARG_MODEL", value));
    assert(value == "/tmp/default.gguf");
    assert(presets.at("named").get_option("LLAMA_ARG_MODEL", value));
    assert(value == "/tmp/named.gguf");
}

static void test_global_preset_is_kept() {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset global;

    auto presets = load_ini(ctx,
        "version = 1\n"
        "\n"
        "[*]\n"
        "ctx-size = 4096\n"
        "\n"
        "[named]\n"
        "model = /tmp/named.gguf\n",
        global);

    assert(presets.size() == 1);
    assert(presets.find(COMMON_PRESET_DEFAULT_NAME) == presets.end());
    assert(presets.find("named") != presets.end());
    assert(global.name == "*");

    std::string value;
    assert(global.get_option("LLAMA_ARG_CTX_SIZE", value));
    assert(value == "4096");
}

int main(void) {
    test_metadata_only_default_is_skipped();
    test_default_with_options_is_kept();
    test_global_preset_is_kept();

    printf("test-preset: all tests OK\n");
}
