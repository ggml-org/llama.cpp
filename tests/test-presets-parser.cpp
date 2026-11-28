#include "preset.h"
#include "log.h"
#include "ini.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

#undef NDEBUG
#include <cassert>

static void test_comments_v1(const std::filesystem::path & path) {
    common_preset_context ctx(LLAMA_EXAMPLE_COMMON);
    common_preset global;
    for (const std::string invalid : {"[model] ; comment\n", "[model] # comment\n"}) {
        {
            std::ofstream file(path, std::ios::binary);
            file << "version = 1\n" << invalid;
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
            file << "version = 1\n \t" << marker << " comment at EOF";
        }
        assert(ctx.load_from_ini(path, global).empty());
    }
}

static void test_ini_v1(const std::filesystem::path & path) {
    struct ini_case {
        std::string contents;
        std::string prompt;
        std::string error;
        llama_example example = LLAMA_EXAMPLE_COMMON;
        std::string template_kwargs = "";
    };
    const std::vector<ini_case> cases = {
        {"version = 1\n; comment\n # comment\n[model]\nprompt = # literal ; literal", "# literal ; literal", ""},
        {"version = 1\n[model]\nprompt = value\n  with a space\n with zero space\n   with three space", "value\n with a space\nwith zero space\n  with three space", ""},
        {"version = 1\n[model]\n  prompt = value\n   continuation\n    extra space\n  model = next.gguf", "value\ncontinuation\n extra space", ""},
        {"version = 1\n[model]\nprompt = value\n next = literal\n [literal section]\n[other]\nprompt = other", "value\nnext = literal\n[literal section]", ""},
        {"version = 1\n[model]\nprompt = value\n ; comment\n\n # comment\n continuation # literal ; literal", "value\ncontinuation # literal ; literal", ""},
        {"version = 1\n[model]\nchat-template-kwargs = {\n\"custom_instructions\": \"Keep ; and # in values.\"\n}", "", "at line 4", LLAMA_EXAMPLE_SERVER},
        {"version = 1\n[model]\nchat-template-kwargs = {\n   \"custom_instructions\": \"Keep ; and # in values.\"\n }", "", "", LLAMA_EXAMPLE_SERVER, "\"Keep ; and # in values.\""},
        {"version = 1\n[model]\nprompt = value\n \tkeep tab", "value\n\tkeep tab", ""},
        {"version = 1\n[model]\nprompt =\n continuation", "\ncontinuation", ""},
        {"version = 1\n[model]\n continuation", "", "at line 3"},
        {"version = 1\n[model]\nprompt = value\nnot deeper", "", "at line 4"},
        {"version = 1\n[model]\n prompt = value\n same depth", "", "at line 4"},
        {"version = 1\n[model]\n  prompt = value\n less depth", "", "at line 4"},
        {"; comment\n# version = legacy\nversion = 1\nprompt = global\n version = legacy\n[model]\nprompt = # literal", "# literal", ""},
        {"version = 1\n[model]\nprompt = value\n version = legacy", "value\nversion = legacy", ""},
        {"version = 1\n[model]\n version = legacy", "", ""},
        {"version = 1\n[model]\n prompt = value\n version = legacy", "value", ""},
    };
    common_preset global;
    for (const auto & test_case : cases) {
        common_preset_context ctx(test_case.example);
        for (const std::string newline : {"\n", "\r\n", "\r"}) {
            for (bool final_newline : {false, true}) {
                {
                    auto contents = test_case.contents + (final_newline ? "\n" : "");
                    string_replace_all(contents, "\n", newline);
                    std::ofstream file(path, std::ios::binary);
                    file << contents;
                }
                common_presets presets;
                bool valid = true;
                try {
                    presets = ctx.load_from_ini(path, global);
                } catch (const std::runtime_error & e) {
                    valid = false;
                    assert(!test_case.error.empty());
                    assert(std::string(e.what()).find(test_case.error) != std::string::npos);
                    assert(std::string(e.what()).find(path.string()) != std::string::npos);
                }
                assert(valid == test_case.error.empty());
                if (!valid) {
                    continue;
                }
                common_params params;
                presets.at("model").apply_to_params(params);
                assert(params.prompt == test_case.prompt);
                if (!test_case.template_kwargs.empty()) {
                    assert(params.default_template_kwargs.at("custom_instructions") == test_case.template_kwargs);
                }
            }
        }
    }
}

static void test_legacy_warnings(const std::filesystem::path & path) {
    common_preset_context ctx(LLAMA_EXAMPLE_COMMON);
    common_preset global;
    const auto log_path = path.string() + ".log";
    for (bool explicit_version : {false, true}) {
        const std::string version = explicit_version ? "version = legacy\n" : "";
        {
            std::ofstream file(path, std::ios::binary);
            file << version << "[model] ; comment\r\nprompt = value# comment";
        }
        common_log_set_file(common_log_main(), log_path.c_str());
        const auto presets = ctx.load_from_ini(path, global);
        common_log_set_file(common_log_main(), nullptr);
        common_params params;
        presets.at("model").apply_to_params(params);
        assert(params.prompt == "value");
        {
            std::ifstream log_file(log_path);
            const std::string log((std::istreambuf_iterator<char>(log_file)), std::istreambuf_iterator<char>());
            assert((log.find("inline comments use") != std::string::npos) == !explicit_version);
            if (!explicit_version) {
                assert(log.find(path.string() + ":1: inline comments") != std::string::npos);
                assert(log.find(path.string() + ":2: inline comments") != std::string::npos);
            }
        }
        for (const std::string input : {" prompt = value", "prompt = value\n continuation", " prompt = value; comment"}) {
            {
                std::ofstream file(path, std::ios::binary);
                file << version << "\r\n[model]\r" << input;
            }
            common_log_set_file(common_log_main(), log_path.c_str());
            bool valid = true;
            try {
                ctx.load_from_ini(path, global);
            } catch (const std::runtime_error & e) {
                valid = false;
                assert(std::string(e.what()) == "failed to parse server config file: " + path.string());
            }
            common_log_set_file(common_log_main(), nullptr);
            assert(!valid);
            std::ifstream log_file(log_path);
            const std::string log((std::istreambuf_iterator<char>(log_file)), std::istreambuf_iterator<char>());
            assert(log.find("indented keys or continuation lines") == std::string::npos);
            assert(log.find("inline comments use") == std::string::npos);
        }
    }
    std::filesystem::remove(log_path);
}

static void test_version_scope(const std::filesystem::path & path) {
    common_preset_context ctx(LLAMA_EXAMPLE_COMMON);
    common_preset global;
    for (const std::string version : {"", "version = legacy\n", "version = 1\n"}) {
        for (const std::string section : {"model", "*", "default"}) {
            for (const std::string newline : {"\n", "\r\n", "\r"}) {
                {
                    auto contents = version + "[" + section + "]\nversion = first\nversion = arbitrary\nprompt = value";
                    string_replace_all(contents, "\n", newline);
                    std::ofstream file(path, std::ios::binary);
                    file << contents;
                }
                assert(ini_version_from_file(path) == (version == "version = 1\n" ? "1" : "legacy"));
                ctx.load_from_ini(path, global);
                if (version == "version = 1\n") {
                    const auto parsed = parse_ini_from_file_v1(path);
                    assert(parsed.at(section).at("version") == "arbitrary");
                }
            }
        }
    }
    for (const std::string first : {"legacy", "1"}) {
        for (const std::string second : {"legacy", "1"}) {
            {
                std::ofstream file(path, std::ios::binary);
                file << "version = " << first << "\nversion = " << second << "\n[model]";
            }
            bool valid = true;
            try {
                ctx.load_from_ini(path, global);
            } catch (const std::runtime_error & e) {
                valid = false;
                assert(std::string(e.what()).find("only be declared once at line 2") != std::string::npos);
            }
            assert(!valid);
        }
    }
}

int main() {
    const auto path = std::filesystem::temp_directory_path() / ("llama-test-presets-parser-" + std::to_string(std::random_device{}()) + ".ini");
    int result = 0;
    try {
        test_comments_v1(path);
        test_ini_v1(path);
        test_legacy_warnings(path);
        test_version_scope(path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-presets-parser: %s\n", e.what());
        result = 1;
    }
    std::filesystem::remove(path);
    return result;
}
