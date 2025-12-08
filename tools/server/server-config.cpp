#include "server-config.h"

#include "peg-parser.h"
#include "arg.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <set>

namespace {

bool is_option(const std::string & arg) {
    return !arg.empty() && arg[0] == '-';
}

bool is_implicit_value(const std::vector<std::string> & args, size_t index) {
    return index + 1 < args.size() && !is_option(args[index + 1]);
}

std::string relativize(const std::string & path, const std::string & base) {
    if (path.empty()) {
        return path;
    }

    std::error_code ec;
    const auto abs_path = std::filesystem::absolute(path, ec);
    if (ec) {
        return path;
    }
    const auto abs_base = std::filesystem::absolute(base, ec);
    if (ec) {
        return path;
    }

    const auto rel = std::filesystem::relative(abs_path, abs_base, ec);
    if (ec) {
        return path;
    }

    return rel.generic_string();
}

} // namespace

server_config_manager::server_config_manager(const std::string & models_dir)
    : models_dir(models_dir) {
    if (!models_dir.empty()) {
        path = (std::filesystem::path(models_dir) / "config.ini").string();
    }
}

bool server_config_manager::enabled() const {
    return !models_dir.empty();
}

void server_config_manager::ensure_loaded() {
    if (!enabled()) {
        return;
    }

    namespace fs = std::filesystem;

    std::lock_guard<std::mutex> lock(mutex);

    if (!fs::exists(path)) {
        data.clear();
        return;
    }

    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server config file: " + path);
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    static const auto parser = build_peg_parser([](auto & p) {
        // newline ::= "\r\n" / "\n" / "\r"
        auto newline = p.rule("newline", p.literal("\r\n") | p.literal("\n") | p.literal("\r"));

        // ws ::= [ \t]*
        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));

        // comment ::= [;#] (!newline .)*
        auto comment = p.rule("comment", p.chars("[;#]", 1, 1) + p.zero_or_more(p.negate(newline) + p.any()));

        // eol ::= ws comment? (newline / EOF)
        auto eol = p.rule("eol", ws + p.optional(comment) + (newline | p.end()));

        // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
        auto ident = p.rule("ident", p.chars("[a-zA-Z_]", 1, 1) + p.chars("[a-zA-Z0-9_.-]", 0, -1));

        // value ::= (!eol-start .)*
        auto eol_start = p.rule("eol-start", ws + (p.chars("[;#]", 1, 1) | newline | p.end()));
        auto value = p.rule("value", p.zero_or_more(p.negate(eol_start) + p.any()));

        // header-line ::= "[" ws ident ws "]" eol
        auto header_line = p.rule("header-line", "[" + ws + p.tag("section-name", p.chars("[^]]")) + ws + "]" + eol);

        // kv-line ::= ident ws "=" ws value eol
        auto kv_line = p.rule("kv-line", p.tag("key", ident) + ws + "=" + ws + p.tag("value", value) + eol);

        // comment-line ::= ws comment (newline / EOF)
        auto comment_line = p.rule("comment-line", ws + comment + (newline | p.end()));

        // blank-line ::= ws (newline / EOF)
        auto blank_line = p.rule("blank-line", ws + (newline | p.end()));

        // line ::= header-line / kv-line / comment-line / blank-line
        auto line = p.rule("line", header_line | kv_line | comment_line | blank_line);

        // ini ::= line* EOF
        auto ini = p.rule("ini", p.zero_or_more(line) + p.end());

        return ini;
    });

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success()) {
        throw std::runtime_error("failed to parse server config file: " + path);
    }

    std::map<std::string, std::map<std::string, std::string>> parsed;

    std::string current_section;
    std::string current_key;

    ctx.ast.visit(result, [&](const auto & node) {
        if (node.tag == "section-name") {
            const std::string section = std::string(node.text);
            if (section.rfind("LLAMA_ARG_", 0) == 0) {
                current_section.clear();
                return;
            }

            current_section = section;
            parsed[current_section] = {};
        } else if (node.tag == "key") {
            const std::string key = std::string(node.text);
            if (key.rfind("LLAMA_ARG_", 0) == 0) {
                current_key = key;
            } else {
                current_key.clear();
            }
        } else if (node.tag == "value" && !current_key.empty() && !current_section.empty()) {
            parsed[current_section][current_key] = std::string(node.text);
            current_key.clear();
        }
    });

    data = std::move(parsed);
}

// write_locked expects the caller to hold `mutex`.
void server_config_manager::write_locked() {
    if (!enabled()) {
        return;
    }

    namespace fs = std::filesystem;

    if (!path.empty()) {
        auto parent = fs::path(path).parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
    }

    std::ofstream file(path);
    file << "LLAMA_CONFIG_VERSION=1\n\n";

    bool first_section = true;
    for (const auto & [section, args] : data) {
        if (!first_section) {
            file << "\n";
        }
        first_section = false;

        file << "[" << section << "]\n";
        for (const auto & [key, value] : args) {
            file << key << "=";
            if (!value.empty()) {
                file << value;
            }
            file << "\n";
        }
    }

    file.flush();
}

bool is_router_control_arg(const std::string & arg) {
    static const std::set<std::string> blacklist = {
        "--alias",          // set per-child in server_models::load
        "--models-dir",     // router-side discovery only
        "--models-max",     // router capacity control
        "--no-models-autoload", // router autoload policy
        "--port",           // router port differs from child port
        "-m", "--model",  // model path supplied per-child
        "-hf", "--hf-file" // model source supplied per-child
    };
    return blacklist.count(arg) != 0;
}

void server_config_manager::sync(const std::vector<server_local_model> & models, const std::vector<std::string> & base_args) {
    if (!enabled()) {
        return;
    }

    ensure_loaded();

    std::map<std::string, std::string> router_args;

    for (size_t i = 1; i < base_args.size(); ++i) { // skip argv[0]
        const auto & arg = base_args[i];
        if (!is_option(arg)) {
            continue;
        }

        if (is_router_control_arg(arg)) {
            if (is_implicit_value(base_args, i)) {
                ++i;
            }
            continue;
        }

        std::string value = "true";
        if (is_implicit_value(base_args, i)) {
            value = base_args[i + 1];
            ++i;
        }

        const auto env_name = common_arg_get_env_name(arg);
        if (!env_name.empty()) {
            router_args[env_name] = value;
        }
    }

    std::lock_guard<std::mutex> lock(mutex);

    bool changed = !std::filesystem::exists(path);

    const auto model_key  = common_arg_get_env_name("--model");
    const auto model_alias = common_arg_get_env_name("-m");
    const auto mmproj_key = common_arg_get_env_name("--mmproj");

    const std::vector<std::string> model_keys = {
        model_key,
        model_alias,
        "LLAMA_ARG_MODEL",
    };

    const std::vector<std::string> mmproj_keys = {
        mmproj_key,
        "LLAMA_ARG_MMPROJ",
    };

    for (const auto & model : models) {
        auto & section = data[model.name];

        const auto has_any_key = [](const auto & section_map, const std::vector<std::string> & keys) {
            for (const auto & key : keys) {
                if (!key.empty() && section_map.find(key) != section_map.end()) {
                    return true;
                }
            }
            return false;
        };

        if (!model_key.empty() && !has_any_key(section, model_keys)) {
            section[model_key] = relativize(model.path, models_dir);
            changed = true;
        }

        if (!model.path_mmproj.empty() && !mmproj_key.empty() && !has_any_key(section, mmproj_keys)) {
            section[mmproj_key] = relativize(model.path_mmproj, models_dir);
            changed = true;
        }

        for (const auto & router_arg : router_args) {
            if (section.find(router_arg.first) == section.end()) {
                section[router_arg.first] = router_arg.second;
                changed = true;
            }
        }
    }

    if (changed) {
        write_locked();
    }
}

std::map<std::string, std::string> server_config_manager::env_for(const std::string & name) {
    if (!enabled()) {
        return {};
    }

    ensure_loaded();

    std::lock_guard<std::mutex> lock(mutex);

    auto it = data.find(name);
    return it != data.end() ? it->second : std::map<std::string, std::string>{};
}

