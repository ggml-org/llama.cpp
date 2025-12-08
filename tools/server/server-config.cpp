#include "server-config.h"

#include "peg-parser.h"
#include "arg.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <optional>
#include <set>

namespace {

bool is_option(const std::string & arg) {
    return !arg.empty() && arg[0] == '-';
}

std::string trim(const std::string & value) {
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    size_t start = 0;
    while (start < value.size() && is_space(value[start])) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && is_space(value[end - 1])) {
        --end;
    }
    return value.substr(start, end - start);
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
        last_write_time = {};
        return;
    }

    const auto current_write_time = fs::last_write_time(path);
    if (last_write_time == current_write_time) {
        return;
    }

    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server config file: " + path);
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    static const auto & parser = *new common_peg_arena(build_peg_parser([](common_peg_parser_builder & p) {
        const auto ws = p.space();
        const auto new_line = p.choice({p.literal("\r\n"), p.literal("\n"), p.literal("\r")});

        const auto section_name = p.tag("section-name", p.until("]"));
        const auto section_line = p.zero_or_more(ws) + "[" + section_name + "]" + p.optional(p.until_one_of({"\r", "\n"}));

        const auto key = p.tag("key", p.until("="));
        const auto value = p.tag("value", p.until_one_of({"\r", "\n"}));
        const auto key_value_line = p.zero_or_more(ws) + key + p.zero_or_more(ws) + "=" + p.zero_or_more(ws) + p.optional(value);

        const auto comment = p.choice({p.literal(";"), p.literal("#")}) + p.optional(p.until_one_of({"\r", "\n"}));
        const auto comment_line = p.zero_or_more(ws) + comment;

        const auto blank_line = p.zero_or_more(ws) + new_line;

        const auto line = p.choice({
            section_line << p.optional(new_line),
            key_value_line << p.optional(new_line),
            comment_line << p.optional(new_line),
            blank_line,
        });

        return p.rule("ini", p.zero_or_more(line) << p.optional(p.zero_or_more(ws)) << p.end());
    }));

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success() || result.end != contents.size()) {
        throw std::runtime_error("failed to parse server config file: " + path);
    }

    std::map<std::string, std::map<std::string, std::string>> parsed;
    std::string current_section;
    std::optional<std::string> pending_key;

    const auto flush_pending = [&](const std::string & value) {
        if (current_section.empty() || !pending_key) {
            return;
        }

        const auto & key = *pending_key;
        if (key.rfind("LLAMA_ARG_", 0) != 0) {
            return;
        }

        parsed[current_section][key] = value;
    };

    ctx.ast.visit(result, [&](const common_peg_ast_node & node) {
        if (node.tag == "section-name") {
            if (pending_key) {
                flush_pending("");
                pending_key.reset();
            }

            current_section = trim(std::string(node.text));
            return;
        }

        if (node.tag == "key") {
            if (pending_key) {
                flush_pending("");
            }

            pending_key = trim(std::string(node.text));
            return;
        }

        if (node.tag == "value") {
            if (!pending_key) {
                return;
            }

            flush_pending(trim(std::string(node.text)));
            pending_key.reset();
            return;
        }
    });

    if (pending_key) {
        flush_pending("");
    }

    data = std::move(parsed);
    last_write_time = current_write_time;
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
    last_write_time = fs::last_write_time(path);
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

