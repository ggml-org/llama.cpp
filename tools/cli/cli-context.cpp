#include "cli-context.h"

#include "arg.h"
#include "base64.hpp"
#include "log.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

static const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

std::atomic<bool> g_cli_interrupted = false;

static bool should_stop() {
    return g_cli_interrupted.load();
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

// number of values an arg consumes on the command line
static int arg_num_values(const common_arg & opt) {
    if (opt.value_hint_2 != nullptr) {
        return 2;
    }
    if (opt.value_hint != nullptr) {
        return 1;
    }
    return 0;
}

// keep only the args that llama-server understands, so that the remainder
// of the command line can be forwarded to the spawned server child
static std::vector<std::string> filter_server_args(int argc, char ** argv) {
    std::map<std::string, int> cli_n_values; // arg -> number of values
    std::set<std::string>      server_args;

    common_params dummy_cli;
    auto ctx_cli = common_params_parser_init(dummy_cli, LLAMA_EXAMPLE_CLI);
    for (const auto & opt : ctx_cli.options) {
        for (const char * a : opt.args) {
            cli_n_values[a] = arg_num_values(opt);
        }
        for (const char * a : opt.args_neg) {
            cli_n_values[a] = 0;
        }
    }

    common_params dummy_server;
    auto ctx_server = common_params_parser_init(dummy_server, LLAMA_EXAMPLE_SERVER);
    for (const auto & opt : ctx_server.options) {
        for (const char * a : opt.args) {
            server_args.insert(a);
        }
        for (const char * a : opt.args_neg) {
            server_args.insert(a);
        }
    }

    std::vector<std::string> result;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto it = cli_n_values.find(arg);
        if (it == cli_n_values.end()) {
            // not a known arg (should not happen when parsing succeeded)
            continue;
        }
        const bool forward = server_args.count(arg) > 0;
        if (forward) {
            result.push_back(arg);
        }
        for (int j = 0; j < it->second && i + 1 < argc; j++) {
            i++;
            if (forward) {
                result.push_back(argv[i]);
            }
        }
    }
    return result;
}

static std::string format_error_message(const json & err) {
    if (err.contains("error") && err.at("error").is_object()) {
        const auto & e = err.at("error");
        if (e.contains("message") && e.at("message").is_string()) {
            return e.at("message").get<std::string>();
        }
    }
    return err.dump();
}

static std::string media_type_from_ext(const std::string & fname) {
    std::string ext = std::filesystem::path(fname).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".wav" || ext == ".mp3") {
        return "audio";
    }
    if (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".webm") {
        return "video";
    }
    return "image";
}

bool cli_context::init(int argc, char ** argv) {
    if (!params.server_base.empty()) {
        std::string base = params.server_base;
        while (!base.empty() && base.back() == '/') {
            base.pop_back();
        }
        client.server_base = base;

        view.print("Connecting to " + client.server_base + " ... ");
        view.spinner_start();
    } else {
        if (params.model.path.empty() && params.model.url.empty() &&
                params.model.hf_repo.empty() && params.model.docker_repo.empty()) {
            view.print_error("no model specified\n");
            view.print("use -m <file.gguf> or -hf <user/repo> to run a local model,\n"
                       "or --server-base <url> to connect to a running llama-server\n");
            return false;
        }

        const bool pass_output = params.verbosity >= LOG_LEVEL_INFO;

        view.print("Loading model... ");
        view.spinner_start();

        server.emplace();
        if (!server->start(filter_server_args(argc, argv), pass_output)) {
            view.spinner_stop();
            view.print_error("\n" + server->last_error + "\n");
            return false;
        }
        if (!server->wait_ready(should_stop)) {
            view.spinner_stop();
            if (!should_stop()) {
                view.print_error("\nthe server exited before becoming ready\n");
                if (!pass_output) {
                    view.print(server->recent_output());
                }
            }
            return false;
        }
        client.server_base = server->address();
    }

    // for --server-base this is the main availability check; for a spawned
    // server it is a cheap sanity check on top of the ready signal
    auto is_aborted = [this]() {
        return should_stop() || (server && !server->alive());
    };
    bool healthy = false;
    try {
        healthy = client.wait_health(is_aborted);
    } catch (const std::exception & e) {
        client.last_error = e.what();
    }
    if (!healthy) {
        view.spinner_stop();
        if (!should_stop()) {
            view.print_error("\n" + client.last_error + "\n");
        }
        return false;
    }

    fetch_server_props();

    view.spinner_stop();
    view.print("\n");

    return true;
}

void cli_context::fetch_server_props() {
    try {
        json props = client.get_props();
        model_name = props.value("model_alias", "");
        if (model_name.empty()) {
            const std::string path = props.value("model_path", "");
            if (!path.empty()) {
                model_name = std::filesystem::path(path).filename().string();
            }
        }
        build_info = props.value("build_info", "");
        if (props.contains("modalities") && props.at("modalities").is_object()) {
            const auto & modalities = props.at("modalities");
            has_vision = modalities.value("vision", false);
            has_audio  = modalities.value("audio", false);
            has_video  = modalities.value("video", false);
        }
    } catch (const std::exception & e) {
        // /props can be disabled on remote servers; not fatal
        LOG_DBG("failed to fetch /props: %s\n", e.what());
    }
}

void cli_context::add_system_prompt() {
    if (!params.system_prompt.empty()) {
        messages.push_back({
            {"role",    "system"},
            {"content", params.system_prompt}
        });
    }
}

void cli_context::push_user_message(const std::string & text) {
    json content;
    if (pending_media.empty()) {
        content = text;
    } else {
        // multimodal message: media parts first, then the text
        content = pending_media;
        content.push_back({
            {"type", "text"},
            {"text", text}
        });
        pending_media = json::array();
    }
    messages.push_back({
        {"role",    "user"},
        {"content", content}
    });
}

bool cli_context::stage_media_file(const std::string & fname, const std::string & type) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        return false;
    }
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::string encoded = base64::encode(data);

    if (type == "audio") {
        std::string ext = std::filesystem::path(fname).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        pending_media.push_back({
            {"type", "input_audio"},
            {"input_audio", {
                {"data",   encoded},
                {"format", ext == ".mp3" ? "mp3" : "wav"}
            }}
        });
    } else if (type == "video") {
        pending_media.push_back({
            {"type", "input_video"},
            {"input_video", {
                {"data", encoded}
            }}
        });
    } else {
        // the server detects the actual image type from the data
        pending_media.push_back({
            {"type", "image_url"},
            {"image_url", {
                {"url", "data:image/unknown;base64," + encoded}
            }}
        });
    }
    return true;
}

bool cli_context::generate_completion(std::string & assistant_content, cli_timings & timings) {
    json body = {
        {"messages",          messages},
        {"stream",            true},
        // in order to get timings even when we cancel mid-way
        {"timings_per_token", true},
    };

    bool is_thinking   = false;
    bool spinner_alive = true;
    bool stream_error  = false;

    auto stop_spinner = [&]() {
        if (spinner_alive) {
            spinner_alive = false;
            view.spinner_stop();
        }
    };

    view.spinner_start();

    json err = client.create_chat_completion(body, should_stop, [&](const json & chunk) {
        if (chunk.contains("error")) {
            stop_spinner();
            stream_error = true;
            view.print_error("Error: " + format_error_message(chunk) + "\n");
            return;
        }
        if (chunk.contains("timings")) {
            const auto & t = chunk.at("timings");
            timings.prompt_per_second    = t.value("prompt_per_second",    0.0);
            timings.predicted_per_second = t.value("predicted_per_second", 0.0);
        }
        if (!chunk.contains("choices") || !chunk.at("choices").is_array() || chunk.at("choices").empty()) {
            return;
        }
        const auto & choice = chunk.at("choices").at(0);
        if (!choice.contains("delta")) {
            return;
        }
        const auto & delta = choice.at("delta");
        if (delta.contains("reasoning_content") && delta.at("reasoning_content").is_string()) {
            const std::string text = delta.at("reasoning_content").get<std::string>();
            if (!text.empty()) {
                stop_spinner();
                if (!is_thinking) {
                    view.print_reasoning("[Start thinking]\n");
                    is_thinking = true;
                }
                view.print_reasoning(text);
                view.flush();
            }
        }
        if (delta.contains("content") && delta.at("content").is_string()) {
            const std::string text = delta.at("content").get<std::string>();
            if (!text.empty()) {
                stop_spinner();
                if (is_thinking) {
                    view.print_reasoning("\n[End thinking]\n\n");
                    is_thinking = false;
                }
                assistant_content += text;
                view.print(text);
                view.flush();
            }
        }
    });

    stop_spinner();
    g_cli_interrupted.store(false);

    if (!err.is_null()) {
        view.print_error("Error: " + format_error_message(err) + "\n");
        return false;
    }
    return !stream_error;
}

int cli_context::run() {
    std::string modalities = "text";
    if (has_vision) {
        modalities += ", vision";
    }
    if (has_audio) {
        modalities += ", audio";
    }
    if (has_video) {
        modalities += ", video";
    }

    add_system_prompt();

    view.print("\n");
    view.print(LLAMA_ASCII_LOGO);
    view.print("\n");
    if (!build_info.empty()) {
        view.print(string_format("build      : %s\n", build_info.c_str()));
    }
    view.print(string_format("model      : %s\n", model_name.empty() ? "(unknown)" : model_name.c_str()));
    view.print(string_format("server     : %s%s\n", client.server_base.c_str(), server ? " (managed by llama-cli)" : ""));
    view.print(string_format("modalities : %s\n", modalities.c_str()));
    if (!params.system_prompt.empty()) {
        view.print("using custom system prompt\n");
    }
    view.print("\n");
    view.print("available commands:\n");
    view.print("  /exit or Ctrl+C     stop or exit\n");
    view.print("  /regen              regenerate the last response\n");
    view.print("  /clear              clear the chat history\n");
    view.print("  /read <file>        add a text file\n");
    view.print("  /glob <pattern>     add text files using globbing pattern\n");
    if (has_vision) {
        view.print("  /image <file>       add an image file\n");
    }
    if (has_audio) {
        view.print("  /audio <file>       add an audio file\n");
    }
    if (has_video) {
        view.print("  /video <file>       add a video file\n");
    }
    view.print("\n");

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            view.print_error(string_format("file does not exist or cannot be opened: '%s'\n", fname.c_str()));
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cur_msg += "--- File: ";
        cur_msg += fname;
        cur_msg += " ---\n";
        cur_msg += content;
        view.print(string_format("Loaded text from '%s'\n", fname.c_str()));
        return true;
    };

    while (true) {
        std::string buffer;
        if (params.prompt.empty()) {
            view.print_user("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = view.readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                if (!stage_media_file(fname, media_type_from_ext(fname))) {
                    view.print_error(string_format("file does not exist or cannot be opened: '%s'\n", fname.c_str()));
                    break;
                }
                view.print(string_format("Loaded media from '%s'\n", fname.c_str()));
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                view.print_user(string_format("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str()));
            } else {
                view.print_user(string_format("\n> %s\n", buffer.c_str()));
            }
            params.prompt.clear(); // only use it once
        }
        view.print("\n");

        if (should_stop()) {
            g_cli_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() && buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (messages.size() >= 2) {
                size_t last_idx = messages.size() - 1;
                messages.erase(last_idx);
                add_user_msg = false;
            } else {
                view.print_error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            messages.clear();
            add_system_prompt();

            pending_media = json::array();
            view.print("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && has_vision) ||
                (string_starts_with(buffer, "/audio ") && has_audio) ||
                (string_starts_with(buffer, "/video ") && has_video)) {
            std::string type = buffer.substr(1, 5);
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            if (!stage_media_file(fname, type)) {
                view.print_error(string_format("file does not exist or cannot be opened: '%s'\n", fname.c_str()));
                continue;
            }
            view.print(string_format("Loaded media from '%s'\n", fname.c_str()));
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            add_text_file(fname);
            continue;
        } else if (string_starts_with(buffer, "/glob ")) {
            std::error_code ec;
            size_t count = 0;
            auto curdir = std::filesystem::current_path();
            std::string pattern = string_strip(buffer.substr(6));
            std::filesystem::path rel_path;

            auto startglob = pattern.find_first_of("![*?");
            if (startglob != std::string::npos && startglob != 0) {
                auto endpath = pattern.substr(0, startglob).find_last_of('/');
                if (endpath != std::string::npos) {
                    std::string rel_pattern = pattern.substr(0, endpath);
#if !defined(_WIN32)
                    if (string_starts_with(rel_pattern, '~')) {
                        const char * home = std::getenv("HOME");
                        if (home && home[0]) {
                            rel_pattern = home + rel_pattern.substr(1);
                        }
                    }
#endif
                    rel_path = rel_pattern;
                    pattern.erase(0, endpath + 1);
                    curdir /= rel_path;
                }
            }

            for (const auto & entry : std::filesystem::recursive_directory_iterator(curdir,
                    std::filesystem::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string rel = std::filesystem::relative(entry.path(), curdir, ec).string();
                if (ec) {
                    ec.clear();
                    continue;
                }
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(pattern, rel)) {
                    continue;
                }

                if (!add_text_file((rel_path / rel).string())) {
                    continue;
                }

                if (++count >= FILE_GLOB_MAX_RESULTS) {
                    view.print_error(string_format("Maximum number of globbed files allowed (%zu) reached.\n", FILE_GLOB_MAX_RESULTS));
                    break;
                }
            }
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            push_user_message(cur_msg);
            cur_msg.clear();
        }
        cli_timings timings;
        std::string assistant_content;
        generate_completion(assistant_content, timings);
        messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        view.print("\n");

        if (params.show_timings) {
            view.print_info(string_format("\n[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n",
                    timings.prompt_per_second, timings.predicted_per_second));
        }

        if (params.single_turn) {
            break;
        }
    }

    view.print("\nExiting...\n");

    return 0;
}

void cli_context::shutdown() {
    if (server) {
        server->stop();
        server.reset();
    }
}
