#include "cli-context.h"
#include "cli-ui.h"

#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "console.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

using json = nlohmann::ordered_json;

struct cli_context_impl {
    json messages      = json::array();
    json pending_media = json::array(); // staged multimodal content parts
};

cli_context::cli_context(const common_params & params) : params(params), impl(new cli_context_impl()) {}

cli_context::~cli_context() {
    shutdown();
}

std::atomic<bool> & cli_context::interrupted() {
    static std::atomic<bool> flag = false;
    return flag;
}

static bool should_stop() {
    return cli_context::interrupted().load();
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

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

static std::string format_error_message(const json & err) {
    if (err.contains("error") && err.at("error").is_object()) {
        const auto & e = err.at("error");
        if (e.contains("message") && e.at("message").is_string()) {
            return e.at("message").get<std::string>();
        }
    }
    return err.dump();
}

// err is the raw response body of a failed request; it may or may not be JSON
static std::string format_error_message(const std::string & err) {
    json parsed = json::parse(err, nullptr, false);
    if (!parsed.is_discarded()) {
        return format_error_message(parsed);
    }
    return err;
}

static std::string media_type_from_ext(const std::string & fname) {
    std::string ext = std::filesystem::path(fname).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".wav" || ext == ".mp3") {
        return "audio";
    }
    if (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".webm") {
        return "video";
    }
    return "image";
}

bool cli_context::init() {
    ui::init(params);

    std::optional<ui::spinner> spinner;

    bool use_external_server = !params.server_base.empty();
    if (use_external_server) {
        std::string base = params.server_base;
        while (!base.empty() && base.back() == '/') {
            base.pop_back();
        }
        client.server_base = base;

        spinner.emplace("Connecting to server at " + base);
    } else {
        if (params.model.path.empty() && params.model.url.empty() &&
                params.model.hf_repo.empty() && params.model.docker_repo.empty()) {
            ui::show_error(
                "no model specified",
                "use -m <file.gguf> or -hf <user/repo> to run a local model,\n"
                "or --server-base <url> to connect to a running llama-server"
            );
            return false;
        }

        spinner.emplace("\n\nLoading model...");

        server.emplace();
        if (!server->start(params)) {
            ui::show_error("server start failed");
            return false;
        }
        if (!server->wait_ready(should_stop)) {
            if (!should_stop()) {
                ui::show_error("the server exited before becoming ready");
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
        if (!should_stop()) {
            ui::show_error(client.last_error);
        }
        return false;
    }

    if (use_external_server) {
        spinner.reset();
        if (!list_and_ask_models()) {
            return false;
        }
        // restore the spinner for the next step
        spinner.emplace("Waiting for server...");
    }

    fetch_server_props();

    return true;
}

void cli_context::fetch_server_props() {
    try {
        json props = json::parse(client.get("/props"));
        model_name = props.value("model_alias", "");
        if (model_name.empty()) {
            const std::string path = props.value("model_path", "");
            if (!path.empty()) {
                model_name = std::filesystem::path(path).filename().string();
            }
        }
        model_ftype = props.value("model_ftype", "");
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

bool cli_context::list_and_ask_models() {
    json resp = json::parse(client.get("/v1/models"));
    if (!resp.contains("data") || !resp.at("data").is_array()) {
        throw std::runtime_error("invalid response from /v1/models");
    }
    std::vector<std::string> models;
    std::vector<std::string> models_display;
    for (const auto & m : resp.at("data")) {
        if (!m.contains("id") || !m.at("id").is_string()) {
            continue;
        }
        std::string name = m.at("id").get<std::string>();
        std::string display = name;
        if (m.contains("aliases") && m.at("aliases").is_array()) {
            std::vector<std::string> aliases;
            for (const auto & a : m.at("aliases")) {
                if (a.is_string()) {
                    aliases.push_back(a.get<std::string>());
                }
            }
            if (!aliases.empty()) {
                display += " (" + string_join(aliases, ", ") + ")";
            }
        }
        models.push_back(name);
        models_display.push_back(display);
    }

    // only one model: use it without asking
    if (models.size() == 1) {
        model_name = models[0];
        client.model = model_name;
        return true;
    }

    std::string message = "\nAvailable models:";
    for (size_t i = 0; i < models_display.size(); ++i) {
        message += "\n  " + std::to_string(i + 1) + ". " + models_display[i];
    }
    message += "\n";
    ui::show_message(message);
    std::string selection;
    while (selection.empty()) {
        if (should_stop()) {
            return false;
        }
        ui::user_turn user_turn;
        selection = user_turn.read_input(false, "Select model by number: ");
        if (selection.empty()) {
            continue;
        }
        try {
            size_t idx = std::stoul(selection);
            if (idx > 0 && idx <= models.size()) {
                model_name = models[idx - 1];
                client.model = model_name;
                ui::show_message("Selected model: " + model_name);
                break;
            }
        } catch (...) {
            // ignore
        }
        ui::show_error("Invalid selection. Please enter a valid number.");
        selection.clear();
        continue;
    }
    return true;
}

void cli_context::add_system_prompt() {
    if (!params.system_prompt.empty()) {
        impl->messages.push_back({
            {"role",    "system"},
            {"content", params.system_prompt}
        });
    }
}

void cli_context::push_user_message(const std::string & text) {
    json content;
    if (impl->pending_media.empty()) {
        content = text;
    } else {
        // multimodal message: media parts first, then the text
        content = impl->pending_media;
        content.push_back({
            {"type", "text"},
            {"text", text}
        });
        impl->pending_media = json::array();
    }
    impl->messages.push_back({
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
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        impl->pending_media.push_back({
            {"type", "input_audio"},
            {"input_audio", {
                {"data",   encoded},
                {"format", ext == ".mp3" ? "mp3" : "wav"}
            }}
        });
    } else if (type == "video") {
        impl->pending_media.push_back({
            {"type", "input_video"},
            {"input_video", {
                {"data", encoded}
            }}
        });
    } else {
        // the server detects the actual image type from the data
        impl->pending_media.push_back({
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
        {"messages",          impl->messages},
        {"stream",            true},
        // in order to get timings even when we cancel mid-way
        {"timings_per_token", true},
    };
    if (!client.model.empty()) {
        body["model"] = client.model;
    }

    bool stream_error = false;

    ui::assistant_turn a;

    std::string err = client.post_sse("/v1/chat/completions", body.dump(), should_stop, [&](const std::string & payload) {
        json chunk = json::parse(payload, nullptr, false);
        if (chunk.is_discarded()) {
            return;
        }
        if (chunk.contains("error")) {
            stream_error = true;
            ui::show_error(format_error_message(chunk));
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
                a.push(ui::ASSISTANT_DISPLAY_MODE_REASONING, text);
            }
        }
        if (delta.contains("content") && delta.at("content").is_string()) {
            const std::string text = delta.at("content").get<std::string>();
            if (!text.empty()) {
                assistant_content += text;
                a.push(ui::ASSISTANT_DISPLAY_MODE_CONTENT, text);
            }
        }
    });

    cli_context::interrupted().store(false);

    if (!err.empty()) {
        ui::show_error(format_error_message(err));
        return false;
    }
    return !stream_error;
}

int cli_context::run() {
    add_system_prompt();

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

    std::string banner;
    banner += "\n";
    banner += LLAMA_ASCII_LOGO;
    banner += "\n";
    banner += "build      : " + build_info + "\n";
    banner += "model      : " + model_name + "\n";
    if (!model_ftype.empty()) {
        banner += "ftype      : " + model_ftype + "\n";
    }
    banner += "modalities : " + modalities + "\n";
    if (!params.system_prompt.empty()) {
        banner += "using custom system prompt\n";
    }
    banner += "\n";
    banner += "available commands:\n";
    banner += "  /exit or Ctrl+C     stop or exit\n";
    banner += "  /regen              regenerate the last response\n";
    banner += "  /clear              clear the chat history\n";
    banner += "  /read <file>        add a text file\n";
    banner += "  /glob <pattern>     add text files using globbing pattern\n";
    if (has_vision) {
        banner += "  /image <file>       add an image file\n";
    }
    if (has_audio) {
        banner += "  /audio <file>       add an audio file\n";
    }
    if (has_video) {
        banner += "  /video <file>       add a video file\n";
    }
    banner += "\n";

    ui::show_message(banner);

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            ui::show_error(string_format("file does not exist or cannot be opened: '%s'", fname.c_str()));
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cur_msg += "--- File: ";
        cur_msg += fname;
        cur_msg += " ---\n";
        cur_msg += content;
        ui::show_message(string_format("Loaded text from '%s'", fname.c_str()));
        return true;
    };

    while (true) {
        std::string buffer;
        {
            ui::user_turn user_turn;

            if (params.prompt.empty()) {
                buffer = user_turn.read_input(params.multiline_input);
            } else {
                // process input prompt from args
                for (auto & fname : params.image) {
                    if (!stage_media_file(fname, media_type_from_ext(fname))) {
                        ui::show_error(string_format("file does not exist or cannot be opened: '%s'", fname.c_str()));
                        break;
                    }
                    ui::show_message(string_format("Loaded media from '%s'", fname.c_str()));
                }
                buffer = params.prompt;
                user_turn.echo(buffer);
                params.prompt.clear(); // only use it once
            }
        }

        if (should_stop()) {
            cli_context::interrupted().store(false);
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
            if (impl->messages.size() >= 2) {
                size_t last_idx = impl->messages.size() - 1;
                impl->messages.erase(last_idx);
                add_user_msg = false;
            } else {
                ui::show_error("No message to regenerate.");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            impl->messages.clear();
            add_system_prompt();

            impl->pending_media = json::array();
            ui::show_message("Chat history cleared.");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && has_vision) ||
                (string_starts_with(buffer, "/audio ") && has_audio) ||
                (string_starts_with(buffer, "/video ") && has_video)) {
            std::string type = buffer.substr(1, 5);
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            if (!stage_media_file(fname, type)) {
                ui::show_error(string_format("file does not exist or cannot be opened: '%s'", fname.c_str()));
                continue;
            }
            ui::show_message(string_format("Loaded media from '%s'", fname.c_str()));
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
                    ui::show_error(string_format("Maximum number of globbed files allowed (%zu) reached.", FILE_GLOB_MAX_RESULTS));
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
        impl->messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });

        if (params.show_timings) {
            ui::show_info(string_format(
                "\n[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                timings.prompt_per_second,
                timings.predicted_per_second
            ));
        }

        if (params.single_turn) {
            break;
        }
    }

    ui::show_message("\n\nExiting...");

    return 0;
}

void cli_context::shutdown() {
    if (server) {
        server->stop();
        server.reset();
    }
}
