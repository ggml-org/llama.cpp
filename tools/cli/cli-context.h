// controller for llama-cli (the "controller" in MVC)
//
// owns the chat state, drives the view and talks to llama-server through
// cli_client; when no --server-base is given it also manages a local
// llama-server child process via cli_server

#pragma once

#include "common.h"

#include "cli-client.h"
#include "cli-server.h"

#include <atomic>
#include <optional>
#include <string>

struct cli_timings {
    double prompt_per_second    = 0.0;
    double predicted_per_second = 0.0;
};

struct cli_command_info {
    std::string usage;       // e.g. "/read <file>"
    std::string description; // e.g. "add a text file"
};

// properties of the connected server, shown on startup
struct cli_server_info {
    std::string build_info;
    std::string model_name;
    std::string server_base;
    bool is_local_server   = false; // server is spawned and managed by llama-cli
    bool has_system_prompt = false;
    bool has_vision        = false;
    bool has_audio         = false;
    bool has_video         = false;

    std::vector<cli_command_info> commands;
};

// set by the SIGINT handler; cleared once the interrupt has been handled
extern std::atomic<bool> g_cli_interrupted;

struct cli_context {
    common_params params;

    cli_client client;                // always initialized
    std::optional<cli_server> server; // only set when no --server-base is given

    json messages      = json::array();
    json pending_media = json::array(); // staged multimodal content parts

    // properties of the connected server
    std::string model_name;
    std::string build_info;
    bool has_vision = false;
    bool has_audio  = false;
    bool has_video  = false;

    cli_context(const common_params & params) : params(params) {}

    // connect to --server-base or spawn a local llama-server child;
    // argc/argv are needed to forward the server-relevant args to the child
    bool init();

    // run the interactive chat loop, returns the process exit code
    int run();

    // stop the local server child (if any)
    void shutdown();

private:
    bool generate_completion(std::string & assistant_content, cli_timings & timings);
    void fetch_server_props();
    void add_system_prompt();
    void push_user_message(const std::string & text);

    // read a file and stage it as a multimodal content part; type is one of
    // "image", "audio", "video"; returns false if the file cannot be read
    bool stage_media_file(const std::string & fname, const std::string & type);
};
