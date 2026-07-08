#pragma once

#include "common.h"

#include "cli-client.h"
#include "cli-server.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>

struct cli_timings {
    double prompt_per_second    = 0.0;
    double predicted_per_second = 0.0;
};

struct cli_context_impl;

struct cli_context {
    common_params params;

    cli_client client;                // always initialized
    std::optional<cli_server> server; // only set when no --server-base is given

    // properties of the connected server
    // will be populated by fetch_server_props()
    std::string model_name;
    std::string model_ftype;
    std::string build_info;
    bool has_vision = false;
    bool has_audio  = false;
    bool has_video  = false;

    cli_context(const common_params & params);
    ~cli_context();

    // connect to --server-base or spawn a local llama-server child;
    // argc/argv are needed to forward the server-relevant args to the child
    bool init();

    // run the interactive chat loop, returns the process exit code
    int run();

    // stop the local server child (if any)
    void shutdown();

    // set by the SIGINT handler; cleared once the interrupt has been handled
    static std::atomic<bool> & interrupted();

private:
    bool generate_completion(std::string & assistant_content, cli_timings & timings);
    void fetch_server_props();
    void add_system_prompt();
    void push_user_message(const std::string & text);

    // check if server have multiple models (router mode)
    // if yes, list them then ask; do nothing otherwise
    bool list_and_ask_models();

    // read a file and stage it as a multimodal content part; type is one of
    // "image", "audio", "video"; returns false if the file cannot be read
    bool stage_media_file(const std::string & fname, const std::string & type);

    std::unique_ptr<cli_context_impl> impl;
};
