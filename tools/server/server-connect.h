#pragma once

#include "common.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

struct common_subproc;

// spawns llama-connect, which exposes this server to a remote browser over WebRTC
// core binary is in a separate project: https://github.com/ggml-org/llama-connect
struct server_connect {
    server_connect();
    ~server_connect();

    server_connect(const server_connect &) = delete;
    server_connect & operator=(const server_connect &) = delete;

    // path of the llama-connect binary, empty if not found
    static std::string find_binary();

    // why --connect cannot work here, empty if it can
    static std::string unavailable_reason(const common_params & params);

    bool start(const common_params & params);

    // idempotent, also called by the destructor
    void stop();

private:
    std::unique_ptr<common_subproc> proc;
    std::thread log_thread;
    std::atomic<bool> stopping{false}; // tells the log thread the exit is expected
};
