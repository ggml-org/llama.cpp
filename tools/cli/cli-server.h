#pragma once

#include <thread>

#include "http.h"

// spawn llama-server in a thread and interact with it via a random port
// note: in the future, we may have a server running as daemon and the CLI can connect to it automatically

// llama_server will be available as a dynamic library symbol
int llama_server(common_params & params, int argc, char ** argv);

struct cli_server {
    std::thread th;
    int port = -1;

    ~cli_server() {
        stop();
    }

    void stop() {
        if (th.joinable()) {
            th.detach();
        }
    }

    bool start(common_params & params) {
        port = common_http_get_free_port();
        if (port <= 0) {
            fprintf(stderr, "failed to get a free port\n");
            exit(1);
        }

        th = std::thread([&]() {
            // argc / argv are only used in router mode, we can skip them for now
            int res = llama_server(params, 0, nullptr);
            if (res != 0) {
                fprintf(stderr, "llama_server exited with code %d\n", res);
            }
        });

        return true;
    }

    std::string address() const {
        return "http://127.0.0.1:" + std::to_string(port);
    }

    bool wait_ready(std::function<bool()> should_stop) {
        // while (true) {
        //     if (should_stop()) {
        //         break;
        //     }
        //     std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        // }
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        return true;
    }

    bool alive() const {
        return th.joinable();
    }
};
