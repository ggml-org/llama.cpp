#pragma once

#include <thread>

#include "http.h"

// spawn llama-server in a thread and interact with it via a random port
// note: in the future, we may have a server running as daemon and the CLI can connect to it automatically

// llama_server will be available as a dynamic library symbol
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();

struct cli_server {
    std::thread th;
    int port = -1;
    std::atomic<bool> is_alive = false;
    std::atomic<bool> is_stopping = false;

    ~cli_server() {
        stop();
    }

    void stop() {
        if (alive() && !is_stopping.exchange(true)) {
            llama_server_terminate();
            th.join();
        }
    }

    bool start(common_params & params) {
        port = common_http_get_free_port();
        if (port <= 0) {
            fprintf(stderr, "failed to get a free port\n");
            exit(1);
        }

        is_alive.store(true, std::memory_order_release);

        th = std::thread([&]() {
            common_params server_params = params; // copy
            server_params.port = port;
            // argc / argv are only used in router mode, we can skip them for now
            int res = llama_server(server_params, 0, nullptr);
            if (res != 0) {
                fprintf(stderr, "llama_server exited with code %d\n", res);
            }
            is_alive.store(false, std::memory_order_release);
        });

        return true;
    }

    std::string address() const {
        return "http://127.0.0.1:" + std::to_string(port);
    }

    bool wait_ready(std::function<bool()> should_stop) {
        if (!alive()) {
            return false;
        }
        while (!should_stop()) {
            auto [cli, parts] = common_http_client(address());
            cli.set_connection_timeout(1, 0);
            auto res = cli.Get("/health");
            if (res) {
                if (res->status == 200) {
                    return true;
                }
                // any other status means the server is up but not ready yet
                // (e.g. 503 while the model is still loading)
            }
            if (!alive()) {
                // in case server die permanently
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        return true;
    }

    bool alive() const {
        return is_alive.load(std::memory_order_acquire);
    }
};
