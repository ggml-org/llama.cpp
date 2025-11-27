// run.cpp - Interactive chat mode using llama-server infrastructure
//
// This is essentially llama-server in chat mode, without the HTTP server.
// It uses the same server infrastructure for task processing and completion.

#include "arg.h"
#include "common.h"
#include "run-chat.h"
#include "server-context.h"
#include "log.h"

#include <cstdio>
#include <thread>

int main(int argc, char ** argv) {
    // Initialize logging system
    common_init();

    common_params params;

    // Parse command-line arguments
    // Note: If user specifies -v or -lv, it will override the above default
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // Initialize server context
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n",
            params.cpuparams.n_threads, params.cpuparams_batch.n_threads,
            std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    // Load model
    LOG_INF("%s: loading model\n", __func__);
    if (!ctx_server.load_model(params)) {
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        llama_backend_free();
        return 1;
    }

    ctx_server.init();
    LOG_INF("%s: model loaded\n", __func__);

    // Setup signal handlers for graceful shutdown
    setup_signal_handlers([&](int) {
        ctx_server.terminate();
    });

    // Start the task processing thread
    std::thread task_thread([&ctx_server]() {
        ctx_server.start_loop();
    });

    // Run interactive chat mode
    run_chat_mode(params, ctx_server);

    // Clean up
    ctx_server.terminate();
    if (task_thread.joinable()) {
        task_thread.join();
    }
    llama_backend_free();

    return 0;
}
