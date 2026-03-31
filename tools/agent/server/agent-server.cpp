#include "agent-routes.h"
#include "agent-session.h"

// server-context.h already included via agent-session.h -> agent-loop.h
#include "server-http.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

// MCP support (Unix only)
#ifndef _WIN32
#include "../mcp/mcp-server-manager.h"
#include "../mcp/mcp-tool-wrapper.h"
#endif

#include <atomic>
#include <csignal>
#include <functional>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

// Wrapper to handle exceptions in handlers
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = json{{"error", e.what()}}.dump();
            return res;
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 500;
            res->data = json{{"error", e.what()}}.dump();
            LOG_ERR("Handler exception: %s\n", e.what());
            return res;
        } catch (...) {
            auto res = std::make_unique<server_http_res>();
            res->status = 500;
            res->data = json{{"error", "Unknown error"}}.dump();
            return res;
        }
    };
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // Set defaults for agent server
    if (params.n_parallel < 0) {
        params.n_parallel = 4;
        params.kv_unified = true;
    }

    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    common_init();

    // Initialize server context (holds LLM inference)
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("llama-agent-server starting\n");
    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n",
            params.cpuparams.n_threads, params.cpuparams_batch.n_threads,
            std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    // Initialize HTTP context
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("Failed to initialize HTTP server\n");
        return 1;
    }

    // Create session manager (manages agent sessions)
    agent_session_manager session_mgr(ctx_server, params);

    // Create and register routes
    agent_routes routes(session_mgr);

    ctx_http.get("/health", ex_wrapper(routes.get_health));
    ctx_http.get("/v1/agent/health", ex_wrapper(routes.get_health));

    ctx_http.post("/v1/agent/session", ex_wrapper(routes.post_session));
    ctx_http.get("/v1/agent/session/:id", ex_wrapper(routes.get_session));
    ctx_http.post("/v1/agent/session/:id/delete", ex_wrapper(routes.delete_session));
    ctx_http.get("/v1/agent/sessions", ex_wrapper(routes.get_sessions));

    ctx_http.post("/v1/agent/session/:id/chat", ex_wrapper(routes.post_chat));
    ctx_http.get("/v1/agent/session/:id/messages", ex_wrapper(routes.get_messages));

    ctx_http.get("/v1/agent/session/:id/permissions", ex_wrapper(routes.get_permissions));
    ctx_http.post("/v1/agent/permission/:id", ex_wrapper(routes.post_permission));

    ctx_http.get("/v1/agent/tools", ex_wrapper(routes.get_tools));
    ctx_http.get("/v1/agent/session/:id/stats", ex_wrapper(routes.get_stats));

    // Setup cleanup
    auto clean_up = [&ctx_http, &ctx_server]() {
        LOG_INF("Cleaning up before exit...\n");
        ctx_http.stop();
        ctx_server.terminate();
        llama_backend_free();
    };

    // Start HTTP server before loading model (to serve /health)
    if (!ctx_http.start()) {
        clean_up();
        LOG_ERR("Failed to start HTTP server\n");
        return 1;
    }

    // Load the model
    LOG_INF("Loading model...\n");

    if (!ctx_server.load_model(params)) {
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    ctx_http.is_ready.store(true);
    LOG_INF("Model loaded successfully\n");

    // Initialize MCP servers (Unix only)
    // MCP manager must be declared here to outlive session manager (tools hold pointer to it)
#ifndef _WIN32
    mcp_server_manager mcp_mgr;
    int mcp_tools_count = 0;
    std::string working_dir = ".";  // Default working directory for MCP config search
    std::string mcp_config = find_mcp_config(working_dir);
    if (!mcp_config.empty()) {
        LOG_INF("Loading MCP config from: %s\n", mcp_config.c_str());
        if (mcp_mgr.load_config(mcp_config)) {
            int started = mcp_mgr.start_servers();
            if (started > 0) {
                register_mcp_tools(mcp_mgr);
                mcp_tools_count = static_cast<int>(mcp_mgr.list_all_tools().size());
                LOG_INF("MCP: %d servers started, %d tools registered\n", started, mcp_tools_count);
            }
        }
    }
#else
    int mcp_tools_count = 0;
#endif

    // Setup signal handlers
    shutdown_handler = [&ctx_server](int) {
        ctx_server.terminate();
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG_INF("\n");
    LOG_INF("============================================\n");
    LOG_INF("llama-agent-server is listening on %s\n", ctx_http.listening_address.c_str());
    LOG_INF("============================================\n");
    LOG_INF("\n");
    if (mcp_tools_count > 0) {
        LOG_INF("MCP tools: %d\n", mcp_tools_count);
    }
    LOG_INF("API Endpoints:\n");
    LOG_INF("  POST /v1/agent/session           - Create a new session\n");
    LOG_INF("  GET  /v1/agent/session/:id       - Get session info\n");
    LOG_INF("  POST /v1/agent/session/:id/chat  - Send message (streaming SSE)\n");
    LOG_INF("  GET  /v1/agent/session/:id/messages - Get conversation history\n");
    LOG_INF("  GET  /v1/agent/tools             - List available tools\n");
    LOG_INF("  GET  /health                     - Health check\n");
    LOG_INF("\n");

    // Start the main inference loop
    ctx_server.start_loop();

    // Clean up after shutdown
    clean_up();

    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }

    LOG_INF("llama-agent-server stopped\n");
    return 0;
}
