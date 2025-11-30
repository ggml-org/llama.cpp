#include "common.h"
#include "download.h"
#include "log.h"
#include "logging.h"
#include "router-app.h"
#include "router-config.h"
#include "router-constants.h"
#include "router-endpoints.h"

#include <cpp-httplib/httplib.h>

#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

static std::atomic<bool> g_shutdown{false};
static RouterApp *        g_app    = nullptr;
static httplib::Server *  g_server = nullptr;

static void signal_handler(int) {
    g_shutdown = true;
    if (g_server) {
        g_server->stop();
    }
    if (g_app) {
        g_app->stop_all();
    }
}

struct CliOptions {
    bool        show_help = false;
    std::string hf_repo;
    std::string hf_file;
    std::string config_path;
};

static bool parse_cli(int argc, char ** argv, CliOptions & out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            out.show_help = true;
        } else if (arg == "-hf" || arg == "-hfr" || arg == "--hf-repo") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                return false;
            }
            out.hf_repo = argv[++i];
        } else if (arg == "-hff" || arg == "--hf-file") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                return false;
            }
            out.hf_file = argv[++i];
        } else if (arg == "--config") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for --config\n");
                return false;
            }
            out.config_path = argv[++i];
        } else {
            fprintf(stderr, "warning: unknown argument %s\n", arg.c_str());
        }
    }
    return true;
}

static void print_help() {
    printf("usage: llama-router [options]\n\n");
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  --config <path>         Override config path (default: ~/.config/llama.cpp/router-config.json)\n");
    printf("  -hf, -hfr, --hf-repo    Hugging Face repository to download (format <user>/<repo>[:quant])\n");
    printf("  -hff, --hf-file         Specific GGUF filename to fetch from repository\n");
}

static bool handle_download(const CliOptions & opts) {
    if (opts.hf_repo.empty()) {
        return false;
    }

    const char * hf_token = std::getenv("HF_TOKEN");
    std::string  token    = hf_token ? std::string(hf_token) : std::string();

    try {
        auto resolved = common_get_hf_file(opts.hf_repo, token, false);
        std::string repo = resolved.repo;
        std::string file = !opts.hf_file.empty() ? opts.hf_file : resolved.ggufFile;
        if (file.empty()) {
            fprintf(stderr, "error: unable to find GGUF file in repo %s\n", repo.c_str());
            return true;
        }

        std::string url = get_model_endpoint() + repo + "/resolve/main/" + file;
        std::string filename = repo + "_" + file;
        string_replace_all(filename, "/", "_");
        std::string local_path = fs_get_cache_file(filename);

        common_params_model model;
        model.hf_repo = repo;
        model.hf_file = file;
        model.url     = url;
        model.path    = local_path;

        LOG_INF("Downloading %s to %s\n", url.c_str(), local_path.c_str());
        if (!common_download_model(model, token, false)) {
            fprintf(stderr, "download failed\n");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "hf download error: %s\n", e.what());
    }
    return true;
}

int main(int argc, char ** argv) {
    CliOptions cli;
    LOG_INF("Parsing %d CLI arguments for llama-router\n", argc);

    router_log_init();

    if (!parse_cli(argc, argv, cli)) {
        return 1;
    }

    if (cli.show_help) {
        print_help();
        return 0;
    }

    if (handle_download(cli)) {
        LOG_INF("Download-only mode completed, exiting\n");
        return 0;
    }

    std::string config_path = !cli.config_path.empty() ? expand_user_path(cli.config_path) : get_default_config_path();
    LOG_INF("Loading router configuration from %s\n", config_path.c_str());

    RouterConfig cfg;
    try {
        cfg = load_config(config_path);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    LOG_INF("Router configuration loaded: %zu models, base port %d, listen %s:%d\n",
            cfg.models.size(), cfg.router.base_port, cfg.router.host.c_str(), cfg.router.port);

    RouterApp app(cfg);
    g_app = &app;
    LOG_INF("Initialized RouterApp with default spawn command size=%zu\n", cfg.default_spawn.command.size());
    app.start_auto_models();
    LOG_INF("Auto-start requested, last spawned model: %s\n", app.get_last_spawned_model().c_str());

    httplib::Server server;
    g_server = &server;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    register_routes(server, app);

    std::string host = cfg.router.host;
    int         port = cfg.router.port;

    LOG_INF("llama-router listening on %s:%d\n", host.c_str(), port);
    std::atomic<bool> listen_ok{true};
    std::thread       server_thread([&]() {
        if (!server.listen(host.c_str(), port)) {
            listen_ok = false;
            g_shutdown = true;
        }
    });

    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ROUTER_POLL_INTERVAL_MS));
    }

    server.stop();
    if (server_thread.joinable()) {
        server_thread.join();
    }

    LOG_INF("llama-router shutting down, stopping all managed models\n");
    app.stop_all();
    g_app    = nullptr;
    g_server = nullptr;
    return listen_ok ? 0 : 1;
}
