#include "server_app.hpp"
#include "http_routes.hpp"
#include "json_utils.hpp"
#include "model_utils.hpp"

// auto generated files (see README.md for details)
#include "index.html.gz.hpp"
#include "loading.html.hpp"

#include <atomic>
#include <signal.h>
#ifdef _WIN32
#include <windows.h>
#endif

/**
 * @brief Main entry point for the llama.cpp HTTP server application.
 * 
 * This file contains the main() function and handles:
 * - Command line argument parsing using common_params infrastructure
 * - Global initialization of llama.cpp backend
 * - Server application lifecycle management
 * - HTTP server setup and startup
 * - Signal handling for graceful shutdown
 * - Server configuration logging and validation
 * 
 * The application architecture follows a clear separation of concerns:
 * - main.cpp: Entry point, CLI parsing, process lifecycle
 * - LlamaServerApp: Core server logic and model management  
 * - HttpServer: HTTP layer and route handling
 * - Utility modules: JSON, model, and other helpers
 */

// Global shutdown handling
std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

/**
 * @brief Signal handler for graceful shutdown on SIGINT/SIGTERM.
 * 
 * Implements a two-stage shutdown process:
 * 1. First signal: Initiates graceful shutdown via shutdown_handler
 * 2. Second signal: Forces immediate termination
 * 
 * This provides better developer experience during testing while ensuring
 * the application can always be terminated if shutdown hangs.
 * 
 * @param signal Signal number received (SIGINT, SIGTERM, etc.)
 */
inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // In case shutdown hangs, we can force terminate by hitting Ctrl+C twice
        // This is primarily for better developer experience during development
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

/**
 * @brief Log server configuration and system information.
 * 
 * Outputs comprehensive configuration information including:
 * - Threading configuration (main threads, batch threads, total threads)
 * - System information (CPU, memory, etc.)
 * - Server-specific configuration parameters
 * 
 * This information is crucial for debugging performance issues and
 * verifying optimal configuration for the deployment environment.
 * 
 * @param params Parsed command line parameters
 */
void log_server_configuration(const common_params & params) {
    LOG_INF("=== llama.cpp HTTP Server Configuration ===\n");
    LOG_INF("Build info: %s\n", LLAMA_COMMIT);
    LOG_INF("Thread configuration:\n");
    LOG_INF("  Main threads: %d\n", params.cpuparams.n_threads);
    LOG_INF("  Batch threads: %d\n", params.cpuparams_batch.n_threads);
    LOG_INF("  Hardware threads: %d\n", std::thread::hardware_concurrency());
    LOG_INF("  HTTP threads: %d\n", params.n_threads_http > 0 ? params.n_threads_http : 
           std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1));
    
    LOG_INF("Model configuration:\n");
    LOG_INF("  Model path: %s\n", params.model.path.c_str());
    LOG_INF("  Context size: %d\n", params.n_ctx);
    LOG_INF("  Parallel slots: %d\n", params.n_parallel);
    LOG_INF("  Batch size: %d\n", params.n_batch);
    
    LOG_INF("Server configuration:\n");
    LOG_INF("  Listen address: %s:%d\n", params.hostname.c_str(), params.port);
    LOG_INF("  API prefix: %s\n", params.api_prefix.c_str());
    
    if (!params.ssl_file_key.empty() && !params.ssl_file_cert.empty()) {
        LOG_INF("  SSL enabled: key=%s, cert=%s\n", 
               params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
    } else {
        LOG_INF("  SSL: disabled\n");
    }
    
    if (!params.api_keys.empty()) {
        if (params.api_keys.size() == 1) {
            const auto & key = params.api_keys[0];
            LOG_INF("  API key: ****%s\n", 
                   key.substr(std::max((int)(key.length() - 4), 0)).c_str());
        } else {
            LOG_INF("  API keys: %d keys loaded\n", (int)params.api_keys.size());
        }
    } else {
        LOG_INF("  API key: not configured (public access)\n");
    }
    
    LOG_INF("System information:\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("============================================\n\n");
}

/**
 * @brief Validate command line parameters for common errors.
 * 
 * Performs validation of parsed parameters to catch common configuration
 * errors early, before attempting to load models or start the server.
 * This provides better error messages and prevents resource waste.
 * 
 * @param params Parsed command line parameters
 * @return true if parameters are valid
 * @return false if validation fails
 */
bool validate_server_params(const common_params & params) {
    // Validate model path
    if (params.model.path.empty()) {
        LOG_ERR("Model path is required. Use --model <path> to specify.\n");
        return false;
    }
    
    // Validate context size
    if (params.n_ctx < 128) {
        LOG_ERR("Context size must be at least 128 tokens\n");
        return false;
    }
    
    // Validate parallel configuration
    if (params.n_parallel < 1) {
        LOG_ERR("Number of parallel slots must be at least 1\n");
        return false;
    }
    
    if (params.n_ctx / params.n_parallel < 64) {
        LOG_WRN("Very small context per slot (%d tokens). Consider reducing --parallel or increasing --ctx-size\n",
               params.n_ctx / params.n_parallel);
    }
    
    // Validate network configuration
    if (params.port < 1 || params.port > 65535) {
        LOG_ERR("Port must be between 1 and 65535\n");
        return false;
    }
    
    // Validate SSL configuration
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (!params.ssl_file_key.empty() || !params.ssl_file_cert.empty()) {
        if (params.ssl_file_key.empty() || params.ssl_file_cert.empty()) {
            LOG_ERR("Both SSL key and certificate files must be provided\n");
            return false;
        }
    }
#else
    if (!params.ssl_file_key.empty() || !params.ssl_file_cert.empty()) {
        LOG_ERR("SSL requested but server was built without SSL support\n");
        return false;
    }
#endif
    
    return true;
}

/**
 * @brief Display comprehensive help information.
 * 
 * Shows detailed usage information including:
 * - Basic usage syntax
 * - Key command line options
 * - Configuration examples
 * - SSL setup instructions
 * - API endpoint information
 * 
 * This supplements the standard common_params help with server-specific
 * guidance and examples.
 */
void show_server_help() {
    printf("\n");
    printf("=== llama.cpp HTTP Server ===\n");
    printf("\n");
    printf("Provides OpenAI-compatible HTTP API for llama.cpp inference.\n");
    printf("\n");
    printf("Basic usage:\n");
    printf("  %s --model <model.gguf> [options]\n", "llama-server");
    printf("\n");
    printf("Key server options:\n");
    printf("  --host <address>         Server listen address (default: 127.0.0.1)\n");
    printf("  --port <number>          Server listen port (default: 8080)\n");
    printf("  --ctx-size <number>      Context size in tokens (default: 2048)\n");
    printf("  --parallel <number>      Number of parallel processing slots (default: 1)\n");
    printf("  --threads <number>       Number of threads for generation (default: auto)\n");
    printf("  --threads-http <number>  Number of HTTP server threads (default: auto)\n");
    printf("  --api-key <key>          API key for authentication (optional)\n");
    printf("  --api-key-file <file>    Read API keys from file (one per line)\n");
    printf("\n");
    printf("SSL options:\n");
    printf("  --ssl-key-file <file>    SSL private key file (PEM format)\n");
    printf("  --ssl-cert-file <file>   SSL certificate file (PEM format)\n");
    printf("\n");
    printf("Multimodal options:\n");
    printf("  --mmproj <file>          Multimodal projection model for vision/audio\n");
    printf("\n");
    printf("Performance options:\n");
    printf("  --batch-size <number>    Batch size for prompt processing (default: 2048)\n");
    printf("  --memory-f32             Use 32-bit floats instead of 16-bit\n");
    printf("  --mlock                  Lock model in memory to prevent swapping\n");
    printf("  --cache-reuse <number>   Enable KV cache reuse optimization\n");
    printf("\n");
    printf("Example usage:\n");
    printf("  # Basic server\n");
    printf("  %s --model model.gguf --ctx-size 4096\n", "llama-server");
    printf("\n");
    printf("  # Production server with SSL and API key\n");
    printf("  %s --model model.gguf --host 0.0.0.0 --port 443 \\\n", "llama-server");
    printf("           --ssl-key-file server.key --ssl-cert-file server.crt \\\n");
    printf("           --api-key-file api_keys.txt --parallel 4\n");
    printf("\n");
    printf("  # Multimodal server with vision support\n");
    printf("  %s --model llava-model.gguf --mmproj llava-projection.gguf \\\n", "llama-server");
    printf("           --ctx-size 8192 --parallel 2\n");
    printf("\n");
    printf("API endpoints:\n");
    printf("  GET  /health                    - Server health check\n");
    printf("  GET  /v1/models                 - List available models (OpenAI compatible)\n");
    printf("  POST /v1/chat/completions       - Chat completions (OpenAI compatible)\n");
    printf("  POST /v1/completions            - Text completions (OpenAI compatible)\n");
    printf("  POST /v1/embeddings             - Text embeddings (OpenAI compatible)\n");
    printf("  GET  /metrics                   - Prometheus metrics\n");
    printf("  GET  /props                     - Server properties\n");
    printf("  POST /tokenize                  - Tokenize text\n");
    printf("  POST /detokenize                - Detokenize tokens\n");
    printf("\n");
    printf("For complete documentation, visit: https://github.com/ggml-org/llama.cpp/tree/master/examples/server\n");
    printf("\n");
}

/**
 * @brief Main application entry point.
 * 
 * Orchestrates the complete server lifecycle:
 * 1. Parse and validate command line arguments
 * 2. Initialize llama.cpp backend and system resources
 * 3. Create and configure the server application
 * 4. Load the language model and initialize contexts
 * 5. Set up HTTP server and register routes
 * 6. Start server processing loop
 * 7. Handle shutdown and cleanup
 * 
 * The main function handles all global initialization and coordinates
 * the interaction between the different application components.
 * Error handling is comprehensive to provide clear feedback on common
 * configuration and startup issues.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return int Exit code (0 for success, non-zero for error)
 */
int main(int argc, char ** argv) {
    // Initialize command line argument parsing
    common_params params;
    
    // Parse command line arguments using common infrastructure
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        show_server_help();
        return 1;
    }
    
    // Validate parsed parameters
    if (!validate_server_params(params)) {
        return 1;
    }
    
    // Initialize common llama.cpp infrastructure
    common_init();
    
    // Display configuration information
    log_server_configuration(params);
    
    try {
        // Initialize llama.cpp backend with NUMA configuration
        initialize_llama_backend(params.numa);
        
        // Create the main server application
        LlamaServerApp server_app;
        
        // Load the language model and initialize processing contexts
        LOG_INF("Loading model: %s\n", params.model.path.c_str());
        if (!server_app.load_model(params)) {
            LOG_ERR("Failed to load model: %s\n", params.model.path.c_str());
            return 1;
        }
        
        // Initialize server slots and prepare for inference
        LOG_INF("Initializing server contexts and slots...\n");
        server_app.init();
        
        // Create and configure HTTP server
        HttpServer http_server(server_app);
        
        if (!http_server.initialize(params)) {
            LOG_ERR("Failed to initialize HTTP server\n");
            return 1;
        }
        
        // Register all API routes and middleware
        LOG_INF("Registering HTTP routes...\n");
        http_server.register_routes(params);
        
        // Set up signal handling for graceful shutdown
        shutdown_handler = [&server_app, &http_server](int signal) {
            LOG_INF("Received signal %d, shutting down gracefully...\n", signal);
            
            // Stop accepting new HTTP requests
            http_server.stop();
            
            // Shutdown server application and cleanup resources
            server_app.shutdown();
            
            LOG_INF("Server shutdown complete\n");
        };
        
        // Register signal handlers for graceful shutdown
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset(&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
        auto console_ctrl_handler = [](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGTERM), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
        
        // Start the HTTP server (this blocks until shutdown)
        LOG_INF("Starting HTTP server on %s:%d\n", params.hostname.c_str(), params.port);
        
        if (!http_server.start(params.hostname, params.port)) {
            LOG_ERR("Failed to start HTTP server on %s:%d\n", params.hostname.c_str(), params.port);
            return 1;
        }
        
        // Server has been stopped, perform final cleanup
        LOG_INF("HTTP server stopped\n");
        
    } catch (const std::exception & e) {
        LOG_ERR("Server error: %s\n", e.what());
        return 1;
    } catch (...) {
        LOG_ERR("Unknown server error occurred\n");
        return 1;
    }
    
    LOG_INF("llama.cpp server terminated successfully\n");
    return 0;
}