#include "http_routes.hpp"
#include "json_utils.hpp"

// Include generated asset headers
#include "index.html.gz.hpp"
#include "loading.html.hpp"

/**
 * @brief Implementation of HTTP route handlers and middleware
 * 
 * This module provides comprehensive HTTP request handling extracted
 * from the original server.cpp. It includes route registration, handler
 * implementations, middleware, and response formatting.
 */

// HttpServer implementation

HttpServer::HttpServer(LlamaServerApp & app) 
    : server_app(app)
    , initialized(false)
    , ssl_enabled(false) {
    LOG_INF("HttpServer created\n");
}

HttpServer::~HttpServer() {
    if (initialized) {
        stop();
    }
    LOG_INF("HttpServer destroyed\n");
}

bool HttpServer::initialize(const common_params & params) {
    LOG_INF("Initializing HTTP server...\n");
    
    // Determine SSL configuration
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    ssl_enabled = !params.ssl_file_key.empty() && !params.ssl_file_cert.empty();
    
    if (ssl_enabled) {
        LOG_INF("SSL enabled: key=%s, cert=%s\n", 
               params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        
        // Create SSL server
        auto ssl_server = std::make_unique<httplib::SSLServer>(
            params.ssl_file_cert.c_str(), 
            params.ssl_file_key.c_str()
        );
        
        if (!ssl_server->is_valid()) {
            LOG_ERR("Failed to create SSL server with provided certificate/key\n");
            return false;
        }
        
        http_server = std::move(ssl_server);
    } else {
        LOG_INF("SSL disabled\n");
        http_server = std::make_unique<httplib::Server>();
    }
#else
    if (!params.ssl_file_key.empty() || !params.ssl_file_cert.empty()) {
        LOG_ERR("SSL requested but server was built without SSL support\n");
        return false;
    }
    
    http_server = std::make_unique<httplib::Server>();
    ssl_enabled = false;
#endif

    if (!http_server) {
        LOG_ERR("Failed to create HTTP server instance\n");
        return false;
    }
    
    // Configure HTTP server settings
    setup_error_handlers();
    setup_request_logging();
    setup_cors_headers();
    
    // Set default server headers
    http_server->set_default_headers({{"Server", "llama.cpp"}});
    
    // Configure HTTP thread pool size
    int n_threads_http = params.n_threads_http;
    if (n_threads_http < 1) {
        n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    
    http_server->new_task_queue = [n_threads_http] { 
        return new httplib::ThreadPool(n_threads_http); 
    };
    
    LOG_INF("HTTP server configured with %d threads\n", n_threads_http);
    
    initialized = true;
    return true;
}

void HttpServer::register_routes(const common_params & params) {
    if (!initialized || !http_server) {
        LOG_ERR("Cannot register routes: server not initialized\n");
        return;
    }
    
    LOG_INF("Registering HTTP routes...\n");
    
    // Register middleware first
    register_middleware(params);
    
    // Register static routes (/, /loading, etc.)
    register_static_routes();
    
    // Register API routes
    register_api_routes(params);
    
    // Register OpenAI-compatible routes
    register_openai_routes(params);
    
    LOG_INF("HTTP routes registered successfully\n");
}

bool HttpServer::start(const std::string & hostname, int port) {
    if (!initialized || !http_server) {
        LOG_ERR("Cannot start server: not initialized\n");
        return false;
    }
    
    LOG_INF("Starting HTTP server on %s:%d\n", hostname.c_str(), port);
    
    // Start the server (this blocks)
    bool success = http_server->listen(hostname, port);
    
    if (!success) {
        LOG_ERR("Failed to start HTTP server on %s:%d\n", hostname.c_str(), port);
    }
    
    return success;
}

void HttpServer::stop() {
    if (http_server) {
        LOG_INF("Stopping HTTP server...\n");
        http_server->stop();
    }
}

bool HttpServer::is_running() const {
    return http_server && http_server->is_running();
}

// Private helper methods for route registration

void HttpServer::register_static_routes() {
    // Main index page
    http_server->Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, 
                       "text/html; charset=utf-8");
        res.set_header("Content-Encoding", "gzip");
    });
    
    // Loading page for when model is still loading
    http_server->Get("/loading", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len,
                       "text/html; charset=utf-8");
    });
}

void HttpServer::register_api_routes(const common_params & params) {
    std::string api_prefix = params.api_prefix;
    
    // Health check endpoint
    http_server->Get(api_prefix + "/health", 
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_health(req, res, server_app);
        });
    
    // Server metrics endpoint  
    http_server->Get(api_prefix + "/metrics",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_metrics(req, res, server_app);
        });
    
    // Server properties endpoint
    http_server->Get(api_prefix + "/props",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_props(req, res, server_app);
        });
    
    // Server properties update endpoint
    http_server->Post(api_prefix + "/props",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_props_change(req, res, server_app);
        });
    
    // Tokenization endpoints
    http_server->Post(api_prefix + "/tokenize",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_tokenize(req, res, server_app);
        });
        
    http_server->Post(api_prefix + "/detokenize", 
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_detokenize(req, res, server_app);
        });
    
    // Slots management endpoints
    http_server->Get(api_prefix + "/slots",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_slots(req, res, server_app);
        });
        
    http_server->Post(api_prefix + "/slots/:id_slot",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_slots_action(req, res, server_app);
        });
    
    // LoRA adapter endpoints
    http_server->Get(api_prefix + "/lora-adapters",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_lora_adapters_list(req, res, server_app);
        });
        
    http_server->Post(api_prefix + "/lora-adapters",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_lora_adapters_apply(req, res, server_app);
        });
}

void HttpServer::register_openai_routes(const common_params & params) {
    std::string api_prefix = params.api_prefix;
    
    // Model listing endpoints (OpenAI compatible)
    http_server->Get(api_prefix + "/models",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_models(req, res, server_app);
        });
        
    http_server->Get(api_prefix + "/v1/models", 
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_models(req, res, server_app);
        });
    
    // Completion endpoints (OpenAI compatible)
    http_server->Post(api_prefix + "/completions",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_completions(req, res, server_app);
        });
        
    http_server->Post(api_prefix + "/v1/completions",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_completions(req, res, server_app);
        });
    
    // Chat completion endpoints (OpenAI compatible)
    http_server->Post(api_prefix + "/v1/chat/completions",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_chat_completions(req, res, server_app);
        });
    
    // Embeddings endpoints (OpenAI compatible) 
    http_server->Post(api_prefix + "/embeddings",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_embeddings(req, res, server_app);
        });
        
    http_server->Post(api_prefix + "/v1/embeddings",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_embeddings(req, res, server_app);
        });
    
    // Reranking endpoint
    http_server->Post(api_prefix + "/rerank", 
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_rerank(req, res, server_app);
        });
    
    // Code infill endpoint
    http_server->Post(api_prefix + "/infill",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_infill(req, res, server_app);
        });
    
    // Chat template application endpoint  
    http_server->Post(api_prefix + "/apply-template",
        [this](const httplib::Request & req, httplib::Response & res) {
            handle_apply_template(req, res, server_app);
        });
}

void HttpServer::register_middleware(const common_params & params) {
    // Pre-routing handler for CORS and authentication
    http_server->set_pre_routing_handler([this, params](const httplib::Request & req, httplib::Response & res) {
        // Set CORS headers
        middleware_cors(req, res);
        
        // Handle OPTIONS preflight requests
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html");
            return httplib::Server::HandlerResponse::Handled;
        }
        
        // Server state validation
        std::atomic<server_state> state{SERVER_STATE_READY}; // Simplified for now
        if (!middleware_server_state(req, res, state)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        
        // API key validation
        if (!middleware_validate_api_key(req, res, params)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        
        return httplib::Server::HandlerResponse::Unhandled;
    });
}

void HttpServer::setup_cors_headers() {
    // CORS headers are handled in middleware
}

void HttpServer::setup_error_handlers() {
    http_server->set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }
        
        json error_data = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_WRN("HTTP exception: %s\n", safe_json_to_str(error_data).c_str());
        
        send_error_response(res, error_data);
    });
}

void HttpServer::setup_request_logging() {
    http_server->set_logger([](const httplib::Request & req, const httplib::Response & res) {
        // Skip logging for frequent health checks
        if (req.path == "/health" || req.path == "/v1/health") {
            return;
        }
        
        LOG_INF("%s %s - %d\n", req.method.c_str(), req.path.c_str(), res.status);
        LOG_DBG("Request body: %s\n", req.body.c_str());
        LOG_DBG("Response body: %s\n", res.body.c_str());
    });
}

// Route handler implementations (simplified placeholders)

void handle_health(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)req; // Unused parameter
    (void)app; // Unused parameter
    
    json health_data = {
        {"status", "ok"},
        {"timestamp", std::time(nullptr)}
    };
    
    send_success_response(res, health_data);
}

void handle_metrics(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)req; // Unused parameter
    
    // This is a simplified placeholder
    json metrics_data = {
        {"requests_total", 0},
        {"tokens_processed", 0},
        {"slots_available", app.get_context().slots.size()}
    };
    
    send_success_response(res, metrics_data);
}

void handle_props(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)req; // Unused parameter
    
    const auto & ctx = app.get_context();
    
    json props_data = {
        {"model_path", ctx.params_base.model.path},
        {"total_slots", ctx.params_base.n_parallel},
        {"chat_template", ""},
        {"default_generation_settings", ctx.default_generation_settings_for_props}
    };
    
    send_success_response(res, props_data);
}

void handle_props_change(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    
    // Parse JSON request body
    json request_data;
    try {
        request_data = json::parse(req.body);
    } catch (const std::exception & e) {
        json error_data = format_error_response("Invalid JSON in request body", ERROR_TYPE_INVALID_REQUEST);
        send_error_response(res, error_data);
        return;
    }
    
    // This is a placeholder - actual implementation would update server properties
    json response_data = {
        {"message", "Properties update not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_models(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)req; // Unused parameter
    
    const auto & ctx = app.get_context();
    json models_response = create_models_response(ctx.params_base.model.path);
    
    send_success_response(res, models_response);
}

void handle_completions(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    
    // Parse JSON request
    json request_data;
    try {
        request_data = json::parse(req.body);
    } catch (const std::exception & e) {
        json error_data = format_error_response("Invalid JSON in request body", ERROR_TYPE_INVALID_REQUEST);
        send_error_response(res, error_data);
        return;
    }
    
    // This is a placeholder implementation
    json response_data = {
        {"message", "Completions endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_chat_completions(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    
    // Parse JSON request
    json request_data;
    try {
        request_data = json::parse(req.body);
    } catch (const std::exception & e) {
        json error_data = format_error_response("Invalid JSON in request body", ERROR_TYPE_INVALID_REQUEST);
        send_error_response(res, error_data);
        return;
    }
    
    // This is a placeholder implementation
    json response_data = {
        {"message", "Chat completions endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_embeddings(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    
    // Parse JSON request
    json request_data;
    try {
        request_data = json::parse(req.body);
    } catch (const std::exception & e) {
        json error_data = format_error_response("Invalid JSON in request body", ERROR_TYPE_INVALID_REQUEST);
        send_error_response(res, error_data);
        return;
    }
    
    // This is a placeholder implementation
    json response_data = {
        {"message", "Embeddings endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_rerank(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "Rerank endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_infill(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "Infill endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_tokenize(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "Tokenize endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_detokenize(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "Detokenize endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_slots(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)req; // Unused parameter
    
    const auto & ctx = app.get_context();
    
    json slots_data = json::array();
    for (const auto & slot : ctx.slots) {
        json slot_obj = json::object();
        slot_obj["id"] = slot.id;
        slot_obj["state"] = (int)slot.state;
        slot_obj["task_id"] = slot.id_task;
        slots_data.push_back(slot_obj);
    }
    
    send_success_response(res, slots_data);
}

void handle_slots_action(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "Slots action endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_lora_adapters_list(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = json::array(); // Empty array for now
    send_success_response(res, response_data);
}

void handle_lora_adapters_apply(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "LoRA adapter application not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

void handle_apply_template(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app) {
    (void)app; // Unused parameter
    (void)req; // Unused parameter
    
    json response_data = {
        {"message", "Apply template endpoint not yet implemented"}
    };
    
    send_success_response(res, response_data);
}

// Middleware implementations

bool middleware_validate_api_key(const httplib::Request & req, httplib::Response & res, const common_params & params) {
    static const std::unordered_set<std::string> public_endpoints = {
        "/health", "/models", "/v1/models"
    };
    
    // If no API keys configured, allow all requests
    if (params.api_keys.empty()) {
        return true;
    }
    
    // Check if this is a public endpoint
    if (public_endpoints.count(req.path) > 0) {
        return true;
    }
    
    // Check for API key in Authorization header
    std::string auth_header = req.get_header_value("Authorization");
    if (auth_header.empty()) {
        json error_data = format_error_response("Missing Authorization header", ERROR_TYPE_AUTHENTICATION);
        send_error_response(res, error_data);
        return false;
    }
    
    // Extract Bearer token
    const std::string bearer_prefix = "Bearer ";
    if (auth_header.substr(0, bearer_prefix.length()) != bearer_prefix) {
        json error_data = format_error_response("Invalid Authorization header format", ERROR_TYPE_AUTHENTICATION);
        send_error_response(res, error_data);
        return false;
    }
    
    std::string token = auth_header.substr(bearer_prefix.length());
    
    // Validate token against configured API keys
    bool valid = std::find(params.api_keys.begin(), params.api_keys.end(), token) != params.api_keys.end();
    
    if (!valid) {
        json error_data = format_error_response("Invalid API key", ERROR_TYPE_AUTHENTICATION);
        send_error_response(res, error_data);
        return false;
    }
    
    return true;
}

bool middleware_server_state(const httplib::Request & req, httplib::Response & res, const std::atomic<server_state> & server_state_atomic) {
    server_state current_state = server_state_atomic.load();
    
    if (current_state == SERVER_STATE_LOADING_MODEL) {
        // Show loading page for HTML requests
        auto path_parts = string_split<std::string>(req.path, '.');
        if (req.path == "/" || (!path_parts.empty() && path_parts.back() == "html")) {
            res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len,
                           "text/html; charset=utf-8");
            res.status = 503;
        } else if (req.path == "/models" || req.path == "/v1/models") {
            // Allow model endpoint during loading
            return true;
        } else {
            json error_data = format_error_response("Server is loading model", ERROR_TYPE_UNAVAILABLE);
            send_error_response(res, error_data);
        }
        return false;
    }
    
    return true;
}

void middleware_cors(const httplib::Request & req, httplib::Response & res) {
    std::string origin = req.get_header_value("Origin");
    if (!origin.empty()) {
        res.set_header("Access-Control-Allow-Origin", origin);
    } else {
        res.set_header("Access-Control-Allow-Origin", "*");
    }
    
    res.set_header("Access-Control-Allow-Credentials", "true");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
}

// Helper functions

void write_streaming_response(httplib::Response & res, const json & data, bool is_final) {
    std::string event = is_final ? "data: [DONE]" : "data: " + safe_json_to_str(data);
    res.body += event + "\n\n";
    
    if (is_final) {
        res.set_header("Content-Type", "text/plain; charset=utf-8");
    } else {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
    }
}

void send_error_response(httplib::Response & res, const json & error_data) {
    json final_response = {{"error", error_data}};
    res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
    res.status = json_value(error_data, "code", 500);
}

void send_success_response(httplib::Response & res, const json & data) {
    res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
    res.status = 200;
}