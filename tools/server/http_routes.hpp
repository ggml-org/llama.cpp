#pragma once

#include "server_app.hpp"
#include "json_utils.hpp"

// Include httplib properly through utils.hpp which has the correct path
#include "utils.hpp"

/**
 * @brief HTTP route handlers and middleware for the llama.cpp server
 * 
 * This module provides comprehensive HTTP request handling including:
 * - Route registration and handler implementation
 * - OpenAI-compatible API endpoints
 * - Custom llama.cpp specific endpoints
 * - Request validation and authentication
 * - Response formatting and error handling
 * - Server-sent events for streaming responses
 * - CORS and security middleware
 * 
 * All routes maintain compatibility with OpenAI API standards while providing
 * additional functionality specific to llama.cpp capabilities.
 */

// Forward declarations for server types
class LlamaServerApp;

/**
 * @brief HTTP server manager that handles all HTTP-related functionality.
 * 
 * This class manages the HTTP server lifecycle including route registration,
 * middleware setup, SSL configuration, and request routing. It provides a
 * clean interface between the HTTP layer and the core server application.
 */
class HttpServer {
public:
    /**
     * @brief Construct a new HttpServer instance.
     * 
     * @param app Reference to the main server application
     */
    explicit HttpServer(LlamaServerApp & app);

    /**
     * @brief Destroy the HttpServer instance and cleanup resources.
     */
    ~HttpServer();

    /**
     * @brief Initialize the HTTP server with the given parameters.
     * 
     * Sets up the underlying HTTP server (with or without SSL) and configures
     * basic server settings like headers and error handlers.
     * 
     * @param params Server parameters including SSL settings
     * @return true if server initialization succeeded
     * @return false if initialization failed
     */
    bool initialize(const common_params & params);

    /**
     * @brief Register all HTTP routes and middleware.
     * 
     * Configures all API endpoints including OpenAI-compatible routes,
     * llama.cpp specific endpoints, and necessary middleware for
     * authentication, CORS, and request validation.
     * 
     * @param params Server parameters for route configuration
     */
    void register_routes(const common_params & params);

    /**
     * @brief Start the HTTP server and begin accepting requests.
     * 
     * Begins listening for HTTP requests on the configured host and port.
     * This method blocks until the server is stopped.
     * 
     * @param hostname Host address to bind to
     * @param port Port number to listen on
     * @return true if server started successfully
     * @return false if server startup failed
     */
    bool start(const std::string & hostname, int port);

    /**
     * @brief Gracefully stop the HTTP server.
     * 
     * Stops accepting new requests and gracefully shuts down the server,
     * allowing existing requests to complete.
     */
    void stop();

    /**
     * @brief Check if the server is currently running.
     * 
     * @return true if the server is active and accepting requests
     * @return false if the server is stopped or not initialized
     */
    bool is_running() const;

private:
    LlamaServerApp & server_app;                           ///< Reference to main server application
    std::unique_ptr<httplib::Server> http_server;          ///< HTTP server instance
    bool initialized;                                       ///< Initialization state flag
    bool ssl_enabled;                                      ///< SSL configuration flag

    // Private helper methods for route registration
    void register_static_routes();
    void register_api_routes(const common_params & params);
    void register_openai_routes(const common_params & params);
    void register_middleware(const common_params & params);
    void setup_cors_headers();
    void setup_error_handlers();
    void setup_request_logging();
};

/**
 * @brief Core API endpoint handlers
 * 
 * These functions implement the actual logic for each HTTP endpoint.
 * They are designed to be called by the HTTP server route handlers
 * and provide clean separation between HTTP handling and business logic.
 */

/**
 * @brief Handle health check endpoint.
 * 
 * Provides server health and status information for monitoring systems.
 * Always returns 200 OK when the server is operational.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_health(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle server metrics endpoint.
 * 
 * Returns comprehensive server performance metrics including token throughput,
 * request statistics, and resource utilization.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_metrics(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle server properties endpoint.
 * 
 * Returns server configuration and model information including supported
 * features, model metadata, and generation settings.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_props(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle server properties update endpoint.
 * 
 * Allows dynamic modification of certain server properties and generation
 * settings without requiring a server restart.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_props_change(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle model listing endpoint (OpenAI compatible).
 * 
 * Returns information about available models in OpenAI API compatible format.
 * Supports both /models and /v1/models endpoints.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_models(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle text completion endpoint.
 * 
 * Processes text completion requests with support for streaming and
 * non-streaming responses. Compatible with OpenAI completion API.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_completions(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle chat completion endpoint (OpenAI compatible).
 * 
 * Processes chat completion requests with full OpenAI ChatGPT API compatibility
 * including message formatting, function calling, and streaming support.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_chat_completions(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle text embeddings endpoint (OpenAI compatible).
 * 
 * Generates text embeddings using the loaded model with support for
 * batch processing and various normalization options.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_embeddings(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle text reranking endpoint.
 * 
 * Performs document reranking using the model's scoring capabilities
 * for information retrieval and search applications.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_rerank(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle code infill endpoint.
 * 
 * Specialized endpoint for code completion and infilling tasks,
 * particularly useful for IDE integrations and coding assistants.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_infill(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle tokenization endpoint.
 * 
 * Converts text input to tokens using the model's tokenizer,
 * useful for debugging and token counting applications.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_tokenize(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle detokenization endpoint.
 * 
 * Converts token arrays back to text using the model's tokenizer,
 * complementing the tokenization endpoint.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_detokenize(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle slots status endpoint.
 * 
 * Returns information about current inference slots including
 * their status, assigned tasks, and performance metrics.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_slots(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle slot actions endpoint (save/restore/erase).
 * 
 * Manages slot state persistence including saving slot state to disk,
 * restoring from saved state, and erasing slot data.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_slots_action(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle LoRA adapter listing endpoint.
 * 
 * Returns information about currently loaded LoRA adapters
 * and their configurations.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_lora_adapters_list(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle LoRA adapter application endpoint.
 * 
 * Applies or removes LoRA adapters dynamically without requiring
 * server restart or model reloading.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_lora_adapters_apply(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Handle chat template application endpoint.
 * 
 * Applies chat templates to format messages according to the model's
 * expected conversation format.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param app Server application reference
 */
void handle_apply_template(const httplib::Request & req, httplib::Response & res, LlamaServerApp & app);

/**
 * @brief Middleware for API key validation.
 * 
 * Validates API keys for protected endpoints while allowing public
 * endpoints to pass through without authentication.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param params Server parameters including API key configuration
 * @return true if request should proceed
 * @return false if request should be rejected
 */
bool middleware_validate_api_key(const httplib::Request & req, httplib::Response & res, 
                                const common_params & params);

/**
 * @brief Middleware for server state validation.
 * 
 * Ensures the server is in an appropriate state to handle the request,
 * returning loading pages or error responses when necessary.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 * @param server_state Current server state
 * @return true if request should proceed
 * @return false if request should be handled by middleware
 */
bool middleware_server_state(const httplib::Request & req, httplib::Response & res,
                            const std::atomic<server_state> & server_state);

/**
 * @brief Middleware for CORS header configuration.
 * 
 * Sets appropriate CORS headers for cross-origin requests,
 * handling preflight OPTIONS requests properly.
 * 
 * @param req HTTP request object
 * @param res HTTP response object
 */
void middleware_cors(const httplib::Request & req, httplib::Response & res);

/**
 * @brief Helper function for streaming response writing.
 * 
 * Utilities for writing server-sent events and streaming responses
 * for long-running completion requests.
 * 
 * @param res HTTP response object
 * @param data JSON data to stream
 * @param is_final Whether this is the final chunk
 */
void write_streaming_response(httplib::Response & res, const json & data, bool is_final = false);

/**
 * @brief Helper function for error response generation.
 * 
 * Generates standardized error responses with proper HTTP status codes
 * and JSON error formatting.
 * 
 * @param res HTTP response object
 * @param error_data Error information
 */
void send_error_response(httplib::Response & res, const json & error_data);

/**
 * @brief Helper function for success response generation.
 * 
 * Generates standardized success responses with proper JSON formatting
 * and HTTP status codes.
 * 
 * @param res HTTP response object
 * @param data Response data
 */
void send_success_response(httplib::Response & res, const json & data);