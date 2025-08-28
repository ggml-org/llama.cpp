#pragma once

#include "chat.h"
#include "utils.hpp"

#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cinttypes>
#include <deque>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::ordered_json;

// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

constexpr int HTTP_POLLING_SECONDS = 1;

// Enums and constants that are shared across modules
enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED, // TODO: this state is only used for setting up the initial prompt processing; maybe merge it with launch_slot_with_task in the future
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

enum oaicompat_type {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
    OAICOMPAT_TYPE_EMBEDDING,
};

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

// Core utility functions for JSON handling
std::string safe_json_to_str(const json & j);
json format_error_response(const std::string & message, const enum error_type type);

// Forward declarations for complex types defined in server_app.cpp
struct slot_params;
struct server_task;
struct server_task_result;
struct server_slot;
struct server_metrics;
struct server_queue;
struct server_response;
struct server_context;

// Task type utility functions
static bool server_task_type_need_embd(server_task_type task_type) {
    switch (task_type) {
        case SERVER_TASK_TYPE_EMBEDDING:
        case SERVER_TASK_TYPE_RERANK:
            return true;
        default:
            return false;
    }
}

static bool server_task_type_need_logits(server_task_type task_type) {
    switch (task_type) {
        case SERVER_TASK_TYPE_COMPLETION:
        case SERVER_TASK_TYPE_INFILL:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Main server application class that manages the llama.cpp server lifecycle.
 * 
 * This class encapsulates the server's core functionality including:
 * - Model loading and management
 * - Context and slot management
 * - Task queue processing
 * - Server state management
 * 
 * The class follows a clear lifecycle pattern:
 * 1. Construction - Basic initialization
 * 2. load_model() - Load the language model and initialize contexts
 * 3. init() - Initialize slots and prepare for inference
 * 4. start() - Begin processing tasks and serving requests
 * 5. shutdown() - Clean shutdown and resource cleanup
 */
class LlamaServerApp {
public:
    /**
     * @brief Construct a new LlamaServerApp instance.
     * 
     * Initializes the application in a clean state, ready for model loading.
     * The constructor does not perform any heavy operations or resource allocation.
     */
    LlamaServerApp();

    /**
     * @brief Destroy the LlamaServerApp instance.
     * 
     * Ensures proper cleanup of all resources including model contexts,
     * multimodal contexts, and any active tasks.
     */
    ~LlamaServerApp();

    /**
     * @brief Load and initialize the language model.
     * 
     * This method loads the primary language model and optionally a draft model
     * for speculative decoding. It also initializes multimodal capabilities
     * if a multimodal projection model is provided.
     * 
     * @param params Common parameters including model paths and configuration
     * @return true if the model was loaded successfully
     * @return false if model loading failed
     */
    bool load_model(const common_params & params);

    /**
     * @brief Initialize server slots and prepare for inference.
     * 
     * Creates and configures inference slots based on the parallel processing
     * parameters. Each slot manages its own context and state for concurrent
     * request processing.
     */
    void init();

    /**
     * @brief Start the server task processing loop.
     * 
     * Begins the main server loop that processes incoming tasks from the queue.
     * This method blocks until the server is shut down.
     */
    void start();

    /**
     * @brief Gracefully shutdown the server.
     * 
     * Stops task processing, cleans up resources, and prepares the application
     * for termination. This method should be called before destroying the instance.
     */
    void shutdown();

    /**
     * @brief Get the current server context.
     * 
     * @return server_context& Reference to the internal server context
     */
    server_context & get_context();

    /**
     * @brief Get the current server context (const version).
     * 
     * @return const server_context& Const reference to the internal server context
     */
    const server_context & get_context() const;

private:
    std::unique_ptr<server_context> ctx_server;  ///< Main server context
    bool initialized;                             ///< Initialization state flag
    bool model_loaded;                           ///< Model loading state flag
};