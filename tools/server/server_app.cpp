#include "server_app.hpp"
#include "json_utils.hpp"
#include "model_utils.hpp"

// Include the original server.cpp structs and data structures that need to be moved here
// This is a transitional implementation that will contain the core server_context and related structures

/**
 * @brief Core server structures and definitions moved from server.cpp
 * 
 * These structures represent the core data types used throughout the server
 * for managing inference tasks, slots, and server state. They have been 
 * extracted from the monolithic server.cpp to enable better modularization.
 */

// Server tokens type alias for compatibility - using the actual server_tokens from utils.hpp
// using server_tokens = std::vector<llama_token>; // Removed - conflicts with utils.hpp

struct slot_params {
    bool stream        = true;
    bool cache_prompt  = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens = false;

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // mininum line indentation for the generated text in number of whitespace characters

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::vector<common_adapter_lora_info> lora;

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;
    bool timings_per_token = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // OAI-compat fields
    bool                         verbose                   = false;
    oaicompat_type               oaicompat                 = OAICOMPAT_TYPE_NONE;
    std::string                  oaicompat_model;
    std::string                  oaicompat_cmpl_id;
    common_chat_syntax           oaicompat_chat_syntax;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)
    
    // Placeholder implementations for required methods
    json to_json() const { return json{}; }
};

struct server_task {
    int id    = -1; // to be filled by server_queue
    int index = -1; // used when there are multiple prompts (batch request)

    server_task_type type;

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    slot_params   params;
    server_tokens prompt_tokens;
    int id_selected_slot = -1;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int slot_id;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::vector<common_adapter_lora_info> set_lora;

    server_task(server_task_type type) : type(type) {}
};

struct server_task_result {
    int id = -1;
    int index = -1;
    bool stop = false;
    std::string error_message;
    json data;
    
    virtual ~server_task_result() = default;
};

struct server_slot {
    int id = -1;
    int id_task = -1;
    
    slot_state state = SLOT_STATE_IDLE;
    slot_params params;
    
    server_tokens prompt_tokens;
    server_tokens generated_tokens;
    
    llama_context * ctx = nullptr;
    mtmd_context * mctx = nullptr;
    
    int n_ctx = 0;
    int n_past = 0;
    int n_predict = 0;
    
    // Timing information
    int64_t t_start_process_prompt = 0;
    int64_t t_start_generation = 0;
    int64_t t_prompt_processing = 0;
    int64_t t_token_generation = 0;
    
    // Token processing counters
    int n_prompt_tokens_processed = 0;
    int n_decoded = 0;
    
    // Placeholder method implementations
    void reset() {
        state = SLOT_STATE_IDLE;
        id_task = -1;
        prompt_tokens.clear();
        generated_tokens.clear();
        n_past = 0;
        n_prompt_tokens_processed = 0;
        n_decoded = 0;
        t_start_process_prompt = 0;
        t_start_generation = 0;
    }
    
    json to_json() const { return json{}; }
};

struct server_metrics {
    int64_t t_start = 0;
    
    uint64_t n_prompt_tokens_processed = 0;
    uint64_t n_tokens_predicted = 0;
    uint64_t n_decode_total = 0;
    uint64_t n_busy_slots_total = 0;
    int32_t n_processing_slots = 0;
    
    void init() {
        t_start = ggml_time_us();
    }
};

struct server_queue {
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<server_task> tasks;
    
    void push_back(server_task task) {
        std::unique_lock<std::mutex> lock(mutex);
        tasks.push_back(std::move(task));
        condition.notify_one();
    }
    
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex);
        // Add termination logic here
        condition.notify_all();
    }
    
    void start_loop() {
        // Placeholder for main processing loop
    }
    
    void on_update_slots(std::function<void()> callback) {
        (void)callback; // Not implemented yet
        // Placeholder for slot update callback
    }
};

struct server_response {
    // Placeholder for response handling
};

/**
 * @brief Main server context that holds all server state and resources.
 * 
 * This structure represents the core server state including the loaded model,
 * inference contexts, processing slots, and task management infrastructure.
 * It has been extracted from the original server.cpp to enable better modularization.
 */
struct server_context {
    common_params params_base;

    // Model and context management
    common_init_result llama_init;
    common_init_result llama_init_dft;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    const llama_vocab * vocab = nullptr;
    bool vocab_dft_compatible = true;

    llama_model * model_dft = nullptr;
    llama_context_params cparams_dft;

    llama_batch batch {};

    bool clean_kv_cache = true;
    bool add_bos_token  = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    server_queue    queue_tasks;
    server_response queue_results;

    server_metrics metrics;

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    common_chat_templates_ptr chat_templates;
    oaicompat_parser_options  oai_parser_opt;

    ~server_context() {
        mtmd_free(mctx);
        // Additional cleanup will be added
    }

    /**
     * @brief Load and initialize the language model.
     * 
     * Loads the primary language model and optionally a draft model for
     * speculative decoding. Also handles multimodal model loading if configured.
     * 
     * @param params Model loading parameters
     * @return true if model loaded successfully
     * @return false if model loading failed  
     */
    bool load_model(const common_params & params) {
        LOG_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;
        llama_init = common_init_from_params(params_base);

        model = llama_init.model.get();
        ctx   = llama_init.context.get();

        if (model == nullptr) {
            LOG_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);
        n_ctx = llama_n_ctx(ctx);
        add_bos_token = llama_vocab_get_add_bos(vocab);

        // Initialize chat templates
        chat_templates = common_chat_templates_init(model, params_base.chat_template);
        
        // Initialize OAI parser options
        oai_parser_opt.use_jinja = params.use_jinja;
        oai_parser_opt.allow_image = mctx != nullptr;
        oai_parser_opt.allow_audio = mctx != nullptr;
        
        return true;
    }

    /**
     * @brief Initialize inference slots and prepare for processing.
     * 
     * Creates the configured number of parallel processing slots,
     * each with its own context allocation for concurrent inference.
     */
    void init() {
        const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

        LOG_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot slot;

            slot.id = i;
            slot.ctx = ctx;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params_base.n_predict;
            slot.mctx = mctx;

            slots.push_back(std::move(slot));
        }

        metrics.init();
    }

    void update_slots() {
        // Placeholder for slot update logic
    }
};

// LlamaServerApp implementation

LlamaServerApp::LlamaServerApp() 
    : ctx_server(std::make_unique<server_context>())
    , initialized(false)
    , model_loaded(false) {
    LOG_INF("LlamaServerApp created\n");
}

LlamaServerApp::~LlamaServerApp() {
    if (initialized) {
        shutdown();
    }
    LOG_INF("LlamaServerApp destroyed\n");
}

bool LlamaServerApp::load_model(const common_params & params) {
    if (!ctx_server) {
        LOG_ERR("Server context not initialized\n");
        return false;
    }
    
    bool success = ctx_server->load_model(params);
    if (success) {
        model_loaded = true;
        LOG_INF("Model loaded successfully\n");
    } else {
        LOG_ERR("Failed to load model\n");
    }
    
    return success;
}

void LlamaServerApp::init() {
    if (!model_loaded) {
        throw std::runtime_error("Cannot initialize server: model not loaded");
    }
    
    ctx_server->init();
    initialized = true;
    
    LOG_INF("Server application initialized\n");
}

void LlamaServerApp::start() {
    if (!initialized) {
        throw std::runtime_error("Cannot start server: not initialized");
    }
    
    LOG_INF("Starting server task processing...\n");
    
    // Set up task queue processing
    ctx_server->queue_tasks.on_update_slots([this]() {
        ctx_server->update_slots();
    });
    
    // Start the main processing loop (this blocks)
    ctx_server->queue_tasks.start_loop();
    
    LOG_INF("Server task processing stopped\n");
}

void LlamaServerApp::shutdown() {
    if (!initialized) {
        return;
    }
    
    LOG_INF("Shutting down server application...\n");
    
    // Stop task processing
    ctx_server->queue_tasks.terminate();
    
    initialized = false;
    model_loaded = false;
    
    LOG_INF("Server application shutdown complete\n");
}

server_context & LlamaServerApp::get_context() {
    if (!ctx_server) {
        throw std::runtime_error("Server context not available");
    }
    return *ctx_server;
}

const server_context & LlamaServerApp::get_context() const {
    if (!ctx_server) {
        throw std::runtime_error("Server context not available");
    }
    return *ctx_server;
}