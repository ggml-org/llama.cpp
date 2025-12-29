#include "command-schema.h"
#include "command-validator.h"
#include "intent-engine.h"
#include "ipc-handler.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <iostream>
#include <csignal>
#include <atomic>

using namespace semantic_server;

static std::atomic<bool> g_running(true);

static void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_running = false;
    }
}

static void print_usage(const char * argv0) {
    fprintf(stdout, "Usage: %s [options]\n", argv0);
    fprintf(stdout, "\nOptions:\n");
    fprintf(stdout, "  -h, --help            Show this help message\n");
    fprintf(stdout, "  -m, --model FILE      Model file (GGUF format)\n");
    fprintf(stdout, "  --pipe-name NAME      Named pipe name (default: frameforge_semantic)\n");
    fprintf(stdout, "  --interactive         Enable interactive mode (command line input)\n");
    fprintf(stdout, "\nExample:\n");
    fprintf(stdout, "  %s -m llama-3-8b-instruct.gguf --interactive\n", argv0);
    fprintf(stdout, "  %s -m llama-3-8b-instruct.gguf --pipe-name frameforge_semantic\n", argv0);
}

int main(int argc, char ** argv) {
    common_params params;
    std::string pipe_name = "frameforge_semantic";
    bool interactive = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--interactive") {
            interactive = true;
        } else if (arg == "--pipe-name") {
            if (i + 1 < argc) {
                pipe_name = argv[++i];
            } else {
                fprintf(stderr, "Error: --pipe-name requires an argument\n");
                return 1;
            }
        }
    }
    
    // Let common_params handle model loading args
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    
    // Validate model is specified
    if (params.model.name.empty()) {
        fprintf(stderr, "Error: Model file must be specified with -m or --model\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize llama
    common_init();
    
    // Load model
    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.name.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", params.model.name.c_str());
        return 1;
    }
    
    // Create context
    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    
    LOG_INF("Semantic AI Server for FrameForge Studio\n");
    LOG_INF("Model loaded: %s\n", params.model.name.c_str());
    LOG_INF("Context size: %d\n", llama_n_ctx(ctx));
    
    // Create intent engine
    IntentEngine intent_engine(ctx, model);
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (interactive) {
        // Interactive mode - read from stdin
        LOG_INF("Interactive mode enabled. Enter commands (Ctrl+C to exit):\n");
        
        std::string line;
        while (g_running) {
            std::cout << "\n> ";
            std::cout.flush();
            
            if (!std::getline(std::cin, line)) {
                break;
            }
            
            if (line.empty()) {
                continue;
            }
            
            // Exit commands
            if (line == "exit" || line == "quit") {
                break;
            }
            
            // Process input through intent engine
            auto result = intent_engine.process_input(line);
            
            // Display result
            json output = result.to_json();
            std::cout << "\nResult: " << output.dump(2) << std::endl;
        }
    } else {
        // IPC mode - communicate via named pipe
        LOG_INF("Starting IPC server on pipe: %s\n", pipe_name.c_str());
        
        IPCHandler ipc_handler(pipe_name);
        
        // Define callback for incoming messages
        auto message_callback = [&](const std::string & message) {
            LOG_INF("Received message: %s\n", message.c_str());
            
            // Check if message is already JSON (from whisper or other source)
            json input;
            bool is_json_input = false;
            
            try {
                input = json::parse(message);
                is_json_input = true;
                
                // Check if it contains transcribed text
                if (input.contains("text")) {
                    std::string text = input["text"].get<std::string>();
                    auto result = intent_engine.process_input(text);
                    
                    // Send validated command through pipe
                    json output = result.to_json();
                    std::string response = output.dump();
                    ipc_handler.send_message(response);
                    
                    LOG_INF("Sent response: %s\n", response.c_str());
                }
            } catch (...) {
                // Not JSON, treat as plain text
                is_json_input = false;
            }
            
            if (!is_json_input) {
                // Process as plain text command
                auto result = intent_engine.process_input(message);
                
                // Send validated command through pipe
                json output = result.to_json();
                std::string response = output.dump();
                ipc_handler.send_message(response);
                
                LOG_INF("Sent response: %s\n", response.c_str());
            }
        };
        
        if (!ipc_handler.start(message_callback)) {
            fprintf(stderr, "Error: Failed to start IPC handler\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        
        LOG_INF("IPC server running. Press Ctrl+C to stop.\n");
        
        // Main loop - wait for shutdown signal
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        LOG_INF("Shutting down...\n");
        ipc_handler.stop();
    }
    
    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    
    LOG_INF("Semantic AI Server stopped.\n");
    
    return 0;
}
