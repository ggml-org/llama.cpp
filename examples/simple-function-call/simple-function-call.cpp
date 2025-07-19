#include "llama.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "json.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <array>

using json = nlohmann::json;

// Forward declaration
std::string execute_shell_command(const std::string& command);

//=============================================================================
// HELP/USAGE SECTION
//=============================================================================

static void print_usage(int argc, char ** argv) {
    (void)argc;  // Suppress unused parameter warning
    (void)argv;  // Suppress unused parameter warning
    printf("\nSimple Function Call Example - Real Shell Command Execution\n");
    printf("\n");
}

//=============================================================================
// SHELL COMMAND EXECUTION SECTION
//=============================================================================

// Real function to execute shell commands
// Uses popen() to run system commands and capture their output
// Returns the command output as a string
std::string execute_shell_command(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;

    // Use popen to execute the command
    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        return "Error: Failed to execute command";
    }

    // Read the output
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

//=============================================================================
// MAIN PROGRAM ENTRY POINT
//=============================================================================

int main(int argc, char ** argv) {
    //=========================================================================
    // CONFIGURATION VARIABLES SECTION
    //=========================================================================

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt;
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 256;
    // chat template file
    std::string chat_template_file;
    // grammar constraint
    std::string grammar;
    // confirmation flag
    bool confirm_commands = false;

    //=========================================================================
    // COMMAND LINE ARGUMENT PARSING SECTION
    //=========================================================================

    // parse command line arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            // Model file path argument
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // Prompt argument
            else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    prompt = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // Number of tokens to predict argument
            else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // GPU layers argument
            else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // Chat template file argument
            else if (strcmp(argv[i], "--chat-template-file") == 0) {
                if (i + 1 < argc) {
                    chat_template_file = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // Grammar constraint argument
            else if (strcmp(argv[i], "--grammar") == 0) {
                if (i + 1 < argc) {
                    grammar = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            // Command confirmation flag
            else if (strcmp(argv[i], "--confirm") == 0) {
                confirm_commands = true;
            }
            // Help argument
            else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                print_usage(argc, argv);
                return 0;
            }
            // Unknown argument
            else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                print_usage(argc, argv);
                return 1;
            }
        }

        // Validate required arguments
        if (model_path.empty()) {
            fprintf(stderr, "Error: Model file (-m) is required\n");
            print_usage(argc, argv);
            return 1;
        }

        if (prompt.empty()) {
            fprintf(stderr, "Error: Prompt (-p) is required\n");
            print_usage(argc, argv);
            return 1;
        }
    }

    //=========================================================================
    // CONFIGURATION DISPLAY SECTION
    //=========================================================================

    printf("Simple Function Call Example\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: %s\n", prompt.c_str());
    printf("GPU layers: %d\n", ngl);
    printf("Max tokens: %d\n", n_predict);
    if (!chat_template_file.empty()) {
        printf("Chat template: %s\n", chat_template_file.c_str());
    }
    if (!grammar.empty()) {
        printf("Grammar: %s\n", grammar.c_str());
    }
    if (confirm_commands) {
        printf("Command confirmation: enabled\n");
    }
    printf("\n");

    //=========================================================================
    // LLAMA.CPP MODEL INITIALIZATION SECTION
    //=========================================================================

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // Get vocabulary from the model
    const llama_vocab * vocab = llama_model_get_vocab(model);

    //=========================================================================
    // LLAMA.CPP CONTEXT INITIALIZATION SECTION
    //=========================================================================

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = 2048;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = 512;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    //=========================================================================
    // FUNCTION CALLING SYSTEM SETUP SECTION
    //=========================================================================

    // Initialize chat templates for function calling
    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, chat_template_file);

    // Define available functions/tools - single shell command tool
    // This defines what functions the LLM can call and their parameters
    std::vector<common_chat_tool> tools = {
        {
            "shell_command",
            "Execute a shell command and return the output",
            R"({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            })"
        }
    };

    //=========================================================================
    // CHAT MESSAGE INITIALIZATION SECTION
    //=========================================================================

    // Create chat messages
    // This sets up the initial conversation context
    std::vector<common_chat_msg> messages = {
        {
            "system",
            "You are a helpful assistant that can execute shell commands. When the user asks for something that requires a command, generate and execute the appropriate shell command. Be careful and only execute safe commands.",
            {},  // content_parts
            {},  // tool_calls
            "",  // reasoning_content
            "",  // tool_name
            ""   // tool_call_id
        },
        {
            "user",
            prompt,
            {},  // content_parts
            {},  // tool_calls
            "",  // reasoning_content
            "",  // tool_name
            ""   // tool_call_id
        }
    };

    //=========================================================================
    // CHAT TEMPLATE APPLICATION SECTION
    //=========================================================================

    // Set up chat template inputs with tools
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.tools = tools;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    // Apply chat template
    auto chat_params = common_chat_templates_apply(chat_templates.get(), inputs);

    //=========================================================================
    // PROMPT TOKENIZATION SECTION
    //=========================================================================

    // Tokenize the prompt
    const int n_prompt = -llama_tokenize(vocab, chat_params.prompt.c_str(), chat_params.prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, chat_params.prompt.c_str(), chat_params.prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    //=========================================================================
    // SAMPLER INITIALIZATION SECTION
    //=========================================================================

    // initialize the sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    //=========================================================================
    // MAIN GENERATION LOOP SECTION
    //=========================================================================

    // main loop
    int n_decode = 0;
    llama_token new_token_id;
    std::string response_text;

    // Main text generation loop - processes tokens one by one
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // Convert token to text and display it
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            response_text += s;
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n\n");

    //=========================================================================
    // FUNCTION CALL PARSING SECTION
    //=========================================================================

    // Parse the response to check for function calls
    common_chat_syntax syntax;
    syntax.format = chat_params.format;
    syntax.parse_tool_calls = true;

    common_chat_msg parsed_response = common_chat_parse(response_text, false, syntax);

    //=========================================================================
    // FUNCTION CALL EXECUTION SECTION
    //=========================================================================

    // Handle function calls if any
    if (!parsed_response.tool_calls.empty()) {
        printf("Function calls detected:\n");
        for (const auto& tool_call : parsed_response.tool_calls) {
            printf("  Function: %s\n", tool_call.name.c_str());
            printf("  Arguments: %s\n", tool_call.arguments.c_str());

            // Execute the function
            if (tool_call.name == "shell_command") {
                try {
                    // Parse JSON arguments
                    json args = json::parse(tool_call.arguments);
                    std::string command = args["command"];

                    printf("  Command: %s\n", command.c_str());

                    // Ask for confirmation if enabled
                    if (confirm_commands) {
                        printf("  Execute this command? (y/N): ");
                        std::string response;
                        std::getline(std::cin, response);
                        if (response != "y" && response != "Y") {
                            printf("  Command execution cancelled.\n");
                            continue;
                        }
                    }

                    // Execute the command
                    std::string result = execute_shell_command(command);
                    printf("  Result:\n%s", result.c_str());

                } catch (const std::exception& e) {
                    printf("  Error parsing arguments: %s\n", e.what());
                }
            }
        }
    } else if (!parsed_response.content.empty()) {
        printf("Response: %s\n", parsed_response.content.c_str());
    }

    //=========================================================================
    // CLEANUP SECTION
    //=========================================================================

    // Clean up resources
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
