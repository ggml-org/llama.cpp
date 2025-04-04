// chat_memory.cpp
#include "chat_memory.h"
#include <iostream>
#include <regex>
#include <ctime>

void ChatMemoryCommon::process_response(json& response, bool is_final, const WriteCallback& write_callback) {
    // For streaming responses
    if (is_streaming_response(response)) {
        // Process the chunk normally
        process_streaming_chunk(response);

        // On final chunk, check if we need to execute memory commands
        if (is_final) {
            // Extract memory commands from the accumulated content
            std::regex json_pattern(R"(\{[^{}]*"memory_command"[^{}]*\})");
            std::smatch match;

            if (std::regex_search(accumulated_content, match, json_pattern)) {
                std::string json_str = match.str();

                // Execute the memory command
                std::string memory_response = parse_and_execute_command(json_str);

                if (!memory_response.empty()) {
                    // Create a JSON response with the memory results
                    nlohmann::ordered_json memory_chunk = {
                        {"id", "memory_response"},
                        {"object", "chat.completion.chunk"},
                        {"created", (int)time(NULL)},
                        {"model", "memory_system"},
                        {"choices", {{
                            {"index", 0},
                            {"delta", {{"content", "\n\n" + memory_response}}},
                            {"finish_reason", nullptr}
                        }}}
                    };

                    // Format and send the response
                    std::string chunk_str = "data: " + memory_chunk.dump() + "\n\n";
                    write_callback(chunk_str.c_str(), chunk_str.size());
                }

                // Signal the end of the stream
                const std::string done_msg = "data: [DONE]\n\n";
                write_callback(done_msg.c_str(), done_msg.size());
            } else {
                // No memory command detected, just end the stream normally
                const std::string done_msg = "data: [DONE]\n\n";
                write_callback(done_msg.c_str(), done_msg.size());
            }

            // Reset streaming state
            reset_streaming();
        }
    } else {
        // For non-streaming responses, process directly
        process_regular_response(response);
    }
}

// Logging functions implementations
bool ChatMemoryCommon::is_debug_enabled() {
    static bool checked = false;
    static bool enabled = false;

    if (!checked) {
        checked = true;
        // Check environment variable first
        const char* debug_env = std::getenv("LLAMA_MEMORY_DEBUG");
        if (debug_env && (std::string(debug_env) == "1" || std::string(debug_env) == "true")) {
            enabled = true;
        } else {
            // Check compile-time flag
            enabled = CHAT_MEMORY_DEBUG != 0;
        }
    }
    return enabled;
}

void ChatMemoryCommon::log_debug(const std::string& message) const {
    if (!is_debug_enabled()) return;
 
    // Get current time for timestamp
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");

    std::cerr << "[" << timestamp.str() << "] [ChatMemory Debug] " << message << std::endl;
}

void ChatMemoryCommon::log_command(const std::string& command, const nlohmann::ordered_json& response) const {
    if (!ChatMemoryCommon::is_debug_enabled()) return;

    ChatMemoryCommon::log_debug("Command executed: " + command);
    ChatMemoryCommon::log_debug("Response: " + response.dump(2));
}

bool ChatMemoryCommon::is_streaming_response(const json& j) const {
    // Check if it's a direct object with the right type
    if (j.contains("object") && j["object"].get<std::string>() == "chat.completion.chunk") {
        return true;
    }
    
    // Check if it's an array containing objects with the right type
    if (j.is_array() && !j.empty() && j[0].contains("object") && 
        j[0]["object"].get<std::string>() == "chat.completion.chunk") {
        return true;
    }
    
    return false;
}

void ChatMemoryCommon::track_response(const std::string& response) {
    ChatMemoryCommon::log_debug("track_response: Adding response with size " + std::to_string(response.size()) + " bytes");
    
    recent_responses.push_back(response);
    if (recent_responses.size() > max_context_responses) {
        ChatMemoryCommon::log_debug("track_response: Removing oldest response (exceeded max_context_responses)");
        recent_responses.pop_front();
    }
}

// Check if a valid memory command JSON is being used
bool ChatMemoryCommon::is_valid_memory_json(const std::string& output) const {
    ChatMemoryCommon::log_debug("is_valid_memory_json: Checking if \"" + output.substr(0, std::min(output.size(), size_t(50))) + 
              (output.size() > 50 ? "..." : "") + "\" contains valid memory command JSON");
    
    // Look for valid memory_command JSON pattern
    std::regex memory_cmd_pattern(R"(\{"memory_command":[^}]+\})");
    bool valid = std::regex_search(output, memory_cmd_pattern);
    
    ChatMemoryCommon::log_debug("is_valid_memory_json: Result = " + std::string(valid ? "valid" : "invalid") + " memory command JSON");
    return valid;
}   

// Main entry point for processing model output and executing commands
std::string ChatMemoryCommon::parse_and_execute_command(const std::string& output) {
    log_debug("parse_and_execute_command: Processing output for memory commands");

    if (output.find("memory_command") == std::string::npos || output.find('{') == std::string::npos) {
        log_debug("parse_and_execute_command: No memory commands found");
        return "";  // No memory commands found
    }

    // Check if this appears to be a valid JSON command structure
    if (!is_valid_memory_json(output)) {
        log_debug("parse_and_execute_command: Warning - Detected memory-related text without proper JSON format");
        std::cerr << "[ChatMemory] Warning: Detected memory-related text without proper JSON format.\n";
        // Continue anyway as regex might not catch all valid formats
    }
    
    std::regex json_block(R"(\{[^{}]*(\{[^{}]*\}[^{}]*)*\})");
    auto begin = std::sregex_iterator(output.begin(), output.end(), json_block);
    auto end = std::sregex_iterator();

    if (begin == end) {
        log_debug("parse_and_execute_command: No JSON blocks found");
        std::cerr << "[ChatMemory] No JSON blocks found in output.\n";
        return "";
    }   
    
    for (auto it = begin; it != end; ++it) {
        const std::string json_text = it->str();
        if (json_text.find("memory_command") == std::string::npos) {
            continue;
        }
    
        ChatMemoryCommon::log_debug("parse_and_execute_command: Found potential memory command JSON: " + 
                 json_text.substr(0, std::min(json_text.size(), size_t(100))) + 
                 (json_text.size() > 100 ? "..." : ""));
        
        try {
            json j = json::parse(json_text);
            
            // Execute the command and get the human-readable response
            std::string human_response = execute_json_command(j);
            if (!human_response.empty()) {
                // Track the response for context management
                track_response(human_response);
                
                log_debug("parse_and_execute_command: Successfully executed command, returning response");
                return human_response;
            }
        } catch (const std::exception& e) {
            log_debug("parse_and_execute_command: JSON parse error: " + std::string(e.what()));
            std::cerr << "[ChatMemory] JSON parse error: " << e.what() << "\n";
            std::cerr << "[ChatMemory] Offending input: " << json_text << "\n";
        }
    }
    
    log_debug("parse_and_execute_command: No valid memory commands found");
    return "";  // No valid commands found
}   

void ChatMemoryCommon::parse_and_execute_command_json(json& j) { 
    log_debug("parse_and_execute_command_json: Processing JSON response");
    
    std::string model_output;
    
    // Handle different response formats
    if (j.contains("content")) {
        // Chat completions format
        model_output = j["content"].get<std::string>();
        log_debug("parse_and_execute_command_json: Found content field");
    } else if (j.contains("text")) {
        // Regular completions format
        model_output = j["text"].get<std::string>();
        log_debug("parse_and_execute_command_json: Found text field");
    } else {
        // No recognizable output format
        log_debug("parse_and_execute_command_json: No recognizable output format");
        return;
    }
    
    // Process and append any memory responses
    std::string memory_response = parse_and_execute_command(model_output);
    if (!memory_response.empty()) {
        log_debug("parse_and_execute_command_json: Found memory response, appending to output");
        
        // Update the appropriate field
        if (j.contains("content")) {
            j["content"] = model_output + "\n" + memory_response;
        } else if (j.contains("text")) {
            j["text"] = model_output + "\n" + memory_response;
        }
    } else {
        log_debug("parse_and_execute_command_json: No memory response to append");
    }
}

void ChatMemoryCommon::process_streaming_chunk(json& j) {
    try {
        // First check if it's a direct object with choices
        if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
            const auto& first_choice = j["choices"][0];
            if (first_choice.contains("delta") && first_choice["delta"].contains("content")) {
                std::string content = first_choice["delta"]["content"].get<std::string>();
                accumulated_content += content;
                log_debug("Chunk appended: '" + content + "'");
                return;
            }
        }
        // Then check the array case
        else if (j.is_array() && !j.empty()) {
            const auto& choices = j[0]["choices"];
            if (choices.is_array() && !choices.empty()) {
                const auto& delta = choices[0]["delta"];
                if (delta.contains("content")) {
                    std::string content = delta["content"].get<std::string>();
                    accumulated_content += content;
                    log_debug("Chunk appended: '" + content + "'");
                    return;
                }
            }
        }
        
        log_debug("Chunk missing 'content' field: " + j.dump());
    } catch (const std::exception &e) {
        log_debug(std::string("Exception parsing chunk: ") + e.what());
    }
}

void ChatMemoryCommon::process_regular_response(json& j) {
    log_debug("process_regular_response: Processing standard response format");

    std::string model_output;
    bool found_content = false;

    // Handle different response formats
    if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
        auto& first_choice = j["choices"][0];

        if (first_choice.contains("message") && first_choice["message"].contains("content")) {
            model_output = first_choice["message"]["content"].get<std::string>();
            found_content = true;
            log_debug("process_regular_response: Found content in OpenAI format: \"" +
                      model_output.substr(0, std::min(model_output.size(), size_t(100))) +
                      (model_output.size() > 100 ? "..." : "") + "\"");
        } else {
            log_debug("process_regular_response: No content found in OpenAI format");
        }
    } else if (j.contains("content")) {
        model_output = j["content"].get<std::string>();
        found_content = true;
        log_debug("process_regular_response: Found content field: \"" +
                  model_output.substr(0, std::min(model_output.size(), size_t(100))) +
                  (model_output.size() > 100 ? "..." : "") + "\"");
    } else if (j.contains("text")) {
        model_output = j["text"].get<std::string>();
        found_content = true;
        log_debug("process_regular_response: Found text field: \"" +
                  model_output.substr(0, std::min(model_output.size(), size_t(100))) +
                  (model_output.size() > 100 ? "..." : "") + "\"");
    } else {
        log_debug("process_regular_response: No recognizable output format. JSON structure: " +
                  j.dump().substr(0, std::min(j.dump().size(), size_t(500))) +
                  (j.dump().size() > 500 ? "..." : ""));
        return;
    }

    if (!found_content || model_output.empty()) {
        log_debug("process_regular_response: No model output found to process");
        return;
    }

    // Process and append any memory responses
    std::string memory_response = parse_and_execute_command(model_output);
    if (!memory_response.empty()) {
        log_debug("process_regular_response: Found memory response, appending to output");

        // Update the appropriate field
        if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
            auto& first_choice = j["choices"][0];
            if (first_choice.contains("message") && first_choice["message"].contains("content")) {
                first_choice["message"]["content"] = model_output + "\n" + memory_response;
                log_debug("process_regular_response: Updated content in OpenAI format");
            } else {
                log_debug("process_regular_response: Couldn't update content in OpenAI format");
            }
        } else if (j.contains("content")) {
            j["content"] = model_output + "\n" + memory_response;
            log_debug("process_regular_response: Updated content field");
        } else if (j.contains("text")) {
            j["text"] = model_output + "\n" + memory_response;
            log_debug("process_regular_response: Updated text field");
        } else {
            log_debug("process_regular_response: Couldn't find field to update with memory response");
        }
    } else {
        log_debug("process_regular_response: No memory response to append");
    }
}

void ChatMemoryCommon::reset_streaming() {
    log_debug("reset_streaming: Resetting streaming state");
    in_streaming_mode = false;
    accumulated_content.clear();
}

std::string ChatMemoryCommon::execute_json_command(nlohmann::ordered_json &j) {
    return "ChatMemoryCommon";
}
