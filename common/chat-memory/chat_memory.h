// chat_memory.h
#pragma once

#include "json.hpp"

#include <deque>
#include <functional>
#include <string>

// Debug logging control
#ifndef CHAT_MEMORY_DEBUG
#define CHAT_MEMORY_DEBUG 1  // Set to 1 to enable debug logging at compile time
#endif

// Forward declarations
using json = nlohmann::ordered_json;
using WriteCallback = std::function<void(const char*, size_t)>;

// Abstract base class defining the interface
class ChatMemory {
public:
    virtual ~ChatMemory() = default;

    // Core interface required by server.cpp
    virtual void process_response(json& response, bool is_final, const WriteCallback& write_callback) = 0;
    virtual std::string parse_and_execute_command(const std::string& output) = 0;
    virtual void parse_and_execute_command_json(json& j) = 0;
    virtual std::string format_injection_prompt() const = 0;

protected:
    // Helper methods that derived classes should implement
    virtual bool is_streaming_response(const json& j) const = 0;
    virtual void process_streaming_chunk(json& j) = 0;
    virtual void process_regular_response(json& j) = 0;
    virtual void reset_streaming() = 0;
};

// A common implementation base class that can be reused
class ChatMemoryCommon : public ChatMemory {
public:
    // Implement the interface from ChatMemory
    void process_response(json& response, bool is_final, const WriteCallback& write_callback) override;
    std::string parse_and_execute_command(const std::string& output) override;
    void parse_and_execute_command_json(json& j) override;
    std::string format_injection_prompt() const override = 0;

protected:
    bool in_streaming_mode = false;

    // Context Management
    std::deque<std::string> recent_responses;
    size_t max_context_responses = 5;  // Default limit

    // Default logging methods
    static bool is_debug_enabled();
    void log_debug(const std::string& message) const;
    void log_command(const std::string& command, const nlohmann::ordered_json& response) const;

    // Default implementations of helper methods
    virtual std::string execute_json_command(nlohmann::ordered_json &j);
    bool is_streaming_response(const json& j) const override;
    bool is_valid_memory_json(const std::string& output) const;
    void process_streaming_chunk(json& j) override;
    void process_regular_response(json& j) override;
    void reset_streaming() override;
    void track_response(const std::string& response);

    // Shared state
    std::string accumulated_content;
};

// Global function to get a ChatMemory object by ID
ChatMemory& get_or_create_chat_memory(const std::string& id);
