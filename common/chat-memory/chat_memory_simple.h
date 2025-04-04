// chat_memory_simple.h
#pragma once

#include "chat_memory.h"
#include <unordered_map>

// Simple ChatMemory implementation that inherits from ChatMemoryCommon
class ChatMemorySimple : public ChatMemoryCommon {
public:
    ChatMemorySimple();
    virtual ~ChatMemorySimple() override = default;

    // Override the abstract methods from ChatMemory
    std::string format_injection_prompt() const override;

private:
    // Storage
    std::unordered_map<std::string, std::string> kv;
    static constexpr size_t MEMORY_QUOTA_BYTES = 16 * 1024 * 1024;
    
    // Core memory functions
    void set(const std::string& key, const std::string& value);
    std::string get(const std::string& key) const;
    void del(const std::string& key);
    bool has(const std::string& key) const;
    std::vector<std::string> list_keys() const;
    size_t count() const;
    size_t usage_bytes() const;
    size_t quota_bytes() const;
    
    // Security and protection methods
    bool is_protected_key(const std::string& key) const;
    std::string format_memory_size(size_t bytes) const;
    std::string get_memory_fullness_assessment() const;
    bool validate_instruction_content() const;
    
    // Command implementation methods
    std::string cmd_get_quota() const;
    std::string cmd_get_usage() const;
    std::string cmd_count_keys() const;
    std::string cmd_list_keys() const;
    std::string cmd_check_key(const std::string& key) const;
    std::string cmd_get_key(const std::string& key) const;
    std::string cmd_set_key(const std::string& key, const std::string& value);
    std::string cmd_del_key(const std::string& key);
    std::string cmd_get_memory_summary() const;
    
    // Self-reinforcement and protective commands
    std::string cmd_refresh_memory_rules() const;
    std::string cmd_get_deletion_recommendation() const;
    std::string cmd_get_memory_facts() const;
    std::string cmd_verify_memory_integrity() const;
    std::string cmd_restore_memory_instructions();
    
    // Other helper methods
    std::string execute_json_command(nlohmann::ordered_json &j) override;

    // Logging functions
    void log_key_value(const std::string& key, const std::string& value) const;
    void log_memory_state(const std::string& context) const;
};

// Helper function to get default memory instructions
std::string get_default_memory_instructions();

// Register this implementation with the factory function
ChatMemory* create_simple_chat_memory();
