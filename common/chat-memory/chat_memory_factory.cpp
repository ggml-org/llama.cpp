// chat_memory_factory.cpp
#include "chat_memory.h"
#include "chat_memory_simple.h"
#include <unordered_map>
#include <memory>

// Create a static map to store ChatMemory instances
namespace {
    std::unordered_map<std::string, std::unique_ptr<ChatMemory>> memory_instances;
}

// Implementation of the factory function
ChatMemory& get_or_create_chat_memory(const std::string& id) {
    auto it = memory_instances.find(id);
    if (it == memory_instances.end()) {
        // Create a new instance of ChatMemorySimple
        memory_instances[id] = std::unique_ptr<ChatMemory>(new ChatMemorySimple());
        return *memory_instances[id];
    }
    return *it->second;
}
