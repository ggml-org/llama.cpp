#ifndef LLAMA_KV_PAGE_H
#define LLAMA_KV_PAGE_H

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <set> // Using std::set for seq_ids for simplicity

// TODO: Potentially replace std::set with a more memory-efficient bitset or similar structure
// if the number of sequence IDs is very large.

struct llama_kv_page {
    int32_t id;          // Unique identifier for the page
    uint8_t * data;      // Pointer to the memory block for the page
    size_t size;         // Size of the page in bytes
    size_t used_tokens;  // Number of tokens currently stored in the page
    std::set<int32_t> seq_ids; // Sequence IDs that use this page

    // Default constructor
    llama_kv_page() : id(-1), data(nullptr), size(0), used_tokens(0) {}

    // Constructor
    llama_kv_page(int32_t page_id, size_t page_size)
        : id(page_id), data(nullptr), size(page_size), used_tokens(0) {
        // Memory for data should be allocated separately, e.g., by a memory manager
    }

    // Destructor
    // ~llama_kv_page() {
    //     // Data is not owned by this struct, so no deallocation here.
    //     // The memory manager that allocates 'data' should be responsible for freeing it.
    // }

    // Method to add a sequence ID
    void add_sequence(int32_t seq_id) {
        seq_ids.insert(seq_id);
    }

    // Method to remove a sequence ID
    void remove_sequence(int32_t seq_id) {
        seq_ids.erase(seq_id);
    }

    // Method to check if the page is used by any sequence
    bool is_used() const {
        return !seq_ids.empty();
    }

    // Method to check if the page has space for more tokens
    // This is a simplified check; actual logic might depend on token size
    bool has_space(size_t token_size_bytes = 1) const { // Assuming 1 byte per token for simplicity
        return (used_tokens * token_size_bytes) < size;
    }
};

#endif // LLAMA_KV_PAGE_H
