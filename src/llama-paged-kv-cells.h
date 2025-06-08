#ifndef LLAMA_PAGED_KV_CELLS_H
#define LLAMA_PAGED_KV_CELLS_H

#include "llama-kv-page.h" // Include the definition of llama_kv_page (from subtask 1)
#include <vector>
#include <list>
#include <map>
#include <cstdint>   // For int32_t, uint8_t
#include <cstddef>   // For size_t
#include <stdexcept> // For exceptions

// Defines a mapping key for (sequence ID, token position)
struct TokenKey {
    int32_t seq_id;
    int32_t token_pos;

    TokenKey(int32_t s_id, int32_t t_pos) : seq_id(s_id), token_pos(t_pos) {}

    bool operator<(const TokenKey& other) const {
        if (seq_id != other.seq_id) {
            return seq_id < other.seq_id;
        }
        return token_pos < other.token_pos;
    }
};

// Defines the value for the token_to_page_offset map
struct PageOffset {
    int32_t page_id; // The ID of the llama_kv_page
    size_t offset;   // Byte offset within the page's data buffer where this token's KV data starts
                     // Or could be element offset if all tokens have same K+V size per layer.
                     // Let's assume byte offset for flexibility with K/V structures.

    PageOffset(int32_t p_id, size_t off) : page_id(p_id), offset(off) {}
    PageOffset() : page_id(-1), offset(0) {} // Default constructor
};

class llama_paged_kv_cells {
public:
    // Constructor
    // page_size_bytes: The fixed size of each llama_kv_page's data buffer.
    // page_memory_pool: A large, pre-allocated (ideally by GGML paged allocator) memory region.
    // page_memory_pool_size_bytes: Total size of the page_memory_pool.
    // initial_pages_to_fill_from_pool: Number of llama_kv_page objects to create and assign memory to initially.
    llama_paged_kv_cells(
        size_t page_size_bytes, // Size of each individual page
        uint8_t* page_memory_pool,
        size_t page_memory_pool_size_bytes,
        size_t initial_pages_to_fill_from_pool);

    // Destructor
    ~llama_paged_kv_cells();

    // Allocates a llama_kv_page object and assigns it memory from the pool.
    // Returns the ID of the allocated page, or -1 on failure (e.g., pool exhausted).
    int32_t allocate_page();

    // Marks a llama_kv_page (by its ID) as free. Its memory can be reused.
    void free_page(int32_t page_id);

    // Finds an existing page or allocates a new one for a given token's KV cache.
    // This is a placeholder for more complex logic that considers token data size and page capacity.
    // For now, assumes one "token" fits in a page and uses some part of page->used_tokens.
    PageOffset find_or_allocate_page_for_token(int32_t seq_id, int32_t token_pos, size_t token_kv_size);

    // Returns the page ID and offset for a given token.
    PageOffset get_page_and_offset(int32_t seq_id, int32_t token_pos) const;

    // Returns a pointer to the llama_kv_page struct by its ID.
    llama_kv_page* get_page(int32_t page_id);
    const llama_kv_page* get_page(int32_t page_id) const;

    // Get the configured page size in bytes
    size_t get_page_size_bytes() const { return page_size_bytes_; }

private:
    std::vector<llama_kv_page> pages_;       // Stores all page metadata objects. Indexed by page_id.
                                             // page_id is the index in this vector.
    std::list<int32_t> free_page_indices_;   // List of indices (page_ids) in `pages_` that are free.

    // Maps (sequence ID, token position) to (page ID, offset within page data)
    std::map<TokenKey, PageOffset> token_to_page_offset_;

    size_t page_size_bytes_;                 // Size of each page's data buffer in bytes.
    int32_t next_page_id_counter_;           // Counter for generating unique page IDs if needed,
                                             // but if page_id is just vector index, this is more like current count.

    uint8_t* page_memory_pool_;              // The large memory pool provided externally.
    size_t page_memory_pool_size_bytes_;     // Total size of the pool.
    size_t page_memory_pool_used_bytes_;     // How much of the pool has been carved out for pages.

    // Helper to get a new block of memory from the page_memory_pool_ for a new page.
    // Returns true on success, false if pool is exhausted.
    // This will also update the page.data pointer.
    bool assign_memory_to_new_page(llama_kv_page &page);
};

#endif // LLAMA_PAGED_KV_CELLS_H
