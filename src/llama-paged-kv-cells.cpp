#include "llama-paged-kv-cells.h"
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <cstdlib>   // For malloc, free (should not be used for page data)
#include <algorithm> // For std::find
#include <iostream>  // For debugging (optional)

// --- llama_paged_kv_cells ---

llama_paged_kv_cells::llama_paged_kv_cells(
    size_t page_size_bytes,
    uint8_t* page_memory_pool,
    size_t page_memory_pool_size_bytes,
    size_t initial_pages_to_fill_from_pool)
    : page_size_bytes_(page_size_bytes),
      next_page_id_counter_(0), // Page IDs will be indices into the pages_ vector
      page_memory_pool_(page_memory_pool),
      page_memory_pool_size_bytes_(page_memory_pool_size_bytes),
      page_memory_pool_used_bytes_(0) {

    if (page_size_bytes_ == 0) {
        throw std::invalid_argument("Page size cannot be zero.");
    }
    if (page_memory_pool_ == nullptr && page_memory_pool_size_bytes_ > 0) {
        // If a size is given, a pool must be provided. Or, allow null pool if size is also 0 (dynamic growth not yet supported here)
        throw std::invalid_argument("Page memory pool cannot be null if pool size is greater than zero.");
    }

    // Pre-assign memory to initial pages from the pool
    for (size_t i = 0; i < initial_pages_to_fill_from_pool; ++i) {
        // Create a new page metadata object
        pages_.emplace_back(next_page_id_counter_, page_size_bytes_); // id is its index
        llama_kv_page& new_page_meta = pages_.back();

        if (assign_memory_to_new_page(new_page_meta)) {
            free_page_indices_.push_back(new_page_meta.id); // Add to free list
            next_page_id_counter_++;
        } else {
            // Ran out of pool memory during initial allocation
            pages_.pop_back(); // Remove the metadata object that couldn't get memory
            std::cerr << "Warning: Ran out of page memory pool during initial page allocation. Allocated " << i << " pages." << std::endl;
            break;
        }
    }
}

llama_paged_kv_cells::~llama_paged_kv_cells() {
    // The page_memory_pool_ is owned by an external entity (e.g., llama_paged_kv_cache, via GGML).
    // This class does not free the page_memory_pool_ itself.
    // Individual page.data pointers are offsets into this pool, so no individual free calls needed.
    pages_.clear();
    free_page_indices_.clear();
    token_to_page_offset_.clear();
}

bool llama_paged_kv_cells::assign_memory_to_new_page(llama_kv_page &page) {
    if (page_memory_pool_used_bytes_ + page_size_bytes_ > page_memory_pool_size_bytes_) {
        return false; // Not enough space in the pool
    }
    page.data = page_memory_pool_ + page_memory_pool_used_bytes_;
    page_memory_pool_used_bytes_ += page_size_bytes_;
    page.size = page_size_bytes_; // Ensure page knows its actual data region size
    return true;
}

int32_t llama_paged_kv_cells::allocate_page() {
    if (!free_page_indices_.empty()) {
        int32_t page_id = free_page_indices_.front();
        free_page_indices_.pop_front();
        llama_kv_page* page_ptr = get_page(page_id); // page_id is the index
        if (page_ptr) {
            page_ptr->used_tokens = 0;
            page_ptr->seq_ids.clear();
            return page_id;
        }
        // Should not happen if free_page_indices_ is consistent
        throw std::runtime_error("Internal error: page_id from free list is invalid.");
    }

    // Try to create a new page metadata and assign memory from pool
    if (pages_.size() < (1 << 20)) { // Arbitrary limit on total number of page metadata objects
        int32_t new_page_id = static_cast<int32_t>(pages_.size());
        pages_.emplace_back(new_page_id, page_size_bytes_);
        llama_kv_page& new_page_meta = pages_.back();

        if (assign_memory_to_new_page(new_page_meta)) {
            // next_page_id_counter_ is implicitly pages_.size() after emplace_back
            new_page_meta.used_tokens = 0; // Reset for new use
            new_page_meta.seq_ids.clear();
            return new_page_meta.id;
        } else {
            pages_.pop_back(); // Couldn't assign memory, remove metadata
            // No more memory in the pool
            return -1;
        }
    }
    return -1; // Max page metadata objects reached or pool exhausted
}

void llama_paged_kv_cells::free_page(int32_t page_id) {
    llama_kv_page* page_ptr = get_page(page_id);
    if (!page_ptr) {
        // Trying to free a non-existent page or already handled
        return;
    }

    // Check if it's already in the free list to prevent double freeing
    for (int32_t free_id : free_page_indices_) {
        if (free_id == page_id) {
            return; // Already marked as free
        }
    }

    page_ptr->used_tokens = 0;
    page_ptr->seq_ids.clear();
    free_page_indices_.push_front(page_id); // Add to front for potential LIFO reuse
}

PageOffset llama_paged_kv_cells::find_or_allocate_page_for_token(int32_t seq_id, int32_t token_pos, size_t token_kv_size) {
    TokenKey key(seq_id, token_pos);
    auto it = token_to_page_offset_.find(key);
    if (it != token_to_page_offset_.end()) {
        return it->second;
    }

    // Simplified allocation: find any page associated with this seq_id that has space,
    // or any completely free page, or allocate a new page.
    // This needs to be much smarter, considering token_kv_size.
    // For now, assume page.used_tokens counts abstract "slots" and one token uses one slot.
    // The offset returned will be byte offset: page_ptr->used_tokens * (some fixed K/V element size per token).

    int32_t target_page_id = -1;
    size_t offset_in_page_bytes = 0; // This should be calculated based on actual K/V layout

    // Try to find an existing page for this sequence that *might* have space
    // (This simple check doesn't know if the *remaining* space is enough for token_kv_size)
    for (size_t i = 0; i < pages_.size(); ++i) {
        llama_kv_page& page = pages_[i];
        bool is_page_free = false;
        for (int32_t free_id : free_page_indices_) { if (free_id == page.id) { is_page_free = true; break; } }
        if (is_page_free) continue; // Skip pages in the free list

        if (page.seq_ids.count(seq_id)) {
            // Placeholder: assume page.used_tokens is # of items, and offset is based on this.
            // A real implementation needs to check `page_size_bytes_ - current_byte_offset_of_used_tokens >= token_kv_size`.
            if (page.used_tokens * token_kv_size + token_kv_size <= page.size) { // Simplified check
                target_page_id = page.id;
                offset_in_page_bytes = page.used_tokens * token_kv_size; // This is a byte offset
                break;
            }
        }
    }

    // If no suitable page found, allocate a new one
    if (target_page_id == -1) {
        target_page_id = allocate_page();
        if (target_page_id == -1) {
            throw std::runtime_error("Failed to allocate page for token (pool exhausted or limit reached).");
        }
        offset_in_page_bytes = 0; // Start of a new page
    }

    llama_kv_page* page_ptr = get_page(target_page_id);
    if (!page_ptr) {
        throw std::runtime_error("Internal error: allocated page is invalid.");
    }

    page_ptr->add_sequence(seq_id);
    // The actual K/V data is copied by the caller into: page_ptr->data + offset_in_page_bytes
    page_ptr->used_tokens++; // Increment count of items stored in this page.

    PageOffset result(target_page_id, offset_in_page_bytes);
    token_to_page_offset_[key] = result;
    return result;
}

PageOffset llama_paged_kv_cells::get_page_and_offset(int32_t seq_id, int32_t token_pos) const {
    TokenKey key(seq_id, token_pos);
    auto it = token_to_page_offset_.find(key);
    if (it != token_to_page_offset_.end()) {
        return it->second;
    }
    throw std::out_of_range("Token not found in any page.");
}

llama_kv_page* llama_paged_kv_cells::get_page(int32_t page_id) {
    if (page_id < 0 || static_cast<size_t>(page_id) >= pages_.size()) {
        return nullptr;
    }
    // Assuming page_id is a direct index and pages_[page_id].id == page_id
    if (pages_[page_id].id != page_id) { // Consistency check
        // This indicates an issue if page IDs are not dense array indices
        return nullptr;
    }
    return &pages_[page_id];
}

const llama_kv_page* llama_paged_kv_cells::get_page(int32_t page_id) const {
    if (page_id < 0 || static_cast<size_t>(page_id) >= pages_.size()) {
        return nullptr;
    }
    if (pages_[page_id].id != page_id) {
        return nullptr;
    }
    return &pages_[page_id];
}
