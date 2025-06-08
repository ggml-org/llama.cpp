#include "llama-paged-kv-cache.h"
#include "llama-context.h" // For llama_token, llama_pos, etc.
#include "ggml.h" // For ggml_tensor, GGML_TYPE_F16 etc.
#include <stdexcept>
#include <iostream> // For debugging
#include <algorithm> // For std::min, std::max, std::remove_if
#include <cstring> // For memcpy (TODO: replace with safer alternatives if possible)

// --- llama_paged_kv_cache ---

// Helper to calculate the size of K or V data for a single token per layer.
// This needs to match how ggml stores K/V cache tensors.
// Typically, for K cache: n_embd_head * n_head_kv * sizeof(element_type)
// For V cache: n_embd * sizeof(element_type) (if not using GQA/MQA effectively)
// Or more generally: (n_embd / n_head) * n_head_kv * sizeof(element_type) for K
// and (n_embd / n_head) * n_head * sizeof(element_type) for V.
// Helper to calculate the size of K AND V data for a single token across all layers.
// This is what one "slot" in a page effectively needs to store if pages don't span tokens.
size_t llama_paged_kv_cache::get_kv_item_size_bytes() const {
    const size_t size_k_element = ggml_type_size(type_k_);
    const size_t size_v_element = ggml_type_size(type_v_);

    // Size of K-cache per token, per layer
    const size_t size_k_token_layer = (n_embd_head_ * n_head_kv_) * size_k_element;
    // Size of V-cache per token, per layer
    // Note: n_head_kv_ is used for V as well, assuming GQA/MQA where n_head_v == n_head_kv effectively for storage.
    // If V has n_head (full MHA-style V), this would be (n_embd_head_ * n_head_) * size_v_element.
    // For simplicity with common GQA models, using n_head_kv for V's "width" calculation for cache.
    const size_t size_v_token_layer = (n_embd_head_ * n_head_kv_) * size_v_element;

    return (size_k_token_layer + size_v_token_layer) * n_layer_;
}


llama_paged_kv_cache::llama_paged_kv_cache(
    const struct llama_model_params & mparams,   // llama.h model_params
    const struct llama_context_params & cparams, // llama.h context_params
    ggml_backend_buffer_type_t paged_kv_buffer_type,
    struct ggml_context * kv_mem_ctx)
    : n_embd_(mparams.n_embd),
      n_layer_(mparams.n_layer),
      n_ctx_(cparams.n_ctx),
      n_head_kv_(mparams.n_head_kv),
      n_embd_head_(mparams.n_embd / mparams.n_head), // n_embd_head = d_k = d_v
      type_k_(cparams.type_k),
      type_v_(cparams.type_v),
      kv_mem_ctx_(kv_mem_ctx),
      paged_kv_buffer_type_(paged_kv_buffer_type),
      main_page_pool_tensor_(nullptr),
      main_page_pool_data_(nullptr),
      main_page_pool_size_bytes_(0),
      default_page_size_bytes_(0),
      initial_page_count_(0)
{
    if (!kv_mem_ctx_) {
        throw std::runtime_error("KV memory ggml_context is null for paged KV cache.");
    }
    if (!paged_kv_buffer_type_) {
        // In a real setup, this buffer type must be configured to use a paged ggml_dyn_tallocr.
        throw std::runtime_error("Paged KV buffer type is null.");
    }

    const size_t kv_item_size = get_kv_item_size_bytes();
    if (kv_item_size == 0) {
        throw std::runtime_error("K/V item size is zero, check model/context parameters.");
    }

    // Determine page size in bytes for llama_paged_kv_cells
    if (cparams.kv_page_size > 0) {
        default_page_size_bytes_ = cparams.kv_page_size * kv_item_size;
    } else {
        // Default: aim for roughly 2MB pages, then adjust to be multiple of kv_item_size.
        // Or, use a default number of tokens like 2048.
        size_t default_tokens_per_page = 2048; // A common choice
        default_page_size_bytes_ = default_tokens_per_page * kv_item_size;
        // It's good if default_page_size_bytes_ aligns somewhat with GGML_ALLOCATOR_DEFAULT_PAGE_SIZE,
        // but not strictly necessary as the underlying paged allocator handles GGML pages.
    }
    // Ensure page size is at least one item.
    if (default_page_size_bytes_ < kv_item_size) {
        default_page_size_bytes_ = kv_item_size;
    }
    // TODO: Align default_page_size_bytes_ to some hardware-friendly boundary if desired,
    // e.g., multiple of 256 bytes or GGML_ALLOCATOR_DEFAULT_PAGE_SIZE.
    // For now, it's purely based on token capacity.

    // Determine initial number of pages to allocate in the pool
    // Example: enough for n_ctx / 2 tokens, or a fixed number like 32.
    // Max tokens to cache = n_ctx typically.
    initial_page_count_ = (n_ctx_ > 0 ? (n_ctx_ / 2) : 512) * kv_item_size / default_page_size_bytes_ ;
    if (initial_page_count_ == 0) initial_page_count_ = 1; // At least one page
    // A more robust initial count might be a certain fraction of n_ctx, e.g., enough pages for n_ctx/2 tokens.
    // initial_page_count_ = (n_ctx_ * kv_item_size) / default_page_size_bytes_ / 2;
    // if (initial_page_count_ < 4) initial_page_count_ = 4; // Minimum number of pages
    // For now, let's try a fixed number of initial pages for simplicity of example.
    initial_page_count_ = 32; // e.g. 32 pages.

    main_page_pool_size_bytes_ = initial_page_count_ * default_page_size_bytes_;

    LLAMA_LOG_INFO("%s: Initializing paged KV cache with: total pool size %.2f MiB, page size %.2f KiB, %zu initial pages\n",
        __func__,
        main_page_pool_size_bytes_ / (1024.0*1024.0),
        default_page_size_bytes_ / 1024.0,
        initial_page_count_);

    // Allocate the main page pool tensor using the provided context and buffer type
    main_page_pool_tensor_ = ggml_new_tensor_1d(kv_mem_ctx_, GGML_TYPE_I8, main_page_pool_size_bytes_);
    if (!main_page_pool_tensor_) {
        throw std::runtime_error("Failed to create main page pool tensor for paged KV cache.");
    }
    ggml_set_name(main_page_pool_tensor_, "paged_kv_main_pool");

    // This is the crucial step: associate the tensor with the paged buffer type.
    // The allocator for kv_mem_ctx_ (a ggml_gallocr_t) must be configured such that
    // this paged_kv_buffer_type_ uses a paged ggml_dyn_tallocr.
    // This typically happens in llama.cpp when ggml_gallocr_new_n is called and buffer types are set up.
    enum ggml_status status = ggml_allocr_alloc(ggml_backend_buft_get_allocator(paged_kv_buffer_type_), main_page_pool_tensor_);
    if (status != GGML_STATUS_SUCCESS || main_page_pool_tensor_->data == nullptr) {
         ggml_free_tensor(main_page_pool_tensor_); // Free the tensor struct if allocation failed
         main_page_pool_tensor_ = nullptr;
        throw std::runtime_error("Failed to allocate main page pool buffer using paged allocator. GGML Error: " + std::to_string(status));
    }
    main_page_pool_data_ = (uint8_t*)main_page_pool_tensor_->data;

    // Initialize paged_cells_ with the allocated pool
    new (&paged_cells_) llama_paged_kv_cells(
        default_page_size_bytes_,
        main_page_pool_data_,
        main_page_pool_size_bytes_,
        initial_page_count_ // Number of pages to "carve out" from the pool metadata initially
    );

    if (default_page_size_bytes_ < get_kv_item_size_bytes()) { // Final check with actual item size
        std::cerr << "llama_paged_kv_cache: Warning: Effective page size (" << default_page_size_bytes_
                  << " bytes) is smaller than one K/V item size (" << get_kv_item_size_bytes()
                  << " bytes). This is likely an error." << std::endl;
    }
}

llama_paged_kv_cache::~llama_paged_kv_cache() {
    // The main_page_pool_tensor_ is allocated within kv_mem_ctx_ using its allocator.
    // It will be freed when kv_mem_ctx_ is freed or when the allocator itself is freed/reset,
    // assuming the allocator owns the buffer from which main_page_pool_tensor_ was sub-allocated.
    // If main_page_pool_tensor_ represents a buffer allocated directly by a backend buffer type
    // (e.g. via ggml_backend_buft_alloc_buffer), then that buffer would need explicit freeing
    // if not managed by an allocator.
    // Given we used ggml_allocr_alloc, the allocator associated with paged_kv_buffer_type_
    // manages this tensor's memory.

    // Explicitly call destructor for paged_cells_ if it was placement-newed
    // and not a direct member object (though it is a direct member here, so C++ handles it).
    // paged_cells_.~llama_paged_kv_cells(); // Not needed for direct members.
}

llama_memory_state_i * llama_paged_kv_cache::init_batch(const llama_batch & batch, uint32_t n_ubatch, bool embd_pooled, bool logits_all) {
    return new llama_paged_kv_cache_state(*this, batch, n_ubatch, embd_pooled, logits_all);
}

llama_memory_state_i * llama_paged_kv_cache::init_full() {
    // This typically pre-allocates or prepares the KV cache for all tokens in the context.
    // For a paged system, it might mean ensuring enough free pages are available
    // or pre-mapping some sequence IDs if known.
    // For now, just return a state object.
    return new llama_paged_kv_cache_state(*this);
}

llama_memory_state_i * llama_paged_kv_cache::init_update(llama_context * lctx, bool optimize) {
    // This is used for incremental updates to the KV cache.
    return new llama_paged_kv_cache_state(*this, lctx, optimize);
}

bool llama_paged_kv_cache::get_can_shift() const {
    // Paged KV cache should ideally support shifting tokens without recopying all data,
    // by just adjusting metadata in token_to_page_offset.
    // However, a simple implementation might still involve some copying if pages become fragmented.
    return true; // Placeholder: assume it can support shifting efficiently.
}

void llama_paged_kv_cache::clear(bool data) {
    // data = true means clear KV data, false means only clear metadata (like sequence associations)
    // This needs to iterate through all token mappings and effectively free them.
    // Then, if data is true, it should also clear the actual memory in pages if desired,
    // though typically pages are just added back to the free list.

    // Option 1: Clear all mappings and return all pages to free list.
    paged_cells_.token_to_page_offset.clear(); // Assuming this member is public or accessible
    for (auto& page : paged_cells_.pages) { // Assuming 'pages' is accessible
        if (page.id != -1 && page.data != nullptr) { // Valid page
             // If 'data' is true, one might zero out the page.data, but it's not strictly necessary
             // as it will be overwritten. More important is to mark it as free.
            paged_cells_.free_page(page.id); // This resets used_tokens and seq_ids
        }
    }
    // Ensure free_pages_ids contains all pages if we cleared them all.
    // The free_page logic should handle this.
    // Note: This is a simplified way. A more robust clear might need more careful handling
    // of the paged_cells_ internal state.
    std::cout << "llama_paged_kv_cache::clear() called. All mappings removed and pages (intended to be) freed." << std::endl;
}

void llama_paged_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 == -1) {
        // Remove all tokens for this sequence ID from p0 onwards
        // Or, if p0 is also special, remove all tokens for seq_id
        // This needs clarification based on llama.cpp behavior.
        // Assume p1 = infinity or max_pos if -1.
        // For now, let's assume it means remove from p0 to max_pos for that sequence.
        llama_pos current_max_pos = seq_pos_max(seq_id);
        if (current_max_pos == -1 && p0 == 0) { // No tokens for this seq_id, or remove all
             p1 = -1; // Special marker to remove all
        } else {
            p1 = current_max_pos +1; // Iterate up to and including current_max_pos
        }
    }

    // This function needs to interact with llama_paged_kv_cells to correctly update page metadata.
    // Let's assume llama_paged_kv_cells will get a method remove_token_mappings_for_sequence_range
    // or we iterate here and call a simpler remove_token(key) on paged_cells.

    auto& cells = paged_cells_.get_paged_cells(); // Get reference to the map and page list
    std::vector<TokenKey> keys_to_remove;

    // Collect keys to remove
    for (auto it = cells.token_to_page_offset_.begin(); it != cells.token_to_page_offset_.end(); ++it) {
        const TokenKey& key = it->first;
        if (key.seq_id == seq_id || seq_id < 0) { // seq_id < 0 means all sequences
            if (p1 == -1 && key.token_pos >= p0) { // remove from p0 to end
                keys_to_remove.push_back(key);
            } else if (key.token_pos >= p0 && key.token_pos < p1) { // remove from [p0, p1)
                keys_to_remove.push_back(key);
            } else if (p0 == 0 && p1 == -1 && seq_id >= 0) { // remove all for a specific sequence
                 keys_to_remove.push_back(key);
            }
        }
    }

    // Process removals
    for (const auto& key_to_remove : keys_to_remove) {
        auto it = cells.token_to_page_offset_.find(key_to_remove);
        if (it != cells.token_to_page_offset_.end()) {
            PageOffset po = it->second;
            llama_kv_page* page = cells.get_page(po.page_id); // get_page is public in cells
            if (page) {
                page->remove_sequence(key_to_remove.seq_id); // Remove this seq_id's association
                // Decrement used_tokens for the page. This count represents actual stored items.
                if (page->used_tokens > 0) {
                    page->used_tokens--;
                }
                // If no sequences refer to any token on this page anymore AND no tokens are stored, free the page.
                // A simpler rule: if used_tokens is 0, the page is free.
                // seq_ids being empty is a stronger condition that might not always be met if another seq shares a different token on the same page.
                if (page->used_tokens == 0) {
                    cells.free_page(po.page_id); // free_page is public in cells
                }
            }
            cells.token_to_page_offset_.erase(it); // Remove the mapping
        }
    }
    // TODO: More robust page freeing. If a page's seq_ids becomes empty, it means no
    // sequence *currently being tracked by this specific call to seq_rm* uses it.
    // But other sequences (not part of this seq_rm call, e.g. if seq_id >= 0) might still use tokens on that page.
    // The `page->used_tokens--` and `if (page->used_tokens == 0)` is the most reliable way.
    LLAMA_LOG_INFO("llama_paged_kv_cache::seq_rm(seq=%d, p0=%d, p1=%d) processed %zu token mappings.\n", seq_id, p0, p1, keys_to_remove.size());
}

void llama_paged_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // This is complex. It requires copying token data from src pages to dst pages.
    // May involve allocating new pages for seq_id_dst.
    if (p1 == -1) { // convention: copy all from p0 to end of seq_id_src
        p1 = seq_pos_max(seq_id_src) + 1; // seq_pos_max is inclusive, so add 1 for exclusive upper bound
        if (p1 == 0 && seq_pos_min(seq_id_src) == -1) return; // seq_pos_max was -1 (empty seq), so nothing to copy
    }

    const size_t token_kv_item_size = get_kv_item_size_bytes(); // Size of all K/V data for one token position

    for (llama_pos pos = p0; pos < p1; ++pos) {
        try {
            PageOffset src_po = paged_cells_.get_page_and_offset(seq_id_src, pos); // Can throw if src token not found
            llama_kv_page* src_page = paged_cells_.get_page(src_po.page_id);

            if (!src_page || !src_page->data) {
                std::cerr << "seq_cp: Source page data not found for seq_id=" << seq_id_src << ", pos=" << pos << std::endl;
                continue;
            }
            uint8_t* src_data_ptr = src_page->data + src_po.offset; // Assuming offset is byte offset

            // Find or allocate space for destination
            // We need to ensure space for the token in the destination.
            // find_or_allocate_page_for_token needs the size of the item it's allocating for.
            PageOffset dst_po = paged_cells_.find_or_allocate_page_for_token(seq_id_dst, pos, token_kv_item_size);
            llama_kv_page* dst_page = paged_cells_.get_page(dst_po.page_id);

            if (!dst_page || !dst_page->data) { // Should not happen if find_or_allocate throws on failure
                 std::cerr << "seq_cp: Destination page data could not be retrieved after find_or_allocate for seq_id=" << seq_id_dst << ", pos=" << pos << std::endl;
                continue;
            }
            uint8_t* dst_data_ptr = dst_page->data + dst_po.offset;

            // Check bounds: src_po.offset is within src_page, dst_po.offset is within dst_page.
            // The find_or_allocate_page_for_token should have ensured dst_page has space for token_kv_item_size at dst_po.offset.
            // And src_page must contain the data for token_kv_item_size at src_po.offset.
            if (src_po.offset + token_kv_item_size > src_page->size) {
                std::cerr << "seq_cp: Source read out of bounds for seq_id=" << seq_id_src << ", pos=" << pos
                          << ". Offset=" << src_po.offset << ", ItemSize=" << token_kv_item_size << ", PageSize=" << src_page->size << std::endl;
                continue;
            }
             if (dst_po.offset + token_kv_item_size > dst_page->size) {
                std::cerr << "seq_cp: Destination write out of bounds for seq_id=" << seq_id_dst << ", pos=" << pos
                          << ". Offset=" << dst_po.offset << ", ItemSize=" << token_kv_item_size << ", PageSize=" << dst_page->size << std::endl;
                continue;
            }

            std::memcpy(dst_data_ptr, src_data_ptr, token_kv_item_size);
            // find_or_allocate_page_for_token should have already associated seq_id_dst with the page.

        } catch (const std::out_of_range& e) { // From get_page_and_offset if src token not found
            std::cerr << "seq_cp: Token not found for seq_id_src=" << seq_id_src << ", pos=" << pos << ". What: " << e.what() << std::endl;
            // If source doesn't exist, we can't copy it.
        } catch (const std::runtime_error& e) {
            std::cerr << "seq_cp: Runtime error during copy for pos=" << pos << ". What: " << e.what() << std::endl;
        }
    }
    std::cout << "llama_paged_kv_cache::seq_cp(src=" << seq_id_src << ", dst=" << seq_id_dst << ", p0=" << p0 << ", p1=" << p1 << ") called." << std::endl;
}

void llama_paged_kv_cache::seq_keep(llama_seq_id seq_id) {
    // Remove all sequence IDs except this one.
    std::vector<TokenKey> keys_to_remove;
    for (auto const& [key, val] : paged_cells_.get_paged_cells().token_to_page_offset) {
        if (key.seq_id != seq_id) {
            keys_to_remove.push_back(key);
        }
    }
    for (const auto& key_to_remove : keys_to_remove) {
         // Similar to seq_rm, remove mapping and update page metadata.
        PageOffset po = paged_cells_.get_page_and_offset(key_to_remove.seq_id, key_to_remove.token_pos);
        llama_kv_page* page = paged_cells_.get_page(po.page_id);
        if (page) {
            page->remove_sequence(key_to_remove.seq_id);
        }
        paged_cells_.token_to_page_offset.erase(key_to_remove);
    }
    // TODO: Add logic to check if pages associated with removed tokens can be fully freed.
    std::cout << "llama_paged_kv_cache::seq_keep(seq=" << seq_id << ") called." << std::endl;
}

void llama_paged_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // This is effectively a "shift" operation on token positions for a sequence.
    // If p1 is -1, shift all tokens from p0 onwards.
    // All token positions p >= p0 are changed to p + shift.
    // This can be done by updating keys in token_to_page_offset.
    // A common use case is removing tokens at the beginning (shift < 0) to make space.
    // Or compacting after a seq_div.

    if (p1 == -1) {
        p1 = seq_pos_max(seq_id) + 1; // Make p1 exclusive end
         if (p1 == 0 && seq_pos_min(seq_id) == -1) return; // Empty sequence
    }

    auto& cells = paged_cells_.get_paged_cells();
    std::vector<std::pair<TokenKey, PageOffset>> items_to_remap;

    // Collect items to remap by iterating and erasing matching old keys directly
    // This avoids concurrent modification issues if new keys overlap with not-yet-processed old keys.
    auto it = cells.token_to_page_offset_.begin();
    while (it != cells.token_to_page_offset_.end()) {
        if (it->first.seq_id == seq_id && it->first.token_pos >= p0 && it->first.token_pos < p1) {
            items_to_remap.push_back(*it);
            it = cells.token_to_page_offset_.erase(it); // Erase and get next valid iterator
        } else {
            ++it;
        }
    }

    // Re-insert with new positions
    for (const auto& item_pair : items_to_remap) {
        const TokenKey& old_key = item_pair.first;
        const PageOffset& val = item_pair.second;
        llama_pos new_pos = old_key.token_pos + shift;

        if (new_pos < 0) { // Token shifted out of bounds (negative position)
            // This token is effectively removed. We need to update page metadata.
            llama_kv_page* page = cells.get_page(val.page_id);
            if (page) {
                page->remove_sequence(old_key.seq_id); // Remove this specific sequence's association
                if (page->used_tokens > 0) {
                    page->used_tokens--;
                }
                if (page->used_tokens == 0) { // If page becomes empty
                    cells.free_page(val.page_id);
                }
            }
            // Do not re-insert this token's mapping.
        } else {
            TokenKey new_key(seq_id, new_pos);
            // TODO: Handle collision if new_key already exists (e.g. from a different original token).
            // This typically implies an error in how seq_add is used or that the target range should be clear.
            // For now, assume overwrite or direct insertion is fine.
            cells.token_to_page_offset_[new_key] = val;
            // The page itself (val.page_id) and offset within page (val.offset) remain the same.
            // Only the logical position (token_pos) changes.
            // If the token was remapped (not dropped), its original page's seq_id association remains.
        }
    }
    LLAMA_LOG_INFO("llama_paged_kv_cache::seq_add(seq=%d, p0=%d, p1=%d, shift=%d) processed %zu items.\n", seq_id, p0, p1, shift, items_to_remap.size());
}


void llama_paged_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 0) {
        throw std::invalid_argument("Division by zero in seq_div.");
    }
    if (p1 == -1) {
        p1 = seq_pos_max(seq_id) + 1; // Make p1 exclusive end
        if (p1 == 0 && seq_pos_min(seq_id) == -1) return; // Empty sequence
    }

    auto& cells = paged_cells_.get_paged_cells();
    std::vector<std::pair<TokenKey, PageOffset>> items_to_remap;

    // Collect and erase old mappings
    auto it = cells.token_to_page_offset_.begin();
    while (it != cells.token_to_page_offset_.end()) {
        if (it->first.seq_id == seq_id && it->first.token_pos >= p0 && it->first.token_pos < p1) {
            items_to_remap.push_back(*it);
            it = cells.token_to_page_offset_.erase(it);
        } else {
            ++it;
        }
    }

    // Re-insert with new positions. Handle potential collisions by keeping only one token per new_pos.
    // This is a simple strategy; more complex ones might involve merging or erroring.
    std::map<llama_pos, std::pair<TokenKey, PageOffset>> unique_new_pos_mappings;

    for (const auto& item_pair : items_to_remap) {
        const TokenKey& old_key = item_pair.first;
        const PageOffset& val = item_pair.second;
        llama_pos new_pos = old_key.token_pos / d; // Integer division

        // If new_pos already mapped, the first token encountered for that new_pos "wins".
        // Other tokens that would map to the same new_pos are effectively dropped.
        if (unique_new_pos_mappings.find(new_pos) == unique_new_pos_mappings.end()) {
            unique_new_pos_mappings[new_pos] = item_pair; // Store original key and val
        } else {
            // This token slot is "lost" due to collision after division. Free its page resources.
            llama_kv_page* page = cells.get_page(val.page_id);
            if (page) {
                page->remove_sequence(old_key.seq_id);
                if (page->used_tokens > 0) {
                    page->used_tokens--;
                }
                if (page->used_tokens == 0) {
                    cells.free_page(val.page_id);
                }
            }
        }
    }

    // Insert the unique new position mappings
    for(const auto& mapping_pair : unique_new_pos_mappings){
        const TokenKey& old_key_of_winner = mapping_pair.second.first; // unused, just for context
        const PageOffset& val_of_winner = mapping_pair.second.second;
        llama_pos new_pos = mapping_pair.first;
        cells.token_to_page_offset_[TokenKey(seq_id, new_pos)] = val_of_winner;
    }
    LLAMA_LOG_INFO("llama_paged_kv_cache::seq_div(seq=%d, p0=%d, p1=%d, d=%d) remapped %zu unique items from %zu original items.\n",
        seq_id, p0, p1, d, unique_new_pos_mappings.size(), items_to_remap.size());
}

llama_pos llama_paged_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos min_pos = -1;
    for (auto const& [key, val] : paged_cells_.get_paged_cells().token_to_page_offset) {
        if (key.seq_id == seq_id) {
            if (min_pos == -1 || key.token_pos < min_pos) {
                min_pos = key.token_pos;
            }
        }
    }
    return min_pos;
}

llama_pos llama_paged_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos max_pos = -1;
    for (auto const& [key, val] : paged_cells_.get_paged_cells().token_to_page_offset) {
        if (key.seq_id == seq_id) {
            if (max_pos == -1 || key.token_pos > max_pos) {
                max_pos = key.token_pos;
            }
        }
    }
    return max_pos;
}

size_t llama_paged_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::cout << "llama_paged_kv_cache::state_write() called. (Not Implemented)" << std::endl;
    // This would involve serializing token_to_page_offset and the relevant page data.
    // Complex due to pointers and paged nature.
    // For now, this is a conceptual sketch.
    size_t bytes_written = 0;

    // Write basic parameters
    io.write_val(n_embd_);              bytes_written += sizeof(n_embd_);
    io.write_val(n_layer_);             bytes_written += sizeof(n_layer_);
    io.write_val(n_ctx_);               bytes_written += sizeof(n_ctx_); // n_ctx_ from constructor
    io.write_val(n_head_kv_);           bytes_written += sizeof(n_head_kv_);
    io.write_val(n_embd_head_);         bytes_written += sizeof(n_embd_head_);
    io.write_val(type_k_);              bytes_written += sizeof(type_k_);
    io.write_val(type_v_);              bytes_written += sizeof(type_v_);

    // Write paged_cells_ state
    // This requires paged_cells_ to expose its members or have its own serialize method.
    // For simplicity, assuming direct access or helper methods in paged_cells for its state.
    const auto& cells = paged_cells_.get_paged_cells();
    io.write_val(cells.page_size_bytes_);          bytes_written += sizeof(cells.page_size_bytes_);
    io.write_val(cells.pages_.size());             bytes_written += sizeof(cells.pages_.size());
    io.write_val(cells.page_memory_pool_size_bytes_); bytes_written += sizeof(cells.page_memory_pool_size_bytes_);
    io.write_val(cells.page_memory_pool_used_bytes_); bytes_written += sizeof(cells.page_memory_pool_used_bytes_);

    // Write individual page metadata and data
    for (const auto& page : cells.pages_) {
        io.write_val(page.id);                     bytes_written += sizeof(page.id);
        io.write_val(page.used_tokens);            bytes_written += sizeof(page.used_tokens);
        io.write_val(page.size);                   bytes_written += sizeof(page.size);
        // Serialize seq_ids set
        size_t num_seq_ids = page.seq_ids.size();
        io.write_val(num_seq_ids);                 bytes_written += sizeof(num_seq_ids);
        for (int32_t s_id : page.seq_ids) {
            io.write_val(s_id);                    bytes_written += sizeof(s_id);
        }
        // Write page data if it's valid and part of the pool (it should be)
        if (page.data && page.size > 0 && cells.page_memory_pool_ &&
            page.data >= cells.page_memory_pool_ &&
            page.data < cells.page_memory_pool_ + cells.page_memory_pool_used_bytes_) {
            io.write_raw(page.data, page.size);    bytes_written += page.size;
        } else if (page.size > 0) {
            // This case should ideally not happen if page.data is always from the pool
            // or indicates an uninitialized/problematic page. Write zeros or handle error.
            std::vector<uint8_t> zeros(page.size, 0);
            io.write_raw(zeros.data(), page.size); bytes_written += page.size;
            LLAMA_LOG_WARN("Warning: writing zeroed data for page %d as its data pointer was invalid or size was zero during state_write.\n", page.id);
        }
    }

    // Write token_to_page_offset_ map
    size_t map_size = cells.token_to_page_offset_.size();
    io.write_val(map_size);                        bytes_written += sizeof(map_size);
    for (const auto& pair : cells.token_to_page_offset_) {
        io.write_val(pair.first.seq_id);           bytes_written += sizeof(pair.first.seq_id);
        io.write_val(pair.first.token_pos);        bytes_written += sizeof(pair.first.token_pos);
        io.write_val(pair.second.page_id);         bytes_written += sizeof(pair.second.page_id);
        io.write_val(pair.second.offset);          bytes_written += sizeof(pair.second.offset);
    }

    // Write free_page_indices_ list
    size_t free_list_size = cells.free_page_indices_.size();
    io.write_val(free_list_size);                  bytes_written += sizeof(free_list_size);
    for (int32_t page_idx : cells.free_page_indices_) {
        io.write_val(page_idx);                    bytes_written += sizeof(page_idx);
    }

    LLAMA_LOG_INFO("llama_paged_kv_cache::state_write() wrote %zu bytes.\n", bytes_written);
    GGML_UNUSED(seq_id); // TODO: Implement partial state write for a specific sequence
    return bytes_written;
}

size_t llama_paged_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    // This would involve deserializing and reconstructing the cache state.
    // This is a conceptual sketch and needs robust error handling and GGML memory re-acquisition.
    size_t bytes_read = 0;

    // Read basic parameters (and expect them to match current config, or reconfigure)
    uint32_t read_n_embd, read_n_layer, read_n_ctx, read_n_head_kv, read_n_embd_head;
    ggml_type read_type_k, read_type_v;
    io.read_val(read_n_embd);           bytes_read += sizeof(read_n_embd);
    // ... (read all other parameters and validate against current mparams/cparams) ...
    // If mismatch, this state is likely incompatible. For now, assume they match.

    // Read paged_cells_ state
    auto& cells = paged_cells_.get_paged_cells();
    size_t read_page_size_bytes, read_num_pages_meta, read_pool_size, read_pool_used;
    io.read_val(read_page_size_bytes);       bytes_read += sizeof(read_page_size_bytes);
    io.read_val(read_num_pages_meta);        bytes_read += sizeof(read_num_pages_meta);
    io.read_val(read_pool_size);             bytes_read += sizeof(read_pool_size);
    io.read_val(read_pool_used);             bytes_read += sizeof(read_pool_used);

    // Critical: Re-allocate main_page_pool_tensor_ with the read size using kv_mem_ctx_ and paged_kv_buffer_type_
    // This assumes the context and buffer type are already correctly set up for paged allocation.
    // If main_page_pool_tensor_ already exists, it might need to be freed/reallocated if size changed.
    // For simplicity, assume this is called on a newly constructed or cleared cache.
    if (main_page_pool_tensor_) { /* handle existing tensor, maybe free it */ }
    main_page_pool_size_bytes_ = read_pool_size;
    main_page_pool_tensor_ = ggml_new_tensor_1d(kv_mem_ctx_, GGML_TYPE_I8, main_page_pool_size_bytes_);
    if (!main_page_pool_tensor_) throw std::runtime_error("Failed to reallocate main page pool tensor during state_read.");
    ggml_set_name(main_page_pool_tensor_, "paged_kv_main_pool_loaded");
    enum ggml_status status = ggml_allocr_alloc(ggml_backend_buft_get_allocator(paged_kv_buffer_type_), main_page_pool_tensor_);
    if (status != GGML_STATUS_SUCCESS || main_page_pool_tensor_->data == nullptr) {
        throw std::runtime_error("Failed to allocate main page pool buffer during state_read.");
    }
    main_page_pool_data_ = (uint8_t*)main_page_pool_tensor_->data;
    cells.page_memory_pool_ = main_page_pool_data_;
    cells.page_memory_pool_size_bytes_ = main_page_pool_size_bytes_;
    cells.page_memory_pool_used_bytes_ = read_pool_used; // Important to restore this
    cells.page_size_bytes_ = read_page_size_bytes;


    // Reconstruct pages_ vector
    cells.pages_.clear();
    cells.pages_.resize(read_num_pages_meta);
    for (size_t i = 0; i < read_num_pages_meta; ++i) {
        llama_kv_page& page = cells.pages_[i];
        io.read_val(page.id);                      bytes_read += sizeof(page.id);
        io.read_val(page.used_tokens);             bytes_read += sizeof(page.used_tokens);
        io.read_val(page.size);                    bytes_read += sizeof(page.size);
        // Point page.data to the correct offset in the newly allocated pool
        page.data = cells.page_memory_pool_ + (page.id * cells.page_size_bytes_); // Assumes contiguous layout by ID

        size_t num_seq_ids;
        io.read_val(num_seq_ids);                  bytes_read += sizeof(num_seq_ids);
        page.seq_ids.clear();
        for (size_t j = 0; j < num_seq_ids; ++j) {
            int32_t s_id;
            io.read_val(s_id);                     bytes_read += sizeof(s_id);
            page.seq_ids.insert(s_id);
        }
        if (page.data && page.size > 0) { // Read page content
            io.read_raw(page.data, page.size);     bytes_read += page.size;
        } else if (page.size > 0) {
            LLAMA_LOG_WARN("Warning: page %d had size %zu but no data pointer during state_read, skipping data read.\n", page.id, page.size);
        }
    }
    cells.next_page_id_counter_ = read_num_pages_meta; // Assuming IDs were dense 0 to N-1

    // Rebuild token_to_page_offset_ map
    cells.token_to_page_offset_.clear();
    size_t map_size;
    io.read_val(map_size);                         bytes_read += sizeof(map_size);
    for (size_t i = 0; i < map_size; ++i) {
        TokenKey key(0,0);
        PageOffset val(0,0);
        io.read_val(key.seq_id);                   bytes_read += sizeof(key.seq_id);
        io.read_val(key.token_pos);                bytes_read += sizeof(key.token_pos);
        io.read_val(val.page_id);                  bytes_read += sizeof(val.page_id);
        io.read_val(val.offset);                   bytes_read += sizeof(val.offset);
        cells.token_to_page_offset_[key] = val;
    }

    // Rebuild free_page_indices_ list
    cells.free_page_indices_.clear();
    size_t free_list_size;
    io.read_val(free_list_size);                   bytes_read += sizeof(free_list_size);
    for (size_t i = 0; i < free_list_size; ++i) {
        int32_t page_idx;
        io.read_val(page_idx);                     bytes_read += sizeof(page_idx);
        cells.free_page_indices_.push_back(page_idx);
    }

    LLAMA_LOG_INFO("llama_paged_kv_cache::state_read() read %zu bytes.\n", bytes_read);
    GGML_UNUSED(seq_id); // TODO: Implement partial state read
    return bytes_read;
}


// --- llama_paged_kv_cache_state ---

llama_paged_kv_cache_state::llama_paged_kv_cache_state(
    llama_paged_kv_cache & cache_ref,
    const llama_batch & batch,
    uint32_t n_ubatch_in,
    bool embd_pooled_in,
    bool logits_all_in)
    : cache_(cache_ref),
      batch_ref_(batch), // This might need to be a copy if batch lifetime is shorter
      n_ubatch_total_(n_ubatch_in),
      current_ubatch_idx_(0),
      embd_pooled_(embd_pooled_in),
      logits_all_(logits_all_in),
      lctx_ref_(nullptr),
      optimize_(false),
      status_(LLAMA_MEMORY_STATUS_OK)
{
    // Prepare for the first ubatch
    if (n_ubatch_total_ > 0) {
        prepare_kv_view_for_ubatch();
    } else {
        status_ = LLAMA_MEMORY_STATUS_ERROR; // Or some other appropriate status
    }
}

llama_paged_kv_cache_state::llama_paged_kv_cache_state(llama_paged_kv_cache & cache_ref)
    : cache_(cache_ref),
      batch_ref_({0, nullptr, nullptr, nullptr, nullptr, 0, 0, 0}), // Dummy batch
      n_ubatch_total_(1), // Typically init_full might be considered one "operation"
      current_ubatch_idx_(0),
      embd_pooled_(false),
      logits_all_(false),
      lctx_ref_(nullptr),
      optimize_(false),
      status_(LLAMA_MEMORY_STATUS_OK)
{
    // For init_full, there isn't really a "batch" in the same sense.
    // The apply() method might do global setup.
    // For now, make it behave like one ubatch.
    current_kv_view_.n_ctx = cache_.get_n_ctx();
    current_kv_view_.n_head = cache_.get_n_head_kv(); // Using n_head_kv, assuming GQA/MQA context
    current_kv_view_.n_embd_head = cache_.get_n_embd_head();
    current_kv_view_.n_layer = cache_.get_n_layer();
    // k_data, v_data, q_data, etc. will be set by apply() or during get_ubatch() by interacting with paged_cells.
}

llama_paged_kv_cache_state::llama_paged_kv_cache_state(llama_paged_kv_cache & cache_ref, llama_context * lctx_in, bool optimize_in)
    : cache_(cache_ref),
      batch_ref_({0, nullptr, nullptr, nullptr, nullptr, 0, 0, 0}), // Dummy batch
      n_ubatch_total_(1), // init_update might be one operation
      current_ubatch_idx_(0),
      embd_pooled_(false), // Not relevant for update?
      logits_all_(false),  // Not relevant for update?
      lctx_ref_(lctx_in),
      optimize_(optimize_in),
      status_(LLAMA_MEMORY_STATUS_OK)
{
    current_kv_view_.n_ctx = cache_.get_n_ctx();
    current_kv_view_.n_head = cache_.get_n_head_kv();
    current_kv_view_.n_embd_head = cache_.get_n_embd_head();
    current_kv_view_.n_layer = cache_.get_n_layer();
}


llama_paged_kv_cache_state::~llama_paged_kv_cache_state() {
    // Nothing specific to clean up here unless current_kv_view_ owns memory not handled by paged_cells.
}

void llama_paged_kv_cache_state::prepare_kv_view_for_ubatch() {
    if (current_ubatch_idx_ >= n_ubatch_total_) {
        status_ = LLAMA_MEMORY_STATUS_NO_SPACE; // Or some "finished" status
        return;
    }

    // This is the core logic for init_batch's state.
    // It needs to figure out which tokens are in the current ubatch,
    // then for each of those tokens, find/allocate pages for their K/V data.
    // The `current_kv_view_` should then point to the memory regions in these pages.
    // This is extremely complex because the pages might not be contiguous.
    // The `llama_kv_cache_view` struct expects contiguous `k_data` and `v_data` pointers (or ggml_tensors).
    // A paged KV cache fundamentally breaks this assumption for the *entire batch*.

    // A more realistic `llama_kv_cache_view` for a paged system would be a list of
    // (token_idx_in_batch, layer_idx, K_or_V_ptr) or similar.
    // Or, `get_ubatch()` would return a view that, when its `data()` method is called for a specific
    // (seq_id, pos, layer), it resolves to the correct page and offset.

    // For now, this is a massive simplification / placeholder:
    current_kv_view_.n_ctx = cache_.get_n_ctx(); // Max context
    current_kv_view_.n_head = cache_.get_n_head_kv();
    current_kv_view_.n_embd_head = cache_.get_n_embd_head();
    current_kv_view_.n_layer = cache_.get_n_layer();

    // The actual pointers k_data, v_data, etc. in current_kv_view_ cannot be easily set
    // to a single contiguous block for a paged KV store if the ubatch processes multiple tokens
    // whose KV data lands on different pages.
    // This implies that the compute kernels (ggml) need to be aware of this paged structure,
    // or we need a temporary contiguous buffer where data for the current ubatch is gathered,
    // and then scattered back after computation. This is inefficient.

    // Let's assume for now that `apply()` will handle the direct interaction with paged_cells
    // and the compute side (ggml) will be given pointers token by token or through a modified API.
    // `current_kv_view_` might be more of a metadata container in this paged context.

    current_out_ids_.clear();
    // Simplified: assume all sequences in the batch are processed in each ubatch.
    // A real implementation would slice the batch.
    // For now, let's just consider all sequences in the batch for out_ids.
    if (batch_ref_.n_tokens > 0) { // if there is a batch
        std::set<llama_seq_id> unique_seq_ids;
        for (int i = 0; i < batch_ref_.n_tokens; ++i) {
             for (int j = 0; j < batch_ref_.n_seq_id[i]; ++j) {
                unique_seq_ids.insert(batch_ref_.seq_id[i][j]);
            }
        }
        current_out_ids_.assign(unique_seq_ids.begin(), unique_seq_ids.end());
    }


    status_ = LLAMA_MEMORY_STATUS_OK;
    std::cout << "llama_paged_kv_cache_state::prepare_kv_view_for_ubatch() ubatch " << current_ubatch_idx_ << std::endl;
}


bool llama_paged_kv_cache_state::next() {
    current_ubatch_idx_++;
    if (current_ubatch_idx_ >= n_ubatch_total_) {
        status_ = LLAMA_MEMORY_STATUS_NO_SPACE; // Or some "finished" status
        return false;
    }
    prepare_kv_view_for_ubatch();
    return true;
}

void llama_paged_kv_cache_state::apply() {
    if (status_ != LLAMA_MEMORY_STATUS_OK && status_ != LLAMA_MEMORY_STATUS_PARTIAL) {
        // Don't apply if there was an error or already finished
        return;
    }

    // This is where the K/V data for the current ubatch (described by batch_ref_ and current_ubatch_idx_)
    // should be written into the paged_cells_.
    // The `current_kv_view_` should have been prepared by `ggml_graph_plan` with pointers
    // to where the new K/V data for this ubatch *will be computed*.
    // After computation (e.g. `ggml_graph_compute`), this `apply` method is called to commit it to our store.

    // Example logic for init_batch:
    // Iterate through tokens in the current ubatch of batch_ref_.
    // For each token (seq_id, pos):
    //   1. Determine the source of its K and V data (from current_kv_view_, which points to ggml computation results).
    //   2. Call cache_.paged_cells_.find_or_allocate_page_for_token(seq_id, pos) to get destination PageOffset.
    //      This needs to be adapted: the "offset" from paged_cells must be understood in terms of
    //      the full K+V data size for all layers for that token.
    //      Let full_token_kv_size = cache_.get_kv_token_size_bytes() * cache_.get_n_layer().
    //      The paged_cells `used_tokens` and `offset` should operate on units of this size.
    //   3. Get destination page pointer `dst_page_ptr = cache_.paged_cells_.get_page(dst_po.page_id)->data + dst_po.offset;`
    //   4. For each layer:
    //      a. Calculate where K-data for this (token,layer) is in `current_kv_view_` (e.g., `current_kv_view_.k_data`).
    //      b. Calculate where V-data for this (token,layer) is in `current_kv_view_` (e.g., `current_kv_view_.v_data`).
    //      c. Copy K-data to `dst_page_ptr + offset_for_K_layer_N`.
    //      d. Copy V-data to `dst_page_ptr + offset_for_V_layer_N`.

    // This is highly complex due to the mismatch between ggml's contiguous tensor expectations for a batch
    // and the paged, potentially non-contiguous storage.
    // The current `current_kv_view_.k_data` (if it's a ggml_tensor) is likely a contiguous block for the whole ubatch.
    // We need to pick out the slice for each token and copy it to its page.

    size_t ubatch_start_token_idx = 0; // Needs to be calculated based on current_ubatch_idx_ and batch slicing logic
    size_t ubatch_end_token_idx = batch_ref_.n_tokens; // Needs to be calculated

    size_t per_token_all_layer_kv_bytes = cache_.get_kv_token_size_bytes() * cache_.get_n_layer();


    // This loop is conceptual for what apply would do if it had the computed K/V data.
    // In reality, `current_kv_view_.k_data` and `v_data` are usually set up by `llama.cpp`
    // to point to the *destination* in the KV cache *before* `ggml_graph_compute` is called.
    // So, `ggml_graph_compute` writes *directly* into the memory provided by `current_kv_view_`.
    // THUS, for a paged KV cache, `prepare_kv_view_for_ubatch` or `get_ubatch` is the critical part.
    // It must set up `current_kv_view_.k_data` and `v_data` (potentially as lists of pointers or using
    // ggml's upcoming support for scattered data access) to point to the correct locations in the pages.
    // `apply()` then might just be a metadata commit step, or do nothing if ggml wrote directly.

    // Given the current structure of llama_kv_cache_view, it expects contiguous k_data/v_data.
    // This implies we might need a temporary contiguous buffer for each ubatch.
    // 1. In `prepare_kv_view_for_ubatch` or `get_ubatch`:
    //    - Allocate temp contiguous buffers for K and V for the ubatch.
    //    - Set `current_kv_view_.k_data` and `v_data` to these temp buffers.
    // 2. `ggml_graph_compute` writes into these temp buffers.
    // 3. In `apply()`:
    //    - Iterate tokens in ubatch.
    //    - For each token, find/allocate its page(s).
    //    - Copy K/V data from the temp ubatch buffer to the respective page(s).
    // This is the "gather-compute-scatter" approach, which has performance overheads.

    // For now, let's assume `apply` is responsible for the "scatter" part.
    // This assumes `current_kv_view_` (specifically its k_data, v_data ggml_tensors)
    // holds the *computed* K and V values for the current ubatch, and these are contiguous.

    if (!batch_ref_.token || !current_kv_view_.k_data || !current_kv_view_.v_data) {
        // Not enough info to apply, or it's not an init_batch style state.
        // For init_full or init_update, apply might do other things.
        if (lctx_ref_ && optimize_){
            // Handle init_update specific logic if any.
            // e.g. compacting pages, etc.
        }
        std::cout << "llama_paged_kv_cache_state::apply() called (no batch data or not init_batch, or post-optimization cleanup)." << std::endl;
        return;
    }

    // Simplified scatter logic:
    // This requires knowing how tokens in batch_ref_ map to slices in current_kv_view_.k_data/v_data
    // Assume a direct 1:1 mapping for tokens in this ubatch.
    // And current_kv_view has k_data and v_data as flat arrays for the ubatch.

    // This part is extremely hand-wavy due to not knowing the exact structure of K/V in current_kv_view_
    // or how tokens are distributed into ubaches.
    // For each token `i` in the current ubatch:
    //   llama_seq_id seq_id = batch_ref_.seq_id[i][0]; // Assuming one seq_id per token for simplicity
    //   llama_pos    pos    = batch_ref_.pos[i];
    //   uint8_t* computed_k_for_token_i = (uint8_t*)current_kv_view_.k_data->data + offset_to_token_i_k_data;
    //   uint8_t* computed_v_for_token_i = (uint8_t*)current_kv_view_.v_data->data + offset_to_token_i_v_data;
    //
    //   PageOffset po = cache_.get_paged_cells().find_or_allocate_page_for_token(seq_id, pos);
    //   llama_kv_page* page = cache_.get_paged_cells().get_page(po.page_id);
    //   uint8_t* dest_ptr = page->data + po.offset; // Assuming po.offset is byte offset
    //   memcpy(dest_ptr, computed_k_for_token_i, size_of_k_for_one_token_all_layers);
    //   memcpy(dest_ptr + size_of_k_for_one_token_all_layers, computed_v_for_token_i, size_of_v_for_one_token_all_layers);

    std::cout << "llama_paged_kv_cache_state::apply() for ubatch " << current_ubatch_idx_ << " called. (Conceptual scatter)" << std::endl;
    status_ = LLAMA_MEMORY_STATUS_OK; // Or some "applied" status
}


const std::vector<llama_seq_id> & llama_paged_kv_cache_state::out_ids() const {
    return current_out_ids_;
}

llama_kv_cache_view llama_paged_kv_cache_state::get_ubatch() {
    // This method is supposed to return a view that ggml can use to *write* K/V data into.
    // As discussed in `apply()`, for a paged KV cache, this is the hard part if ggml expects contiguous memory.
    // If we use temporary contiguous buffers:
    //   1. Allocate/resize temp_k_buffer, temp_v_buffer for this ubatch's size.
    //   2. Set current_kv_view_.k_data and current_kv_view_.v_data (and their ggml_tensor wrappers)
    //      to point to these temp buffers.
    //   3. Return current_kv_view_.
    // `apply()` will then copy from these temp buffers to pages.

    // If ggml can handle scattered writes (e.g., via a list of pointers per token/layer):
    //   1. For each token in ubatch, for each layer:
    //      a. Find/allocate page.
    //      b. Store pointer `page->data + offset` into a list.
    //   2. Set `current_kv_view_` (or a modified version of it) to use these lists of pointers.
    //   3. Return this view. `apply()` might then be minimal (just metadata).

    // For now, returning the current_kv_view_ which was partially set up in prepare_kv_view_for_ubatch.
    // This is INCOMPLETE for actual computation.
    std::cout << "llama_paged_kv_cache_state::get_ubatch() for ubatch " << current_ubatch_idx_ << " called." << std::endl;
    if (status_ != LLAMA_MEMORY_STATUS_OK && status_ != LLAMA_MEMORY_STATUS_PARTIAL) {
        // Return an invalid view or handle error
        llama_kv_cache_view invalid_view = {0};
        return invalid_view;
    }
    return current_kv_view_;
}

llama_memory_status llama_paged_kv_cache_state::get_status() const {
    return status_;
}
