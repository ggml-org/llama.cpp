#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256 // TODO: Evaluate if this is suitable for page runs

// Default page size for paged allocators, e.g., 2MB. Can be made configurable.
// For now, KV cache related allocations would align to this.
#define GGML_ALLOCATOR_DEFAULT_PAGE_SIZE (2 * 1024 * 1024)

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)


static bool ggml_is_view(const struct ggml_tensor * t) {
    return t->view_src != NULL;
}

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

// ops that return true for this function must not use restrict pointers for their backend implementations
static bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;

        default:
            return false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    // TODO: Determine if this tallocr instance should be paged.
    // This might depend on the buffer type or a flag.
    // For now, assume not paged by default for ggml_tallocr.
    // If a specific ggml_tallocr needs to ensure its allocations are page-sized multiples,
    // a wrapper or a modified creation method would be needed.
    // Let's assume page_size_for_allocs = 0 means standard behavior.
    // A non-zero value would enforce page-multiple sizing.
    size_t page_size_for_allocs = 0; // Example: Could be passed or derived from buffer properties

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer               = */ buffer,
        /*.base                 = */ base,
        /*.alignment            = */ align,
        /*.offset               = */ aligned_offset(base, 0, align),
        /*.page_size_for_allocs = */ page_size_for_allocs, // if > 0, allocs are rounded up to this size
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t tensor_alloc_size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size_t effective_size = tensor_alloc_size;

    if (talloc->page_size_for_allocs > 0) {
        // If this allocator instance is designated to make page-sized allocations,
        // round up the tensor's size to the nearest multiple of that page size.
        effective_size = ((tensor_alloc_size + talloc->page_size_for_allocs - 1) / talloc->page_size_for_allocs) * talloc->page_size_for_allocs;
    }

    // Then, ensure the effective_size is padded to the base alignment requirement.
    effective_size = GGML_PAD(effective_size, talloc->alignment);

    if (talloc->offset + effective_size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (tensor size %zu, effective size %zu, needed %zu, available %zu)\n",
                __func__, tensor->name, tensor_alloc_size, effective_size, effective_size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += effective_size;

    assert(((uintptr_t)addr % talloc->alignment) == 0); // Base alignment check
    // If paged, and page_size_for_allocs is a multiple of alignment (usually is), this should hold.
    // Also, individual tensors within this effective_size block need to be aligned.
    // ggml_backend_tensor_alloc handles setting tensor->data, which should respect internal alignment needs
    // if the start `addr` is sufficiently aligned.

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

// Represents a run of free pages or a contiguous block of memory.
// When used for paged allocation, 'offset' can be page index and 'size' can be number of pages.
// For byte-based allocation, 'offset' is byte offset and 'size' is bytes.
struct free_block { // Renaming to free_run might be clearer if exclusively pages
    size_t offset; // Byte offset or page index
    size_t size;   // Size in bytes or number of pages
};

struct ggml_dyn_tallocr {
    size_t alignment; // For byte-based alignment of allocations
    int n_free_blocks; // Number of free runs/blocks
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    size_t max_size;   // Maximum size reached by this allocator (bytes)

    bool paged;      // If true, this allocator manages pages
    size_t page_size;  // Page size in bytes, valid if paged is true

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        size_t offset;
    } allocated_tensors[1024];
#endif
};

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, size_t offset, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].offset = offset;
            return;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, size_t offset, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].offset == offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static size_t ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size_t request_size = size; // Original requested size in bytes
    size_t alloc_unit_size; // The unit of allocation (bytes or pages)
    size_t num_alloc_units; // Number of allocation units (e.g. number of pages)

    if (alloc->paged) {
        // Align size to page size for page-based allocation
        // All allocations in paged mode are in multiples of page_size
        alloc_unit_size = alloc->page_size;
        num_alloc_units = (size + alloc->page_size - 1) / alloc->page_size;
        size = num_alloc_units * alloc->page_size; // Total bytes to be occupied by pages
        // The returned offset will be page-aligned by nature.
        // Individual tensor alignment within a page needs separate handling if tensors are smaller than pages.
        // For KV cache, tensors will likely be page-sized or occupy full pages.
    } else {
        // Byte-based allocation, ensure alignment
        size = aligned_offset(NULL, size, alloc->alignment);
        alloc_unit_size = 1; // Unit is bytes
        num_alloc_units = size;
    }

    AT_PRINTF("%s: allocating %s (requested %zu, effective %zu bytes, paged: %d) - ", __func__, tensor->name, request_size, size, alloc->paged);

    size_t max_avail = 0; // Max available units (bytes or pages)

    // find the best fitting free block
    int best_fit_block_idx = -1;
    size_t best_fit_block_size = SIZE_MAX; // In units (bytes or pages)

    for (int i = 0; i < alloc->n_free_blocks; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        max_avail = MAX(max_avail, block->size); // block->size is in units
        // block->size is num_pages if paged, or num_bytes if not paged
        // num_alloc_units is num_pages_needed if paged, or num_bytes_aligned if not paged
        if (block->size >= num_alloc_units && block->size <= best_fit_block_size) {
            best_fit_block_idx = i;
            best_fit_block_size = block->size;
        }
    }

    if (best_fit_block_idx == -1) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu %s for %s (requested %zu bytes), largest block available %zu %s\n",
                __func__, num_alloc_units, alloc->paged ? "pages" : "bytes",
                tensor->name, request_size,
                max_avail, alloc->paged ? "pages" : "bytes");
        GGML_ABORT("not enough space in the buffer");
    }

    struct free_block * block_to_alloc_from = &alloc->free_blocks[best_fit_block_idx];
    size_t actual_offset; // This will always be in bytes

    if (alloc->paged) {
        actual_offset = block_to_alloc_from->offset * alloc->page_size; // block_to_alloc_from->offset is page index
        block_to_alloc_from->offset += num_alloc_units; // Advance page index
    } else {
        actual_offset = block_to_alloc_from->offset; // block_to_alloc_from->offset is byte offset
        block_to_alloc_from->offset += num_alloc_units; // Advance byte offset (size = num_alloc_units here)
    }
    block_to_alloc_from->size -= num_alloc_units; // Reduce size in units (pages or bytes)

    if (block_to_alloc_from->size == 0) {
        // remove block if empty
        alloc->n_free_blocks--;
        for (int j = best_fit_block_idx; j < alloc->n_free_blocks; j++) {
            alloc->free_blocks[j] = alloc->free_blocks[j+1];
        }
    }

    AT_PRINTF("block %d, offset %zu (bytes)\n", best_fit_block_idx, actual_offset);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, offset, tensor);
    size_t cur_max = offset + size;
    if (cur_max > alloc->max_size) {
        // sort allocated_tensors by offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (alloc->allocated_tensors[i].offset > alloc->allocated_tensors[j].offset) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    size_t tmp_offset = alloc->allocated_tensors[i].offset;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].offset = alloc->allocated_tensors[j].offset;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].offset = tmp_offset;
                }
            }
        }
        GGML_LOG_DEBUG("max_size = %.2f MB: tensors: ", cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].offset,
                    alloc->allocated_tensors[i].offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    alloc->max_size = MAX(alloc->max_size, actual_offset + size); // size is effective_size in bytes

    return actual_offset; // Return byte offset

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_tensor(struct ggml_dyn_tallocr * alloc, size_t byte_offset, size_t original_size_bytes, const struct ggml_tensor * tensor) {
    size_t size_in_units; // bytes or pages
    size_t offset_in_units; // byte offset or page index

    if (alloc->paged) {
        size_in_units = (original_size_bytes + alloc->page_size - 1) / alloc->page_size;
        offset_in_units = byte_offset / alloc->page_size;
        GGML_ASSERT(byte_offset % alloc->page_size == 0); // Must be page aligned for paged allocator
    } else {
        size_in_units = aligned_offset(NULL, original_size_bytes, alloc->alignment);
        offset_in_units = byte_offset;
    }

    AT_PRINTF("%s: freeing %s at %zu (bytes), size %zu (bytes), %zu %s - n_free_blocks = %d\n",
        __func__, tensor->name, byte_offset, original_size_bytes, size_in_units, alloc->paged ? "pages" : "bytes", alloc->n_free_blocks);

#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, offset, tensor);
#endif

    // see if we can merge with an existing block (logic assumes sorted free_blocks by offset_in_units)
    for (int i = 0; i < alloc->n_free_blocks; i++) {
        struct free_block * block = &alloc->free_blocks[i];
        // check if freed block is adjacent to the end of the current free block
        if (block->offset + block->size == offset_in_units) {
            block->size += size_in_units;
            // check if we can merge with the next block
            if (i < alloc->n_free_blocks - 1 && block->offset + block->size == alloc->free_blocks[i+1].offset) {
                block->size += alloc->free_blocks[i+1].size;
                alloc->n_free_blocks--;
                for (int j = i+1; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
        // check if freed block is adjacent to the beginning of the current free block
        if (offset_in_units + size_in_units == block->offset) {
            block->offset = offset_in_units;
            block->size += size_in_units;
            // check if we can merge with the previous block
            if (i > 0 && alloc->free_blocks[i-1].offset + alloc->free_blocks[i-1].size == block->offset) {
                alloc->free_blocks[i-1].size += block->size;
                alloc->n_free_blocks--;
                for (int j = i; j < alloc->n_free_blocks; j++) {
                    alloc->free_blocks[j] = alloc->free_blocks[j+1];
                }
            }
            return;
        }
    }

    // otherwise, add a new block, keeping blocks sorted by offset_in_units
    GGML_ASSERT(alloc->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    int insert_pos = 0;
    while (insert_pos < alloc->n_free_blocks && alloc->free_blocks[insert_pos].offset < offset_in_units) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = alloc->n_free_blocks; i > insert_pos; i--) {
        alloc->free_blocks[i] = alloc->free_blocks[i-1];
    }
    // insert the new block
    alloc->free_blocks[insert_pos].offset = offset_in_units;
    alloc->free_blocks[insert_pos].size = size_in_units;
    alloc->n_free_blocks++;

    GGML_UNUSED(tensor);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    alloc->n_free_blocks = 1;
    // free_blocks[0].offset is 0 (either page index 0 or byte offset 0)
    // free_blocks[0].size is "all available space" in relevant units (pages or bytes)
    // For a measure allocator, total size is not known upfront.
    // It's set to a very large value and max_size tracks actual usage.
    // If paged, this initial size should represent max possible pages.
    // However, since total buffer size isn't known here, we use a large number.
    // The actual number of pages will be implicitly limited by max_size / page_size later.
    alloc->free_blocks[0].offset = 0; // Page index 0 or byte offset 0
    alloc->free_blocks[0].size = SIZE_MAX / (alloc->paged ? alloc->page_size : 1) / 2; // Max units (pages or bytes)
    alloc->max_size = 0; // Max bytes used

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

// Creates a new dynamic tensor allocator.
// Can be either byte-based or paged.
// For paged allocator, pass paged=true and page_size. Alignment is still used for base alignment.
// For byte-based allocator, pass paged=false, page_size is ignored (can be 0).
static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new_impl(size_t alignment, bool paged, size_t page_size_param) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment     = */ alignment,
        /*.n_free_blocks = */ 0,
        /*.free_blocks   = */ {{0}},
        /*.max_size      = */ 0,
        /*.paged         = */ paged,
        /*.page_size     = */ paged ? page_size_param : 1, // Ensure page_size is valid if paged
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };
    if (alloc->paged && alloc->page_size == 0) {
        GGML_LOG_WARN("%s: paged allocator created with page_size=0. Defaulting to %zu\n", __func__, (size_t)GGML_ALLOCATOR_DEFAULT_PAGE_SIZE);
        alloc->page_size = GGML_ALLOCATOR_DEFAULT_PAGE_SIZE;
    }


    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

// Public constructor for a standard (byte-based) dynamic allocator
struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment) {
    return ggml_dyn_tallocr_new_impl(alignment, false, 0);
}

// Public constructor for a paged dynamic allocator
// TODO: Expose this via header if needed, or make it an option in ggml_gallocr_new_n
// For now, it's internal, and ggml_gallocr can be modified to create one of these
// if a buffer_type indicates it needs paged allocation.
GGML_CALL struct ggml_dyn_tallocr * ggml_dyn_tallocr_new_paged(size_t alignment, size_t page_size) {
    return ggml_dyn_tallocr_new_impl(alignment, true, page_size == 0 ? GGML_ALLOCATOR_DEFAULT_PAGE_SIZE : page_size);
}


static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc) {
    return alloc->max_size;
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    size_t offset; // offset within the buffer
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    size_t offset;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ggml_backend_buffer_t * buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(ggml_backend_buffer_t));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
                // TODO: Here we need to decide if the buffer type `bufts[i]` implies paged allocation.
                // This requires extending ggml_backend_buffer_type_t or having a parallel flags array.
                // For now, assume non-paged by default.
                // If KV cache tensors are assigned to a specific buffer_id that IS paged, then
                // galloc->buf_tallocs[buffer_id] should be a paged allocator.
                // This implies ggml_gallocr_new_n might need more info about which bufts are paged.
                // Let's assume for now all are non-paged here.
                // The actual paged allocator would be created and used by llama.cpp's memory management for KV.
                // However, if ggml-alloc itself needs to manage paged KV tensors within a graph, this needs to change.
                // The subtask implies modifying ggml_dyn_tallocr for paged KV. So, if a KV tensor is part of a graph
                // and computed by ggml, its buffer (if not pre-allocated by CPU backend) needs this.

                // For now, to make progress, let's assume standard allocator here.
                // The paged variant `ggml_dyn_tallocr_new_paged` can be used explicitly where needed.
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_backend_buffer_free(galloc->buffers[i]);
            }
        }
        if (galloc->buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    return &galloc->hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return t->data != NULL || ggml_gallocr_hash_get(galloc, t)->allocated;
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_is_view(node)) {
        hn->allocated = true;
        assert(hn->offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->offset == p_hn->offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->offset = p_hn->offset;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->offset = p_hn->offset;
                        p_hn->allocated = false; // avoid freeing the parent
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        size_t offset = ggml_dyn_tallocr_alloc(alloc, size, node);
        hn->buffer_id = buffer_id;
        hn->offset = offset;
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    size_t offset = hn->offset;
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);
    ggml_dyn_tallocr_free_tensor(alloc, offset, size, node);
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        if (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.offset = SIZE_MAX;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.offset    = hn->offset;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].offset = SIZE_MAX;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].offset   = hn->offset;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.offset = SIZE_MAX;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.offset = hn->offset;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        size_t cur_size = galloc->buffers[i] ? ggml_backend_buffer_get_size(galloc->buffers[i]) : 0;
        size_t new_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i]);

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        if (new_size > cur_size || galloc->buffers[i] == NULL) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif

            ggml_backend_buffer_free(galloc->buffers[i]);
            galloc->buffers[i] = ggml_backend_buft_alloc_buffer(galloc->bufts[i], new_size);
            if (galloc->buffers[i] == NULL) {
                GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                return false;
            }
            ggml_backend_buffer_set_usage(galloc->buffers[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
    }

    return true;
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    int buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buffer_get_alloc_size(galloc->buffers[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->offset != SIZE_MAX);
            assert(ggml_backend_buffer_get_alloc_size(galloc->buffers[buffer_id], tensor) <= tensor_alloc->size_max);
            void * base = ggml_backend_buffer_get_base(galloc->buffers[buffer_id]);
            void * addr = (char *)base + tensor_alloc->offset;
            ggml_backend_tensor_alloc(galloc->buffers[buffer_id], tensor, addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    size_t node_size = 0;
    if (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                return true;
            }
        }
    }

    return false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_backend_buffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }

    for (int i = 0; i < buffer_id; i++) {
        if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            return 0;
        }
    }

    return ggml_backend_buffer_get_size(galloc->buffers[buffer_id]);
}

// utils

static void free_buffers(ggml_backend_buffer_t ** buffers, const size_t * n_buffers) {
    for (size_t i = 0; i < *n_buffers; i++) {
        ggml_backend_buffer_free((*buffers)[i]);
    }
    free(*buffers);
}

static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
        free_buffers(buffers, n_buffers);
        return false;
    }

    *buffers = realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
    (*buffers)[(*n_buffers)++] = buffer;

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                status = ggml_tallocr_alloc(&tallocr, t);
            } else if (t->buffer == NULL) {
                status = ggml_backend_view_init(t);
            }
        } else {
            if (t->view_src != NULL && t->buffer == NULL) {
                // view of a pre-allocated tensor
                status = ggml_backend_view_init(t);
            }
        }
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t->name);
            free_buffers(buffers, n_buffers);
            return false;
        }
    }

    return true;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
        return NULL;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    free(buffers);
    return buffer;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
