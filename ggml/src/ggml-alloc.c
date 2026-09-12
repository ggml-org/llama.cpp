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
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL:
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
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
        case GGML_OP_CLAMP:
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

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    if (talloc->buffer->binding) {
        return GGML_STATUS_FAILED;
    }
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
    int chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(struct buffer_address a, struct buffer_address b) {
    return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};

static void ggml_dyn_tallocr_insert_block(struct tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

static void ggml_dyn_tallocr_remove_block(struct tallocr_chunk * chunk, int idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    for (int i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

static int ggml_dyn_tallocr_new_chunk(struct ggml_dyn_tallocr * alloc, size_t min_size) {
    if (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        return -1;
    }
    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    if (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    return alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            return;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    if (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        for (int c = 0; c < alloc->n_chunks; ++c) {
            struct tallocr_chunk * chunk = alloc->chunks[c];
            if (chunk->n_free_blocks > 0) {
                struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                if (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    if (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    if (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block    * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    if (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    struct buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        free(alloc->chunks[i]);
        alloc->chunks[i] = NULL;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < alloc->n_chunks; ++i) {
        free(alloc->chunks[i]);
    }
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc, int chunk) {
    return chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

static void ggml_vbuffer_free(struct vbuffer * buf) {
    if (buf == NULL) {
        return;
    }
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}

static size_t ggml_vbuffer_chunk_size(struct vbuffer * buf, int chunk) {
    return buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t ggml_vbuffer_size(struct vbuffer * buf) {
    size_t size = 0;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    return size;
}

static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc, enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}

static enum ggml_status ggml_vbuffer_tensor_alloc(struct vbuffer * buf, struct ggml_tensor * tensor, struct buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    GGML_ASSERT(base || buf_addr.offset == 0);
    void * addr = base ? (char *)base + buf_addr.offset : NULL;
    return ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

static void ggml_vbuffer_reset(struct vbuffer * buf) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

struct ggml_gallocr_domain {
    ggml_backend_buffer_type_t buft;
    struct ggml_dyn_tallocr * alloc;
};

struct ggml_gallocr_graph;

struct ggml_gallocr_composite_plan {
    const struct ggml_backend_buffer_type_alloc_i * iface;
    struct ggml_gallocr_domain * domains;
    size_t n_domains;
    struct ggml_gallocr_graph * graph;
    void * preparation;
};

struct ggml_gallocr_tensor_requirement {
    struct tensor_alloc native;
    int buffer_id;
    int src[GGML_MAX_SRC];
    int view_src;
    size_t view_offs;
    size_t first_shard;
    size_t n_shards;
    bool external;
};

struct ggml_gallocr_shard_requirement {
    struct buffer_address addr;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    size_t size;
    size_t view_offs;
    int flags;
};

struct ggml_gallocr_plan {
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    struct ggml_gallocr_composite_plan ** composites; // [n_buffers], optional
    struct ggml_gallocr_tensor_requirement * tensors;
    struct ggml_gallocr_shard_requirement * shards;
    int * node_ids;
    int * leaf_ids;
    int * hash_ids;
    size_t n_tensors;
    size_t n_shards;
    bool valid;
    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;
};

struct ggml_gallocr_composite_storage {
    ggml_backend_buffer_t owner;
    struct ggml_backend_buffer_set * domains;
    bool released;
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    struct vbuffer ** buffers; // [n_buffers]
    struct ggml_gallocr_composite_storage ** composite_buffers;
    int n_buffers;
    struct ggml_gallocr_plan plan;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->plan.buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->plan.buf_tallocs != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->plan.buf_tallocs[i] = galloc->plan.buf_tallocs[j];
                break;
            }
        }

        if (galloc->plan.buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->plan.buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }

        const struct ggml_backend_buffer_type_alloc_i * iface = ggml_backend_buft_get_alloc_interface(bufts[i]);
        if (iface) {
            if (!galloc->plan.composites) {
                galloc->plan.composites = calloc(n_bufs, sizeof(galloc->plan.composites[0]));
                galloc->composite_buffers = calloc(n_bufs, sizeof(galloc->composite_buffers[0]));
                GGML_ASSERT(galloc->plan.composites && galloc->composite_buffers);
            }
            for (int j = 0; j < i; j++) {
                if (bufts[i] == bufts[j]) {
                    galloc->plan.composites[i] = galloc->plan.composites[j];
                    galloc->composite_buffers[i] = galloc->composite_buffers[j];
                    break;
                }
            }
            if (!galloc->plan.composites[i]) {
                struct ggml_gallocr_composite_plan * composite = calloc(1, sizeof(*composite));
                GGML_ASSERT(composite);
                composite->iface = iface;
                composite->n_domains = iface->n_domains(bufts[i]);
                GGML_ASSERT(composite->n_domains && composite->n_domains <= SIZE_MAX/sizeof(composite->domains[0]));
                composite->domains = calloc(composite->n_domains, sizeof(composite->domains[0]));
                GGML_ASSERT(composite->domains);
                for (size_t domain = 0; domain < composite->n_domains; domain++) {
                    struct ggml_gallocr_domain * physical = &composite->domains[domain];
                    physical->buft = iface->get_domain(bufts[i], domain);
                    GGML_ASSERT(!ggml_backend_buft_get_alloc_interface(physical->buft));
                    physical->alloc = ggml_dyn_tallocr_new(ggml_backend_buft_get_alignment(physical->buft), ggml_backend_buft_get_max_size(physical->buft));
                }
                galloc->plan.composites[i] = composite;
                struct ggml_gallocr_composite_storage * storage = calloc(1, sizeof(*storage));
                GGML_ASSERT(storage);
                storage->domains = calloc(composite->n_domains, sizeof(storage->domains[0]));
                GGML_ASSERT(storage->domains);
                galloc->composite_buffers[i] = storage;
            }
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
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        if (galloc->plan.buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->plan.buf_tallocs[j] == galloc->plan.buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_dyn_tallocr_free(galloc->plan.buf_tallocs[i]);
            }
        }
        if (galloc->plan.composites && galloc->plan.composites[i]) {
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->plan.composites[j] == galloc->plan.composites[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                struct ggml_gallocr_composite_plan * composite = galloc->plan.composites[i];
                struct ggml_gallocr_composite_storage * storage = galloc->composite_buffers[i];
                ggml_backend_buffer_free(storage->owner);
                for (size_t domain = 0; domain < composite->n_domains; domain++) {
                    ggml_backend_buffer_set_free(&storage->domains[domain]);
                    ggml_dyn_tallocr_free(composite->domains[domain].alloc);
                }
                free(storage->domains);
                free(storage);
                free(composite->domains);
                free(composite);
            }
        }
    }

    ggml_hash_set_free(&galloc->plan.hash_set);
    free(galloc->plan.hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->composite_buffers);
    free(galloc->plan.buf_tallocs);
    free(galloc->plan.composites);
    free(galloc->plan.tensors);
    free(galloc->plan.shards);
    free(galloc->plan.node_ids);
    free(galloc->plan.leaf_ids);
    free(galloc->plan.hash_ids);
    free(galloc->plan.node_allocs);
    free(galloc->plan.leaf_allocs);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->plan.hash_set, t);
    return &galloc->plan.hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_owns_buffer(ggml_gallocr_t galloc, ggml_backend_buffer_t buffer) {
    if (!buffer || !galloc->plan.composites) {
        return false;
    }
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->composite_buffers[i] && galloc->composite_buffers[i]->owner == buffer) {
            return true;
        }
        if (galloc->buffers[i]) {
            for (int c = 0; c < GGML_VBUFFER_MAX_CHUNKS; c++) {
                if (galloc->buffers[i]->chunks[c] == buffer) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool ggml_gallocr_is_external(ggml_gallocr_t galloc, const struct ggml_tensor * tensor) {
    if (ggml_gallocr_owns_buffer(galloc, tensor->buffer)) {
        return false;
    }
    if (!tensor->buffer && tensor->view_src && tensor->view_src->buffer &&
            ggml_backend_buft_get_alloc_interface(tensor->view_src->buffer->buft)) {
        return false;
    }
    return tensor->buffer || ggml_backend_tensor_is_bound(tensor);
}

static struct ggml_gallocr_tensor_requirement * ggml_gallocr_requirement(ggml_gallocr_t galloc, const struct ggml_tensor * tensor) {
    if (!galloc->plan.composites) {
        return NULL;
    }
    size_t slot = ggml_hash_find(&galloc->plan.hash_set, tensor);
    GGML_ASSERT(slot != GGML_HASHSET_FULL && galloc->plan.hash_ids[slot]);
    return &galloc->plan.tensors[galloc->plan.hash_ids[slot] - 1];
}

static bool ggml_gallocr_can_reuse_shards(ggml_gallocr_t galloc, const struct ggml_tensor * node, const struct ggml_tensor * parent) {
    struct ggml_gallocr_tensor_requirement * dst = ggml_gallocr_requirement(galloc, node);
    struct ggml_gallocr_tensor_requirement * src = ggml_gallocr_requirement(galloc, parent);
    if (!dst || (!dst->n_shards && !src->n_shards)) {
        return true;
    }
    if (!dst->n_shards || dst->n_shards != src->n_shards || src->view_src >= 0 ||
            galloc->plan.composites[dst->buffer_id] != galloc->plan.composites[src->buffer_id]) {
        return false;
    }
    for (size_t i = 0; i < dst->n_shards; i++) {
        struct ggml_gallocr_shard_requirement * a = &galloc->plan.shards[dst->first_shard + i];
        struct ggml_gallocr_shard_requirement * b = &galloc->plan.shards[src->first_shard + i];
        if (a->size > b->size || memcmp(a->ne, b->ne, sizeof(a->ne)) || memcmp(a->nb, b->nb, sizeof(a->nb))) {
            return false;
        }
    }
    return true;
}

static void ggml_gallocr_allocate_shards(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    struct ggml_gallocr_tensor_requirement * requirement = ggml_gallocr_requirement(galloc, node);
    if (!requirement || !requirement->n_shards) {
        return;
    }
    struct ggml_gallocr_tensor_requirement * source = parent ? ggml_gallocr_requirement(galloc, parent) : NULL;
    struct ggml_gallocr_composite_plan * composite = galloc->plan.composites[requirement->buffer_id];
    for (size_t i = 0; i < requirement->n_shards; i++) {
        struct ggml_gallocr_shard_requirement * shard = &galloc->plan.shards[requirement->first_shard + i];
        struct ggml_dyn_tallocr * alloc = composite->domains[i].alloc;
        if (source) {
            struct ggml_gallocr_shard_requirement * previous = &galloc->plan.shards[source->first_shard + i];
            shard->addr = previous->addr;
            size_t old_size = GGML_PAD(previous->size, alloc->alignment);
            size_t new_size = GGML_PAD(shard->size, alloc->alignment);
            if (old_size > new_size) {
                struct buffer_address tail = shard->addr;
                tail.offset += new_size;
                ggml_dyn_tallocr_free_bytes(alloc, tail, old_size - new_size);
            }
        } else {
            shard->addr = ggml_dyn_tallocr_alloc(alloc, shard->size, node);
        }
    }
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_is_external(galloc, t)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    struct ggml_dyn_tallocr * p_alloc = galloc->plan.buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

    if (parent_size > node_size) {
        struct buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    struct ggml_gallocr_tensor_requirement * requirement = ggml_gallocr_requirement(galloc, node);
    if (requirement) {
        buffer_id = requirement->buffer_id;
    }
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

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
                if (!ggml_gallocr_can_reuse_shards(galloc, node, parent)) {
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_impl_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && parent->view_offs == 0) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            ggml_gallocr_allocate_shards(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        ggml_gallocr_allocate_shards(galloc, node, parent);
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->plan.buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
        ggml_gallocr_allocate_shards(galloc, node, NULL);
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    struct ggml_gallocr_tensor_requirement * requirement = ggml_gallocr_requirement(galloc, node);
    if (requirement && requirement->n_shards) {
        struct ggml_gallocr_composite_plan * composite = galloc->plan.composites[requirement->buffer_id];
        for (size_t i = 0; i < requirement->n_shards; i++) {
            struct ggml_gallocr_shard_requirement * shard = &galloc->plan.shards[requirement->first_shard + i];
#ifdef GGML_ALLOCATOR_DEBUG
            remove_allocated_tensor(composite->domains[i].alloc, shard->addr, node);
#endif
            ggml_dyn_tallocr_free_bytes(composite->domains[i].alloc, shard->addr, shard->size);
        }
    }
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->plan.buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
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
        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
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
                if (ggml_impl_is_view(parent)) {
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

struct ggml_gallocr_graph {
    ggml_gallocr_t galloc;
    struct ggml_tensor ** tensors;
    int * hash_ids;
    uint8_t * state;
    bool * declared;
    size_t count;
};

static int ggml_gallocr_graph_tensor_id(struct ggml_gallocr_graph * graph, struct ggml_tensor * tensor, int buffer_id) {
    if (buffer_id < 0 || buffer_id >= graph->galloc->n_buffers) {
        return -1;
    }
    struct ggml_hash_set * hash = &graph->galloc->plan.hash_set;
    size_t slot = ggml_hash_find(hash, tensor);
    if (slot == GGML_HASHSET_FULL) {
        return -1;
    }
    if (!graph->hash_ids[slot]) {
        GGML_ASSERT(graph->count < hash->size && graph->count < INT_MAX);
        ggml_hash_find_or_insert(hash, tensor);
        int id = (int) graph->count++;
        graph->hash_ids[slot] = id + 1;
        graph->tensors[id] = tensor;
        graph->galloc->plan.tensors[id].buffer_id = buffer_id;
    }
    return graph->hash_ids[slot] - 1;
}

static ggml_backend_buffer_type_t ggml_gallocr_source_buft(void * context, const struct ggml_tensor * tensor, enum ggml_backend_buffer_usage * usage) {
    struct ggml_gallocr_composite_plan * composite = context;
    struct ggml_gallocr_graph * graph = composite->graph;
    size_t slot = ggml_hash_find(&graph->galloc->plan.hash_set, tensor);
    GGML_ASSERT(slot != GGML_HASHSET_FULL && graph->hash_ids[slot]);
    int id = graph->hash_ids[slot] - 1;
    *usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    return graph->galloc->bufts[graph->galloc->plan.tensors[id].buffer_id];
}

static struct ggml_tensor * ggml_gallocr_source_tensor(void * context, const struct ggml_tensor * tensor, size_t device) {
    struct ggml_gallocr_composite_plan * composite = context;
    struct ggml_gallocr_graph * graph = composite->graph;
    size_t slot = ggml_hash_find(&graph->galloc->plan.hash_set, tensor);
    GGML_ASSERT(slot != GGML_HASHSET_FULL && graph->hash_ids[slot]);
    int id = graph->hash_ids[slot] - 1;
    int buffer_id = graph->galloc->plan.tensors[id].buffer_id;
    struct ggml_gallocr_composite_plan * source = graph->galloc->plan.composites[buffer_id];
    if (!source || source == composite || !source->preparation || device >= source->n_domains) {
        return NULL;
    }
    return source->iface->get_tensor(source->preparation, tensor, device);
}

static bool ggml_gallocr_source_replaced(void * context, ggml_backend_buffer_t buffer) {
    struct ggml_gallocr_composite_plan * composite = context;
    return ggml_gallocr_owns_buffer(composite->graph->galloc, buffer);
}

static void ggml_gallocr_graph_free(struct ggml_gallocr_graph * graph) {
    struct ggml_gallocr_plan * plan = &graph->galloc->plan;
    for (int i = 0; i < graph->galloc->n_buffers; i++) {
        struct ggml_gallocr_composite_plan * composite = plan->composites[i];
        if (composite) {
            if (composite->preparation) {
                composite->iface->free_preparation(composite->preparation);
            }
            composite->preparation = NULL;
            composite->graph = NULL;
        }
    }
    free(graph->tensors);
    free(graph->state);
    free(graph->declared);
    ggml_hash_set_reset(&plan->hash_set);
    memset(plan->hash_set.keys, 0, plan->hash_set.size*sizeof(plan->hash_set.keys[0]));
    memset(graph, 0, sizeof(*graph));
}

static bool ggml_gallocr_prepare_source(struct ggml_gallocr_composite_plan * composite, const struct ggml_tensor * tensor) {
    if (!tensor || composite->iface->get_tensor(composite->preparation, tensor, 0)) {
        return true;
    }
    if (!ggml_gallocr_prepare_source(composite, tensor->view_src)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != tensor && !ggml_gallocr_prepare_source(composite, tensor->src[i])) {
            return false;
        }
    }
    return composite->iface->prepare_tensor(composite->preparation, tensor) == GGML_STATUS_SUCCESS;
}

static bool ggml_gallocr_prepare_graph_tensor(struct ggml_gallocr_graph * graph, int id) {
    if (graph->state[id]) {
        return graph->state[id] == 2;
    }
    graph->state[id] = 1;
    struct ggml_tensor * tensor = graph->tensors[id];
    struct ggml_gallocr_tensor_requirement * requirement = &graph->galloc->plan.tensors[id];
    requirement->external = ggml_gallocr_is_external(graph->galloc, tensor);
    requirement->view_offs = tensor->view_offs;
    requirement->view_src = -1;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        requirement->src[i] = -1;
    }
    if (graph->declared[id] || !requirement->external || !ggml_backend_tensor_is_bound(tensor)) {
        if (tensor->view_src) {
            requirement->view_src = ggml_gallocr_graph_tensor_id(graph, tensor->view_src, requirement->buffer_id);
            if (requirement->view_src < 0 || !ggml_gallocr_prepare_graph_tensor(graph, requirement->view_src)) {
                return false;
            }
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (!tensor->src[i]) {
                continue;
            }
            requirement->src[i] = ggml_gallocr_graph_tensor_id(graph, tensor->src[i], requirement->buffer_id);
            if (requirement->src[i] < 0 || (tensor->src[i] != tensor && !ggml_gallocr_prepare_graph_tensor(graph, requirement->src[i]))) {
                return false;
            }
        }
    }
    struct ggml_gallocr_composite_plan * composite = graph->galloc->plan.composites[requirement->buffer_id];
    if (composite && !ggml_gallocr_prepare_source(composite, tensor)) {
        return false;
    }
    graph->state[id] = 2;
    return true;
}

static bool ggml_gallocr_prepare_graph(ggml_gallocr_t galloc, struct ggml_cgraph * cgraph, const int * node_buffer_ids, const int * leaf_buffer_ids, struct ggml_gallocr_graph * retained) {
    struct ggml_gallocr_plan * plan = &galloc->plan;
    size_t capacity = plan->hash_set.size;
    struct ggml_gallocr_graph graph = { .galloc = galloc };
    graph.tensors = calloc(capacity, sizeof(graph.tensors[0]));
    free(plan->hash_ids);
    plan->hash_ids = calloc(capacity, sizeof(plan->hash_ids[0]));
    graph.hash_ids = plan->hash_ids;
    graph.state = calloc(capacity, sizeof(graph.state[0]));
    graph.declared = calloc(capacity, sizeof(graph.declared[0]));
    GGML_ASSERT(graph.tensors && graph.hash_ids && graph.state && graph.declared);
    free(plan->tensors);
    free(plan->shards);
    free(plan->node_ids);
    free(plan->leaf_ids);
    plan->tensors = calloc(capacity, sizeof(plan->tensors[0]));
    plan->shards = NULL;
    plan->node_ids = calloc(MAX(1, cgraph->n_nodes), sizeof(plan->node_ids[0]));
    plan->leaf_ids = calloc(MAX(1, cgraph->n_leafs), sizeof(plan->leaf_ids[0]));
    plan->n_tensors = plan->n_shards = 0;
    GGML_ASSERT(plan->tensors && plan->node_ids && plan->leaf_ids);
    bool result = false;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        int buffer_id = get_node_buffer_id(leaf_buffer_ids, i);
        int id = ggml_gallocr_graph_tensor_id(&graph, cgraph->leafs[i], buffer_id);
        if (id < 0) {
            goto cleanup;
        }
        plan->leaf_ids[i] = id;
        plan->tensors[id].buffer_id = buffer_id;
        graph.declared[id] = true;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);
        int id = ggml_gallocr_graph_tensor_id(&graph, cgraph->nodes[i], buffer_id);
        if (id < 0) {
            goto cleanup;
        }
        plan->node_ids[i] = id;
        plan->tensors[id].buffer_id = buffer_id;
        graph.declared[id] = true;
    }
    for (int i = 0; i < galloc->n_buffers; i++) {
        struct ggml_gallocr_composite_plan * composite = plan->composites[i];
        if (!composite || composite->preparation) {
            continue;
        }
        struct ggml_backend_alloc_source_i sources = {ggml_gallocr_source_buft, ggml_gallocr_source_tensor, composite, NULL, ggml_gallocr_source_replaced};
        composite->graph = &graph;
        composite->preparation = composite->iface->new_preparation(galloc->bufts[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE, capacity, &sources);
        if (!composite->preparation) {
            goto cleanup;
        }
    }
    for (size_t i = 0; i < graph.count; i++) {
        if (!ggml_gallocr_prepare_graph_tensor(&graph, (int) i)) {
            goto cleanup;
        }
    }
    for (size_t i = 0; i < graph.count; i++) {
        struct ggml_gallocr_tensor_requirement * requirement = &plan->tensors[i];
        struct ggml_gallocr_composite_plan * composite = plan->composites[requirement->buffer_id];
        requirement->first_shard = plan->n_shards;
        requirement->n_shards = composite ? composite->n_domains : 0;
        if (requirement->n_shards > SIZE_MAX/sizeof(plan->shards[0]) - plan->n_shards) {
            goto cleanup;
        }
        plan->n_shards += requirement->n_shards;
    }
    plan->shards = calloc(MAX((size_t) 1, plan->n_shards), sizeof(plan->shards[0]));
    GGML_ASSERT(plan->shards);
    for (size_t i = 0; i < graph.count; i++) {
        struct ggml_gallocr_tensor_requirement * requirement = &plan->tensors[i];
        struct ggml_gallocr_composite_plan * composite = plan->composites[requirement->buffer_id];
        for (size_t device = 0; device < requirement->n_shards; device++) {
            struct ggml_tensor * simple = composite->iface->get_tensor(composite->preparation, graph.tensors[i], device);
            if (!simple) {
                goto cleanup;
            }
            struct ggml_gallocr_shard_requirement * shard = &plan->shards[requirement->first_shard + device];
            shard->addr = GGML_BUFFER_ADDRESS_INVALID;
            memcpy(shard->ne, simple->ne, sizeof(shard->ne));
            memcpy(shard->nb, simple->nb, sizeof(shard->nb));
            shard->view_offs = simple->view_offs;
            shard->flags = simple->flags;
            shard->size = simple->view_src ? 0 : ggml_backend_buft_get_alloc_size(composite->domains[device].buft, simple);
        }
    }
    plan->n_tensors = graph.count;
    result = true;
cleanup:
    if (result && retained) {
        *retained = graph;
        for (int i = 0; i < galloc->n_buffers; i++) {
            if (plan->composites[i]) {
                plan->composites[i]->graph = retained;
            }
        }
    } else {
        // The integer hash map is needed by the lifetime traversal after measurement preparation.
        for (int i = 0; i < galloc->n_buffers; i++) {
            struct ggml_gallocr_composite_plan * composite = plan->composites[i];
            if (composite) {
                if (composite->preparation) {
                    composite->iface->free_preparation(composite->preparation);
                }
                composite->preparation = NULL;
                composite->graph = NULL;
            }
        }
        free(graph.tensors);
        free(graph.state);
        free(graph.declared);
    }
    if (!result) {
        plan->n_tensors = plan->n_shards = 0;
    }
    return result;
}

static bool ggml_gallocr_reserve_plan(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, struct ggml_gallocr_graph * retained) {
    size_t min_hash_size = MAX((size_t) 1, (size_t) graph->n_nodes + graph->n_leafs);
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;
    if (galloc->plan.composites) {
        min_hash_size = MAX(min_hash_size, 2*((size_t) graph->n_nodes + graph->n_leafs) + GGML_MAX_SRC);
    }

    // initialize hash table
    if (galloc->plan.hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->plan.hash_set);
        galloc->plan.hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->plan.hash_set.keys != NULL);

        free(galloc->plan.hash_values);
        galloc->plan.hash_values = malloc(sizeof(struct hash_node) * galloc->plan.hash_set.size);
        GGML_ASSERT(galloc->plan.hash_values != NULL);
    }

    ggml_hash_set_reset(&galloc->plan.hash_set);
    memset(galloc->plan.hash_values, 0, sizeof(struct hash_node) * galloc->plan.hash_set.size);
    galloc->plan.valid = false;
    if (galloc->plan.composites && !ggml_gallocr_prepare_graph(galloc, graph, node_buffer_ids, leaf_buffer_ids, retained)) {
        ggml_hash_set_reset(&galloc->plan.hash_set);
        memset(galloc->plan.hash_set.keys, 0, galloc->plan.hash_set.size*sizeof(galloc->plan.hash_set.keys[0]));
        return false;
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->plan.buf_tallocs[i]);
        struct ggml_gallocr_composite_plan * composite = galloc->plan.composites ? galloc->plan.composites[i] : NULL;
        if (composite) {
            for (size_t domain = 0; domain < composite->n_domains; domain++) {
                ggml_dyn_tallocr_reset(composite->domains[domain].alloc);
            }
        }
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    if (galloc->plan.n_nodes < graph->n_nodes) {
        free(galloc->plan.node_allocs);
        galloc->plan.node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->plan.node_allocs != NULL);
    }
    galloc->plan.n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->plan.node_allocs[i];
        if (node->view_src || ggml_gallocr_is_external(galloc, node)) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || ggml_gallocr_is_external(galloc, src)) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->plan.n_leafs < graph->n_leafs) {
        free(galloc->plan.leaf_allocs);
        galloc->plan.leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->plan.leaf_allocs[0]));
        GGML_ASSERT(galloc->plan.leaf_allocs != NULL);
    }
    galloc->plan.n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || ggml_gallocr_is_external(galloc, leaf)) {
            galloc->plan.leaf_allocs[i].leaf.buffer_id = -1;
            galloc->plan.leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->plan.leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->plan.leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->plan.leaf_allocs[i].leaf.addr = hn->addr;
            galloc->plan.leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    if (galloc->plan.composites) {
        for (size_t slot = 0; slot < galloc->plan.hash_set.size; slot++) {
            if (!galloc->plan.hash_ids[slot]) {
                continue;
            }
            struct ggml_gallocr_tensor_requirement * requirement = &galloc->plan.tensors[galloc->plan.hash_ids[slot] - 1];
            struct hash_node * node = &galloc->plan.hash_values[slot];
            requirement->native.buffer_id = requirement->buffer_id;
            requirement->native.addr = node->addr;
            requirement->native.size_max = requirement->n_shards || requirement->external || requirement->view_src >= 0 ? 0 :
                ggml_backend_buft_get_alloc_size(galloc->bufts[requirement->buffer_id], galloc->plan.hash_set.keys[slot]);
        }
    }
    if (!retained) {
        ggml_hash_set_reset(&galloc->plan.hash_set);
        memset(galloc->plan.hash_set.keys, 0, galloc->plan.hash_set.size*sizeof(galloc->plan.hash_set.keys[0]));
    }
#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < galloc->n_buffers; i++) {
        memset(galloc->plan.buf_tallocs[i]->allocated_tensors, 0, sizeof(galloc->plan.buf_tallocs[i]->allocated_tensors));
        struct ggml_gallocr_composite_plan * composite = galloc->plan.composites ? galloc->plan.composites[i] : NULL;
        if (composite) {
            for (size_t domain = 0; domain < composite->n_domains; domain++) {
                memset(composite->domains[domain].alloc->allocated_tensors, 0, sizeof(composite->domains[domain].alloc->allocated_tensors));
            }
        }
    }
#endif
    galloc->plan.valid = true;
    return true;
}

static bool ggml_gallocr_reserve_composite(ggml_gallocr_t galloc, int buffer_id) {
    struct ggml_gallocr_composite_plan * composite = galloc->plan.composites[buffer_id];
    struct ggml_gallocr_composite_storage * storage = galloc->composite_buffers[buffer_id];
    if (storage->owner && !storage->released) {
        if (composite->iface->release_buffers(storage->owner, storage->domains, composite->n_domains) != GGML_STATUS_SUCCESS) {
            return false;
        }
        storage->released = true;
    }
    for (size_t device = 0; device < composite->n_domains; device++) {
        struct ggml_gallocr_domain * domain = &composite->domains[device];
        struct ggml_backend_buffer_set * buffers = &storage->domains[device];
        size_t count = MAX(buffers->n_buffers, (size_t) domain->alloc->n_chunks);
        if (count > buffers->n_buffers) {
            ggml_backend_buffer_t * next = realloc(buffers->buffers, count*sizeof(next[0]));
            if (!next) {
                return false;
            }
            memset(next + buffers->n_buffers, 0, (count - buffers->n_buffers)*sizeof(next[0]));
            buffers->buffers = next;
            buffers->n_buffers = count;
        }
        for (int chunk = 0; chunk < domain->alloc->n_chunks; chunk++) {
            size_t size = ggml_dyn_tallocr_max_size(domain->alloc, chunk);
            if (!buffers->buffers[chunk] || buffers->buffers[chunk]->size < size) {
                ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(domain->buft, size);
                if (!buffer) {
                    return false;
                }
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                ggml_backend_buffer_free(buffers->buffers[chunk]);
                buffers->buffers[chunk] = buffer;
            }
        }
        while (buffers->n_buffers && !buffers->buffers[buffers->n_buffers - 1]) {
            buffers->n_buffers--;
        }
    }
    return true;
}

static bool ggml_gallocr_reserve_buffers(ggml_gallocr_t galloc) {
    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->plan.composites && galloc->plan.composites[i]) {
            bool shared = false;
            for (int j = 0; j < i; j++) {
                shared |= galloc->plan.composites[j] == galloc->plan.composites[i];
            }
            if (!shared && !ggml_gallocr_reserve_composite(galloc, i)) {
                return false;
            }
            continue;
        }
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->plan.buf_tallocs[j] == galloc->plan.buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;
        for (int c = 0; c < galloc->plan.buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->plan.buf_tallocs[i], c);
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        if (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                if (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->plan.buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            if (galloc->buffers[i] == NULL) {
                GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                return false;
            }
        }
    }

    return true;
}

static bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, bool no_alloc) {
    if (!ggml_gallocr_reserve_plan(galloc, graph, node_buffer_ids, leaf_buffer_ids, NULL)) {
        return false;
    }
    return no_alloc || ggml_gallocr_reserve_buffers(galloc);
}

void ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    for (int i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        bool shared = false;
        for (int j = 0; j < i; j++) {
            shared |= galloc->bufts[j] == galloc->bufts[i];
        }
        if (shared) {
            continue;
        }
        struct ggml_gallocr_composite_plan * composite = galloc->plan.composites ? galloc->plan.composites[i] : NULL;
        size_t n_domains = composite ? composite->n_domains : 1;
        for (size_t device = 0; device < n_domains; device++) {
            struct ggml_dyn_tallocr * alloc = composite ? composite->domains[device].alloc : galloc->plan.buf_tallocs[i];
            for (int c = 0; c < alloc->n_chunks; c++) {
                sizes[i] += ggml_dyn_tallocr_max_size(alloc, c);
            }
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static enum ggml_status ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    if (tensor->view_src) {
        if (tensor->buffer || !tensor->view_src->buffer) {
            return ggml_backend_tensor_is_bound(tensor) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
        if (!ggml_backend_tensor_is_bound(tensor->view_src)) {
            return GGML_STATUS_FAILED;
        }
        return ggml_backend_view_init(tensor);
    }
    if (ggml_backend_tensor_is_bound(tensor)) {
        return GGML_STATUS_SUCCESS;
    }
    int buffer_id = tensor_alloc->buffer_id;
    if (tensor->buffer || buffer_id < 0 || buffer_id >= galloc->n_buffers || !galloc->buffers[buffer_id]) {
        return GGML_STATUS_FAILED;
    }
    GGML_ASSERT(tensor_alloc->addr.offset != SIZE_MAX);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
    return ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    size_t node_size = 0;
    if (!ggml_gallocr_is_external(galloc, node) && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (!galloc->plan.valid) {
        return true;
    }
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (!galloc->buffers[i]) {
            return true;
        }
    }

    if (galloc->plan.n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->plan.n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->plan.node_allocs[i];

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

static bool ggml_gallocr_bind_graph_tensor(struct ggml_gallocr_graph * graph, int id, bool publish) {
    if (graph->state[id]) {
        return graph->state[id] == 2;
    }
    graph->state[id] = 1;
    ggml_gallocr_t galloc = graph->galloc;
    struct ggml_gallocr_tensor_requirement * requirement = &galloc->plan.tensors[id];
    struct ggml_tensor * tensor = graph->tensors[id];
    if (requirement->external) {
        if (!ggml_backend_tensor_is_bound(tensor)) {
            return false;
        }
        graph->state[id] = 2;
        return true;
    }
    if (requirement->view_src >= 0 && !ggml_gallocr_bind_graph_tensor(graph, requirement->view_src, publish)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        int src = requirement->src[i];
        if (src >= 0 && src != id && !ggml_gallocr_bind_graph_tensor(graph, src, publish)) {
            return false;
        }
    }
    int buffer_id = requirement->buffer_id;
    struct ggml_gallocr_composite_plan * composite = galloc->plan.composites[buffer_id];
    if (publish) {
        if (composite) {
            tensor->buffer = NULL;
            tensor->data = NULL;
            ggml_backend_buffer_t owner = galloc->composite_buffers[buffer_id]->owner;
            if (ggml_backend_buffer_init_tensor(owner, tensor) != GGML_STATUS_SUCCESS) {
                return false;
            }
        }
    } else if (composite) {
        for (size_t device = 0; device < composite->n_domains; device++) {
            struct ggml_tensor * simple = composite->iface->get_tensor(composite->preparation, tensor, device);
            struct ggml_gallocr_shard_requirement * shard = &galloc->plan.shards[requirement->first_shard + device];
            if (!simple) {
                return false;
            }
            if (simple->view_src) {
                if (!ggml_backend_tensor_is_bound(simple->view_src) || ggml_backend_view_init(simple) != GGML_STATUS_SUCCESS) {
                    return false;
                }
            } else {
                struct ggml_backend_buffer_set * buffers = &galloc->composite_buffers[buffer_id]->domains[device];
                if (shard->addr.chunk < 0 || (size_t) shard->addr.chunk >= buffers->n_buffers) {
                    return false;
                }
                ggml_backend_buffer_t buffer = buffers->buffers[shard->addr.chunk];
                void * base = ggml_backend_buffer_get_base(buffer);
                if ((!base && shard->addr.offset) || shard->addr.offset > buffer->size || shard->size > buffer->size - shard->addr.offset) {
                    return false;
                }
                void * address = base ? (char *) base + shard->addr.offset : NULL;
                if (ggml_backend_tensor_alloc(buffer, simple, address) != GGML_STATUS_SUCCESS) {
                    return false;
                }
            }
        }
    } else {
        tensor->buffer = NULL;
        tensor->data = NULL;
        if (ggml_gallocr_init_tensor(galloc, tensor, &requirement->native) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    graph->state[id] = 2;
    return true;
}

static bool ggml_gallocr_materialize_graph(struct ggml_gallocr_graph * graph) {
    ggml_gallocr_t galloc = graph->galloc;
    ggml_backend_buffer_t * owners = calloc(galloc->n_buffers, sizeof(owners[0]));
    if (!owners) {
        return false;
    }
    bool result = false;
    memset(graph->state, 0, graph->count*sizeof(graph->state[0]));
    for (size_t i = 0; i < graph->count; i++) {
        if (!ggml_gallocr_bind_graph_tensor(graph, (int) i, false)) {
            goto cleanup;
        }
    }
    for (int i = 0; i < galloc->n_buffers; i++) {
        struct ggml_gallocr_composite_plan * composite = galloc->plan.composites[i];
        if (!composite || !composite->preparation) {
            continue;
        }
        owners[i] = composite->iface->materialize(composite->preparation, galloc->composite_buffers[i]->domains, composite->n_domains);
        if (!owners[i]) {
            goto cleanup;
        }
        composite->preparation = NULL;
    }
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (owners[i]) {
            struct ggml_gallocr_composite_storage * storage = galloc->composite_buffers[i];
            ggml_backend_buffer_t previous = storage->owner;
            storage->owner = owners[i];
            storage->released = false;
            owners[i] = previous;
        }
    }
    memset(graph->state, 0, graph->count*sizeof(graph->state[0]));
    for (size_t i = 0; i < graph->count; i++) {
        if (!ggml_gallocr_bind_graph_tensor(graph, (int) i, true)) {
            for (int j = 0; j < galloc->n_buffers; j++) {
                if (galloc->composite_buffers[j]) {
                    ggml_backend_buffer_reset(galloc->composite_buffers[j]->owner);
                }
            }
            goto cleanup;
        }
    }
    result = true;
cleanup:
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_backend_buffer_free(owners[i]);
    }
    free(owners);
    return result;
}

static bool ggml_gallocr_alloc_composite_graph(ggml_gallocr_t galloc, struct ggml_cgraph * cgraph) {
    if (galloc->n_buffers > 1 && (!galloc->plan.valid || galloc->plan.n_nodes != cgraph->n_nodes || galloc->plan.n_leafs != cgraph->n_leafs)) {
        return false;
    }
    int * node_ids = calloc(MAX(1, cgraph->n_nodes), sizeof(node_ids[0]));
    int * leaf_ids = calloc(MAX(1, cgraph->n_leafs), sizeof(leaf_ids[0]));
    GGML_ASSERT(node_ids && leaf_ids);
    if (galloc->n_buffers > 1) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            node_ids[i] = galloc->plan.tensors[galloc->plan.node_ids[i]].buffer_id;
        }
        for (int i = 0; i < cgraph->n_leafs; i++) {
            leaf_ids[i] = galloc->plan.tensors[galloc->plan.leaf_ids[i]].buffer_id;
        }
    }
    struct ggml_gallocr_graph graph = {0};
    bool result = ggml_gallocr_reserve_plan(galloc, cgraph, node_ids, leaf_ids, &graph);
    free(node_ids);
    free(leaf_ids);
    if (!result) {
        return false;
    }
    // Reject assignment-only external tensors before retiring current bindings.
    for (size_t i = 0; i < graph.count; i++) {
        if (galloc->plan.tensors[i].external && !ggml_backend_tensor_is_bound(graph.tensors[i])) {
            ggml_gallocr_graph_free(&graph);
            return false;
        }
    }
    if (ggml_gallocr_reserve_buffers(galloc)) {
        for (int i = 0; i < galloc->n_buffers; i++) {
            if (galloc->buffers[i]) {
                ggml_vbuffer_reset(galloc->buffers[i]);
            }
        }
        result = ggml_gallocr_materialize_graph(&graph);
    } else {
        result = false;
    }
    if (!result) {
        for (int i = 0; i < galloc->n_buffers; i++) {
            struct ggml_gallocr_composite_storage * storage = galloc->composite_buffers[i];
            if (!storage) {
                continue;
            }
            for (size_t device = 0; device < galloc->plan.composites[i]->n_domains; device++) {
                struct ggml_backend_buffer_set * buffers = &storage->domains[device];
                for (size_t chunk = 0; chunk < buffers->n_buffers; chunk++) {
                    if (buffers->buffers[chunk]) {
                        ggml_backend_buffer_reset(buffers->buffers[chunk]);
                    }
                }
            }
        }
    }
    ggml_gallocr_graph_free(&graph);
    return result;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->plan.composites) {
        return ggml_gallocr_alloc_composite_graph(galloc, graph);
    }
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
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->plan.leaf_allocs[i];
        if (ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->plan.node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]) != GGML_STATUS_SUCCESS) {
                return false;
            }
        }
        if (ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst) != GGML_STATUS_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool ggml_gallocr_has_active_bindings(ggml_gallocr_t galloc) {
    for (int i = 0; galloc->composite_buffers && i < galloc->n_buffers; i++) {
        struct ggml_gallocr_composite_storage * storage = galloc->composite_buffers[i];
        if (storage && storage->owner && !storage->released) {
            return true;
        }
    }
    return false;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->composite_buffers && galloc->composite_buffers[buffer_id]) {
        struct ggml_gallocr_composite_storage * storage = galloc->composite_buffers[buffer_id];
        for (int i = 0; i < buffer_id; i++) {
            if (galloc->composite_buffers[i] == storage) {
                return 0;
            }
        }
        size_t size = storage->owner ? storage->owner->size : 0;
        for (size_t device = 0; device < galloc->plan.composites[buffer_id]->n_domains; device++) {
            struct ggml_backend_buffer_set * buffers = &storage->domains[device];
            for (size_t i = 0; i < buffers->n_buffers; i++) {
                if (buffers->buffers[i]) {
                    size += buffers->buffers[i]->size;
                }
            }
        }
        return size;
    }

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

    return ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

struct ggml_gallocr_domain_plan_info ggml_gallocr_get_domain_plan_info(ggml_gallocr_t galloc, int buffer_id, size_t domain) {
    GGML_ASSERT(galloc->plan.valid && buffer_id >= 0 && buffer_id < galloc->n_buffers);
    struct ggml_gallocr_composite_plan * composite = galloc->plan.composites ? galloc->plan.composites[buffer_id] : NULL;
    GGML_ASSERT(composite ? domain < composite->n_domains : domain == 0);
    struct ggml_dyn_tallocr * alloc = composite ? composite->domains[domain].alloc : galloc->plan.buf_tallocs[buffer_id];
    struct ggml_gallocr_domain_plan_info result = {0, (size_t) alloc->n_chunks};
    for (int i = 0; i < alloc->n_chunks; i++) {
        result.size += ggml_dyn_tallocr_max_size(alloc, i);
    }
    return result;
}

// utils

void ggml_backend_buffer_set_free(struct ggml_backend_buffer_set * buffers) {
    for (size_t i = 0; i < buffers->n_buffers; i++) {
        ggml_backend_buffer_free(buffers->buffers[i]);
    }
    free(buffers->buffers);
    buffers->buffers = NULL;
    buffers->n_buffers = 0;
}

static enum ggml_status alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        struct ggml_backend_buffer_set * buffers) {

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
        ggml_backend_buffer_set_free(buffers);
        return GGML_STATUS_ALLOC_FAILED;
    }

    ggml_backend_buffer_t * next = NULL;
    if (buffers->n_buffers < SIZE_MAX / sizeof(ggml_backend_buffer_t)) {
        next = realloc(buffers->buffers, sizeof(ggml_backend_buffer_t) * (buffers->n_buffers + 1));
    }
    if (next == NULL) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_buffer_set_free(buffers);
        return GGML_STATUS_ALLOC_FAILED;
    }
    buffers->buffers = next;
    buffers->buffers[buffers->n_buffers++] = buffer;

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (!ggml_backend_tensor_is_bound(t)) {
            if (t->view_src == NULL) {
                status = t->buffer ? GGML_STATUS_FAILED : ggml_tallocr_alloc(&tallocr, t);
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
            ggml_backend_buffer_set_free(buffers);
            return status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_alloc_ctx_tensors_from_buft_impl(
        struct ggml_context * ctx, ggml_backend_buffer_type_t buft, size_t * nbytes_total, bool no_alloc,
        struct ggml_backend_buffer_set * buffers) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);
    GGML_ASSERT(buffers->buffers == NULL && buffers->n_buffers == 0);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    *nbytes_total = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (!ggml_backend_tensor_is_bound(t) && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!no_alloc) {
                enum ggml_status status = alloc_tensor_range(ctx, first, t, buft, cur_buf_size, buffers);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            first = t;
            *nbytes_total += cur_buf_size;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        *nbytes_total += cur_buf_size;
        if (!no_alloc) {
            enum ggml_status status = alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, buffers);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
    }

    if (no_alloc) {
        return GGML_STATUS_SUCCESS;
    }

    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src && !t->buffer && t->view_src->buffer) {
            enum ggml_status status = ggml_backend_tensor_is_bound(t->view_src) ? ggml_backend_view_init(t) : GGML_STATUS_FAILED;
            if (status != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_set_free(buffers);
                return status;
            }
        }
    }

    if (buffers->n_buffers == 0) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
        GGML_ASSERT(!buffers->buffers);
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_alloc_ctx_tensors_from_buft_set(
        struct ggml_context * ctx, ggml_backend_buffer_type_t buft, struct ggml_backend_buffer_set * buffers) {
    GGML_ASSERT(buffers && buffers->buffers == NULL && buffers->n_buffers == 0);
    if (ggml_backend_buft_is_meta(buft)) {
        return GGML_STATUS_FAILED;
    }
    size_t nbytes_total = 0;
    return ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, false, buffers);
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_meta(buft)) {
        return ggml_backend_meta_alloc_ctx_tensors_from_buft_size(ctx, buft);
    }
    size_t nbytes_total = 0;
    struct ggml_backend_buffer_set buffers = { NULL, 0 };
    enum ggml_status status = ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, true, &buffers);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && buffers.buffers == NULL && buffers.n_buffers == 0);
    return nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_meta(buft)) {
        return ggml_backend_meta_alloc_ctx_tensors_from_buft(ctx, buft);
    }
    struct ggml_backend_buffer_set buffers = { NULL, 0 };
    enum ggml_status status = ggml_backend_alloc_ctx_tensors_from_buft_set(ctx, buft, &buffers);
    if (status != GGML_STATUS_SUCCESS || buffers.n_buffers == 0) {
        return NULL;
    }
    ggml_backend_buffer_t buffer = buffers.n_buffers == 1 ? buffers.buffers[0] :
        ggml_backend_multi_buffer_alloc_buffer(buffers.buffers, buffers.n_buffers);
    free(buffers.buffers);
    return buffer;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
