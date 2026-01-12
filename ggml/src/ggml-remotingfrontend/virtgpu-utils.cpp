#include "virtgpu-utils.h"

#include <malloc.h>
#include <stdlib.h>

#include <cstring>

#define NODE_ALLOC_ALIGN 64
#define NODE_PTR_MASK    (~((uintptr_t) NODE_ALLOC_ALIGN - 1))
#define NODE_LEVEL_MASK  ((uintptr_t) NODE_ALLOC_ALIGN - 1)
#define NULL_NODE        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            free(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

static inline uint64_t util_logbase2_64(uint64_t n) {
#if defined(HAVE___BUILTIN_CLZLL)
    return ((sizeof(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#else
    uint64_t pos = 0ull;
    if (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    if (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    if (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    if (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    if (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    if (n >= 1ull << 1) {
        pos += 1;
    }
    return pos;
#endif
}

void util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t node_size) {
    memset(arr, 0, sizeof(*arr));
    arr->elem_size      = elem_size;
    arr->node_size_log2 = util_logbase2_64(node_size);
    assert(node_size >= 2 && node_size == (1ull << arr->node_size_log2));
}

static inline void * os_malloc_aligned(size_t size, size_t alignment) {
    void * ptr;
    alignment = (alignment + sizeof(void *) - 1) & ~(sizeof(void *) - 1);
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static inline void * _util_sparse_array_node_data(uintptr_t handle) {
    return (void *) (handle & NODE_PTR_MASK);
}

static inline unsigned _util_sparse_array_node_level(uintptr_t handle) {
    return handle & NODE_LEVEL_MASK;
}

static inline void _util_sparse_array_node_finish(util_sparse_array * arr, uintptr_t node) {
    if (_util_sparse_array_node_level(node) > 0) {
        uintptr_t * children  = (uintptr_t *) _util_sparse_array_node_data(node);
        size_t      node_size = 1ull << arr->node_size_log2;
        for (size_t i = 0; i < node_size; i++) {
            if (children[i]) {
                _util_sparse_array_node_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(_util_sparse_array_node_data(node));
}

static inline uintptr_t _util_sparse_array_node(void * data, unsigned level) {
    assert(data != NULL);
    assert(((uintptr_t) data & NODE_LEVEL_MASK) == 0);
    assert((level & NODE_PTR_MASK) == 0);
    return (uintptr_t) data | level;
}

inline uintptr_t _util_sparse_array_node_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    if (level == 0) {
        size = arr->elem_size << arr->node_size_log2;
    } else {
        size = sizeof(uintptr_t) << arr->node_size_log2;
    }

    void * data = os_malloc_aligned(size, NODE_ALLOC_ALIGN);
    memset(data, 0, size);

    return _util_sparse_array_node(data, level);
}

static inline uintptr_t _util_sparse_array_set_or_free_node(uintptr_t * node_ptr, uintptr_t cmp_node, uintptr_t node) {
    uintptr_t prev_node = p_atomic_cmpxchg(node_ptr, cmp_node, node);

    if (prev_node != cmp_node) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(_util_sparse_array_node_data(node));
        return prev_node;
    } else {
        return node;
    }
}

void * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    const unsigned node_size_log2 = arr->node_size_log2;
    uintptr_t      root           = p_atomic_read(&arr->root);
    if (unlikely(!root)) {
        unsigned root_level = 0;
        uint64_t idx_iter   = idx >> node_size_log2;
        while (idx_iter) {
            idx_iter >>= node_size_log2;
            root_level++;
        }
        uintptr_t new_root = _util_sparse_array_node_alloc(arr, root_level);
        root               = _util_sparse_array_set_or_free_node(&arr->root, NULL_NODE, new_root);
    }

    while (1) {
        unsigned root_level = _util_sparse_array_node_level(root);
        uint64_t root_idx   = idx >> (root_level * node_size_log2);
        if (likely(root_idx < (1ull << node_size_log2))) {
            break;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_root = _util_sparse_array_node_alloc(arr, root_level + 1);

        uintptr_t * new_root_children = (uintptr_t *) _util_sparse_array_node_data(new_root);
        new_root_children[0]          = root;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        root = _util_sparse_array_set_or_free_node(&arr->root, root, new_root);
    }

    void *   node_data  = _util_sparse_array_node_data(root);
    unsigned node_level = _util_sparse_array_node_level(root);
    while (node_level > 0) {
        uint64_t child_idx = (idx >> (node_level * node_size_log2)) & ((1ull << node_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) node_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        if (unlikely(!child)) {
            child = _util_sparse_array_node_alloc(arr, node_level - 1);
            child = _util_sparse_array_set_or_free_node(&children[child_idx], NULL_NODE, child);
        }

        node_data  = _util_sparse_array_node_data(child);
        node_level = _util_sparse_array_node_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << node_size_log2) - 1);
    return (void *) ((char *) node_data + (elem_idx * arr->elem_size));
}
