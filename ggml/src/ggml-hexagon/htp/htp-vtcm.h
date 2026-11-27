#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#include <stddef.h>
#include <stdint.h>

// Reserve one HVX vector at the end of physical VTCM so tail hvx_vmemu loads cannot fault past the page end
#define HTP_VTCM_GUARD_SIZE 128

static inline size_t htp_vtcm_usable_size(size_t raw_size) {
    return (raw_size - HTP_VTCM_GUARD_SIZE) & ~(size_t) 127;
}

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) do { (L)->field = (off); (off) += (sz); } while (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) do { if (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } else { (L)->field = 0; } } while (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : NULL)

#endif // HTP_VTCM_H
