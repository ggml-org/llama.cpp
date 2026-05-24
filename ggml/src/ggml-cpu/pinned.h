#ifndef GGML_PINNED_H
#define GGML_PINNED_H

#include "ggml.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Allocate page-locked (pinned) memory suitable for GPU DMA.
 *  Falls back to regular malloc if locking fails. Returns NULL on OOM. */
GGML_API void * ggml_cpu_pinned_alloc(size_t size);

/** Free memory allocated by ggml_cpu_pinned_alloc. */
GGML_API void ggml_cpu_pinned_free(void * ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif
