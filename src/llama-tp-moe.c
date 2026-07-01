#include "llama-tp-moe.h"
#include <stdint.h>

// unpack base/count (this rank's local expert range) from the userdata pointer:
//   ud = ((uintptr_t)count << 32) | (uintptr_t)base,  base = rank*count, count = n_expert/size.
static void unpack(void * ud, int32_t * base, int32_t * count) {
    uintptr_t p = (uintptr_t) ud;
    *base  = (int32_t) (p & 0xffffffffu);
    *count = (int32_t) (p >> 32);
}

// custom1 on selected_experts (i32): dst = (global - base) if the expert is local, else 0.
// NOTE: selected_experts is a strided top-k view; callers MUST pass a contiguous copy (ggml_cont).
void tp_moe_local_ids_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                         int ith, int nth, void * userdata) {
    if (ith != 0) { return; }
    (void) nth;
    int32_t base, count;
    unpack(userdata, &base, &count);
    const int32_t * src = (const int32_t *) a->data;
    int32_t * out = (int32_t *) dst->data;
    const int64_t n = ggml_nelements(a);
    for (int64_t i = 0; i < n; i++) {
        const int32_t v = src[i] - base;
        out[i] = (v >= 0 && v < count) ? v : 0;
    }
}

// custom2 on (weights f32, selected_experts i32): dst = weights where the expert is local, else 0.
// Both inputs must be contiguous and share element ordering (weights [1,n_used,n_tok], sel [n_used,n_tok]).
void tp_moe_mask_weights_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                            const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    if (ith != 0) { return; }
    (void) nth;
    int32_t base, count;
    unpack(userdata, &base, &count);
    const float   * w   = (const float   *) a->data;
    const int32_t * sel = (const int32_t *) b->data;
    float * out = (float *) dst->data;
    const int64_t n = ggml_nelements(a);
    for (int64_t i = 0; i < n; i++) {
        const int32_t g = sel[i];
        out[i] = (g >= base && g < base + count) ? w[i] : 0.0f;
    }
}
