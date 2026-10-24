#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// A mailbox is page-locked host memory that a GPU backend can read and write while the CPU polls it: a GPU op stores its
// data and then a flag, a CPU op sees the flag and reads the data, with no stream synchronization and no copy between.
// llama.cpp's MoE cache hands a layer's input to the CPU (which computes the experts the GPU does not hold) and takes the
// result back this way, one hand-over costing a few microseconds instead of a synchronous copy's tens.
//
// Both ops are a GGML_OP_CUSTOM whose function is ggml_mailbox_cpu_stub and whose userdata is a ggml_mailbox_op. The GPU
// backend runs them; the stub aborts if one ever reaches the CPU.

#define GGML_MAILBOX_MAGIC 0x4d424f58u // "MBOX"

enum ggml_mailbox_kind {
    GGML_MAILBOX_SEND = 1, // GPU -> CPU: copy src[0] (F32, n_f32) and src[1] (I32, n_i32) into the mailbox, then flag := *epoch
    GGML_MAILBOX_RECV = 2, // CPU -> GPU: copy n_f32 floats from the mailbox into dst
};

struct ggml_mailbox_op {
    uint32_t magic;     // GGML_MAILBOX_MAGIC
    int32_t  kind;      // enum ggml_mailbox_kind
    int32_t  n_f32;
    int32_t  n_i32;
    // the mailbox memory as the GPU addresses it (the CPU's addresses differ under WSL2)
    void     * dev_f32;
    void     * dev_i32;
    volatile uint32_t * dev_flag;
    const volatile uint32_t * dev_epoch;
};

GGML_API void ggml_mailbox_cpu_stub(struct ggml_tensor * dst, int ith, int nth, void * userdata);

// allocates a mailbox buffer of `size` bytes of page-locked host memory that the GPU maps; *dev_base is the GPU's address of
// the buffer's base. Tensors placed in it are read by the CPU in place and need no copy to run on either side. NULL if the
// backend has no mailboxes. Get it from the backend registry: ggml_backend_reg_get_proc_address(reg, "ggml_backend_mailbox_buffer")
typedef ggml_backend_buffer_t (*ggml_backend_mailbox_buffer_t)(size_t size, void ** dev_base);

#ifdef __cplusplus
}
#endif
