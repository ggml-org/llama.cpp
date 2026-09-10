// Regression test for malformed mmproj metadata handling in the CLIP loader.
//
// Two crafted projector files that previously aborted the process (SIGABRT /
// std::bad_alloc) in clip_init must now be rejected cleanly (null contexts):
//   1. a scalar key declared with the wrong GGUF type (type confusion)
//   2. an implausibly large clip.vision.block_count (unbounded allocation)
//
// Before the fix this test process aborts (test fails). After the fix clip_init
// returns null contexts and the asserts pass.

#include "clip.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>

static clip_context_params make_params() {
    clip_context_params p{};
    p.use_gpu           = false;
    p.warmup            = false;
    p.no_alloc          = true;
    p.image_min_tokens  = -1;
    p.image_max_tokens  = -1;
    return p;
}

// A minimal vision mmproj whose hparams parse fully (reaching the layer-count
// path). Tensors are absent on purpose: a valid-hparams file stops later at
// tensor loading, so any earlier clean rejection is attributable to the field
// under test, not to a missing required key.
static void write_base(gguf_context * ctx) {
    gguf_set_val_str  (ctx, "general.architecture",                    "clip");
    gguf_set_val_bool (ctx, "clip.has_vision_encoder",                 true);
    gguf_set_val_str  (ctx, "clip.projector_type",                     "mlp");
    gguf_set_val_u32  (ctx, "clip.vision.embedding_length",            32);
    gguf_set_val_u32  (ctx, "clip.vision.block_count",                 2);
    gguf_set_val_u32  (ctx, "clip.vision.feed_forward_length",         64);
    gguf_set_val_u32  (ctx, "clip.vision.attention.head_count",        4);
    gguf_set_val_u32  (ctx, "clip.vision.projection_dim",             32);
    gguf_set_val_f32  (ctx, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_u32  (ctx, "clip.vision.image_size",                 32);
    gguf_set_val_u32  (ctx, "clip.vision.patch_size",                16);
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float std_[3] = {0.5f, 0.5f, 0.5f};
    gguf_set_arr_data (ctx, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data (ctx, "clip.vision.image_std",  GGUF_TYPE_FLOAT32, std_, 3);
}

static bool loads_without_context(const char * path) {
    clip_init_result r = clip_init(path, make_params());
    bool ok = (r.ctx_v == nullptr && r.ctx_a == nullptr);
    if (r.ctx_v) clip_free(r.ctx_v);
    if (r.ctx_a) clip_free(r.ctx_a);
    return ok;
}

int main() {
    const char * f_badtype  = "test-clip-badtype.gguf";
    const char * f_bigcount = "test-clip-bigcount.gguf";

    // 1. wrong-typed scalar: has_vision_encoder declared as u32 instead of bool.
    {
        gguf_context * ctx = gguf_init_empty();
        write_base(ctx);
        // overwrite the marker key with the WRONG type (u32 instead of bool)
        gguf_set_val_u32(ctx, "clip.has_vision_encoder", 1);
        gguf_write_to_file(ctx, f_badtype, false);
        gguf_free(ctx);
    }

    // 2. implausible block_count -> would attempt a huge allocation.
    {
        gguf_context * ctx = gguf_init_empty();
        write_base(ctx);
        gguf_set_val_u32(ctx, "clip.vision.block_count", 201326592u); // ~200M
        gguf_write_to_file(ctx, f_bigcount, false);
        gguf_free(ctx);
    }

    printf("test: wrong-typed scalar key must be rejected cleanly\n");
    bool ok1 = loads_without_context(f_badtype);
    printf("  -> %s\n", ok1 ? "PASS" : "FAIL");

    printf("test: implausible block_count must be rejected cleanly\n");
    bool ok2 = loads_without_context(f_bigcount);
    printf("  -> %s\n", ok2 ? "PASS" : "FAIL");

    remove(f_badtype);
    remove(f_bigcount);

    if (!ok1 || !ok2) {
        fprintf(stderr, "clip metadata regression test FAILED\n");
        return 1;
    }
    printf("all clip metadata regression checks passed\n");
    return 0;
}
