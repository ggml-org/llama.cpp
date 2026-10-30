#include "llama.h"

#undef NDEBUG
#include <assert.h>

int main(void) {
    const struct llama_context_params params = llama_context_default_params();
    assert(!params.moe_direct_io);
    assert(params.moe_slots == 4096);
    assert(params.moe_chunk_max == 64);
    assert(params.moe_gguf_path == NULL);
    assert(params.moe_pack_dir == NULL);
    assert(params.moe_profile_path == NULL);
    return 0;
}
