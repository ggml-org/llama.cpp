#include "llama-kv-codec.h"
#include "llama-context.h"

#include <string.h>

struct llama_kv_codec_params llama_kv_codec_default_params(void) {
    struct llama_kv_codec_params result;
    memset(&result, 0, sizeof(result));
    return result;
}

void llama_set_kv_codec(
    struct llama_context * ctx,
    const struct llama_kv_codec_params * params) {
    if (!ctx) {
        return;
    }

    // Store codec params in context
    if (params && params->codec) {
        ctx->kv_codec_params = *params;
        ctx->has_kv_codec = true;
    } else {
        ctx->has_kv_codec = false;
        memset(&ctx->kv_codec_params, 0, sizeof(ctx->kv_codec_params));
    }
}

bool llama_get_kv_codec_info(
    const struct llama_context * ctx,
    struct llama_kv_codec_params * params) {
    if (!ctx || !ctx->has_kv_codec || !params) {
        return false;
    }
    *params = ctx->kv_codec_params;
    return true;
}
