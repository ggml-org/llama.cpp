#include <stdio.h>
#include <assert.h>

#include "mtmd.h"
#include "mtmd-helper.h"

int main(void) {
    printf("\n\nTesting libmtmd C API...\n");
    printf("--------\n\n");

    struct mtmd_context_params params = mtmd_context_params_default();
    printf("Default image marker: %s\n", params.image_marker);

    mtmd_input_chunks * chunks = mtmd_test_create_input_chunks();

    if (!chunks) {
        fprintf(stderr, "Failed to create input chunks\n");
        return 1;
    }

    // simple test for the helper
    size_t n_tokens_total = mtmd_helper_get_n_tokens(chunks);
    printf("Total tokens in chunks: %zu\n", n_tokens_total);
    assert(n_tokens_total > 0);

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    printf("Number of chunks: %zu\n", n_chunks);
    assert(n_chunks > 0);

    mtmd_prompt_plan * plan = mtmd_prompt_plan_init(chunks);
    if (plan == NULL ||
            !mtmd_prompt_plan_validate(plan) ||
            mtmd_prompt_plan_get_n_spans(plan) != n_chunks ||
            mtmd_prompt_plan_get_n_tokens(plan) != n_tokens_total ||
            mtmd_prompt_plan_get_n_positions(plan) == 0 ||
            mtmd_prompt_plan_get_modality_mask(plan) != 1 ||
            mtmd_prompt_plan_get_attention_layout(plan) != MTMD_ATTENTION_LAYOUT_MULTIMODAL_PREFILL) {
        fprintf(stderr, "Invalid multimodal prompt plan\n");
        mtmd_prompt_plan_free(plan);
        mtmd_input_chunks_free(chunks);
        return 2;
    }
    for (size_t i = 0; i < n_chunks; ++i) {
        struct mtmd_prompt_span span;
        if (!mtmd_prompt_plan_get_span(plan, i, &span) ||
                span.token_length == 0 ||
                span.decoder_position_length == 0) {
            fprintf(stderr, "Invalid multimodal prompt span %zu\n", i);
            mtmd_prompt_plan_free(plan);
            mtmd_input_chunks_free(chunks);
            return 3;
        }
        printf("Plan span %zu: tokens=%u positions=%u\n",
               i, span.token_length, span.decoder_position_length);
    }
    mtmd_prompt_plan_free(plan);

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        assert(chunk != NULL);
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        printf("Chunk %zu type: %d\n", i, type);

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            printf("    Text chunk with %zu tokens\n", n_tokens);
            assert(tokens != NULL);
            assert(n_tokens > 0);
            for (size_t j = 0; j < n_tokens; j++) {
                assert(tokens[j] >= 0);
                printf("    > Token %zu: %d\n", j, tokens[j]);
            }

        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            size_t n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
            // get position of the last token, which should be (nx - 1, ny - 1)
            struct mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(image_tokens, 0, n_tokens - 1);
            size_t nx = pos.x + 1;
            size_t ny = pos.y + 1;
            const char * id = mtmd_image_tokens_get_id(image_tokens);
            assert(n_tokens > 0);
            assert(nx > 0);
            assert(ny > 0);
            assert(id != NULL);
            printf("    Image chunk with %zu tokens\n", n_tokens);
            printf("    Image size: %zu x %zu\n", nx, ny);
            printf("    Image ID: %s\n", id);
        }
    }

    // Free the chunks
    mtmd_input_chunks_free(chunks);

    printf("\n\nDONE: test libmtmd C API...\n");

    return 0;
}
