#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    // test chunk save/load round-trip
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        assert(chunk != NULL);
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);

        // query the required buffer size (out_buf == NULL)
        size_t expected_len = 0;
        int32_t rc = mtmd_input_chunk_save(chunk, NULL, 0, &expected_len);
        printf("    Chunk %zu: save query rc = %d, expected_len = %zu\n", i, rc, expected_len);
        assert(rc == 0);
        assert(expected_len > 0);

        // saving into a too-small buffer must fail, not crash
        char tiny_buf[1];
        rc = mtmd_input_chunk_save(chunk, tiny_buf, sizeof(tiny_buf), NULL);
        printf("    Chunk %zu: save into too-small buffer rc = %d (expect non-zero)\n", i, rc);
        assert(rc != 0);

        // save into a properly-sized buffer
        char * buf = (char *) malloc(expected_len);
        assert(buf != NULL);
        rc = mtmd_input_chunk_save(chunk, buf, expected_len, NULL);
        assert(rc == 0);

        // loading from a truncated buffer must fail gracefully, not crash
        if (expected_len > 1) {
            mtmd_input_chunk * bad = mtmd_input_chunk_load(buf, expected_len - 1);
            printf("    Chunk %zu: load from truncated buffer = %p (expect NULL)\n", i, (void *) bad);
            assert(bad == NULL);
        }

        // load it back
        mtmd_input_chunk * loaded = mtmd_input_chunk_load(buf, expected_len);
        assert(loaded != NULL);

        // metadata must match the original chunk
        assert(mtmd_input_chunk_get_type(loaded) == type);
        assert(mtmd_input_chunk_get_n_tokens(loaded) == mtmd_input_chunk_get_n_tokens(chunk));
        assert(mtmd_input_chunk_get_n_pos(loaded) == mtmd_input_chunk_get_n_pos(chunk));

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tok_orig, n_tok_loaded;
            const llama_token * tok_orig   = mtmd_input_chunk_get_tokens_text(chunk, &n_tok_orig);
            const llama_token * tok_loaded = mtmd_input_chunk_get_tokens_text(loaded, &n_tok_loaded);
            printf("    Chunk %zu: loaded %zu text tokens (orig %zu), first token %d (orig %d)\n",
                i, n_tok_loaded, n_tok_orig,
                n_tok_loaded > 0 ? tok_loaded[0] : -1,
                n_tok_orig   > 0 ? tok_orig[0]   : -1);
            assert(n_tok_orig == n_tok_loaded);
            for (size_t j = 0; j < n_tok_orig; j++) {
                assert(tok_orig[j] == tok_loaded[j]);
            }
        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            const char * id_orig   = mtmd_input_chunk_get_id(chunk);
            const char * id_loaded = mtmd_input_chunk_get_id(loaded);
            printf("    Chunk %zu: loaded id '%s' (orig '%s')\n", i, id_loaded, id_orig);
            assert(id_orig != NULL && id_loaded != NULL);
            assert(strcmp(id_orig, id_loaded) == 0);
        }

        mtmd_input_chunk_free(loaded);
        free(buf);
    }
    printf("Chunk save/load round-trip OK\n");

    // Free the chunks
    mtmd_input_chunks_free(chunks);

    printf("\n\nDONE: test libmtmd C API...\n");

    return 0;
}
