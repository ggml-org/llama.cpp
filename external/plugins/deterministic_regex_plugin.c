#include "deterministic_regex_plugin.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Simple helper: check if token is letters and/or whitespace only.
// Whitespace is allowed so the constrained output stays readable instead of
// collapsing into a single unbroken letter run.
static bool is_alpha_token(const char *token, int32_t len) {
    if (token == NULL || len <= 0) return false;
    for (int32_t i = 0; i < len; i++) {
        char c = token[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              c == ' ' || c == '\n' || c == '\t' || c == '\r')) {
            return false;
        }
    }
    return true;
}

typedef struct {
    char ** tokens;
    int32_t vocab_size;
    bool    initialized;
    int32_t last_slot;
    int32_t * stop_tokens;
    int32_t   n_stop;
} RegexPluginState;

DeterministicDraftPlugin * deterministic_draft_create(void) {
    RegexPluginState *state = malloc(sizeof(RegexPluginState));
    if (state == NULL) return NULL;
    state->tokens = NULL;
    state->vocab_size = 0;
    state->initialized = false;
    state->last_slot = -1;
    state->stop_tokens = NULL;
    state->n_stop = 0;
    return (DeterministicDraftPlugin *) state;
}

void deterministic_draft_destroy(DeterministicDraftPlugin *state) {
    if (state == NULL) return;
    RegexPluginState *s = (RegexPluginState *) state;
    if (s->tokens != NULL) {
        for (int32_t i = 0; i < s->vocab_size; i++) {
            free(s->tokens[i]);
        }
        free(s->tokens);
    }
    free(s->stop_tokens);
    free(s);
}

uint32_t deterministic_draft_get_capabilities(DeterministicDraftPlugin *state) {
    (void) state;
    return DETERMINISTIC_DRAFT_CAPABILITY_BITMASK;
}

bool deterministic_draft_set_vocab(
        DeterministicDraftPlugin *state,
        const char **vocab_entries,
        int32_t vocab_size,
        const int32_t *stop_tokens,
        int32_t n_stop) {
    if (state == NULL) return false;
    RegexPluginState *s = (RegexPluginState *) state;
    if (s->initialized) return true;

    if (s->tokens != NULL) {
        for (int32_t i = 0; i < s->vocab_size; i++) {
            free(s->tokens[i]);
        }
        free(s->tokens);
    }

    s->tokens = malloc(vocab_size * sizeof(char *));
    if (s->tokens == NULL) return false;

    for (int32_t i = 0; i < vocab_size; i++) {
        s->tokens[i] = strdup(vocab_entries[i]);
        if (s->tokens[i] == NULL) {
            for (int32_t j = 0; j < i; j++) {
                free(s->tokens[j]);
            }
            free(s->tokens);
            return false;
        }
    }

    s->vocab_size = vocab_size;
    s->initialized = true;

    // store stop tokens so EOG always passes the filter (termination)
    s->stop_tokens = malloc((size_t) n_stop * sizeof(int32_t));
    if (s->stop_tokens == NULL && n_stop > 0) {
        s->initialized = false;
        return false;
    }
    for (int32_t i = 0; i < n_stop; i++) {
        s->stop_tokens[i] = stop_tokens[i];
    }
    s->n_stop = n_stop;

    return true;
}

bool deterministic_draft_fill_bitmask(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        uint32_t *bitmask,
        int32_t vocab_size) {
    if (state == NULL) return false;
    RegexPluginState *s = (RegexPluginState *) state;
    if (!s->initialized) return false;
    if (bitmask == NULL) return false;

    (void) vocab_size;

    int32_t bits = (s->vocab_size + 31) / 32;
    memset(bitmask, 0, bits * sizeof(uint32_t));

    for (int32_t i = 0; i < s->vocab_size; i++) {
        if (is_alpha_token(s->tokens[i], strlen(s->tokens[i]))) {
            int32_t word = i / 32;
            int32_t bit = i % 32;
            bitmask[word] |= (1u << bit);
        }
    }

    // always allow EOG/stop tokens so the sequence can terminate
    for (int32_t i = 0; i < s->n_stop; i++) {
        int32_t tok = s->stop_tokens[i];
        if (tok >= 0 && tok < s->vocab_size) {
            bitmask[tok / 32] |= (1u << (tok % 32));
        }
    }

    (void) slot_id;
    return true;
}

const char * deterministic_draft_get_jump_forward(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        int32_t *out_length) {
    (void) state;
    (void) slot_id;
    if (out_length != NULL) {
        *out_length = 0;
    }
    return NULL;
}

void deterministic_draft_commit(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        int32_t token_id,
        const char *token_text,
        int32_t token_length) {
    (void) state;
    (void) slot_id;
    (void) token_id;
    (void) token_text;
    (void) token_length;
}

bool deterministic_draft_rollback(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        int32_t n_tokens) {
    (void) state;
    (void) slot_id;
    (void) n_tokens;
    return true;
}

bool deterministic_draft_is_terminated(
        DeterministicDraftPlugin *state,
        int32_t slot_id) {
    (void) state;
    (void) slot_id;
    return false;
}

void deterministic_draft_reset(
        DeterministicDraftPlugin *state,
        int32_t slot_id) {
    (void) state;
    (void) slot_id;
}

int32_t deterministic_draft_state_get_size(
        DeterministicDraftPlugin *state,
        int32_t slot_id) {
    (void) state;
    (void) slot_id;
    return -1;
}

int32_t deterministic_draft_state_get_data(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        uint8_t *buffer,
        int32_t buffer_size) {
    (void) state;
    (void) slot_id;
    (void) buffer;
    (void) buffer_size;
    return -1;
}

bool deterministic_draft_state_set_data(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        const uint8_t *data,
        int32_t size) {
    (void) state;
    (void) slot_id;
    (void) data;
    (void) size;
    return false;
}

const char * deterministic_draft_get_version(DeterministicDraftPlugin *state) {
    (void) state;
    return "1.0.0";
}

const char * deterministic_regex_plugin_get_version(void) {
    return "1.0.0";
}

int32_t deterministic_draft_filter_draft(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        const int32_t *tokens,
        int32_t n_tokens) {
    (void) slot_id;
    if (state == NULL || tokens == NULL || n_tokens <= 0) return 0;

    RegexPluginState *s = (RegexPluginState *) state;
    if (!s->initialized) return 0;

    int32_t valid = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= s->vocab_size) break;
        const char *text = s->tokens[tokens[i]];
        bool is_stop = false;
        for (int32_t j = 0; j < s->n_stop; j++) {
            if (s->stop_tokens[j] == tokens[i]) { is_stop = true; break; }
        }
        if (!is_stop && (text == NULL || !is_alpha_token(text, (int32_t) strlen(text)))) {
            break;
        }
        valid++;
    }
    return valid;
}

bool deterministic_draft_apply_bitmask(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        uint32_t *bitmask,
        int32_t vocab_size,
        float *logits) {
    if (!deterministic_draft_fill_bitmask(state, slot_id, bitmask, vocab_size)) {
        return false;
    }

    for (int32_t i = 0; i < vocab_size; i++) {
        if (((bitmask[i / 32] >> (i % 32)) & 1u) == 0) {
            logits[i] = -1e30f;
        }
    }
    return true;
}

void deterministic_draft_commit_tokens(
        DeterministicDraftPlugin *state,
        int32_t slot_id,
        const int32_t *tokens,
        int32_t n_tokens) {
    (void) state;
    (void) slot_id;
    (void) tokens;
    (void) n_tokens;
}
