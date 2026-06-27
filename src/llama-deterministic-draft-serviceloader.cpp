// src/llama-deterministic-draft-serviceloader.cpp -- Deterministic draft ServiceLoader
//
// ServiceLoader for the deterministic-draft SPI (include/deterministic_draft_plugin.h):
// it dlopen/dlsym's a Service Provider's plugin (.so/.dylib/.dll) and resolves
// the contract methods at runtime when --det-draft-model is passed.
//
// Built unconditionally into libllama so that the llama_deterministic_draft_*
// API declared in include/llama_deterministic_draft.h is always available.
//
// The same source is also used by external/CMakeLists.txt to build
// libdeterministic_draft_spec.so (a standalone distributable for plugin
// authors who don't want to link against full libllama), gated by
// DETERMINISTIC_SPEC_ENABLED.

#include "deterministic_draft_plugin.h"
#include "llama_deterministic_draft.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// LLAMA_DETERMINISTIC_DRAFT_INTERNAL is defined only for the libllama build
// (src/CMakeLists.txt); the standalone SDK build has no llama dependency and
// falls back to stderr.
#ifdef LLAMA_DETERMINISTIC_DRAFT_INTERNAL
#    include "llama-impl.h"
#    define DET_LOG_ERROR(...) LLAMA_LOG_ERROR(__VA_ARGS__)
#else
#    define DET_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifdef _WIN32
#    include <windows.h>
#    define DL_HANDLE            HMODULE
#    define DL_OPEN(path)        LoadLibraryA(path)
#    define DL_SYM(handle, name) GetProcAddress(handle, name)
#    define DL_CLOSE(handle)     FreeLibrary(handle)
#elif defined(__EMSCRIPTEN__)
// no dynamic loading on Emscripten - load() always fails
#    define DL_HANDLE            void *
#    define DL_OPEN(path)        nullptr
#    define DL_SYM(handle, name) nullptr
#    define DL_CLOSE(handle)     ((void) (handle))
#else
#    include <dlfcn.h>
#    define DL_HANDLE            void *
#    define DL_OPEN(path)        dlopen(path, RTLD_NOW | RTLD_LOCAL)
#    define DL_SYM(handle, name) dlsym(handle, name)
#    define DL_CLOSE(handle)     dlclose(handle)
#endif

// The struct layout is known only in this file; external code accesses the
// plugin through the C API (opaque pointer).

struct llama_deterministic_draft {
    DL_HANDLE plugin_handle;

    DeterministicDraftPlugin * (*create)(void);
    void (*destroy)(DeterministicDraftPlugin *);
    void (*commit)(DeterministicDraftPlugin *, int32_t, int32_t, const char *, int32_t);
    bool (*rollback)(DeterministicDraftPlugin *, int32_t, int32_t);
    void (*reset)(DeterministicDraftPlugin *, int32_t);
    const char * (*get_language)(DeterministicDraftPlugin *, int32_t);
    const char * (*get_version)(DeterministicDraftPlugin *);
    bool (*set_language)(DeterministicDraftPlugin *, int32_t, const char *);

    // capabilities API (optional - NULL for plugins that don't implement it)
    uint32_t (*get_capabilities)(DeterministicDraftPlugin *);
    bool (*set_vocab)(DeterministicDraftPlugin *, const char **, int32_t, const int32_t *, int32_t);
    bool (*set_grammar)(DeterministicDraftPlugin *, const char *, const char *);
    bool (*fill_bitmask)(DeterministicDraftPlugin *, int32_t, uint32_t *, int32_t);
    const char * (*get_jump_forward)(DeterministicDraftPlugin *, int32_t, int32_t *);

    // high-level filter helpers (optional - NULL for plugins that don't implement them)
    int32_t (*filter_draft)(DeterministicDraftPlugin *, int32_t, const int32_t *, int32_t);
    bool (*apply_bitmask)(DeterministicDraftPlugin *, int32_t, uint32_t *, int32_t, float *);
    void (*commit_tokens)(DeterministicDraftPlugin *, int32_t, const int32_t *, int32_t);
    bool (*is_terminated)(DeterministicDraftPlugin *, int32_t);
    bool (*is_detecting)(DeterministicDraftPlugin *, int32_t);

    // state serialization (optional - NULL for plugins that don't implement it)
    int32_t (*state_get_size)(DeterministicDraftPlugin *, int32_t);
    int32_t (*state_get_data)(DeterministicDraftPlugin *, int32_t, uint8_t *, int32_t);
    bool (*state_set_data)(DeterministicDraftPlugin *, int32_t, const uint8_t *, int32_t);

    DeterministicDraftPlugin * state;

    bool load(const std::string & path);
    void unload();
};

// --- Plugin Loader --------------------------------------------------

bool llama_deterministic_draft::load(const std::string & path) {
    plugin_handle = DL_OPEN(path.c_str());
    if (!plugin_handle) {
        DET_LOG_ERROR("%s: failed to load plugin '%s'\n", __func__, path.c_str());
        return false;
    }

#define LOAD_SYM(name)                                                                     \
    name = (decltype(name)) DL_SYM(plugin_handle, "deterministic_draft_" #name);           \
    if (!name) {                                                                           \
        DET_LOG_ERROR("%s: symbol 'deterministic_draft_%s' not found in %s\n", __func__,   \
                #name, path.c_str());                                                      \
        DL_CLOSE(plugin_handle);                                                           \
        return false;                                                                      \
    }

    LOAD_SYM(create);
    LOAD_SYM(destroy);
    LOAD_SYM(commit);
    LOAD_SYM(reset);
    LOAD_SYM(get_language);
    LOAD_SYM(get_version);

    set_language = (decltype(set_language)) DL_SYM(plugin_handle, "deterministic_draft_set_language");
    rollback     = (decltype(rollback))     DL_SYM(plugin_handle, "deterministic_draft_rollback");

    // optional, graceful fallback for older plugins
    get_capabilities = (decltype(get_capabilities)) DL_SYM(plugin_handle, "deterministic_draft_get_capabilities");
    set_vocab        = (decltype(set_vocab))        DL_SYM(plugin_handle, "deterministic_draft_set_vocab");
    set_grammar      = (decltype(set_grammar))      DL_SYM(plugin_handle, "deterministic_draft_set_grammar");
    fill_bitmask     = (decltype(fill_bitmask))     DL_SYM(plugin_handle, "deterministic_draft_fill_bitmask");
    get_jump_forward = (decltype(get_jump_forward)) DL_SYM(plugin_handle, "deterministic_draft_get_jump_forward");

    filter_draft  = (decltype(filter_draft))  DL_SYM(plugin_handle, "deterministic_draft_filter_draft");
    apply_bitmask = (decltype(apply_bitmask)) DL_SYM(plugin_handle, "deterministic_draft_apply_bitmask");
    commit_tokens = (decltype(commit_tokens)) DL_SYM(plugin_handle, "deterministic_draft_commit_tokens");
    is_terminated = (decltype(is_terminated)) DL_SYM(plugin_handle, "deterministic_draft_is_terminated");
    is_detecting  = (decltype(is_detecting))  DL_SYM(plugin_handle, "deterministic_draft_is_detecting");

    state_get_size = (decltype(state_get_size)) DL_SYM(plugin_handle, "deterministic_draft_state_get_size");
    state_get_data = (decltype(state_get_data)) DL_SYM(plugin_handle, "deterministic_draft_state_get_data");
    state_set_data = (decltype(state_set_data)) DL_SYM(plugin_handle, "deterministic_draft_state_set_data");

    state = create();
    if (!state) {
        DET_LOG_ERROR("%s: plugin create() returned NULL\n", __func__);
        DL_CLOSE(plugin_handle);
        return false;
    }

    return true;
}

void llama_deterministic_draft::unload() {
    if (state && destroy) {
        destroy(state);
        state = nullptr;
    }
    if (plugin_handle) {
        DL_CLOSE(plugin_handle);
        plugin_handle = nullptr;
    }
}

// --- C API ----------------------------------------------------------

extern "C" {

LLAMA_DET_API struct llama_deterministic_draft * llama_deterministic_draft_init(const char * plugin_path) {
    auto * draft = new llama_deterministic_draft{};
    if (plugin_path && !draft->load(plugin_path)) {
        draft->unload();
        delete draft;
        return nullptr;
    }
    return draft;
}

LLAMA_DET_API void llama_deterministic_draft_free(struct llama_deterministic_draft * draft) {
    if (!draft) {
        return;
    }
    draft->unload();
    delete draft;
}

LLAMA_DET_API void llama_deterministic_draft_commit(struct llama_deterministic_draft * draft,
                                                    int32_t                            slot_id,
                                                    int32_t                            token_id,
                                                    const char *                       token_text,
                                                    int32_t                            token_length) {
    if (!draft || !draft->commit || !draft->state) {
        return;
    }
    draft->commit(draft->state, slot_id, token_id, token_text, token_length);
}

LLAMA_DET_API bool llama_deterministic_draft_rollback(struct llama_deterministic_draft * draft,
                                                      int32_t                            slot_id,
                                                      int32_t                            n_tokens) {
    if (!draft || !draft->rollback || !draft->state) {
        return false;
    }
    return draft->rollback(draft->state, slot_id, n_tokens);
}

LLAMA_DET_API void llama_deterministic_draft_reset(struct llama_deterministic_draft * draft, int32_t slot_id) {
    if (!draft || !draft->reset || !draft->state) {
        return;
    }
    draft->reset(draft->state, slot_id);
}

LLAMA_DET_API bool llama_deterministic_draft_set_language(struct llama_deterministic_draft * draft,
                                                          int32_t                            slot_id,
                                                          const char *                       lang) {
    if (!draft || !draft->set_language || !draft->state) {
        return false;
    }
    return draft->set_language(draft->state, slot_id, lang);
}

LLAMA_DET_API const char * llama_deterministic_draft_get_language(struct llama_deterministic_draft * draft, int32_t slot_id) {
    if (!draft || !draft->get_language || !draft->state) {
        return nullptr;
    }
    return draft->get_language(draft->state, slot_id);
}

LLAMA_DET_API const char * llama_deterministic_draft_get_version(struct llama_deterministic_draft * draft) {
    if (!draft || !draft->get_version || !draft->state) {
        return "unknown";
    }
    return draft->get_version(draft->state);
}

LLAMA_DET_API uint32_t llama_deterministic_draft_get_capabilities(struct llama_deterministic_draft * draft) {
    if (!draft || !draft->state) {
        return 0;
    }
    if (!draft->get_capabilities) {
        return 0;
    }
    return draft->get_capabilities(draft->state);
}

LLAMA_DET_API bool llama_deterministic_draft_set_vocab(struct llama_deterministic_draft * draft,
                                                       const char **                     vocab_entries,
                                                       int32_t                           vocab_size,
                                                       const int32_t *                   stop_tokens,
                                                       int32_t                           n_stop) {
    if (!draft || !draft->state) {
        return false;
    }
    if (!draft->set_vocab) {
        return true;
    }
    return draft->set_vocab(draft->state, vocab_entries, vocab_size, stop_tokens, n_stop);
}

LLAMA_DET_API bool llama_deterministic_draft_set_grammar(struct llama_deterministic_draft * draft,
                                                         const char *                       ebnf_str,
                                                         const char *                       root_rule) {
    if (!draft || !draft->state) {
        return false;
    }
    if (!draft->set_grammar) {
        return false;
    }
    return draft->set_grammar(draft->state, ebnf_str, root_rule);
}

LLAMA_DET_API bool llama_deterministic_draft_fill_bitmask(struct llama_deterministic_draft * draft,
                                                          int32_t                            slot_id,
                                                          uint32_t *                         bitmask,
                                                          int32_t                            vocab_size) {
    if (!draft || !draft->state || !bitmask || vocab_size <= 0) {
        return false;
    }
    if (!draft->fill_bitmask) {
        return false;
    }
    return draft->fill_bitmask(draft->state, slot_id, bitmask, vocab_size);
}

LLAMA_DET_API const char * llama_deterministic_draft_get_jump_forward(struct llama_deterministic_draft * draft,
                                                                      int32_t                            slot_id,
                                                                      int32_t *                          out_length) {
    if (!draft || !draft->state) {
        if (out_length) {
            *out_length = 0;
        }
        return nullptr;
    }
    if (!draft->get_jump_forward) {
        if (out_length) {
            *out_length = 0;
        }
        return nullptr;
    }
    return draft->get_jump_forward(draft->state, slot_id, out_length);
}

LLAMA_DET_API int32_t llama_deterministic_draft_filter_draft(struct llama_deterministic_draft * draft,
                                                             int32_t                            slot_id,
                                                             const int32_t *                    tokens,
                                                             int32_t                            n_tokens) {
    if (!draft || !draft->filter_draft || !draft->state) {
        return 0;
    }
    return draft->filter_draft(draft->state, slot_id, tokens, n_tokens);
}

LLAMA_DET_API bool llama_deterministic_draft_apply_bitmask(struct llama_deterministic_draft * draft,
                                                           int32_t                            slot_id,
                                                           uint32_t *                         bitmask,
                                                           int32_t                            vocab_size,
                                                           float *                            logits) {
    if (!draft || !draft->apply_bitmask || !draft->state) {
        return false;
    }
    return draft->apply_bitmask(draft->state, slot_id, bitmask, vocab_size, logits);
}

LLAMA_DET_API void llama_deterministic_draft_commit_tokens(struct llama_deterministic_draft * draft,
                                                           int32_t                            slot_id,
                                                           const int32_t *                    tokens,
                                                           int32_t                            n_tokens) {
    if (!draft || !draft->commit_tokens || !draft->state) {
        return;
    }
    draft->commit_tokens(draft->state, slot_id, tokens, n_tokens);
}

LLAMA_DET_API bool llama_deterministic_draft_is_terminated(struct llama_deterministic_draft * draft, int32_t slot_id) {
    if (!draft || !draft->is_terminated || !draft->state) {
        return false;
    }
    return draft->is_terminated(draft->state, slot_id);
}

LLAMA_DET_API bool llama_deterministic_draft_is_detecting(struct llama_deterministic_draft * draft, int32_t slot_id) {
    if (!draft || !draft->is_detecting || !draft->state) {
        return false;
    }
    return draft->is_detecting(draft->state, slot_id);
}

LLAMA_DET_API int32_t llama_deterministic_draft_state_get_size(struct llama_deterministic_draft * draft, int32_t slot_id) {
    if (!draft || !draft->state_get_size || !draft->state) {
        return -1;
    }
    return draft->state_get_size(draft->state, slot_id);
}

LLAMA_DET_API int32_t llama_deterministic_draft_state_get_data(struct llama_deterministic_draft * draft,
                                                               int32_t                            slot_id,
                                                               uint8_t *                          buffer,
                                                               int32_t                            buffer_size) {
    if (!draft || !draft->state_get_data || !draft->state || !buffer || buffer_size <= 0) {
        return -1;
    }
    return draft->state_get_data(draft->state, slot_id, buffer, buffer_size);
}

LLAMA_DET_API bool llama_deterministic_draft_state_set_data(struct llama_deterministic_draft * draft,
                                                            int32_t                            slot_id,
                                                            const uint8_t *                    data,
                                                            int32_t                            size) {
    if (!draft || !draft->state_set_data || !draft->state || !data || size <= 0) {
        return false;
    }
    return draft->state_set_data(draft->state, slot_id, data, size);
}

}  // extern "C"
