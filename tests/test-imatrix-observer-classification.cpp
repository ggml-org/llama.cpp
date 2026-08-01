#include "ggml.h"
#include "llama.h"

// llama_cparams is declared in src/llama-cparams.h, which the public llama.h
// header does not pull in. We need it here to assert the per-scope filter
// storage contract.
#include "../src/llama-graph.h"

#include <cassert>
#include <cstddef>
#include <cstring>

namespace {

// Marker values that simulate the two collectors a real imatrix session
// would attach: one for the verifier model, one for the drafter.
int g_verifier_marker = 0x1;
int g_drafter_marker  = 0x2;

bool verifier_filter(const char * /*tensor_name*/, void * user_data) {
    return user_data == &g_verifier_marker;
}

bool drafter_filter(const char * /*tensor_name*/, void * user_data) {
    return user_data == &g_drafter_marker;
}

} // namespace

int main() {
    // -- Part 1: ggml observer tensor classification --
    // The stats view is the half of an imatrix observer cast whose offset and
    // element count identify it as the importance accumulator; the storage
    // and activation halves are the activation path. This is the contract
    // ggml_imatrix_observer_is_stats depends on.
    {
        ggml_init_params params = {
            /*.mem_size   =*/16 * 1024 * 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml_context * ctx = ggml_init(params);
        assert(ctx != nullptr);

        ggml_tensor * activations     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2048, 128, 2, 1);
        ggml_tensor * weight_anchor   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 2048);
        ggml_tensor * stats           = nullptr;
        ggml_tensor * activation_view = ggml_imatrix_observer_cast(ctx, activations, weight_anchor, &stats);
        ggml_tensor * storage         = activation_view->view_src;

        assert(storage != nullptr);
        assert(storage == stats->view_src);
        assert(storage->op == GGML_OP_IMATRIX_OBSERVER);
        assert(storage->type == GGML_TYPE_F16);
        assert(activation_view->op == GGML_OP_VIEW);
        assert(activation_view->type == GGML_TYPE_F16);
        assert(stats->op == GGML_OP_VIEW);
        assert(stats->type == GGML_TYPE_F32);

        assert(!ggml_imatrix_observer_is_stats(storage));
        assert(!ggml_imatrix_observer_is_stats(activation_view));
        assert(ggml_imatrix_observer_is_stats(stats));

        const size_t original_offset = stats->view_offs;
        stats->view_offs             = 0;
        assert(!ggml_imatrix_observer_is_stats(stats));
        stats->view_offs = original_offset;

        const int64_t original_elements = stats->ne[0];
        stats->ne[0]                    = original_elements - 1;
        assert(!ggml_imatrix_observer_is_stats(stats));
        stats->ne[0] = original_elements;
        assert(ggml_imatrix_observer_is_stats(stats));

        ggml_free(ctx);
    }

    // -- Part 2: per-scope observer filter storage --
    // The verifier and drafter scopes are independent storage slots in
    // llama_cparams. Setting one must not clobber the other, and the active
    // scope is the value the user last set via
    // llama_set_imatrix_observer_scope.
    {
        llama_cparams cparams{};
        assert(cparams.imatrix_observer_scope == LLAMA_OBSERVER_SCOPE_VERIFIER);
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER] == nullptr);
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]  == nullptr);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] == nullptr);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER]  == nullptr);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] == 0);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]  == 0);

        // Set the verifier slot to the verifier filter + marker.
        cparams.imatrix_observer_scope                                       = LLAMA_OBSERVER_SCOPE_VERIFIER;
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER]       = verifier_filter;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] = &g_verifier_marker;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER]       = 5;

        // Setting the drafter slot must not disturb the verifier slot.
        cparams.imatrix_observer_scope                                       = LLAMA_OBSERVER_SCOPE_DRAFTER;
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]       = drafter_filter;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER] = &g_drafter_marker;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]       = 9;

        // Both scopes retain their own filter, user_data, and epoch.
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER] == verifier_filter);
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]  == drafter_filter);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] == &g_verifier_marker);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER]  == &g_drafter_marker);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] == 5);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]  == 9);

        // The active scope is what the user last set. This is the value the
        // graph build dispatches against.
        assert(cparams.imatrix_observer_scope == LLAMA_OBSERVER_SCOPE_DRAFTER);
        cparams.imatrix_observer_scope = LLAMA_OBSERVER_SCOPE_VERIFIER;
        assert(cparams.imatrix_observer_scope == LLAMA_OBSERVER_SCOPE_VERIFIER);
        // Per-scope state survived the scope switch.
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER] == verifier_filter);
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]  == drafter_filter);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] == &g_verifier_marker);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER]  == &g_drafter_marker);

        // Filter dispatch is independent per scope. The verifier filter must
        // reject the drafter's user_data (and vice versa) so a context can
        // carry two distinct filter contracts at once.
        const auto v_filter = cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER];
        const auto d_filter = cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER];
        assert(v_filter != nullptr);
        assert(d_filter != nullptr);
        assert(v_filter("blk.0.attn_q.weight", cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER]));
        assert(!v_filter("blk.0.attn_q.weight", cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER]));
        assert(d_filter("blk.0.attn_q.weight", cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER]));
        assert(!d_filter("blk.0.attn_q.weight", cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER]));
    }

    return 0;
}
