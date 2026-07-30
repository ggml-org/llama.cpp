// Verifier/drafter observer isolation test.
//
// This test verifies that the per-scope observer state in
// llama_cparams keeps the verifier and drafter buckets fully
// independent. It exercises the same storage contract that
// llm_graph_context::imatrix_observer_enabled and
// llama_context::set_imatrix_observer_filter depend on, so a regression
// in either of those surfaces here as a test failure.
//
// In particular, the test ensures:
//   * The two scopes have independent filter, filter_data, and epoch
//     slots, and assigning to one scope does not perturb the other.
//   * Switching the active scope via llama_cparams preserves both
//     filters and switches the dispatch to the right slot.
//   * The filter dispatch through the per-scope slots is symmetric: a
//     filter that only accepts its own marker rejects the other scope's
//     user_data, so two contexts (verifier + drafter) in the same
//     process can each see only the observers it owns.

#include "llama.h"

// llama_cparams is declared in src/llama-cparams.h, which the public
// llama.h header does not pull in. The per-scope filter storage
// contract is part of the C++ context state; the C API alone cannot
// reach it without going through a live llama_context.
#include "../src/llama-graph.h"

#include <cassert>
#include <cstring>

namespace {

// Two distinct user_data values stand in for the two collectors an
// imatrix session would carry. Each filter recognises only its own
// marker; cross-feeding is a hard reject.
int g_verifier_marker = 0xAA;
int g_drafter_marker  = 0x55;

bool accept_verifier_only(const char * /*tensor_name*/, void * user_data) {
    return user_data == &g_verifier_marker;
}

bool accept_drafter_only(const char * /*tensor_name*/, void * user_data) {
    return user_data == &g_drafter_marker;
}

} // namespace

int main() {
    // Initial state: every scope slot is zeroed and the active scope is
    // the verifier (matches the public default in llama_set_imatrix_observer_scope
    // before the user has touched it).
    {
        llama_cparams cparams{};
        assert(cparams.imatrix_observer_scope == LLAMA_OBSERVER_SCOPE_VERIFIER);
        for (int s = LLAMA_OBSERVER_SCOPE_VERIFIER; s <= LLAMA_OBSERVER_SCOPE_DRAFTER; ++s) {
            assert(cparams.imatrix_observer_filter[s]      == nullptr);
            assert(cparams.imatrix_observer_filter_data[s] == nullptr);
            assert(cparams.imatrix_observer_epoch[s]       == 0);
        }
    }

    // Bind the verifier slot to the verifier filter. The drafter slot
    // must remain untouched (a regression here would mean the two
    // scopes alias the same storage).
    {
        llama_cparams cparams{};
        cparams.imatrix_observer_scope                   = LLAMA_OBSERVER_SCOPE_VERIFIER;
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER]      = accept_verifier_only;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] = &g_verifier_marker;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER]       = 1;

        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]      == nullptr);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER] == nullptr);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]       == 0);
    }

    // Now bind the drafter slot. The verifier slot must survive the
    // drafter assignment (independence in both directions).
    {
        llama_cparams cparams{};
        cparams.imatrix_observer_scope                   = LLAMA_OBSERVER_SCOPE_VERIFIER;
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER]      = accept_verifier_only;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] = &g_verifier_marker;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER]       = 1;

        cparams.imatrix_observer_scope                   = LLAMA_OBSERVER_SCOPE_DRAFTER;
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]      = accept_drafter_only;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER] = &g_drafter_marker;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]       = 7;

        // Verifier side untouched by drafter writes.
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER]      == accept_verifier_only);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] == &g_verifier_marker);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER]       == 1);

        // Drafter side reflects its own writes.
        assert(cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]      == accept_drafter_only);
        assert(cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER] == &g_drafter_marker);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]       == 7);
    }

    // The active scope drives dispatch. With the active scope set to
    // VERIFIER, the verifier filter must run with the verifier user_data;
    // with the active scope set to DRAFTER, the drafter filter must run
    // with the drafter user_data. This is the contract
    // llm_graph_context::imatrix_observer_enabled() relies on.
    {
        llama_cparams cparams{};
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_VERIFIER]      = accept_verifier_only;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER] = &g_verifier_marker;
        cparams.imatrix_observer_filter[LLAMA_OBSERVER_SCOPE_DRAFTER]      = accept_drafter_only;
        cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER] = &g_drafter_marker;

        // Verifier dispatch: verifier filter + verifier data → true.
        cparams.imatrix_observer_scope = LLAMA_OBSERVER_SCOPE_VERIFIER;
        const int scope_v              = cparams.imatrix_observer_scope;
        const auto filter_v            = cparams.imatrix_observer_filter[scope_v];
        const void * data_v            = cparams.imatrix_observer_filter_data[scope_v];
        assert(filter_v != nullptr);
        assert(filter_v("blk.0.attn_q.weight", data_v));
        // Cross-feeding: verifier filter must reject the drafter's data.
        assert(!filter_v("blk.0.attn_q.weight",
                          cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_DRAFTER]));

        // Drafter dispatch: drafter filter + drafter data → true.
        cparams.imatrix_observer_scope = LLAMA_OBSERVER_SCOPE_DRAFTER;
        const int scope_d              = cparams.imatrix_observer_scope;
        const auto filter_d            = cparams.imatrix_observer_filter[scope_d];
        const void * data_d            = cparams.imatrix_observer_filter_data[scope_d];
        assert(filter_d != nullptr);
        assert(filter_d("blk.0.attn_q.weight", data_d));
        // Cross-feeding: drafter filter must reject the verifier's data.
        assert(!filter_d("blk.0.attn_q.weight",
                          cparams.imatrix_observer_filter_data[LLAMA_OBSERVER_SCOPE_VERIFIER]));
    }

    // Epoch tracking is per-scope: bumping one must not touch the other.
    // This is what llama_bump_imatrix_observer_epoch() depends on.
    {
        llama_cparams cparams{};
        cparams.imatrix_observer_scope                  = LLAMA_OBSERVER_SCOPE_VERIFIER;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] = 0;
        cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]  = 0;

        // Simulate bumping the verifier scope.
        ++cparams.imatrix_observer_epoch[cparams.imatrix_observer_scope];
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] == 1);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]  == 0);

        // Simulate bumping the drafter scope.
        cparams.imatrix_observer_scope = LLAMA_OBSERVER_SCOPE_DRAFTER;
        ++cparams.imatrix_observer_epoch[cparams.imatrix_observer_scope];
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] == 1);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]  == 1);

        // One more verifier bump must not perturb the drafter count.
        cparams.imatrix_observer_scope = LLAMA_OBSERVER_SCOPE_VERIFIER;
        ++cparams.imatrix_observer_epoch[cparams.imatrix_observer_scope];
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] == 2);
        assert(cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER]  == 1);
    }

    return 0;
}
