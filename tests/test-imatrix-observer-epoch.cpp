#include "../src/llama-graph.h"

#include <cassert>

int main() {
    // Graph reuse is safe while the observer selection is unchanged.  A
    // progressive freeze/probe transition changes graph topology, however,
    // and must force the context to rebuild it.  Both the verifier and
    // drafter scopes own an independent epoch so a transition in either
    // bucket invalidates the cached topology.
    llm_graph_params previous{};
    llm_graph_params current{};

    assert(previous.allow_reuse(current));

    // Verifier-scope transition invalidates reuse.
    current.cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] = 1;
    assert(!previous.allow_reuse(current));

    previous.cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_VERIFIER] = 1;
    assert(previous.allow_reuse(current));

    // Drafter-scope transition invalidates reuse even when the verifier
    // epoch is unchanged.
    current.cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER] = 7;
    assert(!previous.allow_reuse(current));

    previous.cparams.imatrix_observer_epoch[LLAMA_OBSERVER_SCOPE_DRAFTER] = 7;
    assert(previous.allow_reuse(current));

    return 0;
}
