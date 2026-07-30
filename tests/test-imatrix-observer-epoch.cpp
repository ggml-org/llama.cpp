#include "../src/llama-graph.h"

#include <cassert>

int main() {
    // Graph reuse is safe while the observer selection is unchanged.  A
    // progressive freeze/probe transition changes graph topology, however,
    // and must force the context to rebuild it.
    llm_graph_params previous{};
    llm_graph_params current{};

    assert(previous.allow_reuse(current));

    current.cparams.imatrix_observer_epoch = 1;
    assert(!previous.allow_reuse(current));

    previous.cparams.imatrix_observer_epoch = 1;
    assert(previous.allow_reuse(current));

    return 0;
}
