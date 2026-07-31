#pragma once

//
// tessera-adapt.h
//
// Adaptation engine (docs/self-improving-loop-design.md section 4.5 and
// ratified decision 3: in-repo, first-class). v1 is the guarded
// skeleton: score the candidate eval, run the collapse guard against an
// optional baseline, and write a schema-versioned receipt. It never
// adapts into a regression - if the general-competence guard fails, the
// run is recorded as blocked and signalled to the caller.
//

struct ts_adapt_params {
    bool   dry_run;                    // record intent only; no adapter is produced either way in v1
    char   input_eval_path[1024];      // versioned capability-eval JSON (tessera-capability-eval)
    char   output_receipt_path[1024];  // schema-versioned adaptation receipt output
    double guard_epsilon;              // general-competence regression tolerance
};

void ts_adapt_default_params(ts_adapt_params * params);

// Run one adaptation step.
// Returns:
//    0  success (guard passed, receipt written)
//    1  collapse guard FAILED (receipt still written, recording the block)
//   -1  error (bad args, unreadable/malformed eval, unwritable receipt)
int ts_adapt_run(const ts_adapt_params * params);
