#pragma once

//
// tessera-dataset.h
//
// Dataset preparation from llama.tessera.spec.v1 telemetry JSONL (produced by
// llama-imatrix --telemetry-out --telemetry-topk K). Converts raw per-step
// acceptance traces into training data for the drafter fine-tuning pipeline.
//
// Three output modes:
//   text  - accepted token sequences as space-separated token IDs, one
//           sequence per line. Suitable for llama-finetune --train-data
//           after detokenization, or for a token-ID-aware training loop.
//   pairs - rejection-sampling pairs as JSONL:
//           {"context":[...],"drafted":[...],"accepted":[...],"n_acc":N}
//           One record per spec step. The drafted sequence is the negative
//           example; the accepted sequence is the positive.
//   lk    - LK loss training data as JSONL:
//           {"position":I,"p_tokens":[...],"p_probs":[...],
//            "q_tokens":[...],"q_probs":[...],"accepted":bool}
//           One record per position per spec step. p = verifier, q = drafter.
//           This is the input format for direct acceptance-rate optimization.
//   dflash - block-structured data for DFlash/D-PACE training as JSONL:
//           {"schema":"llama.tessera.dflash-block.v1","block_size":B,
//            "target_tokens":[...],"acceptance_probs":[...],
//            "dpace_weights":[...],"decay_weights":[...],
//            "n_acc":M,"n_dft":N,"surrogate":S}
//           One record per spec step (one drafted block). target_tokens[j] is
//           the verifier argmax at drafted position j (the ground-truth token
//           the block drafter should emit); acceptance_probs[j] is the
//           verifier softmax prob of the drafter's pick (the D-PACE acceptance
//           proxy). dpace_weights are the smoothed+normalized adaptive D-PACE
//           weights, baked in so the training driver can pre-weight a standard
//           cross-entropy label without recomputing them. decay_weights are the
//           normalized fixed DFlash exponential-decay baseline for A/B.
//
// D-PACE needs no dedicated ggml-opt loss type. The weights are detached from
// the gradient, so the training driver multiplies a standard one-hot/dense
// cross-entropy label row by dpace_weights[j] per position and feeds the
// existing CE objective. Baking the weights here keeps that path a data
// concern rather than a new graph op.
//

#include <string>

enum ts_dataset_mode {
    TS_DATASET_MODE_TEXT,
    TS_DATASET_MODE_PAIRS,
    TS_DATASET_MODE_LK,
    TS_DATASET_MODE_DFLASH,  // block-structured data for DFlash/D-PACE training
};

struct ts_dataset_params {
    char input_path[1024];   // llama.tessera.spec.v1 JSONL
    char output_path[1024];  // output file
    ts_dataset_mode mode;
    int  min_accepted;       // skip steps with fewer accepted tokens (default 1)
    float dpace_alpha;       // D-PACE asymmetric smoothing floor (dflash mode)
    float dflash_gamma;      // DFlash decay baseline gamma (dflash mode)
};

void ts_dataset_default_params(ts_dataset_params * p);

// Parse a mode string: "text", "pairs", "lk", "dflash". Returns 0 on success.
int ts_dataset_mode_from_string(const char * s, ts_dataset_mode * out);

// Run dataset preparation. Returns 0 on success, non-zero on error.
// *n_records_out is set to the number of output records written.
// *n_skipped_out, if non-null, is set to the number of input records skipped
// (wrong schema, below min_accepted, or missing fields for the mode).
int ts_dataset_run(const ts_dataset_params * params,
                   int * n_records_out,
                   int * n_skipped_out,
                   std::string * err_msg);
