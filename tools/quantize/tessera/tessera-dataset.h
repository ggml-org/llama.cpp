#pragma once

//
// tessera-dataset.h
//
// Dataset preparation from llama.spec_calib.v2 telemetry JSONL (produced by
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
//

#include <string>

enum ts_dataset_mode {
    TS_DATASET_MODE_TEXT,
    TS_DATASET_MODE_PAIRS,
    TS_DATASET_MODE_LK,
};

struct ts_dataset_params {
    char input_path[1024];   // llama.spec_calib.v2 JSONL
    char output_path[1024];  // output file
    ts_dataset_mode mode;
    int  min_accepted;       // skip steps with fewer accepted tokens (default 1)
};

void ts_dataset_default_params(ts_dataset_params * p);

// Parse a mode string: "text", "pairs", "lk". Returns 0 on success.
int ts_dataset_mode_from_string(const char * s, ts_dataset_mode * out);

// Run dataset preparation. Returns 0 on success, non-zero on error.
// *n_records_out is set to the number of output records written.
int ts_dataset_run(const ts_dataset_params * params,
                   int * n_records_out,
                   std::string * err_msg);
