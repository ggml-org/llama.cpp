// System One (Jev-compatible) typed decision readout.
//
// One forward pass over [state | question 1 .. question n], reading the label-token
// logits at each question's answer slot. No decoding, no grammar: the answer is a
// distribution over a small label set, which is what makes the output typed by
// construction.
//
// The prompt format belongs to the checkpoint: a Jinja template stored in the GGUF as the
// named chat template `system_one`. There is no built-in default, so a model without one is
// an error.
//
// The state and each question block are tokenized separately and concatenated, which differs
// from a single-pass encode at the seams. The template marks those seams with
// `system_one.segment_separator` (U+001E).

#pragma once

#include "llama.h"

#include <string>
#include <vector>

// These calls throw: std::invalid_argument when the request or template as stated cannot be
// answered, std::runtime_error when the checkpoint cannot, or when the numbers handed back do
// not fit the plan.

enum system_one_kind {
    SYSTEM_ONE_KIND_NOUL,
    SYSTEM_ONE_KIND_CHOICE,
    SYSTEM_ONE_KIND_SCORE,
};

// Where an answer is read, and what is read there.
//   LETTER_SLOT  the next-token distribution at the end of a question's segment
//   MASKED_SLOT  the distribution at a mask token inside it (bidirectional models)
//   RANK_HEAD    one sequence per option, scored by the model's classification head -- the
//                reranker shape, and the only readout that costs K forward passes
enum system_one_readout {
    SYSTEM_ONE_READOUT_LETTER_SLOT,
    SYSTEM_ONE_READOUT_MASKED_SLOT,
    SYSTEM_ONE_READOUT_RANK_HEAD,
};

const char * system_one_readout_name(enum system_one_readout readout);

struct system_one_question {
    enum system_one_kind     kind = SYSTEM_ONE_KIND_NOUL;
    std::string              text;
    std::vector<std::string> options;   // noul: {"no","yes"}; score: levels in order
    std::vector<std::string> descs;     // empty, or one per option
};

// What the checkpoint says about its own prompt and readout. Three of these come from
// `system_one.*` keys; the rest are standard GGUF metadata, because a mask token, a leading BOS
// and an attention direction all have homes of their own already.
struct system_one_params {
    std::string template_src;                             // tokenizer.chat_template.system_one
    std::string segment_separator = "\x1e";               // system_one.segment_separator

    // Derived from the model's capabilities in system_one_params_from_model(), not required
    // in the GGUF: a classification head means RANK_HEAD, a bidirectional model means
    // MASKED_SLOT, and a causal one means LETTER_SLOT.
    enum system_one_readout readout = SYSTEM_ONE_READOUT_LETTER_SLOT;

    // system_one.labels: option i is labelled labels[i] and that token is what gets read.
    // Defaulted to A-Za-z when the checkpoint says nothing, which is why a failure to resolve
    // one has to say where the label came from -- the caller may never have chosen it.
    std::vector<std::string> labels;
    bool labels_are_default = false;

    // from the standard metadata
    // {arch}.attention.causal, and what the readout is derived from. It is also the whole of
    // whether a prefix's KV can be reused: see system_one_sequence::n_reusable.
    bool causal = true;

    llama_token mask_token = -1;    // tokenizer.ggml.mask_token_id
    std::string mask_text;          // its piece, so a template can write it
    std::string bos_text;           // likewise for BOS: the template writes it, as chat
    std::string eos_text;           // templates do, instead of a flag saying "prepend one"
};

// Reads the metadata. Fails when the model carries no System One template.
system_one_params system_one_params_from_model(const llama_model * model);

// Render the template, then split on the separator. The last `n_questions` segments are the
// question blocks.
std::vector<std::string> system_one_render_segments(const system_one_params & cfg,
                                                    const std::string & state,
                                                    const std::vector<system_one_question> & qs);

struct system_one_tokenized {
    std::vector<llama_token> ids;
    std::vector<int>         slots;   // one per question: where its answer is read
};

// One sequence the request turns into. A slot readout plans exactly one; a rank_head readout
// plans one per option.
struct system_one_sequence {
    // stage one: what this sequence renders to. The trailing n_question_segments are the
    // question blocks; everything before them is the state.
    std::vector<std::string> segments;
    size_t n_question_segments = 0;

    size_t question = 0;   // rank_head: the question this sequence scores
    size_t option   = 0;   // rank_head: the option it scores

    // stage two: what that tokenized to, and where the answers sit in it
    system_one_tokenized tok;

    // A prefix's KV may be reused only where the prefix's representation does not depend
    // on what follows it -- that is, only on a causal model. Both numbers are already zero
    // when it does not hold, so a caller never has to ask again.

    // How far a reused prefix may extend into this sequence. Short of the whole sequence
    // when an answer is read from inside it: a slot covered by the reused part is never
    // decoded and has no logits.
    size_t n_reusable  = 0;

    // How many leading tokens this sequence has in common with an earlier one of the same
    // question, and which that is -- the state and the question, for a readout that scores
    // one sequence per option. Decode that once and copy its KV rather than repeating it.
    size_t shares_with = 0;
    size_t n_shared    = 0;
};

// A request, resolved: which sequences to run, where the answers are in them, and what the
// numbers coming back mean. Every decision that depends on the readout is made here, so a
// caller only has to run the sequences and hand the numbers back.
struct system_one_plan {
    std::vector<system_one_sequence> sequences;
    std::vector<int>                 n_options;   // per question, in request order
    std::vector<llama_token>         labels;      // label token ids; empty for rank_head

    bool rank_pooling = false;  // sequences are scored by the model's head, not read at a slot
};

// Stage one: what the request is, before anything is tokenized.
system_one_plan system_one_build_plan(const system_one_params & cfg,
                                      const llama_vocab * vocab,
                                      const std::string & state,
                                      const std::vector<system_one_question> & qs_in);

// Stage two, for a caller whose input is all text.
void system_one_tokenize_plan(const llama_vocab * vocab, const system_one_params & cfg,
                              system_one_plan & p);

system_one_tokenized system_one_resolve_question_slots(const llama_vocab * vocab,
                                                       const system_one_params & cfg,
                                                       const std::vector<std::string> & question_segments,
                                                       size_t prefix_positions);


// One BOS (when the model asks for it), then every segment with add_special = false.
system_one_tokenized system_one_tokenize_segments(const llama_vocab * vocab,
                                                  const system_one_params & cfg,
                                                  const std::vector<std::string> & segments,
                                                  size_t n_questions);


// Token ids of the label pieces, in `labels` order. Every label must be a single token, since
// the answer is read as one position's distribution over them.
std::vector<llama_token> system_one_label_tokens(const llama_vocab * vocab,
                                                 const std::vector<std::string> & labels,
                                                 bool labels_are_default = false);

struct system_one_answer {
    int                choice = 0;      // argmax over the labels
    float              confidence = 0;  // 1 - H(p)/ln K
    std::vector<float> logits;
    std::vector<float> probs;
};

// Turn one row of vocab logits into a typed answer: gather the first `n_options` label
// tokens, softmax over just those, argmax, and the entropy-normalised confidence.
system_one_answer system_one_answer_from_logits(const float * row,
                                                const std::vector<llama_token> & letter_ids,
                                                int n_options);

// Rescale an answer by a temperature and recompute the probabilities, argmax and confidence.
// T = 1 returns immediately.
void system_one_apply_temperature(system_one_answer & a, float t);

// Expected value over an ordered rubric: sum(i * p_i), the `score` question type.
float system_one_score_expectation(const system_one_answer & a);

// The same, from one score per option -- what a classification head produces.
system_one_answer system_one_answer_from_scores(const std::vector<float> & scores);

// A whole request's answers in one call. The plan supplies the labels and the option counts,
// so the caller only says where the numbers are:
//
//   _from_logits  a slot readout: one row of vocab logits per answer slot, in question order
//   _from_scores  a rank_head readout: one score per planned sequence, in plan order
std::vector<system_one_answer> system_one_answers_from_logits(const system_one_plan & p,
                                                              const std::vector<const float *> & rows);

std::vector<system_one_answer> system_one_answers_from_scores(const system_one_plan & p,
                                                              const std::vector<float> & scores);

// What the readout requires of a context: a rank_head answer is read from a pooled embedding.
struct system_one_context_needs {
    bool                    embeddings   = false;
    enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
};

system_one_context_needs system_one_required_context(const system_one_params & cfg);
