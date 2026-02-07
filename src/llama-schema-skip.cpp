// CPUOPTI: Schema-aware token skipping

#include "llama-schema-skip.h"
#include "llama-vocab.h"
#include "llama-impl.h"

#include <string>
#include <vector>

//
// Public API
//

llama_opt_skip_result llama_opt_schema_query_next(
        const llama_grammar * grammar,
        const llama_vocab   * vocab) {

    llama_opt_skip_result result;
    result.can_skip     = false;
    result.forced_token = -1;

    if (grammar == nullptr || vocab == nullptr) {
        return result;
    }

    // If grammar is in lazy mode and still awaiting trigger, don't skip
    if (grammar->lazy && grammar->awaiting_trigger) {
        return result;
    }

    // If no active stacks, grammar is done
    if (grammar->stacks.empty()) {
        return result;
    }

    // Check if the grammar allows exactly one token
    // Strategy: iterate vocab, build candidates, check which survive grammar rejection
    // This is O(vocab_size) — future optimization: precompute forced-token table

    llama_token forced = -1;
    int n_allowed = 0;

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    for (llama_token t = 0; t < n_vocab; t++) {
        // Use the internal vocab method for cached string access
        const std::string & piece = vocab->token_to_piece(t);
        if (piece.empty()) {
            continue;
        }

        // Convert string to code points for grammar checking
        std::vector<uint32_t> code_points;
        for (size_t i = 0; i < piece.size(); ) {
            uint8_t byte = (uint8_t) piece[i];
            uint32_t cp;
            if (byte < 0x80) {
                cp = byte;
                i += 1;
            } else if (byte < 0xE0) {
                cp = (byte & 0x1F) << 6;
                if (i + 1 < piece.size()) {
                    cp |= ((uint8_t)piece[i + 1] & 0x3F);
                }
                i += 2;
            } else if (byte < 0xF0) {
                cp = (byte & 0x0F) << 12;
                if (i + 1 < piece.size()) {
                    cp |= ((uint8_t)piece[i + 1] & 0x3F) << 6;
                }
                if (i + 2 < piece.size()) {
                    cp |= ((uint8_t)piece[i + 2] & 0x3F);
                }
                i += 3;
            } else {
                cp = (byte & 0x07) << 18;
                if (i + 1 < piece.size()) {
                    cp |= ((uint8_t)piece[i + 1] & 0x3F) << 12;
                }
                if (i + 2 < piece.size()) {
                    cp |= ((uint8_t)piece[i + 2] & 0x3F) << 6;
                }
                if (i + 3 < piece.size()) {
                    cp |= ((uint8_t)piece[i + 3] & 0x3F);
                }
                i += 4;
            }
            code_points.push_back(cp);
        }
        code_points.push_back(0); // null terminator

        // Check against each grammar stack
        bool rejected_by_all = true;
        for (const auto & stack : grammar->stacks) {
            llama_grammar_candidates cands = {
                { 0, code_points.data(), {0, 0}, t }
            };
            auto rejects = llama_grammar_reject_candidates_for_stack(
                grammar->rules, stack, cands);

            if (rejects.empty()) {
                // Not rejected by this stack — token is allowed
                rejected_by_all = false;
                break;
            }
        }

        if (!rejected_by_all) {
            n_allowed++;
            forced = t;

            // Early exit: if more than 1 token is allowed, can't skip
            if (n_allowed > 1) {
                return result;
            }
        }
    }

    if (n_allowed == 1) {
        result.can_skip     = true;
        result.forced_token = forced;
    }

    return result;
}

llama_opt_skip_sequence llama_opt_schema_query_sequence(
        const llama_grammar * grammar,
        const llama_vocab   * vocab,
        uint32_t              max_lookahead) {

    llama_opt_skip_sequence seq;
    seq.n_skip = 0;

    if (grammar == nullptr || vocab == nullptr || max_lookahead == 0) {
        return seq;
    }

    // Clone the grammar to walk forward without mutating the original
    llama_grammar * temp_grammar = llama_grammar_clone_impl(*grammar);
    if (temp_grammar == nullptr) {
        return seq;
    }

    for (uint32_t step = 0; step < max_lookahead; step++) {
        llama_opt_skip_result r = llama_opt_schema_query_next(temp_grammar, vocab);

        if (!r.can_skip) {
            break;
        }

        seq.tokens.push_back(r.forced_token);
        seq.n_skip++;

        // Advance the grammar state with this token
        const std::string & piece = vocab->token_to_piece(r.forced_token);
        llama_grammar_accept_str(*temp_grammar, piece);
    }

    llama_grammar_free_impl(temp_grammar);

    return seq;
}

//
// Skip accumulator
//

void llama_opt_skip_accumulator::push(llama_token token, llama_pos pos) {
    tokens_.push_back(token);
    positions_.push_back(pos);
}

const std::vector<llama_token> & llama_opt_skip_accumulator::tokens() const {
    return tokens_;
}

const std::vector<llama_pos> & llama_opt_skip_accumulator::positions() const {
    return positions_;
}

bool llama_opt_skip_accumulator::has_pending() const {
    return !tokens_.empty();
}

uint32_t llama_opt_skip_accumulator::n_pending() const {
    return (uint32_t) tokens_.size();
}

void llama_opt_skip_accumulator::clear() {
    tokens_.clear();
    positions_.clear();
}
