// Parallel Lazy Beam Search for llama.cpp
// Optimized for encoder-decoder models (NLLB, T5, etc.)

#pragma once

#include "llama.h"
#include <vector>
#include <functional>

namespace llama_beam {

// Configuration for beam search
struct beam_search_params {
    int beam_size = 5;                    // Number of beams to maintain
    float length_penalty_alpha = 1.0f;    // Length penalty (1.0 = neutral, >1.0 = favor longer)
    int max_length = 200;                 // Maximum tokens to generate
    bool early_stopping = true;           // Stop when all beams finish
    int min_length = 1;                   // Minimum tokens to generate
    float diversity_penalty = 0.0f;       // Diversity penalty (0.0 = disabled)
    
    // Advanced options
    int top_k_per_beam = 0;               // Top-K candidates per beam (0 = use all)
    float score_threshold = -1e9f;        // Minimum score threshold
    bool normalize_scores = true;         // Normalize scores by length
};

// Single beam hypothesis
struct beam_hypothesis {
    std::vector<llama_token> tokens;      // Generated tokens
    float score;                          // Cumulative log probability
    float normalized_score;               // Score / length^alpha
    llama_seq_id seq_id;                  // Sequence ID in KV cache
    bool finished;                        // Has this beam finished (EOS)?
    
    beam_hypothesis() 
        : score(0.0f), normalized_score(0.0f), seq_id(-1), finished(false) {}
};

// Candidate during expansion (before pruning)
struct beam_candidate {
    beam_hypothesis hyp;                  // The hypothesis
    int parent_beam_idx;                  // Which beam it came from
    llama_seq_id parent_seq_id;           // Parent's seq_id
    llama_token last_token;               // Token that was just added
    float token_log_prob;                 // Log prob of last token
    
    beam_candidate() 
        : parent_beam_idx(-1), parent_seq_id(-1), last_token(-1), token_log_prob(0.0f) {}
};

// Result of beam search
struct beam_search_result {
    std::vector<beam_hypothesis> hypotheses;  // All final hypotheses (sorted by score)
    int n_steps;                              // Number of decode steps taken
    bool stopped_early;                       // Did we hit early stopping?
    
    // Get best hypothesis
    const beam_hypothesis & best() const {
        return hypotheses.empty() ? 
            *(beam_hypothesis*)nullptr : hypotheses[0];
    }
};

// Main beam search engine
class beam_search_engine {
public:
    // Constructor
    beam_search_engine(
        llama_context * ctx,
        const beam_search_params & params
    );
    
    // Destructor
    ~beam_search_engine();
    
    // Run beam search
    // initial_tokens: Starting tokens (e.g., [EOS, target_lang])
    // is_eos: Function to check if token is EOS
    beam_search_result search(
        const std::vector<llama_token> & initial_tokens,
        std::function<bool(llama_token)> is_eos
    );
    
    // Step-by-step interface (for advanced control)
    void initialize(const std::vector<llama_token> & initial_tokens);
    bool step(std::function<bool(llama_token)> is_eos);  // Returns false when done
    beam_search_result get_results();
    
    // Callbacks for monitoring
    using step_callback_t = std::function<void(int step, const std::vector<beam_hypothesis>&)>;
    void set_step_callback(step_callback_t callback);
    
private:
    llama_context * ctx_;
    beam_search_params params_;
    
    std::vector<beam_hypothesis> beams_;
    std::vector<beam_candidate> candidates_;
    
    int current_step_;
    bool initialized_;
    
    step_callback_t step_callback_;
    
    // Internal methods
    void expand_beams(std::function<bool(llama_token)> is_eos);
    void prune_candidates();
    void rearrange_kv_caches();
    float compute_score(const beam_hypothesis & hyp) const;
    float apply_length_penalty(float score, int length) const;
    
    // Helper to get top-K tokens from logits
    std::vector<std::pair<llama_token, float>> get_top_k_tokens(
        const float * logits,
        int n_vocab,
        int k
    ) const;
};

// Utility functions

// Default EOS checker
inline bool is_eos_token(llama_token token, const llama_vocab * vocab) {
    return llama_vocab_is_eog(vocab, token);
}

// Print hypothesis for debugging
void print_hypothesis(
    const beam_hypothesis & hyp,
    const llama_vocab * vocab,
    const char * prefix = ""
);

// Compare hypotheses by score (for sorting)
inline bool compare_hypotheses_by_score(
    const beam_hypothesis & a,
    const beam_hypothesis & b
) {
    return a.normalized_score > b.normalized_score;
}

inline bool compare_candidates_by_score(
    const beam_candidate & a,
    const beam_candidate & b
) {
    return a.hyp.normalized_score > b.hyp.normalized_score;
}

} // namespace llama_beam



