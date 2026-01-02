// Parallel Lazy Beam Search Implementation

#include "beam-search.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace llama_beam {

// Constructor
beam_search_engine::beam_search_engine(
    llama_context * ctx,
    const beam_search_params & params
) : ctx_(ctx),
    params_(params),
    current_step_(0),
    initialized_(false),
    step_callback_(nullptr)
{
    if (!ctx_) {
        fprintf(stderr, "beam_search_engine: ctx is null\n");
        return;
    }
    
    // Reserve space for beams and candidates
    beams_.reserve(params_.beam_size);
    candidates_.reserve(params_.beam_size * 10);  // Heuristic: beam_size * avg_top_k
}

// Destructor
beam_search_engine::~beam_search_engine() {
    // Cleanup: remove all sequences from KV cache
    llama_memory_t mem = llama_get_memory(ctx_);
    for (const auto & beam : beams_) {
        if (beam.seq_id >= 0) {
            llama_memory_seq_rm(mem, beam.seq_id, -1, -1);
        }
    }
}

// Initialize beam search
void beam_search_engine::initialize(const std::vector<llama_token> & initial_tokens) {
    if (initial_tokens.empty()) {
        fprintf(stderr, "beam_search_engine: initial_tokens is empty\n");
        return;
    }
    
    // Clear any previous state
    beams_.clear();
    candidates_.clear();
    current_step_ = 0;
    
    // Create initial beam
    beam_hypothesis initial_beam;
    initial_beam.tokens = initial_tokens;
    initial_beam.score = 0.0f;
    initial_beam.normalized_score = 0.0f;
    initial_beam.seq_id = 0;  // Use seq_id 0 for first beam
    initial_beam.finished = false;
    
    beams_.push_back(initial_beam);
    
    initialized_ = true;
    
    fprintf(stderr, "[BeamSearch] Initialized with %zu tokens, beam_size=%d\n",
            initial_tokens.size(), params_.beam_size);
}

// Get top-K tokens from logits
std::vector<std::pair<llama_token, float>> beam_search_engine::get_top_k_tokens(
    const float * logits,
    int n_vocab,
    int k
) const {
    if (k <= 0 || k > n_vocab) {
        k = n_vocab;  // Use all if k is invalid
    }
    
    // Create pairs of (token, log_prob)
    std::vector<std::pair<llama_token, float>> token_probs;
    token_probs.reserve(n_vocab);
    
    for (int i = 0; i < n_vocab; ++i) {
        token_probs.push_back({i, logits[i]});
    }
    
    // Partial sort to get top-K
    std::partial_sort(
        token_probs.begin(),
        token_probs.begin() + k,
        token_probs.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; }
    );
    
    // Return top-K
    token_probs.resize(static_cast<size_t>(k));
    return token_probs;
}

// Apply length penalty
float beam_search_engine::apply_length_penalty(float score, int length) const {
    if (!params_.normalize_scores || params_.length_penalty_alpha == 0.0f) {
        return score;
    }
    
    // Formula: score / (length ^ alpha)
    float penalty = std::pow(static_cast<float>(length), params_.length_penalty_alpha);
    return score / penalty;
}

// Compute normalized score
float beam_search_engine::compute_score(const beam_hypothesis & hyp) const {
    return apply_length_penalty(hyp.score, hyp.tokens.size());
}

// Expand all beams in parallel
void beam_search_engine::expand_beams(std::function<bool(llama_token)> is_eos) {
    candidates_.clear();
    
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_));
    const int n_vocab = llama_vocab_n_tokens(vocab);
    
    // Determine how many candidates to generate per beam
    int k_per_beam = params_.top_k_per_beam > 0 ? 
                     params_.top_k_per_beam : 
                     params_.beam_size;  // Default: beam_size candidates per beam
    
    // Step 1: Batch decode all active beams
    llama_batch batch = llama_batch_init(params_.beam_size, 0, params_.beam_size);
    int n_active = 0;
    
    for (size_t b = 0; b < beams_.size(); ++b) {
        if (beams_[b].finished) {
            continue;  // Skip finished beams
        }
        
        const auto & beam = beams_[b];
        llama_token last_token = beam.tokens.back();
        int pos = beam.tokens.size() - 1;
        
        batch.token[n_active] = last_token;
        batch.pos[n_active] = pos;
        batch.n_seq_id[n_active] = 1;
        batch.seq_id[n_active][0] = beam.seq_id;
        batch.logits[n_active] = true;  // We need logits
        
        n_active++;
    }
    
    if (n_active == 0) {
        // All beams finished
        batch.n_tokens = 0;
        llama_batch_free(batch);
        return;
    }
    
    batch.n_tokens = n_active;
    
    // Decode all beams in one forward pass
    if (llama_decode(ctx_, batch) != 0) {
        fprintf(stderr, "[BeamSearch] llama_decode failed at step %d\n", current_step_);
        llama_batch_free(batch);
        return;
    }
    
    // Step 2: Expand each beam (lazy - don't copy KV caches yet)
    int active_idx = 0;
    for (int b = 0; b < static_cast<int>(beams_.size()); ++b) {
        if (beams_[b].finished) {
            continue;
        }
        
        const auto & beam = beams_[b];
        
        // Get logits for this beam
        const float * logits = llama_get_logits_ith(ctx_, active_idx);
        active_idx++;
        
        // Get top-K tokens
        auto top_k = get_top_k_tokens(logits, n_vocab, k_per_beam);
        
        // Create candidates
        for (const auto & [token, log_prob] : top_k) {
            // Check if we should skip this token (EOS before min_length, etc.)
            if (is_eos(token) && (int)beam.tokens.size() < params_.min_length) {
                continue;  // Don't allow EOS before min_length
            }
            
            // Create candidate
            beam_candidate candidate;
            candidate.hyp = beam;  // Copy beam
            candidate.hyp.tokens.push_back(token);
            candidate.hyp.score = beam.score + log_prob;
            candidate.hyp.normalized_score = compute_score(candidate.hyp);
            candidate.hyp.finished = is_eos(token);
            
            candidate.parent_beam_idx = b;
            candidate.parent_seq_id = beam.seq_id;
            candidate.last_token = token;
            candidate.token_log_prob = log_prob;
            
            // Apply score threshold
            if (candidate.hyp.normalized_score < params_.score_threshold) {
                continue;
            }
            
            candidates_.push_back(candidate);
        }
    }
    
    llama_batch_free(batch);
    
    fprintf(stderr, "[BeamSearch] Step %d: Generated %zu candidates from %d active beams\n",
            current_step_, candidates_.size(), n_active);
}

// Prune candidates to top beam_size
void beam_search_engine::prune_candidates() {
    if (candidates_.empty()) {
        return;
    }
    
    // Sort candidates by normalized score (descending)
    std::sort(candidates_.begin(), candidates_.end(), compare_candidates_by_score);
    
    // Keep top beam_size (or all finished beams + top incomplete beams)
    int n_finished = 0;
    for (const auto & c : candidates_) {
        if (c.hyp.finished) {
            n_finished++;
        }
    }
    
    int n_keep = params_.beam_size;
    if (params_.early_stopping && n_finished >= params_.beam_size) {
        // Keep all finished beams
        n_keep = n_finished;
    }
    
    n_keep = std::min(n_keep, (int)candidates_.size());
    candidates_.resize(n_keep);
    
    fprintf(stderr, "[BeamSearch] Pruned to %d candidates (%d finished)\n",
            n_keep, n_finished);
}

// Rearrange KV caches to match new beam assignments
void beam_search_engine::rearrange_kv_caches() {
    // Now we need to assign the top candidates to seq_ids [0, beam_size-1]
    // This is where the "lazy" optimization happens:
    // - Only copy KV cache if the winner's parent_seq_id != target seq_id
    
    llama_memory_t mem = llama_get_memory(ctx_);
    
    std::vector<beam_hypothesis> new_beams;
    new_beams.reserve(params_.beam_size);
    
    for (int i = 0; i < (int)candidates_.size() && i < params_.beam_size; ++i) {
        const auto & candidate = candidates_[i];
        beam_hypothesis new_beam = candidate.hyp;
        
        // Assign seq_id
        int target_seq_id = i;
        
        if (candidate.parent_seq_id != target_seq_id) {
            // Need to copy KV cache from parent to target slot
            fprintf(stderr, "[BeamSearch] Copying KV cache: seq %d → seq %d\n",
                    candidate.parent_seq_id, target_seq_id);
            
            // Clear target slot first
            llama_memory_seq_rm(mem, target_seq_id, -1, -1);
            
            // Copy from parent
            llama_memory_seq_cp(mem, candidate.parent_seq_id, target_seq_id, -1, -1);
        }
        
        new_beam.seq_id = target_seq_id;
        new_beams.push_back(new_beam);
    }
    
    beams_ = new_beams;
    
    fprintf(stderr, "[BeamSearch] Rearranged to %zu beams\n", beams_.size());
}

// Single step of beam search
bool beam_search_engine::step(std::function<bool(llama_token)> is_eos) {
    if (!initialized_) {
        fprintf(stderr, "[BeamSearch] Not initialized\n");
        return false;
    }
    
    // Check if all beams are finished
    bool all_finished = true;
    for (const auto & beam : beams_) {
        if (!beam.finished) {
            all_finished = false;
            break;
        }
    }
    
    if (all_finished) {
        fprintf(stderr, "[BeamSearch] All beams finished\n");
        return false;
    }
    
    // Check max length
    if (current_step_ >= params_.max_length) {
        fprintf(stderr, "[BeamSearch] Max length reached\n");
        return false;
    }
    
    // Expand beams
    expand_beams(is_eos);
    
    // Prune to top beam_size
    prune_candidates();
    
    // Rearrange KV caches
    rearrange_kv_caches();
    
    // Increment step
    current_step_++;
    
    // Call callback if set
    if (step_callback_) {
        step_callback_(current_step_, beams_);
    }
    
    return true;
}

// Run full beam search
beam_search_result beam_search_engine::search(
    const std::vector<llama_token> & initial_tokens,
    std::function<bool(llama_token)> is_eos
) {
    // Initialize
    initialize(initial_tokens);
    
    // Run steps until done
    while (step(is_eos)) {
        // Continue
    }
    
    // Return results
    return get_results();
}

// Get final results
beam_search_result beam_search_engine::get_results() {
    beam_search_result result;
    result.hypotheses = beams_;
    result.n_steps = current_step_;
    result.stopped_early = false;
    
    // Check if we stopped early
    int n_finished = 0;
    for (const auto & beam : beams_) {
        if (beam.finished) {
            n_finished++;
        }
    }
    
    if (params_.early_stopping && n_finished >= params_.beam_size) {
        result.stopped_early = true;
    }
    
    // Sort by score
    std::sort(result.hypotheses.begin(), result.hypotheses.end(),
              compare_hypotheses_by_score);
    
    return result;
}

// Set step callback
void beam_search_engine::set_step_callback(step_callback_t callback) {
    step_callback_ = callback;
}

// Print hypothesis
void print_hypothesis(
    const beam_hypothesis & hyp,
    const llama_vocab * vocab,
    const char * prefix
) {
    fprintf(stderr, "%sScore: %.4f (normalized: %.4f), Tokens: %zu, Finished: %s\n",
            prefix, hyp.score, hyp.normalized_score, hyp.tokens.size(),
            hyp.finished ? "yes" : "no");
    fprintf(stderr, "%s  Tokens: [", prefix);
    for (size_t i = 0; i < hyp.tokens.size(); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", hyp.tokens[i]);
    }
    fprintf(stderr, "]\n");
}

} // namespace llama_beam



