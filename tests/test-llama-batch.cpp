#include "../src/llama-batch.h"
#include "../common/common.h"
#include "llama.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cstring>

/**
 * llama_batch/sbatch/ubatch Test Program
 * Tests the basic principles and functionality of batch processing
 * Focuses on split_simple operation and state modifications
 * 
 * Data Flow Diagram:
 * ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
 * │   llama_batch   │───▶│  llama_sbatch   │───▶│  llama_ubatch   │
 * │ (raw input)     │    │ (sorted/grouped)│    │ (view/subset)   │
 * │                 │    │                 │    │                 │
 * │ token[]: [A,B,C]│    │ seq[]: groups   │    │ token: ptr→data │
 * │ pos[]:   [0,1,2]│    │ ids[]: [0,1,2]  │    │ n_tokens: count │
 * │ seq_id: [0,0,0] │    │ offset: 0       │    │ equal_seqs: T/F │
 * └─────────────────┘    │ length: 3       │    └─────────────────┘
 *                        └─────────────────┘
 */

struct test_scope {
    const char * name;
    explicit test_scope(const char * name) : name(name) {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(84) << name << " ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";
    }
    ~test_scope() {
        std::cout << "\n✅ " << name << " Test Completed Successfully\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════════════════════\n\n";
    }
};

// Helper function to print batch details
static void print_batch_details(const llama_batch& batch, const std::string& title) {
    std::cout << "\n" << title << " Details:\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Total Tokens: " << batch.n_tokens << "\n";
    
    if (batch.token) {
        std::cout << "Tokens: ";
        for (int i = 0; i < batch.n_tokens; ++i) {
            std::cout << batch.token[i] << " ";
        }
        std::cout << "\n";
    }
    
    if (batch.pos) {
        std::cout << "Positions: ";
        for (int i = 0; i < batch.n_tokens; ++i) {
            std::cout << batch.pos[i] << " ";
        }
        std::cout << "\n";
    }
    
    if (batch.n_seq_id && batch.seq_id) {
        std::cout << "Sequence Details:\n";
        for (int i = 0; i < batch.n_tokens; ++i) {
            std::cout << "  Token[" << i << "]: seq_ids=[";
            for (int j = 0; j < batch.n_seq_id[i]; ++j) {
                std::cout << batch.seq_id[i][j];
                if (j < batch.n_seq_id[i] - 1) std::cout << ",";
            }
            std::cout << "]\n";
        }
    }
    
    if (batch.logits) {
        std::cout << "Output Flags: ";
        for (int i = 0; i < batch.n_tokens; ++i) {
            std::cout << (int)batch.logits[i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "---------------------------------------------\n";
}

// Helper function to print sbatch details
static void print_sbatch_details(const llama_sbatch& sbatch, const std::string& title) {
    std::cout << "\n" << title << " Details:\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Total Tokens: " << sbatch.n_tokens << "\n";
    std::cout << "Sequences: " << sbatch.seq.size() << "\n";
    
    for (size_t i = 0; i < sbatch.seq.size(); ++i) {
        const auto& s = sbatch.seq[i];
        std::cout << "Sequence[" << i << "]: "
                  << "offset=" << s.offset 
                  << ", length=" << s.length << "\n";
        
        if (s.seq_id && s.n_seq_id > 0) {
            std::cout << "  Sequence IDs: [";
            for (int j = 0; j < s.n_seq_id; ++j) {
                std::cout << s.seq_id[j];
                if (j < s.n_seq_id - 1) std::cout << ",";
            }
            std::cout << "]\n";
        }
    }
    
    std::cout << "Sorted Token Order: ";
    for (size_t i = 0; i < sbatch.ids.size(); ++i) {
        std::cout << sbatch.ids[i] << " ";
    }
    std::cout << "\n";
    std::cout << "---------------------------------------------\n";
}

// Helper function to print ubatch details
static void print_ubatch_details(const llama_ubatch& ubatch, const std::string& title) {
    std::cout << "\n" << title << " Details:\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Equal Sequences: " << (ubatch.equal_seqs ? "true" : "false") << "\n";
    std::cout << "Total Tokens: " << ubatch.n_tokens << "\n";
    std::cout << "Tokens per Sequence: " << ubatch.n_seq_tokens << "\n";
    std::cout << "Number of Sequences: " << ubatch.n_seqs << "\n";
    
    if (ubatch.token) {
        std::cout << "Tokens: ";
        for (size_t i = 0; i < ubatch.n_tokens; ++i) {
            std::cout << ubatch.token[i] << " ";
        }
        std::cout << "\n";
    }
    
    if (ubatch.pos) {
        std::cout << "Positions: ";
        for (size_t i = 0; i < ubatch.n_tokens; ++i) {
            std::cout << ubatch.pos[i] << " ";
        }
        std::cout << "\n";
    }
    
    if (ubatch.n_seq_id) {
        std::cout << "Sequence ID Details: ";
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < ubatch.n_seqs; ++i) {
                std::cout << ubatch.n_seq_id[i] << " ";
            }
        } else {
            for (size_t i = 0; i < ubatch.n_tokens; ++i) {
                std::cout << ubatch.n_seq_id[i] << " ";
            }
        }
        std::cout << "\n";
    }
    
    if (ubatch.output) {
        std::cout << "Output Flags: ";
        for (size_t i = 0; i < ubatch.n_tokens; ++i) {
            std::cout << (int)ubatch.output[i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "---------------------------------------------\n";
}

// Test 1: Basic Batch Creation and Conversion
static void test_basic_batch_conversion() {
    test_scope scope("Basic Batch Creation and Conversion");
    
    /*
     * Basic Conversion Flow:
     * 
     * llama_batch (raw input):
     * ┌─────┬─────┬─────┬─────┬─────┐
     * │ 100 │ 101 │ 102 │ 103 │ 104 │  ← tokens
     * │  0  │  1  │  2  │  3  │  4  │  ← positions
     * │  0  │  0  │  0  │  0  │  0  │  ← seq_id
     * └─────┴─────┴─────┴─────┴─────┘
     *                ↓
     * llama_sbatch (simple_split=true):
     * ┌─────────────────────────────────┐
     * │ seq[0]: {n_seq_id=0, offset=0,  │
     * │         length=5}               │
     * │ ids[]: [0,1,2,3,4]              │
     * └─────────────────────────────────┘
     */
    
    // Create a simple batch with 5 tokens in one sequence
    llama_batch batch = llama_batch_init(10, 0, 2);  // max 10 tokens, no embeddings, max 2 seqs
    
    // Add tokens to sequence 0
    llama_seq_id seq_0 = 0;
    common_batch_add(batch, 100, 0, {seq_0}, false);  // token 100 at pos 0
    common_batch_add(batch, 101, 1, {seq_0}, false);  // token 101 at pos 1  
    common_batch_add(batch, 102, 2, {seq_0}, false);  // token 102 at pos 2
    common_batch_add(batch, 103, 3, {seq_0}, false);  // token 103 at pos 3
    common_batch_add(batch, 104, 4, {seq_0}, true);   // token 104 at pos 4, output=true
    
    print_batch_details(batch, "Original Batch");
    
    // Convert to sbatch with simple split mode
    llama_sbatch sbatch(batch, 64, true, false);  // n_embd=64, simple_split=true, logits_all=false
    
    print_sbatch_details(sbatch, "Simple Split SBatch");
    
    // Verify that simple split creates one sequence with n_seq_id = 0
    GGML_ASSERT(sbatch.seq.size() == 1);
    GGML_ASSERT(sbatch.seq[0].n_seq_id == 0);
    GGML_ASSERT(sbatch.seq[0].length == 5);
    GGML_ASSERT(sbatch.seq[0].offset == 0);
    
    llama_batch_free(batch);
}

// Test 2: Testing split_simple Operation and State Modification
static void test_split_simple_modification() {
    test_scope scope("Split Simple Operation and State Modification");
    
    /*
     * split_simple State Modification Visualization:
     * 
     * Initial sbatch state:
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │  ← token data
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *   ▲                                 ▲
     * offset=0                       offset+length=6
     * 
     * After split_simple(2):
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑consumed↑   ▲                   ▲
     *             offset=2         offset+length=6
     * 
     * After split_simple(3):
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑─── consumed ────↑       ▲     ▲
     *                        offset=5  offset+length=6
     * 
     * Key insight: split_simple "consumes" tokens from the head by advancing offset!
     */
    
    // Create a batch with 6 tokens
    llama_batch batch = llama_batch_init(10, 0, 1);
    
    llama_seq_id seq_0 = 0;
    for (int i = 0; i < 6; ++i) {
        //                                          is_logits?
        common_batch_add(batch, 200 + i, i, {seq_0}, i == 5);  // last token outputs
    }
    
    print_batch_details(batch, "Original Batch (6 tokens)");
    
    // Convert to sbatch
    llama_sbatch sbatch(batch, 64, true, false);
    
    print_sbatch_details(sbatch, "Initial SBatch State");
    
    std::cout << "\n=== Testing Multiple split_simple Calls ===\n";
    
    // First split_simple call - take 2 tokens
    std::cout << "\n--- First split_simple(2) ---\n";
    std::cout << "Before split_simple:\n";
    std::cout << "  seq[0].offset = " << sbatch.seq[0].offset << "\n";
    std::cout << "  seq[0].length = " << sbatch.seq[0].length << "\n";
    std::cout << "  sbatch.n_tokens = " << sbatch.n_tokens << "\n";
    
    /*
     * Visual representation of split_simple(2):
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑─ extract these 2 ─↑ ↑─ remaining ─↑
     *    → ubatch1            → sbatch.seq[0]
     */
    
    llama_ubatch ubatch1 = sbatch.split_simple(2);
    
    std::cout << "After split_simple:\n";
    std::cout << "  seq[0].offset = " << sbatch.seq[0].offset << "\n";
    std::cout << "  seq[0].length = " << sbatch.seq[0].length << "\n";
    std::cout << "  sbatch.n_tokens = " << sbatch.n_tokens << "\n";
    
    print_ubatch_details(ubatch1, "First UBatch (2 tokens)");
    
    // Verify the modifications
    GGML_ASSERT(sbatch.seq[0].offset == 2);  // offset advanced by 2
    GGML_ASSERT(sbatch.seq[0].length == 4);  // length reduced by 2
    GGML_ASSERT(sbatch.n_tokens == 4);       // total tokens reduced by 2
    GGML_ASSERT(ubatch1.n_tokens == 2);      // ubatch contains 2 tokens
    
    // Second split_simple call - take 3 tokens
    std::cout << "\n--- Second split_simple(3) ---\n";
    std::cout << "Before split_simple:\n";
    std::cout << "  seq[0].offset = " << sbatch.seq[0].offset << "\n";
    std::cout << "  seq[0].length = " << sbatch.seq[0].length << "\n";
    std::cout << "  sbatch.n_tokens = " << sbatch.n_tokens << "\n";
    
    /*
     * Visual representation of split_simple(3):
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑─consumed─↑ ↑─extract these 3─↑↑─remaining─↑
     *                → ubatch2           → sbatch.seq[0]
     */
    
    llama_ubatch ubatch2 = sbatch.split_simple(3);
    
    std::cout << "After split_simple:\n";
    std::cout << "  seq[0].offset = " << sbatch.seq[0].offset << "\n";
    std::cout << "  seq[0].length = " << sbatch.seq[0].length << "\n";
    std::cout << "  sbatch.n_tokens = " << sbatch.n_tokens << "\n";
    
    print_ubatch_details(ubatch2, "Second UBatch (3 tokens)");
    
    // Verify the modifications
    GGML_ASSERT(sbatch.seq[0].offset == 5);  // offset advanced by 3 more
    GGML_ASSERT(sbatch.seq[0].length == 1);  // length reduced by 3 more
    GGML_ASSERT(sbatch.n_tokens == 1);       // total tokens reduced by 3 more
    GGML_ASSERT(ubatch2.n_tokens == 3);      // ubatch contains 3 tokens
    
    // Third split_simple call - take remaining token
    std::cout << "\n--- Third split_simple(10) (should only get 1 token) ---\n";
    std::cout << "Before split_simple:\n";
    std::cout << "  seq[0].offset = " << sbatch.seq[0].offset << "\n";
    std::cout << "  seq[0].length = " << sbatch.seq[0].length << "\n";
    std::cout << "  sbatch.n_tokens = " << sbatch.n_tokens << "\n";
    
    /*
     * Visual representation - requesting more than available:
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑─────consumed──────────────↑ ↑only 1↑
     *                               remaining
     */
    
    llama_ubatch ubatch3 = sbatch.split_simple(10);  // Request more than available
    
    std::cout << "After split_simple:\n";
    std::cout << "  seq[0].offset = " << sbatch.seq[0].offset << "\n";
    std::cout << "  seq[0].length = " << sbatch.seq[0].length << "\n";
    std::cout << "  sbatch.n_tokens = " << sbatch.n_tokens << "\n";
    
    print_ubatch_details(ubatch3, "Third UBatch (1 token)");
    
    // Verify the modifications
    GGML_ASSERT(sbatch.seq[0].offset == 6);  // offset advanced by 1 more
    GGML_ASSERT(sbatch.seq[0].length == 0);  // length reduced to 0
    GGML_ASSERT(sbatch.n_tokens == 0);       // no more tokens
    GGML_ASSERT(ubatch3.n_tokens == 1);      // ubatch contains 1 token
    
    // Fourth split_simple call - should return empty ubatch
    std::cout << "\n--- Fourth split_simple(1) (should be empty) ---\n";
    
    /*
     * Visual representation - nothing left:
     * ┌─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 200 │ 201 │ 202 │ 203 │ 204 │ 205 │
     * └─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑─────────all consumed────────────↑
     *                                  offset=6, length=0
     */
    
    llama_ubatch ubatch4 = sbatch.split_simple(1);
    print_ubatch_details(ubatch4, "Fourth UBatch (empty)");
    
    GGML_ASSERT(ubatch4.n_tokens == 0);      // no tokens available
    
    std::cout << "\n✓ All state modifications verified correctly!\n";
    
    llama_batch_free(batch);
}

// Test 3: Multi-Sequence Batch Processing
static void test_multi_sequence_batch() {
    test_scope scope("Multi-Sequence Batch Processing");
    
    /*
     * Multi-Sequence Processing Visualization:
     * 
     * Original batch (mixed sequences):
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 300 │ 301 │ 302 │ 400 │ 401 │ 500 │ 999 │
     * │seq:0│seq:0│seq:0│seq:1│seq:1│seq:2│0&1  │
     * │pos:0│pos:1│pos:2│pos:0│pos:1│pos:0│pos:10│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     * 
     * After sbatch sorting (complex mode):
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 999 │ 300 │ 301 │ 302 │ 400 │ 401 │ 500 │
     * │0&1  │seq:0│seq:0│seq:0│seq:1│seq:1│seq:2│
     * │pos:10│pos:0│pos:1│pos:2│pos:0│pos:1│pos:0│
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *   ↑     ↑─────seq 0──────↑ ↑─seq 1─↑ ↑seq2↑
     * shared    (sorted by pos)
     * prompt
     * 
     * Simple split mode treats everything as one sequence:
     * ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     * │ 300 │ 301 │ 302 │ 400 │ 401 │ 500 │ 999 │
     * │     │     │     │     │     │     │     │
     * └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     *  ↑─────────all treated as seq_id=0──────────↑
     */
    
    // Create a batch with multiple sequences
    llama_batch batch = llama_batch_init(20, 0, 3);
    
    llama_seq_id seq_0 = 0;
    llama_seq_id seq_1 = 1;
    llama_seq_id seq_2 = 2;
    
    // Add tokens to different sequences
    common_batch_add(batch, 300, 0, {seq_0}, false);         // seq_0: pos 0
    common_batch_add(batch, 301, 1, {seq_0}, false);         // seq_0: pos 1
    common_batch_add(batch, 302, 2, {seq_0}, true);          // seq_0: pos 2, output
    
    common_batch_add(batch, 400, 0, {seq_1}, false);         // seq_1: pos 0
    common_batch_add(batch, 401, 1, {seq_1}, true);          // seq_1: pos 1, output
    
    common_batch_add(batch, 500, 0, {seq_2}, true);          // seq_2: pos 0, output
    
    // Add a shared prompt token (belongs to multiple sequences)
    common_batch_add(batch, 999, 10, {seq_0, seq_1}, false); // shared between seq_0 and seq_1
    
    print_batch_details(batch, "Multi-Sequence Batch");
    
    // Convert to sbatch with complex split mode (simple_split=false)
    llama_sbatch sbatch_complex(batch, 64, false, false);
    
    print_sbatch_details(sbatch_complex, "Complex SBatch (sorted by seq_id)");
    
    std::cout << "\n=== Testing split_equal and split_seq ===\n";
    
    /*
     * split_equal strategy:
     * - Processes sequences by equal-length batches
     * - Shared prompts processed first (highest priority)
     * - Equal length sequences grouped together
     * 
     * split_seq strategy:
     * - Processes one sequence at a time
     * - Takes from the end of sequence list
     * - Good for sequential processing
     */
    
    // Test split_equal
    llama_ubatch ubatch_equal = sbatch_complex.split_equal(10);
    print_ubatch_details(ubatch_equal, "Split Equal Result");
    
    // Test split_seq  
    llama_ubatch ubatch_seq = sbatch_complex.split_seq(5);
    print_ubatch_details(ubatch_seq, "Split Seq Result");
    
    // Compare with simple split approach
    llama_sbatch sbatch_simple(batch, 64, true, false);
    print_sbatch_details(sbatch_simple, "Simple SBatch");
    
    llama_ubatch ubatch_simple = sbatch_simple.split_simple(10);
    print_ubatch_details(ubatch_simple, "Simple Split Result");
    
    llama_batch_free(batch);
}

// Test 4: Edge Cases and Error Conditions
static void test_edge_cases() {
    test_scope scope("Edge Cases and Error Conditions");
    
    /*
     * Edge Case Testing:
     * 
     * Empty batch:
     * ┌─┐
     * │ │ ← no tokens
     * └─┘
     * 
     * Single token batch:
     * ┌─────┐
     * │ 777 │ ← one token
     * └─────┘
     * 
     * After split:
     * ┌─┐
     * │ │ ← empty sbatch
     * └─┘
     */
    
    // Test empty batch
    llama_batch empty_batch = llama_batch_init(5, 0, 1);
    // Don't add any tokens
    
    print_batch_details(empty_batch, "Empty Batch");
    
    llama_sbatch empty_sbatch(empty_batch, 64, true, false);
    print_sbatch_details(empty_sbatch, "Empty SBatch");
    
    llama_ubatch empty_ubatch = empty_sbatch.split_simple(5);
    print_ubatch_details(empty_ubatch, "Empty UBatch from split_simple");
    
    GGML_ASSERT(empty_ubatch.n_tokens == 0);
    GGML_ASSERT(empty_sbatch.seq.empty());
    
    // Test single token batch
    llama_batch single_batch = llama_batch_init(5, 0, 1);
    common_batch_add(single_batch, 777, 0, {0}, true);
    
    print_batch_details(single_batch, "Single Token Batch");
    
    llama_sbatch single_sbatch(single_batch, 64, true, false);
    print_sbatch_details(single_sbatch, "Single Token SBatch");
    
    llama_ubatch single_ubatch = single_sbatch.split_simple(1);
    print_ubatch_details(single_ubatch, "Single Token UBatch");
    
    GGML_ASSERT(single_ubatch.n_tokens == 1);
    GGML_ASSERT(single_ubatch.token[0] == 777);
    
    // After split, sbatch should be empty
    llama_ubatch post_split_ubatch = single_sbatch.split_simple(1);
    GGML_ASSERT(post_split_ubatch.n_tokens == 0);
    
    llama_batch_free(empty_batch);
    llama_batch_free(single_batch);
}

int main(int argc, char** argv) {
    std::cout << "llama_batch/sbatch/ubatch Test Program\n";
    std::cout << "=====================================\n";
    std::cout << "Testing batch processing principles and split_simple modifications\n";
    
    /*
     * Overall Test Architecture:
     * 
     * ┌─────────────────────────┐
     * │    Input Validation     │
     * │ (test_basic_batch_*)    │
     * └───────────┬─────────────┘
     *             ▼
     * ┌─────────────────────────┐
     * │  Core Functionality     │
     * │(test_split_simple_*)    │ ← Main focus: state modification
     * └───────────┬─────────────┘
     *             ▼
     * ┌─────────────────────────┐
     * │ Complex Scenarios       │
     * │(test_multi_sequence_*)  │
     * └───────────┬─────────────┘
     *             ▼
     * ┌─────────────────────────┐
     * │   Edge Cases &          │
     * │ Data Integrity          │
     * └─────────────────────────┘
     */

    test_basic_batch_conversion();
    test_split_simple_modification();
    test_multi_sequence_batch();
    test_edge_cases();
    
    return 0;
} 