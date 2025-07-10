#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

struct LLM_KV;
struct llama_model_loader;

struct llama_vocab {
    struct token_data {
        std::string      text;
        float            score;
        llama_token_attr attr;
    };

    llama_vocab();
    ~llama_vocab();

    void load(llama_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum llama_vocab_type     get_type()     const;
    enum llama_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (llama_token id) const;
    bool is_unknown     (llama_token id) const;
    bool is_control     (llama_token id) const;
    bool is_byte        (llama_token id) const;
    bool is_user_defined(llama_token id) const;
    bool is_unused      (llama_token id) const;
    bool is_eog         (llama_token id) const;

    uint8_t     token_to_byte(llama_token id) const;
    llama_token byte_to_token(uint8_t ch)     const;

    llama_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(llama_token id) const;

    const char *     token_get_text (llama_token id) const;
    float            token_get_score(llama_token id) const;
    llama_token_attr token_get_attr (llama_token id) const;

    llama_token token_bos() const;
    llama_token token_eos() const;
    llama_token token_eot() const;
    llama_token token_eom() const;
    llama_token token_unk() const;
    llama_token token_sep() const;
    llama_token token_nl () const;
    llama_token token_pad() const;

    llama_token token_prefix() const;
    llama_token token_middle() const;
    llama_token token_suffix() const;

    llama_token token_fim_pre() const;
    llama_token token_fim_suf() const;
    llama_token token_fim_mid() const;
    llama_token token_fim_pad() const;
    llama_token token_fim_rep() const;
    llama_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  llama_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(llama_token token) const;

    int32_t detokenize(
            const llama_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<llama_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// PLaMo-2 Aho-Corasick tokenizer
class llama_vocab_plamo2 {
public:
    // Constants for table structure
    static constexpr int32_t TABLE_PIECE_LENGTH = 0;
    static constexpr int32_t TABLE_TOKEN_ID     = 1;
    static constexpr int32_t TABLE_SCORE        = 2;
    static constexpr int32_t TABLE_PIECE_ID     = 3;

    // Constants for path array
    static constexpr int32_t PATH_TOKEN_LENGTH  = 0;
    static constexpr int32_t PATH_TOKEN_ID      = 1;
    static constexpr int32_t PATH_NUM_TOKENS    = 2;

    // Score constants
    static constexpr int32_t INVALID_SCORE = -20000000;
    static constexpr int32_t UNKNOWN_SCORE = -10000000;

    struct vocab_entry {
        std::string text;
        float score;
        std::string type; // "BYTE" for byte tokens, empty for normal tokens
    };

    llama_vocab_plamo2();
    ~llama_vocab_plamo2();

    // Build the Aho-Corasick automaton from vocabulary
    void build(const std::vector<vocab_entry> & vocab);

    // Encode text to token IDs
    std::vector<llama_token> encode(const std::string & text) const;

    // Get token text by ID
    const std::string & get_token_text(llama_token id) const;

    // Get vocabulary size
    size_t vocab_size() const;

private:
    // Internal structures for Aho-Corasick automaton

    // List of tokens in the vocabulary
    std::vector<std::string> tokens_;

    // Mapping from byte code point to token ID (for byte fallback)
    std::vector<llama_token> bytes_;

    // Mapping from piece code to suffix ID
    std::unordered_map<int64_t, int32_t> to_suffix_id_;

    // Flattened table representing the Trie structure
    // Each row contains: [piece_length, token_id, score, piece_id]
    std::vector<std::vector<int32_t>> table_;

    // Helper functions
    std::vector<llama_token> encode_unicode(const std::vector<uint32_t> & unicode_data) const;
};
