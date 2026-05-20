// Carbon-3B (HybridDNATokenizer) helpers — split out as standalone static
// functions so they are unit-testable without a full llama_vocab.

#pragma once

#include "llama.h"

#include <functional>
#include <string>
#include <vector>

namespace llama_carbon {

using token_lookup_fn = std::function<llama_token(const std::string &)>;
using bpe_fn          = std::function<void(const std::string &, std::vector<llama_token> &)>;

// Tokenize a DNA sequence as fixed-width k-mers. Invalid bases (anything
// outside A/C/G/T after upper-casing ASCII) and any k-mer not in the vocab
// map to `oov_id`. Trailing partial k-mers are right-padded with 'A' to
// match Carbon's Python tokenizer.
inline void emit_dna_kmers(
        const std::string & raw,
        std::size_t k,
        llama_token oov_id,
        const token_lookup_fn & lookup,
        std::vector<llama_token> & output) {
    std::string seq = raw;
    for (char & c : seq) {
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 32);
        }
    }
    auto is_valid_kmer = [](const std::string & s) {
        for (char c : s) {
            if (c != 'A' && c != 'C' && c != 'G' && c != 'T') {
                return false;
            }
        }
        return true;
    };

    std::size_t i = 0;
    for (; i + k <= seq.size(); i += k) {
        const std::string kmer = seq.substr(i, k);
        if (is_valid_kmer(kmer)) {
            const auto tok = lookup(kmer);
            output.push_back(tok != LLAMA_TOKEN_NULL ? tok : oov_id);
        } else {
            output.push_back(oov_id);
        }
    }
    if (i < seq.size()) {
        std::string kmer = seq.substr(i);
        kmer.append(k - kmer.size(), 'A');
        if (is_valid_kmer(kmer)) {
            const auto tok = lookup(kmer);
            output.push_back(tok != LLAMA_TOKEN_NULL ? tok : oov_id);
        } else {
            output.push_back(oov_id);
        }
    }
}

// Walk the input, dispatching <dna>...</dna> regions to k-mer chunking and
// everything else to `bpe_tokenize`. An unterminated `<dna>` swallows the
// rest of the input as DNA, matching the Python reference.
inline void tokenize_carbon(
        const std::string & text,
        llama_token dna_begin_id,
        llama_token dna_end_id,
        llama_token dna_oov_id,
        std::size_t k,
        const token_lookup_fn & lookup,
        const bpe_fn & bpe_tokenize,
        std::vector<llama_token> & output) {
    static const std::string open_tag  = "<dna>";
    static const std::string close_tag = "</dna>";

    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t start = text.find(open_tag, pos);
        if (start == std::string::npos) {
            if (pos < text.size()) {
                bpe_tokenize(text.substr(pos), output);
            }
            break;
        }
        if (start > pos) {
            bpe_tokenize(text.substr(pos, start - pos), output);
        }
        output.push_back(dna_begin_id);

        const std::size_t content_start = start + open_tag.size();
        const std::size_t end = text.find(close_tag, content_start);
        const std::size_t content_end = (end == std::string::npos) ? text.size() : end;

        emit_dna_kmers(text.substr(content_start, content_end - content_start), k, dna_oov_id, lookup, output);

        if (end == std::string::npos) {
            break;
        }
        output.push_back(dna_end_id);
        pos = end + close_tag.size();
    }
}

} // namespace llama_carbon
