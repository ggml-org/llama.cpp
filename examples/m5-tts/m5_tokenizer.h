// Byte-level BPE tokenizer for the M5 text front-end, native edition.
//
// Tokenizer.java does this on the JVM. Keeping it there meant the engine could
// never leave the JVM, so the same pipeline is reproduced here: the M5TK1
// format, the GPT-4 style split pattern, the byte-to-character alphabet and the
// merge ranks. `m5_engine --text` exists to diff this against TokenDump.java.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace m5 {

class Tokenizer {
public:
    bool load(const std::string & path, std::string & error);
    bool loaded() const { return !vocab_.empty(); }

    // Encodes one request body. BOS/EOS are added here, matching Java.
    std::vector<long> encode(const std::string & text) const;

    int bos() const { return bos_; }
    int eos() const { return eos_; }

    // Exposed for tests: the pieces the split pattern produces.
    std::vector<std::pair<size_t, size_t>> split(const char * text, size_t length) const;

private:
    void encode_plain(const char * text, size_t length, std::vector<long> & out) const;
    void encode_piece(const char * text, size_t length, std::vector<long> & out) const;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<uint64_t, uint64_t> merges_;  // (a,b) -> (rank,id)
    std::vector<std::pair<std::string, int>> specials_;  // longest first
    std::string byte_text_[256];
    int byte_id_[256] = {0};
    int bos_ = 0, eos_ = 0;
};

// Unicode class bits used by the tokenizer and the text segmenter.
enum : int {
    kClassLetter = 1,
    kClassNumber = 2,
    kClassSeparator = 4,
};
int unicode_class(uint32_t codepoint);

}  // namespace m5
