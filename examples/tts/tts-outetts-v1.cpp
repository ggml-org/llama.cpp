#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "json.hpp"
#include "llama.h"
#include "default_speaker.h"

#define _USE_MATH_DEFINES // For M_PI on MSVC

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <unordered_set>

using json = nlohmann::ordered_json;

enum outetts_version {
    OUTETTS_V0_2,
    OUTETTS_V0_3,
    OUTETTS_V1_0,
};

// Special Tokens structure
struct SpecialTokens {
    std::string bos = "<|im_start|>";
    std::string eos = "<|im_end|>";
    std::string c1 = "<|c1_{}|>";
    std::string c2 = "<|c2_{}|>";
    std::string text_start = "<|text_start|>";
    std::string text_end = "<|text_end|>";
    std::string voice_characteristic_start = "<|voice_characteristic_start|>";
    std::string voice_characteristic_end = "<|voice_characteristic_end|>";
    std::string emotion_start = "<|emotion_start|>";
    std::string emotion_end = "<|emotion_end|>";
    std::string audio_start = "<|audio_start|>";
    std::string audio_end = "<|audio_end|>";
    std::string time = "<|t_{:.2f}|>";
    std::string code = "<|code|>";
    std::string energy = "<|energy_{}|>";
    std::string spectral_centroid = "<|spectral_centroid_{}|>";
    std::string pitch = "<|pitch_{}|>";
    std::string word_start = "<|word_start|>";
    std::string word_end = "<|word_end|>";
    std::string features = "<|features|>";
    std::string global_features_start = "<|global_features_start|>";
    std::string global_features_end = "<|global_features_end|>";
};

std::string text_normalization(std::string result) {
    // Normalize whitespace characters (newlines, tabs, etc.) to single spaces
    result = std::regex_replace(result, std::regex("\\s+"), " ");

    // Strip leading/trailing whitespace
    auto start = result.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return ""; // String is all whitespace
    }
    auto end = result.find_last_not_of(" \t\n\r\f\v");
    result = result.substr(start, end - start + 1);

    // Normalize common Unicode characters to ASCII equivalents
    result = std::regex_replace(result, std::regex("[""]"), "\""); // Curly quotes to straight quotes
    result = std::regex_replace(result, std::regex("['']"), "'");  // Curly single quotes
    result = std::regex_replace(result, std::regex("[–—]"), "-");  // Various dashes to hyphen

    return result;
}

// Utility function to format strings (simple replacement for Python's format)
std::string format_string(const std::string& format_str, double value) {
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f", value);
    std::string result = format_str;
    size_t pos = result.find("{:.2f}");
    if (pos != std::string::npos) {
        result.replace(pos, 6, buffer);
    }
    return result;
}

std::string format_string(const std::string& format_str, int value) {
    std::string result = format_str;
    size_t pos = result.find("{}");
    if (pos != std::string::npos) {
        result.replace(pos, 2, std::to_string(value));
    }
    return result;
}

std::string format_string(const std::string& format_str, const std::string& value) {
    std::string result = format_str;
    size_t pos = result.find("{}");
    if (pos != std::string::npos) {
        result.replace(pos, 2, value);
    }
    return result;
}

// Function to get features
std::vector<std::string> get_features(const json& f, const SpecialTokens& special_tokens) {
    std::vector<std::string> result;
    
    int energy = f.contains("energy") ? f["energy"].get<int>() : 0;
    int spectral_centroid = f.contains("spectral_centroid") ? f["spectral_centroid"].get<int>() : 0;
    int pitch = f.contains("pitch") ? f["pitch"].get<int>() : 0;
    
    result.push_back(format_string(special_tokens.energy, energy));
    result.push_back(format_string(special_tokens.spectral_centroid, spectral_centroid));
    result.push_back(format_string(special_tokens.pitch, pitch));
    
    return result;
}

// Function to get global features
std::string get_global_features(const json& f, const SpecialTokens& special_tokens, const std::string& global_features_template) {
    std::vector<std::string> features = get_features(f, special_tokens);
    std::string codes;
    for (const auto& feature : features) {
        codes += feature;
    }
    
    std::string result = global_features_template;
    // Replace {fs} with global_features_start
    size_t pos = result.find("{fs}");
    if (pos != std::string::npos) {
        result.replace(pos, 4, special_tokens.global_features_start);
    }
    
    // Replace {codes} with the joined features
    pos = result.find("{codes}");
    if (pos != std::string::npos) {
        result.replace(pos, 7, codes);
    }
    
    // Replace {fe} with global_features_end
    pos = result.find("{fe}");
    if (pos != std::string::npos) {
        result.replace(pos, 4, special_tokens.global_features_end);
    }
    
    return result;
}

// Function to create codes
std::string create_codes(const json& words, const SpecialTokens& special_tokens) {
    std::vector<std::string> codes;
    
    for (const auto& word_item : words) {
        std::string word = word_item["word"].get<std::string>() + special_tokens.features;
        word += format_string(special_tokens.time, word_item["duration"].get<double>());
        
        // Add features
        std::vector<std::string> features = get_features(word_item["features"], special_tokens);
        for (const auto& feature : features) {
            word += feature;
        }
        
        // Add pairs of c1 and c2
        std::vector<std::string> pairs;
        for (size_t idx = 0; idx < word_item["c1"].size(); idx++) {
            std::string c1 = format_string(special_tokens.c1, word_item["c1"][idx].get<int>());
            std::string c2 = format_string(special_tokens.c2, word_item["c2"][idx].get<int>());
            pairs.push_back(c1 + c2);
        }
        
        word += special_tokens.code;
        for (const auto& pair : pairs) {
            word += pair;
        }
        
        codes.push_back(special_tokens.word_start + word + special_tokens.word_end);
    }
    
    // Join codes with newline
    std::string result;
    for (size_t i = 0; i < codes.size(); i++) {
        result += codes[i];
        if (i < codes.size() - 1) {
            result += "\n";
        }
    }
    
    return result;
}

// Function to initialize prompt
std::string init_prompt(const std::string& text, const SpecialTokens& special_tokens, const std::string& input_prompt_template) {
    std::string result = input_prompt_template;
    
    // Replace {bos} with bos
    size_t pos = result.find("{bos}");
    if (pos != std::string::npos) {
        result.replace(pos, 5, special_tokens.bos);
    }
    
    // Replace {text_start} with text_start
    pos = result.find("{text_start}");
    if (pos != std::string::npos) {
        result.replace(pos, 12, special_tokens.text_start);
    }
    
    // Replace {text} with the text
    pos = result.find("{text}");
    if (pos != std::string::npos) {
        result.replace(pos, 6, text);
    }
    
    // Replace {text_end} with text_end
    pos = result.find("{text_end}");
    if (pos != std::string::npos) {
        result.replace(pos, 10, special_tokens.text_end);
    }
    
    // Replace {audio_start} with audio_start
    pos = result.find("{audio_start}");
    if (pos != std::string::npos) {
        result.replace(pos, 13, special_tokens.audio_start);
    }
    
    return result;
}

// Function to get separator based on text
std::string get_separator(const std::string& text) {
    bool has_hiragana = false;
    bool has_katakana = false;
    bool has_han = false;
    bool has_hangul = false;
    
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 0xE3 && uc <= 0xE3) { // Simplified check for hiragana (U+3040-U+309F)
            has_hiragana = true;
        }
        else if (uc >= 0xE3 && uc <= 0xE3) { // Simplified check for katakana (U+30A0-U+30FF)
            has_katakana = true;
        }
        else if (uc >= 0xE4 && uc <= 0xE9) { // Simplified check for han (U+4E00-U+9FFF)
            has_han = true;
        }
        else if (uc >= 0xEA && uc <= 0xED) { // Simplified check for hangul (U+AC00-U+D7AF)
            has_hangul = true;
        }
    }
    
    if (has_hiragana || has_katakana || has_han) {
        return "。";
    }
    else if (has_hangul) {
        return ". ";
    }
    else {
        return ". ";
    }
}

inline std::string trim(std::string_view sv) {
    sv.remove_prefix(std::min(sv.find_first_not_of(" \t\n\r\f\v"), sv.size()));
    auto pos = sv.find_last_not_of(" \t\n\r\f\v");
    if (pos != std::string_view::npos) {
        sv.remove_suffix(sv.size() - pos - 1);
    }
    return std::string(sv);
}

inline bool ends_with(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) return false;
    return value.size() >= ending.size() && 
           value.substr(value.size() - ending.size()) == ending;
}

std::pair<std::string, std::string> merge_speaker_text(const std::string& input_text, const std::string& speaker_text_orig) {
    std::string speaker_text = trim(speaker_text_orig);
    std::string separator = get_separator(speaker_text); 
    
    // Determine allowed endings based on the separator
    std::vector<std::string> allowed_ends;
    if (separator == "。") {
        allowed_ends = {"。", "？", "！", "?", "!"}; 
    } else {
        allowed_ends = {".", "?", "!"};
    }

    std::string rs = ""; // This will be the separator/space to insert
    
    if (!speaker_text.empty()) {
        bool ends_with_allowed_char = false;
        for (const std::string& end_char : allowed_ends) {
            if (ends_with(speaker_text, end_char)) {
                ends_with_allowed_char = true;
                break;
            }
        }

        if (!ends_with_allowed_char) {
            rs = separator;
        } else {
            if (separator != "。") {
                rs = " ";
            }
        }
    }

    std::string output = speaker_text + rs + trim(input_text);
    std::string trimmed_rs = trim(rs);
    return std::make_pair(output, trimmed_rs);
}

// Main function to get completion prompt
std::string get_completion_prompt(const std::string& text, json& speaker) {
    // Initialize special tokens
    SpecialTokens special_tokens;

    // Templates (would normally be passed as parameters)
    std::string input_prompt_template = "{bos}{text_start}{text}{text_end}\n{audio_start}\n";
    std::string global_features_template = "{fs}{codes}{fe}";
    
    // Normalize text

    std::string normalized_text = text_normalization(text);
    
    std::string prompt;
    if (!speaker.is_null()) {
        // Merge speaker text
        auto [merged_text, separator] = merge_speaker_text(normalized_text, speaker["text"]);
        normalized_text = merged_text;
        
        // Update last word with separator if necessary
        if (!separator.empty()) {
            speaker["words"].back()["word"] = speaker["words"].back()["word"].get<std::string>() + separator;
        }
        
        // Create codes
        std::string codes = create_codes(speaker["words"], special_tokens);
        
        // Initialize prompt
        prompt = init_prompt(normalized_text, special_tokens, input_prompt_template);
        
        // Add codes and word_start
        prompt += codes + "\n" + special_tokens.word_start;
    }
    else {
        // Initialize prompt without speaker
        prompt = init_prompt(normalized_text, special_tokens, input_prompt_template);
    }
    
    return prompt;
}

static json speaker_from_file(const std::string & speaker_file) {
    std::ifstream file(speaker_file);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for reading\n", __func__, speaker_file.c_str());
        return json();
    }

    json speaker = json::parse(file);
    return speaker;
}

static outetts_version get_tts_version(llama_model *model, json speaker = json::object()) {
    if (speaker.contains("version")) {
        int version = speaker["interface_version"].get<int>();
        if (version == 1) {
            return OUTETTS_V0_2;
        } else if (version == 2) {
            return OUTETTS_V0_3;
        } else if (version == 3) {
            return OUTETTS_V1_0;
        } else {
            LOG_ERR("%s: Unsupported speaker version '%d'\n", __func__, version);
        }
    }

    // Also could get version from model itself
    const char *chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    } else if (chat_template && std::string(chat_template) == "outetts-1.0") {
        return OUTETTS_V1_0;
    }

    // Use 0.2 as the default version
    return OUTETTS_V0_2;
}

// ------------------------
// Helper functions for UTF-8
// ------------------------

// Return the number of bytes in the current UTF-8 character.
int utf8_char_length(unsigned char c) {
    if (c < 0x80) return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Decode a UTF-8 string into a vector of Unicode code points.
std::vector<uint32_t> decode_utf8(const std::string &s) {
    std::vector<uint32_t> codepoints;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        uint32_t cp = 0;
        int len = utf8_char_length(c);
        if (len == 1) {
            cp = c;
        } else if (len == 2) {
            cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F);
        } else if (len == 3) {
            cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
        } else if (len == 4) {
            cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
        }
        codepoints.push_back(cp);
        i += len;
    }
    return codepoints;
}

// Encode a single Unicode code point into a UTF-8 string.
std::string encode_utf8(uint32_t cp) {
    std::string result;
    if (cp <= 0x7F) {
        result.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        result.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        result.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return result;
}

// Tokenize a UTF-8 string into individual characters.
std::vector<std::string> utf8_tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        int len = utf8_char_length(static_cast<unsigned char>(text[i]));
        tokens.push_back(text.substr(i, len));
        i += len;
    }
    return tokens;
}

// Trim leading and trailing whitespace.
std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// Replace multiple whitespace characters with a single space.
std::string removeExtraSpaces(const std::string &s) {
    std::string result;
    bool in_space = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space) {
                result.push_back(' ');
                in_space = true;
            }
        } else {
            result.push_back(c);
            in_space = false;
        }
    }
    return trim(result);
}

// ------------------------
// Language detection and tokenization
// ------------------------

class LanguageDetector {
public:
    // Return true if any code point is in the ranges for Japanese (Hiragana, Katakana)
    // or Chinese/Japanese ideographs.
    static bool check(const std::string &text) {
        std::vector<uint32_t> cps = decode_utf8(text);
        for (uint32_t cp : cps) {
            if ((cp >= 0x3040 && cp <= 0x309F) ||  // Hiragana
                (cp >= 0x30A0 && cp <= 0x30FF) ||  // Katakana
                (cp >= 0x4E00 && cp <= 0x9FFF)) {  // CJK Unified Ideographs
                return true;
            }
        }
        return false;
    }
};

// Tokenize text based on language. For zh/ja, we use our simple per‐character splitting;
// for others, we split on whitespace.
std::vector<std::string> tokenize_text(const std::string &text) {
    std::string t = trim(text);
    if (t.empty())
        return {};
    if (LanguageDetector::check(text)) {
        return utf8_tokenize(text);
    } else {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        while (iss >> token)
            tokens.push_back(token);
        return tokens;
    }
}

// Count words (tokens) in the text.
int count_words(const std::string &text) {
    return tokenize_text(text).size();
}

// Join tokens into a string. If no_space is true, tokens are concatenated without a separator.
std::string join_tokens(const std::vector<std::string> &tokens, bool no_space) {
    std::string result;
    if (no_space) {
        for (const auto &token : tokens)
            result += token;
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0)
                result += " ";
            result += tokens[i];
        }
    }
    return result;
}

// ------------------------
// Sentence splitting
// ------------------------
//
// We split the text into “sentences” by scanning Unicode code points
// and using a set of sentence-ending characters (including punctuation
// for both Latin and CJK texts). The algorithm accumulates code points
// until a sentence end is found, then reassembles the sentence.
std::vector<std::string> split_into_sentences(const std::string &text) {
    std::vector<uint32_t> cps = decode_utf8(text);
    std::vector< std::vector<uint32_t> > sentenceCps;
    std::vector<uint32_t> currentSentence;
    
    auto isSentenceEnd = [](uint32_t cp) -> bool {
        return cp == 0x002E || cp == 0x0021 || cp == 0x003F ||   // . ! ?
               cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F ||   // 。 ！ ？
               cp == 0xFE56 || cp == 0xFE57;                     // ︕ ︖
    };
    
    for (size_t i = 0; i < cps.size(); ++i) {
        currentSentence.push_back(cps[i]);
        if (isSentenceEnd(cps[i])) {
            // Also include following whitespace as part of the sentence delimiter.
            while (i + 1 < cps.size() &&
                   (cps[i + 1] == 0x0020 || cps[i + 1] == 0x0009 ||
                    cps[i + 1] == 0x000A || cps[i + 1] == 0x000D)) {
                ++i;
                currentSentence.push_back(cps[i]);
            }
            sentenceCps.push_back(currentSentence);
            currentSentence.clear();
        }
    }
    if (!currentSentence.empty()) {
        sentenceCps.push_back(currentSentence);
    }
    
    std::vector<std::string> sentences;
    for (auto &sc : sentenceCps) {
        std::string sentence;
        for (uint32_t cp : sc)
            sentence += encode_utf8(cp);
        sentence = trim(sentence);
        if (!sentence.empty())
            sentences.push_back(sentence);
    }
    return sentences;
}

// ------------------------
// Text chunking
// ------------------------
//
// This function splits the text into chunks that contain between min_words and max_words.
// It uses sentence splitting and then tokenizes each sentence. When a sentence is too long,
// it splits it further. Note that for zh/ja the tokens are joined without spaces,
// while for other languages spaces are inserted.
std::vector<std::string> chunk_text(const std::string &text, int min_words = 10, int max_words = 30) {
    std::string norm = removeExtraSpaces(text);
    norm = trim(norm);
    if (norm.empty())
        return {};
    
    std::vector<std::string> sentences = split_into_sentences(norm);
    std::vector<std::string> chunks;
    std::string current_chunk = "";
    int current_word_count = 0;
    
    for (const auto &sentence : sentences) {
        std::string s = trim(sentence);
        if (s.empty())
            continue;
        
        std::vector<std::string> sentence_tokens = tokenize_text(s);
        int sentence_word_count = sentence_tokens.size();
        
        // If the sentence is longer than max_words, split it into parts.
        if (sentence_word_count > max_words) {
            if (!current_chunk.empty()) {
                chunks.push_back(current_chunk);
                current_chunk = "";
                current_word_count = 0;
            }
            std::vector<std::string> current_part;
            int word_count = 0;
            for (const auto &token : sentence_tokens) {
                current_part.push_back(token);
                ++word_count;
                if (word_count >= max_words) {
                    bool isLang = LanguageDetector::check(s);
                    std::string part = isLang ? join_tokens(current_part, true)
                                              : join_tokens(current_part, false);
                    chunks.push_back(part);
                    current_part.clear();
                    word_count = 0;
                }
            }
            if (!current_part.empty()) {
                bool isLang = LanguageDetector::check(s);
                std::string part = isLang ? join_tokens(current_part, true)
                                          : join_tokens(current_part, false);
                chunks.push_back(part);
            }
            continue;
        }
        
        if (current_word_count + sentence_word_count <= max_words) {
            if (!current_chunk.empty()) {
                current_chunk += LanguageDetector::check(sentence) ? s : " " + s;
            } else {
                current_chunk = s;
            }
            current_word_count += sentence_word_count;
        } else {
            if (current_word_count >= min_words) {
                chunks.push_back(current_chunk);
                current_chunk = s;
                current_word_count = sentence_word_count;
            } else {
                int space_left = max_words - current_word_count;
                std::vector<std::string> current_part(sentence_tokens.begin(),
                                                      sentence_tokens.begin() + space_left);
                std::vector<std::string> remaining_part(sentence_tokens.begin() + space_left,
                                                        sentence_tokens.end());
                bool isLang = LanguageDetector::check(s);
                std::string first_chunk = isLang ?
                    (current_chunk + join_tokens(current_part, true)) :
                    (current_chunk + " " + join_tokens(current_part, false));
                chunks.push_back(first_chunk);
                current_chunk = isLang ? join_tokens(remaining_part, true)
                                       : join_tokens(remaining_part, false);
                current_word_count = remaining_part.size();
            }
        }
    }
    if (!current_chunk.empty())
        chunks.push_back(current_chunk);
    
    return chunks;
}

std::vector<std::vector<int>> extract_codebooks(const std::string& codes) {
    std::vector<int> codebook1;
    std::vector<int> codebook2;
    
    // Use regex to find all matches
    std::regex pattern1("<\\|c1_(\\d+)\\|>");
    std::regex pattern2("<\\|c2_(\\d+)\\|>");
    
    // Iterator for regex matches
    std::sregex_iterator iter1(codes.begin(), codes.end(), pattern1);
    std::sregex_iterator iter2(codes.begin(), codes.end(), pattern2);
    std::sregex_iterator end;
    
    // Extract codebook1 values
    for (; iter1 != end; ++iter1) {
        std::smatch match = *iter1;
        codebook1.push_back(std::stoi(match[1]));
    }
    
    // Extract codebook2 values
    for (; iter2 != end; ++iter2) {
        std::smatch match = *iter2;
        codebook2.push_back(std::stoi(match[1]));
    }
    
    // Truncate to the minimum size of both codebooks
    size_t t = std::min(codebook1.size(), codebook2.size());
    codebook1.resize(t);
    codebook2.resize(t);
    
    return {codebook1, codebook2};
}

//
// Terminal utils
//

#define SQR(X)    ((X) * (X))
#define UNCUBE(x) x < 48 ? 0 : x < 115 ? 1 : (x - 35) / 40

/**
 * Quantizes 24-bit RGB to xterm256 code range [16,256).
 */
static int rgb2xterm256(int r, int g, int b) {
    unsigned char cube[] = {0, 0137, 0207, 0257, 0327, 0377};
    int av, ir, ig, ib, il, qr, qg, qb, ql;
    av = r * .299 + g * .587 + b * .114 + .5;
    ql = (il = av > 238 ? 23 : (av - 3) / 10) * 10 + 8;
    qr = cube[(ir = UNCUBE(r))];
    qg = cube[(ig = UNCUBE(g))];
    qb = cube[(ib = UNCUBE(b))];
    if (SQR(qr - r) + SQR(qg - g) + SQR(qb - b) <=
        SQR(ql - r) + SQR(ql - g) + SQR(ql - b))
        return ir * 36 + ig * 6 + ib + 020;
    return il + 0350;
}

static std::string set_xterm256_foreground(int r, int g, int b) {
    int x = rgb2xterm256(r, g, b);
    std::ostringstream oss;
    oss << "\033[38;5;" << x << "m";
    return oss.str();
}

const std::vector<std::string> k_colors = {
    set_xterm256_foreground(220,   5,  12),
    set_xterm256_foreground(232,  96,  28),
    set_xterm256_foreground(241, 147,  45),
    set_xterm256_foreground(246, 193,  65),
    set_xterm256_foreground(247, 240,  86),
    set_xterm256_foreground(144, 201, 135),
    set_xterm256_foreground( 78, 178, 101),
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello!\"\n", argv[0]);
    LOG("\n");
}

static void prompt_add(llama_tokens & prompt, llama_token token) {
    prompt.push_back(token);
}

static void prompt_add(llama_tokens & prompt, const llama_tokens & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}

static void prompt_add(llama_tokens & prompt, const llama_vocab * vocab, const std::string & txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}

static void prompt_init(llama_tokens & prompt, const llama_vocab * vocab) {
    prompt.clear();
}

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "";

    params.n_predict = 8192;
    params.n_batch   = 8192;
    params.n_ctx     = 8192;

    // Recommended sampling params
    params.sampling.top_k = 40;
    params.sampling.temp = 0.4f;
    params.sampling.penalty_repeat = 1.1f;
    params.sampling.penalty_last_n = 64;
    params.sampling.min_p = 0.05f;

    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    const int n_parallel = params.n_parallel;
    const int n_predict  = params.n_predict;

    common_init();

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_ttc = NULL; // text-to-codes
    llama_context * ctx_ttc = NULL;

    // TODO not implemented
    llama_model * model_cts = NULL; // codes-to-speech
    llama_context * ctx_cts = NULL;

    common_init_result llama_init_ttc = common_init_from_params(params);

    model_ttc = llama_init_ttc.model.get();
    ctx_ttc   = llama_init_ttc.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model_ttc);

    // TODO: refactor in a common struct
    params.model     = params.vocoder.model;
    params.model_url = params.vocoder.model_url;
    params.hf_repo   = params.vocoder.hf_repo;
    params.hf_file   = params.vocoder.hf_file;

    params.embedding = true;

    // TODO DAC not implemented.
    // common_init_result llama_init_cts = common_init_from_params(params);
    // model_cts = llama_init_cts.model.get();
    // ctx_cts   = llama_init_cts.context.get();

    std::vector<common_sampler *> smpl(n_parallel);
    for (int i = 0; i < n_parallel; ++i) {
        params.sampling.no_perf = (i != 0);
        params.sampling.seed = params.sampling.seed + 1;

        smpl[i] = common_sampler_init(model_ttc, params.sampling);
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl[0]));
    LOG_INF("sampler params: \n%s\n", params.sampling.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl[0]).c_str());

    LOG_INF("%s: loading done\n", __func__);

    const auto t_main_start = ggml_time_us();

    //  process prompt and generate codes

    std::vector<llama_token> codes;

    std::vector<std::string> chunks;

    if (params.vocoder.chunked) {
        chunks = chunk_text(params.prompt, 10, 30);
    } else {
        chunks.push_back(params.prompt);
    }

    {
        for (std::string& prompt : chunks) {

            // Reset the context state before processing each new chunk
            llama_kv_cache_clear(ctx_ttc);

            LOG_INF("%s: constructing prompt ..\n", __func__);

            json speaker = nullptr;

            // load speaker if given
            if (!params.vocoder.speaker_file.empty()) {
                LOG_INF("%s: loading speaker ..\n", __func__);
                speaker = speaker_from_file(params.vocoder.speaker_file);

                if (speaker.empty()) {
                    LOG_ERR("%s: Failed to load speaker file '%s'\n", __func__, params.vocoder.speaker_file.c_str());
                    return 1;
                }
            } else {
                speaker = DefaultSpeaker::getJsonData();
            }

            std::vector<llama_token> prompt_inp;

            prompt_init(prompt_inp, vocab);

            // convert the input text into the necessary format expected by OuteTTS
            {
                std::string completion_prompt = get_completion_prompt(prompt, speaker);

                LOG_INF("%s: prompt: '%s'\n", __func__, completion_prompt.c_str());

                prompt_add(prompt_inp, vocab, completion_prompt, false, true);
            }

            // --- generate codes --- // 

            // create a llama_batch
            // we use this object to submit token data for decoding
            llama_batch batch = llama_batch_init(std::max(prompt_inp.size(), (size_t) n_parallel), 0, n_parallel);
            
            std::vector<llama_seq_id> seq_ids(n_parallel, 0);
            for (int32_t i = 0; i < n_parallel; ++i) {
                seq_ids[i] = i;
            }

            // evaluate the initial prompt
            for (size_t i = 0; i < prompt_inp.size(); ++i) {
                common_batch_add(batch, prompt_inp[i], i, seq_ids, false);
            }
            GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

            // llama_decode will output logits only for the last token of the prompt
            batch.logits[batch.n_tokens - 1] = true;

            if (llama_decode(ctx_ttc, batch) != 0) {
                LOG_ERR("%s: llama_decode() failed\n", __func__);
                return 1;
            }

            if (n_parallel > 1) {
                LOG_INF("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
            }

            llama_synchronize(ctx_ttc);

            LOG_INF("%s: time for prompt: %.3f ms\n\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

            const auto t_dec_start = ggml_time_us();
            
            // main loop

            // remember the batch index of the last token for each parallel sequence
            // we need this to determine which logits to sample from
            std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

            int n_past   = batch.n_tokens;
            int n_decode = 0;

            bool next_token_uses_guide_token = true;
            
            while (n_decode <= n_predict) {
                // prepare the next batch
                common_batch_clear(batch);

                // sample the next token for each parallel sequence / stream
                for (int32_t i = 0; i < n_parallel; ++i) {
                    if (i_batch[i] < 0) {
                        // the stream has already finished
                        continue;
                    }

                    llama_token new_token_id = common_sampler_sample(smpl[i], ctx_ttc, i_batch[i]);

                    // Chunked text can be used instead of guide tokens
                    // TODO implement this for v1 if still needed.

                    //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
                    // if (!guide_tokens.empty() && next_token_uses_guide_token && !llama_vocab_is_control(vocab, new_token_id) && !llama_vocab_is_eog(vocab, new_token_id)) {
                    //     llama_token guide_token = guide_tokens[0];
                    //     guide_tokens.erase(guide_tokens.begin());
                    //     new_token_id = guide_token; //ensure correct word fragment is used
                    // }

                    //this is the token id that always precedes a new word
                    next_token_uses_guide_token = (new_token_id == 198);

                    common_sampler_accept(smpl[i], new_token_id, true);

                    codes.push_back(new_token_id);

                    const auto * cands = common_sampler_get_candidates(smpl[i]);

                    // is it an end of generation? -> mark the stream as finished
                    if (llama_vocab_is_eog(vocab, new_token_id) || n_decode == n_predict) {
                        std::string reason;
                        if (llama_vocab_is_eog(vocab, new_token_id)) {
                            reason = "eos";
                        } else {
                            reason = "n_predict";
                        }

                        i_batch[i] = -1;

                        LOG("\n");
                        if (n_parallel > 1) {
                            LOG_CNT("\n");
                            LOG_INF("%s: stream %d finished at n_past = %d, reason = '%s'\n", __func__, i, n_past, reason.c_str());
                        }

                        continue;
                    }

                    {
                        const float p = cands->data[cands->selected].p;

                        const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) ((3*p)*float(k_colors.size()))));

                        LOG_CNT("%s%d%s", k_colors[col].c_str(), i, "\033[0m");
                        //LOG_CNT("%d", i);
                    }

                    i_batch[i] = batch.n_tokens;

                    // push this new token for next evaluation
                    common_batch_add(batch, new_token_id, n_past, { i }, true);
                }

                // all streams are finished
                if (batch.n_tokens == 0) {
                    break;
                }

                n_decode += 1;
                n_past += 1;

                // evaluate the current batch with the transformer model
                if (llama_decode(ctx_ttc, batch)) {
                    LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
                    return 1;
                }
            }

            llama_batch_free(batch);

            LOG("\n");
            LOG_INF("%s: time for decoder:       %.3f ms\n", __func__, (ggml_time_us() - t_dec_start) / 1000.0f);

            common_perf_print(ctx_ttc, smpl[0]);
        }
        
        {
            const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);

            // For DAC decoding
            std::vector<std::vector<int>> codebooks = extract_codebooks(inp_txt);

            // Create string representation of the codebooks for debugging
            std::stringstream cb1_str, cb2_str;
            
            cb1_str << "codebook1: [";
            for (size_t i = 0; i < codebooks[0].size(); ++i) {
                cb1_str << codebooks[0][i];
                if (i < codebooks[0].size() - 1) cb1_str << ", ";
            }
            cb1_str << "]";
            
            cb2_str << "codebook2: [";
            for (size_t i = 0; i < codebooks[1].size(); ++i) {
                cb2_str << codebooks[1][i];
                if (i < codebooks[1].size() - 1) cb2_str << ", ";
            }
            cb2_str << "]";

            LOG("\n");
            LOG_INF("codes: '%s'\n", inp_txt.c_str());
            LOG_INF("%s: codes size: %d\n", __func__, (int) codes.size());
            LOG_INF("%s: codebook sizes: cb1=%d, cb2=%d\n", __func__, (int)codebooks[0].size(), (int)codebooks[1].size());
            LOG_INF("%s: %s\n", __func__, cb1_str.str().c_str());
            LOG_INF("%s: %s\n", __func__, cb2_str.str().c_str());
        }


        // --- Speech Generation --- //
        // TODO: Functionality not yet implemented.
        // Requires integration with the DAC

    }

}
