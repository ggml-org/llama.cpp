#include "parsers.h"

#include "unicode.h"

#include <algorithm>
#include <charconv>
#include <string_view>

namespace {

int digit(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return 16;
}

// Convert JSON-compatible Python literals without evaluating expressions.
class literal_reader {
  public:
    explicit literal_reader(std::string_view text) : text(text) {}

    std::optional<json> parse() {
        auto first = value(0);
        if (!first || !take(',')) {
            return text.empty() ? first : std::nullopt;
        }
        auto tuple = json::array({std::move(*first)});
        while (!text.empty()) {
            auto item = value(0);
            if (!item) {
                return std::nullopt;
            }
            tuple.push_back(std::move(*item));
            if (!take(',')) {
                break;
            }
        }
        return text.empty() ? std::optional<json>(std::move(tuple)) : std::nullopt;
    }

  private:
    std::string_view text;

    void skip_space() {
        while (true) {
            const auto start = text.find_first_not_of(" \t\r\n\f");
            if (start == text.npos) {
                text = {};
                return;
            }
            text.remove_prefix(start);
            if (text.front() != '#') {
                return;
            }
            const auto end = text.find('\n');
            text = end == text.npos ? std::string_view() : text.substr(end + 1);
        }
    }

    bool take(char c) {
        skip_space();
        if (text.empty() || text.front() != c) {
            return false;
        }
        text.remove_prefix(1);
        return true;
    }

    static bool is_quote(char c) {
        return c == '\'' || c == '"';
    }

    bool starts_string() const {
        return !text.empty() && (is_quote(text.front()) ||
            (text.size() >= 2 && std::string_view("rRuU").find(text.front()) != text.npos && is_quote(text[1])));
    }

    bool append_escape(char c, std::string & out) {
        constexpr std::string_view escapes = "'\"\\abfnrtv";
        constexpr std::string_view decoded = "'\"\\\a\b\f\n\r\t\v";
        const auto simple = escapes.find(c);
        if (simple != escapes.npos) {
            out += decoded[simple];
            return true;
        }
        if (c == '\n') {
            return true;
        }

        const bool octal = c >= '0' && c <= '7';
        const size_t width = c == 'x' ? 2 : c == 'u' ? 4 : c == 'U' ? 8 : 0;
        if (!octal && !width) {
            if (c == 'N') { // Named escapes require the Python Unicode name database.
                return false;
            }
            out += '\\';
            out += c;
            return true;
        }

        uint32_t codepoint = octal ? digit(c) : 0;
        size_t i = 0;
        const int base = octal ? 8 : 16;
        for (; i < (octal ? 2 : width) && i < text.size(); ++i) {
            const auto d = digit(text[i]);
            if (d >= base) {
                break;
            }
            codepoint = codepoint * base + d;
        }
        if ((!octal && i != width) || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
        text.remove_prefix(i);
        out += common_unicode_cpt_to_utf8(codepoint);
        return true;
    }

    bool read_quoted(std::string & out) {
        const char prefix = text.front();
        const bool raw = prefix == 'r' || prefix == 'R';
        if (!is_quote(prefix)) {
            text.remove_prefix(1);
        }
        const auto quote = text.substr(0, text.size() >= 3 && text[0] == text[1] && text[0] == text[2] ? 3 : 1);
        text.remove_prefix(quote.size());
        while (!text.empty()) {
            if (text.substr(0, quote.size()) == quote) {
                text.remove_prefix(quote.size());
                return true;
            }
            char c = text.front();
            text.remove_prefix(1);
            if (c == '\\') {
                if (text.empty()) {
                    return false;
                }
                c = text.front();
                text.remove_prefix(1);
                if (raw) {
                    out += '\\';
                    out += c;
                } else if (!append_escape(c, out)) {
                    return false;
                }
            } else {
                if (c == '\0' || (quote.size() == 1 && (c == '\r' || c == '\n'))) {
                    return false;
                }
                out += c;
            }
        }
        return false;
    }

    std::optional<json> read_string() {
        std::string out;
        do {
            if (!read_quoted(out)) {
                return std::nullopt;
            }
            skip_space();
        } while (starts_string());
        return json(std::move(out));
    }

    std::optional<json> number() {
        const auto length = std::min(text.find_first_not_of("0123456789abcdefABCDEFxXoO_+-."), text.size());
        auto token = text.substr(0, length);
        text.remove_prefix(length);
        const bool negative = token.front() == '-';
        if (negative || token.front() == '+') {
            token.remove_prefix(1);
        }
        int base = 10;
        if (token.size() >= 2 && token.front() == '0') {
            switch (token[1]) {
                case 'x': case 'X': base = 16; break;
                case 'o': case 'O': base = 8;  break;
                case 'b': case 'B': base = 2;  break;
                default: break;
            }
            if (base != 10) {
                token.remove_prefix(2);
            }
        }
        std::string clean;
        for (size_t i = 0; i < token.size(); ++i) {
            if (token[i] != '_') {
                clean += token[i];
            } else if (i + 1 == token.size() || digit(token[i + 1]) >= base ||
                       (i == 0 ? base == 10 : digit(token[i - 1]) >= base)) {
                return std::nullopt;
            }
        }

        if (base != 10) {
            uint64_t magnitude;
            const auto parsed = std::from_chars(clean.data(), clean.data() + clean.size(), magnitude, base);
            if (parsed.ec != std::errc() || parsed.ptr != clean.data() + clean.size()) {
                return std::nullopt;
            }
            clean = std::to_string(magnitude);
        } else {
            if (clean.empty() || (clean.front() == '.' && (clean.size() == 1 || digit(clean[1]) >= 10))) {
                return std::nullopt;
            }
            const bool integer = clean.find_first_of(".eE") == clean.npos;
            if (clean.front() == '.') {
                clean.insert(0, "0");
            }
            const auto dot = clean.find('.');
            if (dot != clean.npos && (dot + 1 == clean.size() || clean[dot + 1] == 'e' || clean[dot + 1] == 'E')) {
                clean.insert(dot + 1, "0");
            }
            size_t zeros = 0;
            while (zeros + 1 < clean.size() && clean[zeros] == '0' && digit(clean[zeros + 1]) < 10) {
                ++zeros;
            }
            if (integer && zeros > 0 && clean.find_first_not_of('0') != clean.npos) {
                return std::nullopt;
            }
            clean.erase(0, zeros);
        }
        if (negative) {
            clean.insert(0, "-");
        }
        auto result = json::parse_no_throw(clean);
        return result.is_discarded() ? std::nullopt : std::optional<json>(std::move(result));
    }

    std::optional<json> sequence(char close, size_t depth) {
        auto items = json::array();
        if (take(close)) {
            return items;
        }
        auto first = value(depth);
        if (!first) {
            return std::nullopt;
        }
        if (close == ')' && take(close)) {
            return first;
        }
        items.push_back(std::move(*first));
        while (!take(close)) {
            if (!take(',')) {
                return std::nullopt;
            }
            if (take(close)) {
                break;
            }
            auto item = value(depth);
            if (!item) {
                return std::nullopt;
            }
            items.push_back(std::move(*item));
        }
        return items;
    }

    std::optional<json> object(size_t depth) {
        auto result = json::object();
        while (!take('}')) {
            auto key = value(depth);
            if (!key || key->is_array() || key->is_object() || !take(':')) {
                return std::nullopt;
            }
            auto item = value(depth);
            if (!item) {
                return std::nullopt;
            }
            result[key->is_string() ? key->get<std::string>() : key->dump()] = std::move(*item);
            if (!take(',')) {
                return take('}') ? std::optional<json>(std::move(result)) : std::nullopt;
            }
        }
        return result;
    }

    std::optional<json> value(size_t depth) {
        skip_space();
        // CPython rejects overly nested literals before evaluating them.
        if (text.empty() || depth > 200) {
            return std::nullopt;
        }
        if (starts_string()) {
            return read_string();
        }
        const char first = text.front();
        if (first == '[' || first == '(' || first == '{') {
            text.remove_prefix(1);
            return first == '{' ? object(depth + 1) : sequence(first == '(' ? ')' : ']', depth + 1);
        }
        if (digit(first) < 10 || first == '+' || first == '-' || first == '.') {
            return number();
        }
        for (const std::string_view name : {"True", "False", "None"}) {
            if (text.substr(0, name.size()) == name) {
                text.remove_prefix(name.size());
                return name == "None" ? json(nullptr) : json(name == "True");
            }
        }
        return std::nullopt;
    }
};

} // namespace

json common_chat_parse_minicpm5_value(std::string_view input) {
    auto value = json::parse_no_throw(std::string(input));
    if (!value.is_discarded()) {
        return value;
    }
    auto literal = literal_reader(input).parse();
    return literal ? std::move(*literal) : json(std::string(input));
}
