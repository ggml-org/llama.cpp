#include "jinja/string.h"
#include "jinja/value.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

//
// string_part
//

bool string_part::is_uppercase() const {
    for (char c : val) {
        if (std::islower(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

bool string_part::is_lowercase() const {
    for (char c : val) {
        if (std::isupper(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

//
// string
//

void string::mark_input() {
    for (auto & part : parts) {
        part.is_input = true;
    }
}

std::string string::str() const {
    std::string result;
    result.reserve(length());
    for (const auto & part : parts) {
        result += part.val;
    }
    return result;
}

bool string::empty() const {
    for (const auto & part : parts) {
        if (!part.val.empty()) {
            return false;
        }
    }
    return true;
}

std::optional<char> string::byte_at(size_t index) const {
    for (const auto & part : parts) {
        if (index < part.val.size()) {
            return part.val[index];
        }
        index -= part.val.size();
    }
    return {};
}

bool string::equals(const string & other) const {
    size_t part = 0, offset = 0;
    for (const auto & left : parts) {
        std::string_view text(left.val);
        while (!text.empty()) {
            while (part < other.parts.size() && offset == other.parts[part].val.size()) {
                ++part;
                offset = 0;
            }
            if (part == other.parts.size()) {
                return false;
            }
            const auto & right = other.parts[part].val;
            const auto count = std::min(text.size(), right.size() - offset);
            if (text.substr(0, count) != std::string_view(right).substr(offset, count)) {
                return false;
            }
            text.remove_prefix(count);
            offset += count;
        }
    }
    while (part < other.parts.size() && offset == other.parts[part].val.size()) {
        ++part;
        offset = 0;
    }
    return part == other.parts.size();
}

size_t string::length() const {
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.val.length();
    }
    return len;
}

void string::hash_update(hasher & hash) const noexcept {
    for (const auto & part : parts) {
        hash.update(part.val.data(), part.val.length());
    }
}

bool string::all_parts_are_input() const {
    for (const auto & part : parts) {
        if (!part.is_input) {
            return false;
        }
    }
    return true;
}

bool string::is_uppercase() const {
    for (const auto & part : parts) {
        if (!part.is_uppercase()) {
            return false;
        }
    }
    return true;
}

bool string::is_lowercase() const {
    for (const auto & part : parts) {
        if (!part.is_lowercase()) {
            return false;
        }
    }
    return true;
}

// mark this string as input if other has ALL parts as input
void string::mark_input_based_on(const string & other) {
    if (other.all_parts_are_input()) {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }
}

string & string::append(const string & other) {
    for (const auto & part : other.parts) {
        parts.push_back(part);
    }
    return *this;
}

// in-place transformation

using transform_fn = std::function<std::string(const std::string&)>;
static string apply_transform(string & self, const transform_fn & fn) {
    for (auto & part : self.parts) {
        part.val = fn(part.val);
    }
    return self;
}

string string::uppercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::toupper);
        return res;
    });
}
string string::lowercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
        return res;
    });
}
string string::capitalize() {
    return apply_transform(*this, [](const std::string & s) {
        if (s.empty()) return s;
        std::string res = s;
        res[0] = ::toupper(static_cast<unsigned char>(res[0]));
        std::transform(res.begin() + 1, res.end(), res.begin() + 1, ::tolower);
        return res;
    });
}
string string::titlecase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        bool capitalize_next = true;
        for (char &c : res) {
            if (isspace(static_cast<unsigned char>(c))) {
                capitalize_next = true;
            } else if (capitalize_next) {
                c = ::toupper(static_cast<unsigned char>(c));
                capitalize_next = false;
            } else {
                c = ::tolower(static_cast<unsigned char>(c));
            }
        }
        return res;
    });
}
string string::strip(bool left, bool right, std::optional<const std::string_view> chars) {
    static auto strip_part = [](const std::string & s, bool left, bool right, std::optional<const std::string_view> chars) -> std::string {
        size_t start = 0;
        size_t end = s.length();
        auto match_char = [&chars](unsigned char c) -> bool {
            return chars ? (*chars).find(c) != std::string::npos : isspace(c);
        };
        if (left) {
            while (start < end && match_char(static_cast<unsigned char>(s[start]))) {
                ++start;
            }
        }
        if (right) {
            while (end > start && match_char(static_cast<unsigned char>(s[end - 1]))) {
                --end;
            }
        }
        return s.substr(start, end - start);
    };
    if (parts.empty()) {
        return *this;
    }
    if (left) {
        for (size_t i = 0; i < parts.size(); ++i) {
            parts[i].val = strip_part(parts[i].val, true, false, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                --i;
                continue;
            } else {
                break;
            }
        }
    }
    if (right) {
        for (size_t i = parts.size(); i-- > 0;) {
            parts[i].val = strip_part(parts[i].val, false, true, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                continue;
            } else {
                break;
            }
        }
    }
    return *this;
}

} // namespace jinja
