#pragma once

#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include <algorithm>
#include <cstdint>


namespace jinja {

// allow differentiate between user input strings and template strings
// transformations should handle this information as follows:
// - one-to-one (e.g., uppercase, lowercase): preserve is_input flag
// - one-to-many (e.g., strip): if input string is marked as is_input, all resulting parts should be marked as is_input
// - many-to-one (e.g., concat): if ALL input parts are marked as is_input, resulting part should be marked as is_input
struct string_part {
    bool is_input = false; // may skip parsing special tokens if true
    std::string val;

    bool is_uppercase() const {
        for (char c : val) {
            if (std::islower(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }

    bool is_lowercase() const {
        for (char c : val) {
            if (std::isupper(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }
};

struct string {
    using transform_fn = std::function<std::string(const std::string&)>;

    std::vector<string_part> parts;
    string() = default;
    string(const std::string & v, bool user_input = false) {
        parts.push_back({user_input, v});
    }
    string(int v) {
        parts.push_back({false, std::to_string(v)});
    }
    string(double v) {
        parts.push_back({false, std::to_string(v)});
    }

    void mark_input() {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }

    std::string str() const {
        if (parts.size() == 1) {
            return parts[0].val;
        }
        std::ostringstream oss;
        for (const auto & part : parts) {
            oss << part.val;
        }
        return oss.str();
    }

    size_t length() const {
        size_t len = 0;
        for (const auto & part : parts) {
            len += part.val.length();
        }
        return len;
    }

    bool all_parts_are_input() const {
        for (const auto & part : parts) {
            if (!part.is_input) {
                return false;
            }
        }
        return true;
    }

    bool is_uppercase() const {
        for (const auto & part : parts) {
            if (!part.is_uppercase()) {
                return false;
            }
        }
        return true;
    }

    bool is_lowercase() const {
        for (const auto & part : parts) {
            if (!part.is_lowercase()) {
                return false;
            }
        }
        return true;
    }

    // mark this string as input if other has ALL parts as input
    void mark_input_based_on(const string & other) {
        if (other.all_parts_are_input()) {
            for (auto & part : parts) {
                part.is_input = true;
            }
        }
    }

    string append(const string & other) {
        for (const auto & part : other.parts) {
            parts.push_back(part);
        }
        return *this;
    }

    // in-place transformation

    string apply_transform(const transform_fn & fn) {
        for (auto & part : parts) {
            part.val = fn(part.val);
        }
        return *this;
    }
    string uppercase() {
        return apply_transform([](const std::string & s) {
            std::string res = s;
            std::transform(res.begin(), res.end(), res.begin(), ::toupper);
            return res;
        });
    }
    string lowercase() {
        return apply_transform([](const std::string & s) {
            std::string res = s;
            std::transform(res.begin(), res.end(), res.begin(), ::tolower);
            return res;
        });
    }
    string capitalize() {
        return apply_transform([](const std::string & s) {
            if (s.empty()) return s;
            std::string res = s;
            res[0] = ::toupper(static_cast<unsigned char>(res[0]));
            std::transform(res.begin() + 1, res.end(), res.begin() + 1, ::tolower);
            return res;
        });
    }
    string titlecase() {
        return apply_transform([](const std::string & s) {
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
    string strip(bool left, bool right) {
        static auto strip_part = [](const std::string & s, bool left, bool right) -> std::string {
            size_t start = 0;
            size_t end = s.length();
            if (left) {
                while (start < end && isspace(static_cast<unsigned char>(s[start]))) {
                    ++start;
                }
            }
            if (right) {
                while (end > start && isspace(static_cast<unsigned char>(s[end - 1]))) {
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
                parts[i].val = strip_part(parts[i].val, true, false);
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
                parts[i].val = strip_part(parts[i].val, false, true);
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
};

} // namespace jinja
