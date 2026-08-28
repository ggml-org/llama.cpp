// SPDX-License-Identifier: MIT
#pragma once

#include <climits>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace spec_sidecar_artifact {

struct TensorDesc {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    void * dev = nullptr;
};

// The sidecars intentionally accept only the small, versioned manifest format
// emitted by tools/prepare_assets.py. This parser still consumes complete JSON,
// rather than locating field-looking substrings, so malformed/nested input
// cannot turn missing values into huge sizes or offsets.
class ManifestParser {
public:
    ManifestParser(const std::string & text, std::string & error) : text_(text), error_(error) {}

    bool parse(std::vector<TensorDesc> & tensors) {
        tensors.clear();
        error_.clear();
        bool have_schema = false;
        bool have_tensors = false;
        uint64_t schema = 0;

        space();
        if (!take('{')) return fail("top-level value must be an object");
        space();
        if (!take('}')) {
            for (;;) {
                std::string key;
                if (!string(key)) return false;
                space();
                if (!take(':')) return fail("expected ':' after object key");
                space();
                if (key == "schema") {
                    if (have_schema) return fail("duplicate schema field");
                    have_schema = true;
                    if (!u64(schema)) return false;
                } else if (key == "tensors") {
                    if (have_tensors) return fail("duplicate tensors field");
                    have_tensors = true;
                    if (!tensor_array(tensors)) return false;
                } else if (!skip_value(0)) {
                    return false;
                }
                space();
                if (take('}')) break;
                if (!take(',')) return fail("expected ',' or '}' in object");
                space();
            }
        }
        space();
        if (pos_ != text_.size()) return fail("trailing data after top-level object");
        // Legacy extractor manifests predate the top-level schema field. Keep
        // accepting those, while rejecting every explicitly unsupported value.
        if (have_schema && schema != 1) return fail("manifest schema must be integer 1");
        if (!have_tensors) return fail("missing tensors array");
        return true;
    }

private:
    static constexpr size_t MAX_TENSORS = 256;
    static constexpr size_t MAX_DIMS = 8;
    static constexpr int MAX_DEPTH = 32;

    const std::string & text_;
    std::string & error_;
    size_t pos_ = 0;

    bool fail(const char * message) {
        if (error_.empty()) {
            std::ostringstream out;
            out << message << " at byte " << pos_;
            error_ = out.str();
        }
        return false;
    }

    void space() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
            ++pos_;
        }
    }

    bool take(char expected) {
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool literal(const char * value) {
        size_t n = 0;
        while (value[n] != '\0') ++n;
        if (pos_ + n > text_.size() || text_.compare(pos_, n, value) != 0) return false;
        pos_ += n;
        return true;
    }

    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool string(std::string & out) {
        if (!take('"')) return fail("expected string");
        out.clear();
        while (pos_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ == text_.size()) return fail("unterminated string escape");
            const char e = text_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned value = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (pos_ == text_.size()) return fail("short unicode escape");
                        const int h = hex(text_[pos_++]);
                        if (h < 0) return fail("invalid unicode escape");
                        value = value * 16 + static_cast<unsigned>(h);
                    }
                    // Tensor names/dtypes are ASCII. Decode ASCII escapes and
                    // retain a non-matching marker for other valid code points.
                    out.push_back(value <= 0x7f ? static_cast<char>(value) : '?');
                    break;
                }
                default: return fail("invalid string escape");
            }
        }
        return fail("unterminated string");
    }

    bool u64(uint64_t & out) {
        if (pos_ == text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
            return fail("expected non-negative integer");
        }
        if (text_[pos_] == '0' && pos_ + 1 < text_.size() &&
            text_[pos_ + 1] >= '0' && text_[pos_ + 1] <= '9') {
            return fail("leading zero in integer");
        }
        uint64_t value = 0;
        do {
            const unsigned digit = static_cast<unsigned>(text_[pos_] - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                return fail("integer overflow");
            }
            value = value * 10 + digit;
            ++pos_;
        } while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9');
        if (pos_ < text_.size() && (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
            return fail("integer field cannot be fractional");
        }
        out = value;
        return true;
    }

    bool number() {
        if (take('-') && pos_ == text_.size()) return fail("invalid number");
        if (take('0')) {
            if (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                return fail("leading zero in number");
            }
        } else {
            if (pos_ == text_.size() || text_[pos_] < '1' || text_[pos_] > '9') return fail("invalid number");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (take('.')) {
            if (pos_ == text_.size() || text_[pos_] < '0' || text_[pos_] > '9') return fail("invalid fraction");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ == text_.size() || text_[pos_] < '0' || text_[pos_] > '9') return fail("invalid exponent");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        return true;
    }

    bool skip_value(int depth) {
        if (depth > MAX_DEPTH) return fail("JSON nesting is too deep");
        space();
        if (pos_ == text_.size()) return fail("missing value");
        if (text_[pos_] == '"') {
            std::string ignored;
            return string(ignored);
        }
        if (take('{')) {
            space();
            if (take('}')) return true;
            for (;;) {
                std::string key;
                if (!string(key)) return false;
                space();
                if (!take(':')) return fail("expected ':' after object key");
                if (!skip_value(depth + 1)) return false;
                space();
                if (take('}')) return true;
                if (!take(',')) return fail("expected ',' or '}' in object");
                space();
            }
        }
        if (take('[')) {
            space();
            if (take(']')) return true;
            for (;;) {
                if (!skip_value(depth + 1)) return false;
                space();
                if (take(']')) return true;
                if (!take(',')) return fail("expected ',' or ']' in array");
                space();
            }
        }
        if (text_[pos_] == '-' || (text_[pos_] >= '0' && text_[pos_] <= '9')) return number();
        if (literal("true") || literal("false") || literal("null")) return true;
        return fail("invalid JSON value");
    }

    bool shape(std::vector<int64_t> & dims) {
        if (!take('[')) return fail("tensor shape must be an array");
        space();
        if (take(']')) return true;
        for (;;) {
            if (dims.size() == MAX_DIMS) return fail("tensor has too many dimensions");
            uint64_t value = 0;
            if (!u64(value)) return false;
            if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return fail("shape value overflow");
            dims.push_back(static_cast<int64_t>(value));
            space();
            if (take(']')) return true;
            if (!take(',')) return fail("expected ',' or ']' in shape");
            space();
        }
    }

    bool tensor(TensorDesc & out) {
        if (!take('{')) return fail("tensor descriptor must be an object");
        bool have_name = false, have_dtype = false, have_shape = false;
        bool have_offset = false, have_nbytes = false;
        space();
        if (take('}')) return fail("empty tensor descriptor");
        for (;;) {
            std::string key;
            if (!string(key)) return false;
            space();
            if (!take(':')) return fail("expected ':' after tensor field");
            space();
            if (key == "name") {
                if (have_name) return fail("duplicate tensor name field");
                have_name = true;
                if (!string(out.name)) return false;
            } else if (key == "dtype") {
                if (have_dtype) return fail("duplicate tensor dtype field");
                have_dtype = true;
                if (!string(out.dtype)) return false;
            } else if (key == "shape") {
                if (have_shape) return fail("duplicate tensor shape field");
                have_shape = true;
                if (!shape(out.shape)) return false;
            } else if (key == "offset") {
                if (have_offset) return fail("duplicate tensor offset field");
                have_offset = true;
                if (!u64(out.offset)) return false;
            } else if (key == "nbytes") {
                if (have_nbytes) return fail("duplicate tensor nbytes field");
                have_nbytes = true;
                if (!u64(out.nbytes)) return false;
            } else if (!skip_value(0)) {
                return false;
            }
            space();
            if (take('}')) break;
            if (!take(',')) return fail("expected ',' or '}' in tensor descriptor");
            space();
        }
        if (!have_name || !have_dtype || !have_shape || !have_offset || !have_nbytes) {
            return fail("tensor descriptor is missing a required field");
        }
        if (out.name.empty() || out.name.size() > 256 || out.dtype.empty() || out.dtype.size() > 16) {
            return fail("invalid tensor name or dtype length");
        }
        return true;
    }

    bool tensor_array(std::vector<TensorDesc> & tensors) {
        if (!take('[')) return fail("tensors must be an array");
        space();
        if (take(']')) return true;
        for (;;) {
            if (tensors.size() == MAX_TENSORS) return fail("too many tensors in manifest");
            TensorDesc item;
            if (!tensor(item)) return false;
            tensors.push_back(std::move(item));
            space();
            if (take(']')) return true;
            if (!take(',')) return fail("expected ',' or ']' in tensors array");
            space();
        }
    }
};

inline bool load_manifest(const char * path, std::vector<TensorDesc> & tensors, std::string & error) {
    static constexpr std::streamoff MAX_MANIFEST_BYTES = 1024 * 1024;
    tensors.clear();
    error.clear();
    if (path == nullptr || path[0] == '\0') {
        error = "empty manifest path";
        return false;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot open manifest";
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0 || size > MAX_MANIFEST_BYTES) {
        error = "manifest size is outside 1..1048576 bytes";
        return false;
    }
    std::string text(static_cast<size_t>(size), '\0');
    file.seekg(0, std::ios::beg);
    if (!file.read(&text[0], size)) {
        error = "short manifest read";
        return false;
    }
    ManifestParser parser(text, error);
    return parser.parse(tensors);
}

inline bool make_path(char * dst, size_t capacity, const char * directory, const char * leaf) {
    if (dst == nullptr || capacity == 0 || directory == nullptr || directory[0] == '\0' || leaf == nullptr) return false;
    const int count = std::snprintf(dst, capacity, "%s/%s", directory, leaf);
    return count >= 0 && static_cast<size_t>(count) < capacity;
}

inline bool file_size(FILE * file, uint64_t & size) {
    if (file == nullptr) return false;
#ifdef _WIN32
    if (_fseeki64(file, 0, SEEK_END) != 0) return false;
    const __int64 end = _ftelli64(file);
    if (end < 0 || _fseeki64(file, 0, SEEK_SET) != 0) return false;
#else
    if (fseeko(file, 0, SEEK_END) != 0) return false;
    const off_t end = ftello(file);
    if (end < 0 || fseeko(file, 0, SEEK_SET) != 0) return false;
#endif
    size = static_cast<uint64_t>(end);
    return true;
}

inline bool seek_file(FILE * file, uint64_t offset) {
    if (file == nullptr || offset > static_cast<uint64_t>(INT64_MAX)) return false;
#ifdef _WIN32
    return _fseeki64(file, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

inline bool read_exact_at(FILE * file, uint64_t offset, void * dst, uint64_t bytes) {
    if (dst == nullptr || bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
    return seek_file(file, offset) && std::fread(dst, 1, static_cast<size_t>(bytes), file) == static_cast<size_t>(bytes);
}

inline bool validate_blob_layout(const std::vector<TensorDesc> & tensors, uint64_t blob_size, std::string & error) {
    std::unordered_set<std::string> names;
    uint64_t cursor = 0;
    for (const TensorDesc & tensor : tensors) {
        if (!names.insert(tensor.name).second) {
            error = "duplicate tensor: " + tensor.name;
            return false;
        }
        if (tensor.nbytes == 0) {
            error = "zero-sized tensor: " + tensor.name;
            return false;
        }
        if (tensor.offset != cursor) {
            error = "non-contiguous tensor offset: " + tensor.name;
            return false;
        }
        if (tensor.nbytes > std::numeric_limits<uint64_t>::max() - cursor) {
            error = "tensor range overflow: " + tensor.name;
            return false;
        }
        cursor += tensor.nbytes;
        if (cursor > blob_size) {
            error = "tensor range exceeds weights blob: " + tensor.name;
            return false;
        }
    }
    if (cursor != blob_size) {
        error = "weights blob size does not match manifest ranges";
        return false;
    }
    return true;
}

inline bool validate_remap(const std::vector<int32_t> & ids, int32_t vocab, std::string & error) {
    if (vocab <= 0) {
        error = "invalid vocabulary size";
        return false;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(vocab), 0);
    for (size_t row = 0; row < ids.size(); ++row) {
        const int32_t id = ids[row];
        if (id < 0 || id >= vocab) {
            std::ostringstream out;
            out << "remap id " << id << " at row " << row << " is outside vocabulary";
            error = out.str();
            return false;
        }
        if (seen[static_cast<size_t>(id)] != 0) {
            std::ostringstream out;
            out << "duplicate remap id " << id << " at row " << row;
            error = out.str();
            return false;
        }
        seen[static_cast<size_t>(id)] = 1;
    }
    return true;
}

inline TensorDesc * find_tensor(std::vector<TensorDesc> & tensors, const char * name) {
    for (TensorDesc & tensor : tensors) if (tensor.name == name) return &tensor;
    return nullptr;
}

inline const TensorDesc * find_tensor(const std::vector<TensorDesc> & tensors, const char * name) {
    for (const TensorDesc & tensor : tensors) if (tensor.name == name) return &tensor;
    return nullptr;
}

inline bool tensor_matches(const TensorDesc & tensor, const char * dtype,
                           std::initializer_list<int64_t> shape, uint64_t nbytes) {
    if (tensor.dtype != dtype || tensor.nbytes != nbytes || tensor.shape.size() != shape.size()) return false;
    size_t index = 0;
    for (const int64_t dim : shape) {
        if (tensor.shape[index++] != dim) return false;
    }
    return true;
}

} // namespace spec_sidecar_artifact
