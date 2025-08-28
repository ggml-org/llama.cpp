#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <streambuf>
#include <vector>

#ifdef __APPLE__
#    include <locale>

/// @brief Custom ctype specialization for uint8_t to work around libc++
/// limitation in macOS
template <> struct std::ctype<uint8_t> : public std::ctype_base {
    using char_type = uint8_t;
    static std::locale::id id;

    ctype() : std::ctype_base() {}

    ctype([[maybe_unused]] const std::locale::facet & other) : std::ctype_base() {}

    ctype & operator=(const ctype & other) {
        if (this != &other) {
            std::ctype_base::operator=(other);
        }
        return *this;
    }

    // Required public interface methods
    bool is(mask m, [[maybe_unused]] char_type c) const {
        return (m & space) != 0;  // Treat all uint8_t as non-space
    }

    const char_type * is(const char_type * low, const char_type * high, mask * vec) const {
        for (; low != high; ++low, ++vec) {
            *vec = 0;  // No special character properties
        }
        return high;
    }

    const char_type * scan_is(mask m, const char_type * low, const char_type * high) const {
        for (; low != high; ++low) {
            if (is(m, *low)) {
                return low;
            }
        }
        return high;
    }

    const char_type * scan_not(mask m, const char_type * low, const char_type * high) const {
        for (; low != high; ++low) {
            if (!is(m, *low)) {
                return low;
            }
        }
        return high;
    }

    char_type toupper(char_type c) const {
        return c;  // No case conversion for uint8_t
    }

    const char_type * toupper([[maybe_unused]] char_type * low, const char_type * high) const {
        return high;  // No case conversion for uint8_t
    }

    char_type tolower(char_type c) const {
        return c;  // No case conversion for uint8_t
    }

    const char_type * tolower([[maybe_unused]] char_type * low, const char_type * high) const {
        return high;  // No case conversion for uint8_t
    }

    char_type widen(char c) const { return static_cast<char_type>(c); }

    const char * widen(const char * low, const char * high, char_type * dest) const {
        for (; low != high; ++low, ++dest) {
            *dest = static_cast<char_type>(*low);
        }
        return high;
    }

    char narrow(char_type c, [[maybe_unused]] char dfault) const { return static_cast<char>(c); }

    const char_type * narrow(const char_type * low, const char_type * high, [[maybe_unused]] char dfault,
                             char * dest) const {
        for (; low != high; ++low, ++dest) {
            *dest = static_cast<char>(*low);
        }
        return high;
    }
};
#endif

/// @brief Custom traits for uint8_t for usage in std template classes that use char_traits (e.g. std::basic_streambuf)
template <> struct std::char_traits<uint8_t> {
    using char_type  = uint8_t;
    using int_type   = int;
    using off_type   = std::streamoff;
    using pos_type   = std::streampos;
    using state_type = std::mbstate_t;

    static void assign(char_type & c1, const char_type & c2) noexcept { c1 = c2; }

    static constexpr bool eq(char_type a, char_type b) noexcept { return a == b; }

    static constexpr bool lt(char_type a, char_type b) noexcept { return a < b; }

    static int compare(const char_type * s1, const char_type * s2, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (lt(s1[i], s2[i])) {
                return -1;
            }
            if (lt(s2[i], s1[i])) {
                return 1;
            }
        }
        return 0;
    }

    static std::size_t length(const char_type * s) {
        std::size_t i = 0;
        while (!eq(s[i], char_type())) {
            ++i;
        }
        return i;
    }

    static const char_type * find(const char_type * s, std::size_t n, const char_type & c) {
        for (std::size_t i = 0; i < n; ++i) {
            if (eq(s[i], c)) {
                return s + i;
            }
        }
        return nullptr;
    }

    static char_type * move(char_type * s1, const char_type * s2, std::size_t n) {
        return static_cast<char_type *>(std::memmove(s1, s2, n));
    }

    static char_type * copy(char_type * s1, const char_type * s2, std::size_t n) {
        return static_cast<char_type *>(std::memcpy(s1, s2, n));
    }

    static char_type * assign(char_type * s, std::size_t n, char_type c) {
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = c;
        }
        return s;
    }

    static constexpr int_type not_eof(int_type c) noexcept { return eq_int_type(c, eof()) ? 0 : c; }

    static constexpr char_type to_char_type(int_type c) noexcept {
        return c >= 0 && c <= 255 ? static_cast<char_type>(c) : char_type();
    }

    static constexpr int_type to_int_type(char_type c) noexcept { return static_cast<int_type>(c); }

    static constexpr bool eq_int_type(int_type c1, int_type c2) noexcept { return c1 == c2; }

    static constexpr int_type eof() noexcept { return static_cast<int_type>(-1); }
};

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_CLASS_API __declspec(dllexport)
#        else
#            define GGML_CLASS_API __declspec(dllimport)
#        endif
#    else
#        define GGML_CLASS_API __attribute__((visibility("default")))
#    endif
#else
#    define GGML_CLASS_API
#endif

/// @brief Custom streambuf for uint8_t
class GGML_CLASS_API Uint8BufferStreamBuf : public std::basic_streambuf<uint8_t> {
  public:
    Uint8BufferStreamBuf(std::vector<uint8_t> && _data);

  protected:
    int_type underflow() override;

    /// @brief Efficient bulk reading. The standard implementation specifies that this function can be overridden
    /// to provide a more efficient implementation: sgetn will call this function if it is overridden.
    std::streamsize xsgetn(char_type * s, std::streamsize n) override;

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override;

    pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in) override;

  private:
    std::vector<uint8_t> data;
};
