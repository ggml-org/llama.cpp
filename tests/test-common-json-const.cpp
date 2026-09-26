// Iterating a const common_json must hand out const references, and
// iterating a mutable one must still write through.
#include "json.h"

#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

static_assert(std::is_same<decltype(*std::declval<const common_json &>().begin()), const common_json &>::value,
              "const begin() must yield const elements");
static_assert(std::is_same<decltype(*std::declval<common_json &>().begin()), common_json &>::value,
              "begin() must yield mutable elements");
static_assert(std::is_same<decltype((*std::declval<const common_json &>().items().begin()).v), const common_json &>::value,
              "const items() must yield const values");
static_assert(std::is_same<decltype((*std::declval<common_json &>().items().begin()).v), common_json &>::value,
              "items() must yield mutable values");

int main() {
    common_json obj = common_json::parse(R"({"a": 1, "b": 2})");

    // mutable iteration writes through
    for (auto & v : obj) {
        v = v.get<int>() * 10;
    }

    const common_json & cobj = obj;

    std::string keys;
    int         sum = 0;
    for (const auto & [k, v] : cobj.items()) {
        keys += k;
        sum  += v.get<int>();
    }
    for (const auto & v : cobj) {
        sum += v.get<int>();
    }
    if (keys != "ab" || sum != 60) {
        std::fprintf(stderr, "object iteration: keys=%s sum=%d, expected ab 60\n", keys.c_str(), sum);
        return 1;
    }

    const common_json arr = common_json::parse("[1, 2, 3]");

    size_t n = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        if ((*it).get<int>() != (int) n + 1) {
            std::fprintf(stderr, "array iteration: unexpected element at %zu\n", n);
            return 1;
        }
        n++;
    }
    if (n != 3) {
        std::fprintf(stderr, "array iteration: %zu elements, expected 3\n", n);
        return 1;
    }

    // items() keys an array by index
    n = 0;
    for (const auto & [k, v] : arr.items()) {
        if (k != std::to_string(n) || v.get<int>() != (int) n + 1) {
            std::fprintf(stderr, "array items: unexpected entry at %zu\n", n);
            return 1;
        }
        n++;
    }

    std::fprintf(stderr, "main : const iteration hands out const references\n");
    return 0;
}
