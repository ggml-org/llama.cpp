#pragma once

#include <stdint.h>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace load_input_variant {

struct fname_load_input {
    const std::string &        fname;
    std::vector<std::string> & splits;  // optional, only need if the split does not follow naming scheme
};

struct buffer_load_input {
    std::unique_ptr<std::basic_streambuf<uint8_t>> & streambuf;
};

struct buffer_future_load_input {
    const std::string &        promise_key;
    const std::string &        context;
    std::vector<std::string> & splits;
    const std::string &        tensor_list_file;
};

}  // namespace load_input_variant

using load_input_t = std::variant<load_input_variant::fname_load_input, load_input_variant::buffer_load_input,
                                  load_input_variant::buffer_future_load_input>;

namespace load_input_variant {
const char * identifier(load_input_t & load_input);

fname_load_input split_name_from_variant(load_input_t & load_input);

bool variant_supports_split_load(load_input_t & load_input);

bool variant_supports_split_load_from_memory(load_input_t & load_input);

/// @brief Parse tensor list from future file or nullopt if not a future file
std::optional<std::set<std::string>> parse_tensor_list_from_future(load_input_t & load_input);
}  // namespace load_input_variant
