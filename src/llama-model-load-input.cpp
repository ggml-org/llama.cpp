#include "llama-model-load-input.h"
#include <sstream>
#include "llama-mmap.h"

namespace load_input_variant {

const char * identifier(load_input_t & load_input) {
    if (std::holds_alternative<fname_load_input>(load_input)) {
        const auto & file_input = std::get<fname_load_input>(load_input);
        return file_input.fname.c_str();
    }
    static const char * buffer_id_str = "buffer";
    return buffer_id_str;
}

fname_load_input split_name_from_variant(load_input_t & load_input) {
    if (std::holds_alternative<buffer_future_load_input>(load_input)) {
        auto future_input = std::get<buffer_future_load_input>(load_input);
        return fname_load_input{ future_input.promise_key, future_input.splits };
    }
    auto file_input = std::get<fname_load_input>(load_input);
    return file_input;
}

bool variant_supports_split_load(load_input_t & load_input) {
    return std::holds_alternative<fname_load_input>(load_input) ||
           std::holds_alternative<buffer_future_load_input>(load_input);
}

bool variant_supports_split_load_from_memory(load_input_t & load_input) {
    return std::holds_alternative<buffer_future_load_input>(load_input);
}

std::optional<std::set<std::string>> parse_tensor_list_from_future(load_input_t & load_input) {
    std::set<std::string> tensor_names;

    if (!std::holds_alternative<buffer_future_load_input>(load_input)) {
        return std::nullopt;
    }

    const auto & future_input = std::get<buffer_future_load_input>(load_input);

    // Open and read the tensor list file
    llama_future_file_buffer_ro           tensor_file(future_input.tensor_list_file, future_input.context);
    std::unique_ptr<llama_file_buffer_ro> file_buffer = tensor_file.extract();

    // Read the entire buffer as bytes and convert to string
    std::vector<uint8_t>              buffer;
    std::basic_istream<uint8_t>       stream(file_buffer->streambuf.get());
    std::istreambuf_iterator<uint8_t> begin(stream), end;
    buffer.assign(begin, end);

    // Convert bytes to string and split by newlines
    std::string        content(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    std::istringstream line_stream(content);
    std::string        line;
    while (std::getline(line_stream, line)) {
        tensor_names.insert(line);
    }

    return tensor_names;
}

}  // namespace load_input_variant
