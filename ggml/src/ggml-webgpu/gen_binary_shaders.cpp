#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pre_wgsl.hpp"

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_wgsl> <output_header>" << std::endl;
        return 1;
    }

    std::string shader_src  = read_file(argv[1]);
    std::string output_path = argv[2];

    pre_wgsl::Preprocessor p;

    struct Variant {
        std::string              name;
        std::vector<std::string> defines;
    };

    std::vector<Variant> variants = {
        // ADD
        { "add_f32",         { "TYPE_F32", "OP_ADD" } },
        { "add_f16",         { "TYPE_F16", "OP_ADD" } },
        { "add_f32_inplace", { "TYPE_F32", "OP_ADD", "INPLACE" } },
        { "add_f16_inplace", { "TYPE_F16", "OP_ADD", "INPLACE" } },
        // SUB
        { "sub_f32",         { "TYPE_F32", "OP_SUB" } },
        { "sub_f16",         { "TYPE_F16", "OP_SUB" } },
        { "sub_f32_inplace", { "TYPE_F32", "OP_SUB", "INPLACE" } },
        { "sub_f16_inplace", { "TYPE_F16", "OP_SUB", "INPLACE" } },
        // MUL
        { "mul_f32",         { "TYPE_F32", "OP_MUL" } },
        { "mul_f16",         { "TYPE_F16", "OP_MUL" } },
        { "mul_f32_inplace", { "TYPE_F32", "OP_MUL", "INPLACE" } },
        { "mul_f16_inplace", { "TYPE_F16", "OP_MUL", "INPLACE" } },
        // DIV
        { "div_f32",         { "TYPE_F32", "OP_DIV" } },
        { "div_f16",         { "TYPE_F16", "OP_DIV" } },
        { "div_f32_inplace", { "TYPE_F32", "OP_DIV", "INPLACE" } },
        { "div_f16_inplace", { "TYPE_F16", "OP_DIV", "INPLACE" } },
    };

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Cannot write: " << output_path << std::endl;
        return 1;
    }

    out << "// Auto-generated binary op shaders (via pre_wgsl)\n\n";

    for (const auto & v : variants) {
        std::string result = p.preprocess(shader_src, v.defines);
        out << "const char* wgsl_" << v.name << " = R\"(" << result << ")\";\n\n";
    }

    std::cout << "Generated " << variants.size() << " binary shader variants to " << output_path << std::endl;
    return 0;
}
