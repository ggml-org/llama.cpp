with open("llama.cpp-PoC/src/llama-model-loader.h", "r") as f:
    content = f.read()

if 'is_ffn_tensor' not in content:
    content = content.replace(
        'struct split_info_t {',
        """
inline bool is_ffn_tensor(const char* name) {
    return strstr(name, "ffn_norm")       != nullptr ||
           strstr(name, "ffn_gate")       != nullptr ||
           strstr(name, "ffn_up")         != nullptr ||
           strstr(name, "ffn_down")       != nullptr ||
           strstr(name, "ffn_gate_exps")  != nullptr ||
           strstr(name, "ffn_up_exps")    != nullptr ||
           strstr(name, "ffn_down_exps")  != nullptr;
}
""" + '\nstruct split_info_t {'
    )

    with open("llama.cpp-PoC/src/llama-model-loader.h", "w") as f:
        f.write(content)
