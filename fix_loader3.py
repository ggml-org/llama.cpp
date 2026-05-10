with open("llama.cpp-PoC/src/llama-model.cpp", "r") as f:
    content = f.read()

# Since `ml.mappings` is `std::vector<std::shared_ptr<llama_mmap>>` now, but `pimpl->mappings` expects `std::unique_ptr`, we need to change it back to `std::unique_ptr` in `llama-model-loader.h`.

with open("llama.cpp-PoC/src/llama-model-loader.h", "r") as f:
    hcontent = f.read()

hcontent = hcontent.replace("std::vector<std::shared_ptr<llama_file>> files;", "std::vector<std::unique_ptr<llama_file>> files;")
hcontent = hcontent.replace("std::vector<std::shared_ptr<llama_mmap>> mappings;", "std::vector<std::unique_ptr<llama_mmap>> mappings;")

with open("llama.cpp-PoC/src/llama-model-loader.h", "w") as f:
    f.write(hcontent)
