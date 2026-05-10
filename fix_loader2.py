with open("llama.cpp-PoC/src/llama-model.cpp", "r") as f:
    content = f.read()

# fix `std::unique_ptr<llama_mmap>::unique_ptr(std::shared_ptr<llama_mmap>)` error
content = content.replace("pimpl->mappings.emplace_back(mapping);", "")
content = content.replace("pimpl->mappings.emplace_back(ml.mappings[idx]);", "pimpl->mappings.emplace_back(std::move(ml.mappings[idx]));")
# let's just make it simpler
# Actually, the error might be around line 1539 in `llama-model.cpp`
# let's grep that line

import subprocess
out = subprocess.check_output("cat llama.cpp-PoC/src/llama-model.cpp | grep -n -B 5 -A 5 'mappings.emplace_back'", shell=True).decode()
print(out)
