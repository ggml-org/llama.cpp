#pragma once

// whether a CUDA device reports host MemAvailable (/proc/meminfo) as its free memory
// under WSL2 (/dev/dxg) the Windows host owns an integrated GPU's memory and /proc/meminfo describes only the VM
static inline bool ggml_cuda_uma_free_from_host(bool integrated, bool unified_memory_env, bool wsl) {
    if (integrated && wsl) {
        return false;
    }
    return integrated || unified_memory_env;
}
