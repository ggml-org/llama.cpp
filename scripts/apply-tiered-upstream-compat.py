#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_once(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched {label}: {path.relative_to(ROOT)}")
        return False
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{label}: expected exactly one upstream marker in {path.relative_to(ROOT)}, found {count}; "
            "upstream drift requires a manual compatibility review"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {label}: {path.relative_to(ROOT)}")
    return True


patch_once(
    ROOT / "include/llama.h",
    """        bool no_alloc;        // only load metadata and simulate memory allocations
        bool load_mtp;        // whether to load MTP layers
    };""",
    """        bool no_alloc;        // only load metadata and simulate memory allocations
        bool load_mtp;        // whether to load MTP layers
        bool no_mmap_prefetch; // Summer.cpp: skip eager mmap prefetch for tiered loading
    };""",
    "llama_model_params",
)

patch_once(
    ROOT / "src/llama-model.cpp",
    """    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);""",
    """    ml.init_mappings(!params.no_mmap_prefetch, use_mlock ? &pimpl->mlock_mmaps : nullptr);""",
    "tiered mmap prefetch control",
)

patch_once(
    ROOT / "src/llama-model-loader.cpp",
    """            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }""",
    """            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);

                // Tiered buffers may retain direct pointers into the GGUF mmap. Account for
                // only those byte ranges so upstream's partial-unmap optimization stays active.
                constexpr char tiered_prefix[] = "CUDA_TIERED";
                const char * buffer_name = ggml_backend_buffer_name(cur->buffer);
                if (buffer_name && std::strncmp(buffer_name, tiered_prefix, sizeof(tiered_prefix) - 1) == 0) {
                    auto & mmap_used = mmaps_used[weight->idx];
                    mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                    mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
                }
            }""",
    "tiered mmap lifetime accounting",
)

patch_once(
    ROOT / "src/llama-tiered.cpp",
    """    model_params.mmap_prefetch = false;""",
    """    model_params.no_mmap_prefetch = true;""",
    "tiered model prefetch policy",
)
