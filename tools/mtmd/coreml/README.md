# Apple CoreML / ANE Vision Backend

This directory provides an **Apple CoreML backend** for offloading vision encoder inference to
the Apple Neural Engine (ANE) or GPU, replacing the default CPU/Metal ggml-based ViT pipeline.

## Overview

Instead of running the ViT encoder op-by-op in ggml, the entire vision pipeline
(patch embedding → position embedding → Transformer layers → insert merger → final MLP merger)
is exported as a single `.mlmodelc` bundle using `coremltools`. At runtime, `libmtmd` loads
this bundle via the CoreML framework and executes it in one call.

```
┌──────────────────────────────────────────────────────┐
│  export_coreml.py (Python)                           │
│  HF checkpoint → torch.jit.trace → coremltools       │
│  → .mlpackage → xcrun compile → .mlmodelc            │
└──────────────────────┬───────────────────────────────┘
                       │ .mlmodelc
┌──────────────────────▼───────────────────────────────┐
│  libmtmd CoreML backend (C++ / ObjC++)               │
│  backend.mm: MLModel load / predict                   │
│  models/minicpmv.cpp: adapter (pack pixels, dispatch) │
└──────────────────────────────────────────────────────┘
```

## Directory structure

```
coreml/
├── README.md               ← this file
├── backend.h               # C++ header: load / unload / predict_single_output
├── backend.mm              # ObjC++ implementation (the only file touching CoreML.framework)
├── export_coreml.py        # Python: HF model → .mlpackage export script
└── models/
    ├── models.h            # C++ adapter registry
    ├── minicpmv.cpp        # C++ MiniCPM-V adapter
    └── modeling_siglip.py  # Python SigLIP ViT model definitions (shared by exporters)
```

## Quick start

### 1. Export the vision encoder

```bash
# Install dependencies (one-time)
pip install coremltools safetensors torch

# Export MiniCPM-V 4.6, float16 (ANE-only), 32×32 patch grid
python tools/mtmd/coreml/export_coreml.py \
    -m /path/to/MiniCPM-V-4_6 \
    --precision float16 \
    -o coreml_minicpmv46_vit_all_f16.mlpackage

# float32 variant (ANE + GPU fallback)
python tools/mtmd/coreml/export_coreml.py \
    -m /path/to/MiniCPM-V-4_6 \
    --precision float32 \
    -o coreml_minicpmv46_vit_all_f32.mlpackage
```

**Supported models:**

| Model family | Adapter | Config detection |
|---|---|---|
| MiniCPM-V 4.6 | `coreml/models/minicpmv.cpp` | `vision_config` in `config.json` |

The checkpoint can be a HuggingFace directory or `.zip` archive. Two weight formats are
supported:
- **Old format**: top-level keys prefixed `vpm.`, `vit_merger.`, `resampler.`
- **HF format**: keys prefixed `model.vpm.`, `model.vit_merger.`, `model.merger.`

The script automatically detects the format and maps weights to the correct sub-modules.

### 2. Compile for runtime

```bash
xcrun coremlcompiler compile coreml_minicpmv46_vit_all_f16.mlpackage output_dir/
# Creates: output_dir/coreml_minicpmv46_vit_all_f16.mlmodelc/
```

### 3. Build llama.cpp with CoreML support

```bash
cmake -B build -DMTMD_COREML=ON
cmake --build build
```

### 4. Run inference

```bash
./build/bin/llama-mtmd-cli \
    -m MiniCPM-V-4_6-Q4_K_M.gguf \
    --mmproj output_dir/coreml_minicpmv46_vit_all_f16.mlmodelc \
    --image cat.jpg \
    -p "Describe this image."
```

## Input / output contract

Every exported `.mlmodelc` must follow this schema for runtime compatibility:

| Role | Name | Dtype | Shape |
|---|---|---|---|
| Input | `pixel_values` | float32 | `[1, 3, 14, 14 × max_patches]` |
| Input | `patch_w` | int32 | `[1]` |
| Output | `output` | float32 | `[1, n_tokens, llm_embed_dim]` |

The C++ adapter (`models/minicpmv.cpp`) discovers `n_tokens` and `llm_embed_dim` from
the compiled model's `metadata.json` at load time, so the same adapter works across
variants with different hidden sizes.

## Adding a new model family

1. **Python model definition**: add (or reuse) modules in `models/modeling_*.py`
2. **Export logic**: add a detection branch in `export_coreml.py` `_detect_model()`
3. **C++ adapter**: create `models/your_model.cpp` implementing the `model_adapter` vtable:
   - `detect()` — match input/output schema from `metadata.json`
   - `setup()` — fill `hparams` + build preprocessor from metadata
   - `encode_slice()` — pack pixels → `backend::predict_single_output()`
4. **Register**: add `extern const model_adapter g_adapter;` to `models/models.h` and
   `&your_model::g_adapter` to the registry in `coreml/mtmd-coreml.cpp`

## Notes

- The export script requires **macOS** (coremltools needs Apple frameworks for compilation).
- `.mlmodelc` bundles are platform-specific; export once per major iOS/macOS version.
- `ComputeUnit.CPU_AND_NE` (float16) runs exclusively on ANE + CPU, offering the best
   perf/watt on Apple Silicon.
- `ComputeUnit.ALL` (float32) also allows GPU fallback, useful for precision-sensitive
   models or when running on Mac without Neural Engine.
