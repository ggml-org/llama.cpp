# Docker

> [!NOTE]
> This fork ships **no Dockerfiles** and publishes no images. The `.devops/` directory inherited from upstream (including `cpu.Dockerfile`, `vulkan.Dockerfile`, `intel.Dockerfile`, and `openvino.Dockerfile`) was removed in merge `181a4a8fb`, together with `.github/` and `ci/`. Of the Docker files, only the root `.dockerignore` remains.

## Options

* Build from source on the host instead; see [build.md](build.md).
* To start from the last fork copies of the Dockerfiles, restore them from the parent of that merge, for example `git show 181a4a8fb^1:.devops/vulkan.Dockerfile > vulkan.Dockerfile`. They are not maintained and have not been checked against the current tree.

## Notes

* The upstream `ghcr.io/ggml-org/llama.cpp:*` tags do **not** include the TurboQuant+ codec stack; pull them only if you want upstream mainline with no fork-specific features.
