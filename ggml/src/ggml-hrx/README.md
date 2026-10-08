# HRX Backend

## `hrx-system` backend compatibility

`hrx-minimum-commit.txt` records the minimum required commit in ROCm/hrx-system. It contains one full Git commit SHA. The backend's C++ APIs, Loom syntax, and numerical behavior can all depend on that revision. When `GGML_HRX=ON` and `HRX_SOURCE_DIR` is set, CMake requires the HRX checkout's HEAD to equal or descend from the recorded commit.

For builds using installed packages, check the source revision used to produce both HRX and Loom against this floor before validating a repair. From the staging checkout:

```sh
python3 llama.cpp/ggml/src/ggml-hrx/tools/check_hrx_revision.py hrx-system
```

### Updating the minimum

Raise the minimum in the same llama.cpp commit that introduces a newer HRX dependency. This includes new API calls, new Loom syntax, and removing a numerical workaround after an HRX compiler or runtime fix. Unrelated repairs do not require a bump.

Choose the earliest demonstrated required commit that retains the previous floor in its ancestry. Do not replace the floor with the latest tested revision merely because it is available. Record the dependency and its HRX commit or PR in the change's validation evidence. A minimum is not a promise that every future HRX revision is compatible.
