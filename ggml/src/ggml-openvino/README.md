# Instructions to Modify and Build ggml with OpenVINO

## Step 1: Modify the Source Code

In order to change the frontend `.so` path to the path to `.so` file, you need to add path to the `.so` file in cmake compiler option:
1. Open a terminal and navigate to the root directory of this repo.
2. Run the following commands to configure:
   ```sh
   mkdir build
   cmake -B build -DGGML_OV_FRONTEND="${openvino_repo_dir}/bin/intel64/Release/libopenvino_ggml_frontend.so"
   ```
Where GGML_OV_FRONTEND should point to the path to `libopenvino_ggml_frontend.so` file.

## Step 2: Build the Project

After modifying the source code, you need to build the project using CMake. Follow these steps:

1. (Optional) Enable debug option for ggml-openvino, this will output dump of subgraph sent to OpenVINO, information after convert ggml_cgraph to GraphIterator, and calculation input value/output value of each OP:
   ```sh
    cmake -B build -DGGML_OPENVINO_DEBUG=ON
    ```

2. Run the following commands to configure and build the project:
   ```sh
   cmake -B build -DGGML_OPENVINO=ON
   cmake --build build -j
   ```

This will configure the project with OpenVINO support and build it using multiple cores for faster compilation.

