#!/bin/bash
# Steps to merge the branch to latest
#git clone git@github.com:tsisw/llama.cpp.git
#git remote add upstream https://github.com/ggml-org/llama.cpp.git
#git fetch upstream
#git checkout master
#git merge upstream/master

set -e

# Accept MLIR_COMPILER_DIR and TOOLBOX_DIR as arguments or environment variables
# Usage: ./tsi-pkg-build.sh [release|debug] [MLIR_COMPILER_DIR] [TOOLBOX_DIR]
BUILD_TYPE=${1:-}
MLIR_COMPILER_DIR=${2:-${MLIR_COMPILER_DIR:-}}
TOOLBOX_DIR=${3:-${TOOLBOX_DIR:-}}

# Default to SDK paths if not provided
if [ -z "${MLIR_COMPILER_DIR}" ]; then
  MLIR_SDK_VERSION=${MLIR_SDK_VERSION:-/proj/rel/sw/sdk-r.0.2.2}
  MLIR_COMPILER_DIR=${MLIR_SDK_VERSION}/compiler
  echo "Using default MLIR_COMPILER_DIR: ${MLIR_COMPILER_DIR}"
fi

if [ -z "${TOOLBOX_DIR}" ]; then
  # Derive from MLIR_COMPILER_DIR parent if MLIR_SDK_VERSION exists
  MLIR_SDK_VERSION=${MLIR_SDK_VERSION:-$(dirname ${MLIR_COMPILER_DIR})}
  TOOLBOX_DIR=${MLIR_SDK_VERSION}/toolbox/build/install
  echo "Using default TOOLBOX_DIR: ${TOOLBOX_DIR}"
fi

# Convert to absolute paths (important since script changes directories)
MLIR_COMPILER_DIR=$(cd "${MLIR_COMPILER_DIR}" 2>/dev/null && pwd) || {
  echo "ERROR: MLIR_COMPILER_DIR not found: ${MLIR_COMPILER_DIR}"
  exit 1
}
TOOLBOX_DIR=$(cd "${TOOLBOX_DIR}" 2>/dev/null && pwd) || {
  echo "ERROR: TOOLBOX_DIR not found: ${TOOLBOX_DIR}"
  exit 1
}

echo "MLIR_COMPILER_DIR: ${MLIR_COMPILER_DIR}"
echo "TOOLBOX_DIR: ${TOOLBOX_DIR}"

# Export as environment variables for sub-scripts
export MLIR_SDK_VERSION=${MLIR_SDK_VERSION:-$(dirname ${MLIR_COMPILER_DIR})}
export COMPILER_INSTALL_DIR=${MLIR_COMPILER_DIR}
export TOOLBOX_DIR

# Check if enable_coverage is passed as an argument
ENABLE_COVERAGE_FLAG=""
for arg in "$@"; do
    if [ "$(echo "$arg" | tr '[:upper:]' '[:lower:]')" = "enable_coverage" ]; then
        ENABLE_COVERAGE_FLAG="-DENABLE_COVERAGE=ON"
        echo "enable_coverage argument detected - enabling coverage flags"
        break
    fi
done

#Ensure prerequisites are met as follows
echo 'updating submodule'
git submodule update --recursive --init
cd ggml-tsi-kernel/
#module load gcc/13.3.0
# Set compiler environment variables to use GCC 13.3.0
export CC="/proj/local/gcc-13.3.0/bin/gcc"
export CXX="/proj/local/gcc-13.3.0/bin/g++"
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:${LD_LIBRARY_PATH:-}"


echo 'creating python virtual env'
/proj/local/Python-3.11.12/bin/python3 -m venv blob-creation
source blob-creation/bin/activate
echo 'installing mlir and python dependencies'
pip install --upgrade pip
pip install torch==2.7.0
pip install -r ${MLIR_COMPILER_DIR}/python/requirements-common.txt
# Find and install mlir_external_packages wheel (version may vary)
MLIR_WHL=$(ls ${MLIR_COMPILER_DIR}/python/mlir_external_packages-*.whl 2>/dev/null | head -1)
if [ -n "${MLIR_WHL}" ]; then
  pip install ${MLIR_WHL}
else
  echo "WARNING: mlir_external_packages wheel not found in ${MLIR_COMPILER_DIR}/python/"
fi
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend
#First for FPGA

#echo 'creating fpga kernel'
cd fpga-kernel
cmake -B build-fpga \
  -DTOOLBOX_DIR=${TOOLBOX_DIR} \
  -DCOMPILER_INSTALL_DIR=${MLIR_COMPILER_DIR}
./create-all-kernels.sh
#The for Posix Use cases

echo 'creating posix kernel'
cd ../posix-kernel/
./create-all-kernels.sh

#Change directory to top level llama.cpp

cd ../../

#Compile for posix & fpga with build-posix as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for posix'
if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"   -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni" ${ENABLE_COVERAGE_FLAG}
elif [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"   -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni" ${ENABLE_COVERAGE_FLAG}
else
  cmake -B build-posix -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"   -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"  ${ENABLE_COVERAGE_FLAG}
fi


cmake --build build-posix --config Release

# Fix GLIBC compatibility for TSI binaries
echo 'fixing GLIBC compatibility for TSI binaries'

# Fix simple-backend-tsi
mv build-posix/bin/simple-backend-tsi build-posix/bin/simple-backend-tsi-original
cat > build-posix/bin/simple-backend-tsi << 'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
EOL
chmod +x build-posix/bin/simple-backend-tsi

# Fix llama-cli
mv build-posix/bin/llama-cli build-posix/bin/llama-cli-original
cat > build-posix/bin/llama-cli << 'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/llama-cli-original" "$@"
EOL
chmod +x build-posix/bin/llama-cli

#Compile for fpga with build-fpga as a target folder
# Use toolbox's ARM toolchain file instead of hardcoded paths

echo 'building llama.cp, ggml for tsavorite  and other binary for fpga'
ARM_TOOLCHAIN_FILE=${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake
# Common cmake flags for FPGA build
FPGA_CMAKE_FLAGS="-DCMAKE_TOOLCHAIN_FILE=${ARM_TOOLCHAIN_FILE} \
  -DGGML_TSAVORITE=ON \
  -DGGML_TSAVORITE_TARGET=fpga \
  -DLLAMA_CURL=OFF"

if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-fpga ${FPGA_CMAKE_FLAGS} \
    -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TSAVORITE" \
    -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TSAVORITE" \
    ${ENABLE_COVERAGE_FLAG}
elif [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-fpga ${FPGA_CMAKE_FLAGS} \
    -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TSAVORITE" \
    -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TSAVORITE" \
    ${ENABLE_COVERAGE_FLAG}
else
  cmake -B build-fpga ${FPGA_CMAKE_FLAGS} \
    -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TSAVORITE" \
    -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TSAVORITE" \
    ${ENABLE_COVERAGE_FLAG}
fi

cmake --build build-fpga --config Release


echo 'creating tar bundle for fpga'
TSI_GGML_VERSION=0.2.3
TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
TSI_BLOB_INSTALL_DIR=$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga

if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR} ]; then
   echo "${TSI_GGML_BUNDLE_INSTALL_DIR} exist"
else
   echo "creating ${TSI_GGML_BUNDLE_INSTALL_DIR}"
   mkdir ${TSI_GGML_BUNDLE_INSTALL_DIR}
fi
if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh ]; then
   rm -fr ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
fi

cat > ./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh << EOL
#!/bin/bash
# Set up library paths for GCC 13.3.0 compatibility
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\$(pwd)

tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm"  "swiglu" "add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16")

for kernel in "\${tsi_kernels[@]}"; do
    mkdir -p ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel
    cp blobs ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel/ -r
done
EOL
chmod +x ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
cp ${GGML_TSI_INSTALL_DIR}/fpga/blobs ${TSI_GGML_BUNDLE_INSTALL_DIR}/ -r
cp build-fpga/bin/llama-cli ${TSI_GGML_BUNDLE_INSTALL_DIR}/
cp build-fpga/bin/libggml*.so ${TSI_GGML_BUNDLE_INSTALL_DIR}/
cp build-fpga/bin/libllama*.so ${TSI_GGML_BUNDLE_INSTALL_DIR}/
cp build-fpga/bin/simple-backend-tsi ${TSI_GGML_BUNDLE_INSTALL_DIR}/

tar -cvzf ${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz ${TSI_GGML_BUNDLE_INSTALL_DIR}/*

if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
    cp ${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz ${TSI_GGML_RELEASE_DIR}/

    LATEST_TZ="${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz"
    LATEST_FULL_PATH="${TSI_GGML_RELEASE_DIR}/$(basename "$LATEST_TZ")"

    # Remove old symlinks if they exist
    rm -f "$TSI_GGML_RELEASE_DIR/tsi-ggml-aws-latest.tz"
    rm -f "$TSI_GGML_RELEASE_DIR/tsi-ggml-latest.tz"
    # Create new symbolic links
    ln -s /aws"$LATEST_FULL_PATH" "$TSI_GGML_RELEASE_DIR/tsi-ggml-aws-latest.tz"
    ln -s "$LATEST_FULL_PATH" "$TSI_GGML_RELEASE_DIR/tsi-ggml-latest.tz"

    echo "Symlinks updated to point to $(basename "$LATEST_FULL_PATH")"
fi
