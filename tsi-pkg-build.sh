

# Steps to merge the branch to latest
#git clone git@github.com:tsisw/llama.cpp.git
#git remote add upstream https://github.com/ggml-org/llama.cpp.git
#git fetch upstream
#git checkout master
#git merge upstream/master

set -e

export MLIR_SDK_VERSION=/proj/rel/sw/sdk-r.0.2.2
export TOOLBOX_DIR=${MLIR_SDK_VERSION}/toolbox/build/install
pip install --upgrade pip
pip install torch==2.7.0
pip install -r ${MLIR_SDK_VERSION}/compiler/python/requirements-common.txt
pip install ${MLIR_SDK_VERSION}/compiler/python/mlir_external_packages-1.5.0-py3-none-any.whl
pip install onnxruntime-training
#Ensure prerequisites are met as follows
echo 'updating submodule'
git submodule update --recursive --init
cd ggml-tsi-kernel/
#module load gcc/13.3.0
echo 'creating python virtual env'
/proj/local/Python-3.10.12/bin/python3 -m venv blob-creation
source blob-creation/bin/activate
echo 'installing mlir and python dependencies'
pip install --upgrade pip
pip install torch==2.7.0
pip install -r ${MLIR_SDK_VERSION}/compiler/python/requirements-common.txt
pip install ${MLIR_SDK_VERSION}/compiler/python/mlir_external_packages-1.5.0-py3-none-any.whl
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend
#First for FPGA

#echo 'creating fpga kernel'
cd fpga-kernel
cmake -B build-fpga
./create-all-kernels.sh
#The for Posix Use cases

echo 'creating posix kernel'
cd ../posix-kernel/
./create-all-kernels.sh

#Change directory to top level llama.cpp

cd ../../

#Compile for posix & fpga with build-posix as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for posix'
if [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"   -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"
elif [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"   -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"
else
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"   -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"
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

echo 'building llama.cp, ggml for tsavorite  and other binary for fpga'
export CC="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc"
export CXX="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
export CMAKE_FIND_ROOT_PATH=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/
export TSAVORITE_SYSROOT_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include/
if [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TSAVORITE"   -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include  -DCURL_LIBRARY=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/lib/libcurl.so
elif [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TSAVORITE"   -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include  -DCURL_LIBRARY=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/lib/libcurl.so
else
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TSAVORITE"   -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/include  -DCURL_LIBRARY=/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/lib/libcurl.so
fi

cmake --build build-fpga --config Release


echo 'creating tar bundle for fpga'
TSI_GGML_VERSION=0.2.2
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

if [ "$(echo "$1" | tr '[:upper:]' '[:lower:]')" = "release" ];
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
