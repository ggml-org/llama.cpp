#!/usr/bin/env bash
#
# Build and verify llama.cpp with Qualcomm's ggml-hexagon backend on Linux
# for Android phones equipped with Qualcomm Snapdragon mobile SoC
# (8 Elite is recommended).
#
#
set -e

######## configuration variables ########

PROJECT_ROOT_PATH=$(pwd)

VERBOSE=ON

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#path of built artifacts
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android

#path of toolchain, for purpose of share same toolchain in multiple instance of ggml-hexagon
TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${TOOLCHAIN_PATH}/${ANDROID_NDK_NAME}

# --- Define NDK paths based on the absolute SDK path ---
NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android"

#OpenCL Headers can be found at:
#https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_PATH=${TOOLCHAIN_PATH}/OpenCL_SDK
OPENCL_HEADERS_PATH=${OPENCL_SDK_PATH}/OpenCL-Headers

#fully Qualcomm Hexagon SDK can be found at https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools.
#fully Hexagon SDK must be obtained with Qualcomm Developer Account and follow PKLA&ECA.
#Community Hexagon SDK from GitHub will be used in this script
HEXAGON_SDK_VERSION=6.6.0.0
HEXAGON_TOOLS_VERSION=19.0.07
HEXAGON_SDK_PATH=${TOOLCHAIN_PATH}/Hexagon_SDK/${HEXAGON_SDK_VERSION}
HEXAGON_TOOLS_PATH=${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}

#supported htp arch version:
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka 8 Gen4)
#v81 --- Snapdragon 8 Elite Gen5(aka 8 Gen5)
HTP_ARCH_VERSIONS="v73 v75 v79 v81"

######## models and prompt ########

#default LLM model for inference testing
GGUF_MODEL_NAME=/sdcard/gemma-4-E2B-it-Q4_0.gguf

# Model aliases for quick testing of multiple models
# Usage: ./scripts/build-run-ggmlhexagon-android.sh run_llamacli <alias>
#   qwen3       -> Qwen3.5-2B-Q4_0.gguf
#   gemma4-e2b  -> gemma-4-E2B-it-Q4_0.gguf (2.9 GiB)
#   gemma4-e4b  -> gemma-4-E4B_q4_0-it.gguf (4.9 GiB)
#   qwen1       -> qwen1_5-1_8b-chat-q4_0.gguf
#   llama3      -> Llama-3.2-1B-Instruct-Q4_0.gguf
#   (default)   -> gemma-4-E2B-it-Q4_0.gguf
function resolve_model_name()
{
    case "$1" in
        qwen3)      echo "/sdcard/Qwen3.5-2B-Q4_0.gguf" ;;
        gemma4-e2b) echo "/sdcard/gemma-4-E2B-it-Q4_0.gguf" ;;
        gemma4-e4b) echo "/sdcard/gemma-4-E4B_q4_0-it.gguf" ;;
        qwen1)      echo "/sdcard/qwen1_5-1_8b-chat-q4_0.gguf" ;;
        llama3)     echo "/sdcard/Llama-3.2-1B-Instruct-Q4_0.gguf" ;;
        *)          echo "" ; return 1 ;;
    esac
}

PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"

#command-line parameters used during inference testing
running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --load-mode none -fa on --jinja -st"

######## functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "HEXAGON_SDK_PATH:     ${HEXAGON_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_command_in_host()
{
    set +e
    cmd=$1
    if command -v ${cmd} > /dev/null 2>&1; then
        printf "${cmd} is available on host machine\n"
        echo ""
    else
        printf "${cmd} not exist on host machine, pls install command line utility ${cmd} firstly and accordingly\n"
        exit 1
    fi
    set -e
}


function check_commands_in_host()
{
    check_command_in_host wget
    check_command_in_host xzcat
    check_command_in_host adb
    check_command_in_host md5sum
    check_command_in_host ninja
}


function check_android_phone()
{
    local device_raw
    device_raw=$(adb devices 2>/dev/null | grep -v "List of devices" | awk 'NF>0')

    if [[ -z "$device_raw" ]]; then
        adb kill-server >/dev/null 2>&1
        sleep 0.1
        adb start-server >/dev/null 2>&1
        device_raw=$(adb devices 2>/dev/null | grep -v "List of devices" | awk 'NF>0')
        if [[ -z "$device_raw" ]]; then
            echo "No Android device detected."
            echo "Please check if phone is connected properly.Exiting"
            exit 1
        fi
    fi

    if echo "$device_raw" | grep -q "no permissions"; then
        echo "Device detected but has NO PERMISSIONS."
        echo "Please check if phone is connected properly.Exiting"
        exit 1
    fi

    if echo "$device_raw" | grep -q "unauthorized"; then
        echo "Device detected but UNAUTHORIZED."
        echo "Please check if phone is connected properly.Exiting"
        exit 1
    fi

    if echo "$device_raw" | grep -q "offline"; then
        echo "Device is OFFLINE."
        echo "Please check if phone is connected properly.Exiting"
        exit 1
    fi

    if echo "$device_raw" | awk '{print $2}' | grep -qx "device"; then
        local sn=$(echo "$device_raw" | awk '{print $1}')
        echo "Android device connected successfully: $sn"
        return 0
    fi

    echo "Unknown device error."
    echo "Please check if phone is connected properly.Exiting"
    exit 1
}


#download community Hexagon SDK from GitHub
function check_and_download_hexagon_sdk()
{
    local sdk_tarball="hexagon-sdk-v${HEXAGON_SDK_VERSION}-amd64-lnx.tar.xz"
    local sdk_url="https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v${HEXAGON_SDK_VERSION}/${sdk_tarball}"

    if [ -f ${HEXAGON_TOOLS_PATH}/NOTICE.txt ]; then
        printf "Hexagon SDK already exists: ${HEXAGON_SDK_PATH}\n\n"
        return 0
    fi

    echo "Hexagon SDK not found, downloading ${sdk_tarball}..."
    mkdir -p ${TOOLCHAIN_PATH}/Hexagon_SDK/

    if [ -f ${TOOLCHAIN_PATH}/Hexagon_SDK/${sdk_tarball} ]; then
        echo "${sdk_tarball} already exists"
    else
        wget --no-config --quiet --show-progress -O ${TOOLCHAIN_PATH}/Hexagon_SDK/${sdk_tarball} ${sdk_url}
        if [ $? -ne 0 ]; then
            printf "failed to download ${sdk_tarball}\n"
            exit 1
        fi
    fi

    echo "decompressing ${sdk_tarball}..."
    xzcat ${TOOLCHAIN_PATH}/Hexagon_SDK/${sdk_tarball} | tar -C ${TOOLCHAIN_PATH}/Hexagon_SDK/ -xf -
    if [ $? -ne 0 ]; then
        printf "failed to decompress ${sdk_tarball}\n"
        exit 1
    fi
    printf "Hexagon SDK installed successfully\n\n"

    if [ ! -d ${HEXAGON_SDK_PATH} ]; then
        echo "HEXAGON_SDK_PATH ${HEXAGON_SDK_PATH} not exist, pls install it accordingly..."
        exit 1
    fi
}


function check_and_download_opencl_sdk()
{
    is_opencl_sdk_exist=1

    if [ ! -d ${OPENCL_SDK_PATH} ]; then
        echo -e "OPENCL_SDK_PATH ${OPENCL_SDK_PATH} not exist, download it from ${OPENCL_SDK_URL}...\n"
        is_opencl_sdk_exist=0
    fi
    if [ ! -f ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so ]; then
        echo -e "${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so not exist...\n"
        is_opencl_sdk_exist=0
    fi

    if [ ${is_opencl_sdk_exist} -eq 0 ]; then
        mkdir -p ${OPENCL_SDK_PATH}
        cd ${OPENCL_SDK_PATH}

        if [ ! -d OpenCL-Headers ]; then
            echo "Cloning OpenCL-Headers..."
            git clone https://github.com/KhronosGroup/OpenCL-Headers
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-Headers to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${TOOLCHAIN_PATH}/OpenCL_SDK/OpenCL-Headers
        printf "Copying OpenCL Headers to Android NDK sysroot include: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}"
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        /bin/cp -r -fv CL ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}

        cd ${TOOLCHAIN_PATH}/OpenCL_SDK
        if [ ! -d OpenCL-ICD-Loader ]; then
            echo "Cloning OpenCL-ICD-Loader..."
            git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-ICD-Loader to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${TOOLCHAIN_PATH}/OpenCL_SDK/OpenCL-ICD-Loader
        mkdir -p build
        cd build
        cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DANDROID_STL=c++_shared -DOPENCL_ICD_LOADER_HEADERS_DIR=${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        echo "Building OpenCL-ICD-Loader with ninjia..."
        ninja
        if [ $? -ne 0 ]; then
            printf "failed to build OpenCL-ICD-Loader\n"
            exit 1
        fi
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}
        /bin/cp -fv libOpenCL.so ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}

        echo "OpenCL components setup complete"
        echo "OpenCL Headers are in: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}/CL"
        echo "libOpenCL.so is in:    ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so"

        cd ${PROJECT_ROOT_PATH}
    else
        printf "OpenCL SDK already exist:    ${OPENCL_SDK_PATH} \n\n"
    fi
}


function check_and_download_ndk()
{
    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f ${TOOLCHAIN_PATH}/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${TOOLCHAIN_PATH}/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${TOOLCHAIN_PATH}
        unzip ${ANDROID_NDK_FULLNAME}

        if [ $? -ne 0 ]; then
            printf "failed to download Android NDK to %s \n" "${ANDROID_NDK}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}

        printf "Android NDK saved to ${ANDROID_NDK} \n\n"
    else
        printf "Android NDK already exist:         ${ANDROID_NDK} \n\n"
    fi
}


#build Qualcomm's ggml-hexagon backend
function build_arm64()
{
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache

    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    prepare_ggmlhtp
    update_ggml_libs
    commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
    fi
    show_pwd

    /bin/rm -f CMakeUserPresets.json

    echo "run following command to see the performance of qualcomm's ggml-hexagon backend"
    echo "./scripts/build-run-ggmlhexagon-android.sh run_llamacli"
    echo "./scripts/build-run-ggmlhexagon-android.sh run_llamabench"
}


#push Qualcomm's ggml-hexagon DSP skels to device
function prepare_ggmlhtp()
{
    for ver in ${HTP_ARCH_VERSIONS}; do
        case "$ver" in
            v73 | v75 | v79 | v81)
                echo "adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${ver}.so ${REMOTE_PATH}/libggml-htp-${ver}.so"
                adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${ver}.so ${REMOTE_PATH}/libggml-htp-${ver}.so
            ;;
            *)
                show_usage
                exit 1
            ;;
        esac
    done
}


function remove_temp_dir()
{
    if [ -d ${LOCAL_BUILD_DIR} ]; then
        echo "remove ${LOCAL_BUILD_DIR} directory"
        rm -rf ${LOCAL_BUILD_DIR}
    fi
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    adb shell ls /sdcard/${model_name}
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist on Android phone\n"
    else
        printf "the prebuild LLM model ${model_name} not exist on Android phone\n"
        printf "downloading from ${model_url}\n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        adb push ${PROJECT_ROOT_PATH}/models/${model_name} /sdcard/
    fi

    set -e
}


function check_prebuilt_models()
{
    set +e

    #1.12 GiB
    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf  https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf

    #1.2 GiB
    check_and_download_model Qwen3.5-2B-Q4_0.gguf         https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf

    #2.9 GiB
    check_and_download_model gemma-4-E2B-it-Q4_0.gguf     https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf

    #4.9 GiB
    check_and_download_model gemma-4-E4B_q4_0-it.gguf     https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf

    #737 MiB
    check_and_download_model Llama-3.2-1B-Instruct-Q4_0.gguf     https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf

    set -e
}


# ==============================================================================
# Return codes:
#    0 = NO changes
#    1 = FILE CHANGED
# ==============================================================================
function is_so_file_changed() {
    set +e
    local so_file="$1"
    local md5_file="${so_file}.md5"

    if [ ! -f "$so_file" ]; then
        echo "ERROR: File not found: $so_file"
        return 1
    fi

    local current_md5
    current_md5=$(md5sum "$so_file" | awk '{print $1}')

    if [ ! -f "$md5_file" ]; then
        echo "$current_md5" > "$md5_file"
        echo "Initialized MD5 for $so_file"
        return 1
    fi

    local last_md5
    last_md5=$(cat "$md5_file")

    if [ "$current_md5" = "$last_md5" ]; then
        return 0
    else
        echo "$current_md5" > "$md5_file"
        return 1
    fi
}

# Persist the current MD5 of a .so to its .md5 cache file. Call this ONLY after
# the file has been successfully pushed to the device.
function commit_so_file_md5() {
    local so_file="$1"
    local md5_file="${so_file}.md5"
    if [ -f "$so_file" ]; then
        md5sum "$so_file" | awk '{print $1}' > "$md5_file"
    fi
}


# Push AP-side libs (libggml-*.so, libllama-*.so) from bin/ to device.
function update_ggml_libs()
{
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-base.so                 ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so                  ${REMOTE_PATH}/
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${REMOTE_PATH}/
    fi
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so          ${REMOTE_PATH}/
    fi
    adb push ${LOCAL_BUILD_DIR}/bin/libggml.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-common.so              ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so     ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so          ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama.so                     ${REMOTE_PATH}/
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        echo "invalid param"
        return
    fi
    program=$1

    check_prebuilt_models

    # incremental push: skip lib push if MD5 unchanged and libs exist on device
    local need_update=0
    is_so_file_changed ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
    if [ $? -eq 0 ]; then
        printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so not changed\n"
    else
        printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so has changed or first check\n"
        need_update=1
    fi
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        is_so_file_changed ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        if [ $? -ne 0 ]; then
            printf "${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so has changed or first check\n"
            need_update=1
        else
            printf "${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so not changed\n"
        fi
    fi
    if [ ${need_update} -eq 0 ]; then
        #host-side MD5 matches cache, but verify libs actually exist on device
        if ! adb shell ls ${REMOTE_PATH}/libggml-cpu.so >/dev/null 2>&1; then
            printf "device-side libggml-cpu.so missing, force update ggml libs\n\n"
            need_update=1
        fi
    fi
    if [ ${need_update} -eq 0 ]; then
        printf "reuse cached/uploaded ggml runtime libs on device side\n\n"
    else
        update_ggml_libs
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
            commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        fi
    fi

    #push DSP skels
    prepare_ggmlhtp

    adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb shell chmod +x ${REMOTE_PATH}/${program}

    # configuration for cDSP's logcat
    # FARF bits: 0x01=LOW 0x02=MEDIUM 0x04=HIGH 0x08=ERROR 0x10=FATAL
    # 0x1c = HIGH+ERROR+FATAL (drop LOW+MEDIUM verbose spam; keep diag)
    adb shell "rm -f /data/local/tmp/${program}.farf"
    adb shell "touch /data/local/tmp/${program}.farf"
    adb shell "echo 0x1c > /data/local/tmp/${program}.farf"
}


function run_llamacli()
{
    local model_name=""
    local model_path=""

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-completion

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${GGUF_MODEL_NAME}\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${GGUF_MODEL_NAME}"
}


function run_llamacli_all()
{
    local models=("gemma4-e2b" "qwen3" "qwen1" "llama3" "gemma4-e4b")

    local total=${#models[@]}
    local count=0

    echo "=============================================="
    echo "  Batch inference test: ${#models[@]} models = ${total} tests"
    echo "=============================================="

    for model in "${models[@]}"; do
        count=$(( count + 1 ))
        echo ""
        echo "--- [${count}/${total}] model=${model} ---"
        run_llamacli "${model}"
    done

    echo ""
    echo "=============================================="
    echo "  Batch inference test complete: ${total} tests done"
    echo "=============================================="
}


function run_test-ops()
{
    prog_name=test-backend-ops
    prepare_run_on_phone ${prog_name}

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test"
}


function run_test-op()
{
    prog_name=test-backend-ops
    prog_param="-o ${opname}"
    prepare_run_on_phone ${prog_name}

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test ${prog_param}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test ${prog_param}"
}


function run_perf-op()
{
    prog_name=test-backend-ops
    prepare_run_on_phone ${prog_name}

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} perf -o ${opname}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} perf -o ${opname}"
}


function show_usage()
{
    echo -e "\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 update_ggml_libs                         (incremental: push AP-side libs from bin/ to device only)"

    echo "  $0 build                                    (build Qualcomm's ggml-hexagon backend)"
    echo "  $0 clean"

    echo "  $0 run_testops"
    echo "  $0 run_testop  ADD/MUL_MAT/FLASH_ATTN_EXT   (verify accuracy    of ADD/MUL_MAT/FLASH_ATTN_EXT)"
    echo "  $0 run_perfop  ADD/MUL_MAT/FLASH_ATTN_EXT   (verify performance of ADD/MUL_MAT/FLASH_ATTN_EXT)"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"

    echo "  $0 run_llamacli_all                         (batch test 5 models = 5 tests)"
    echo "    Log capture example:"
    echo "      $0 run_llamacli_all 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_llamacli   [model_alias]"
    echo "  Model aliases for run_llamacli:"
    echo "    qwen3         -> Qwen3.5-2B-Q4_0.gguf"
    echo "    gemma4-e2b    -> gemma-4-E2B-it-Q4_0.gguf (2.9 GiB)"
    echo "    gemma4-e4b    -> gemma-4-E4B_q4_0-it.gguf (4.9 GiB)"
    echo "    qwen1         -> qwen1_5-1_8b-chat-q4_0.gguf"
    echo "    (default)     -> gemma-4-E2B-it-Q4_0.gguf"
    echo "  Examples:"
    echo "    $0 run_llamacli qwen3                     (test qwen3)"
    echo "    $0 run_llamacli gemma4-e2b                (test gemma4-e2b)"
    echo "    $0 run_llamacli gemma4-e4b                (test gemma4-e4b)"
    echo -e "\n"
}


######## entry point ########

show_pwd

check_commands_in_host
check_android_phone
check_and_download_ndk
check_and_download_opencl_sdk
check_and_download_hexagon_sdk
check_prebuilt_models

if [ $# == 0 ]; then
    show_usage
    exit 0
elif [ $# == 1 ]; then
    case "$1" in
        -h | help)
            show_usage
            exit 0
            ;;
        update_ggml_libs)
            update_ggml_libs
            exit 0
            ;;
        build)
            build_ggml_hexagon
            exit 0
            ;;
        clean)
            remove_temp_dir
            exit 0
            ;;
        run_testops)
            run_test-ops
            exit 0
            ;;
        run_llamacli)
            run_llamacli
            exit 0
            ;;
        run_llamabench)
            run_llamabench
            exit 0
            ;;
        run_llamacli_all)
            run_llamacli_all
            exit 0
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
elif [ $# == 2 ]; then
    case "$1" in
        run_testop)
            opname=$2
            run_test-op
            exit 0
            ;;
        run_perfop)
            opname=$2
            run_perf-op
            exit 0
            ;;
        run_llamacli)
            if [ -z "$(resolve_model_name "$2")" ]; then
                echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3, gemma4-e2b, gemma4-e4b, qwen1, llama3"
                show_usage
                exit 1
            fi
            run_llamacli "$2"
            exit 0
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
else
    show_usage
    exit 1
fi
