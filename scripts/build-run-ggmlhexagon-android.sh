#!/usr/bin/env bash
#
set -e

######## part-1: public macros & vars ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

VERBOSE=OFF
VERBOSE=ON

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#path of built artifacts
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android

TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r29
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${TOOLCHAIN_PATH}/${ANDROID_NDK_NAME}

# --- Define NDK paths based on the absolute SDK path ---
NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android"

#OpenCL Headers can be found at:
#https://https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_PATH=${TOOLCHAIN_PATH}/OpenCL_SDK
OPENCL_HEADERS_PATH=${OPENCL_SDK_PATH}/OpenCL-Headers

#fully Qualcomm Hexagon SDK can be found at https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools.
#fully Hexagon SDK must be obtained with Qualcomm Developer Account and follow PKLA&ECA.
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

# default HTP_ARCH
HTP_ARCH_VERSION=${HTP_ARCH_VERSIONS%% *}

######## part-2: prompt and LLM models ########

#2.9 GiB, default model, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-4-E2B-it-Q4_0.gguf

# Model aliases for quick testing of multiple models
# Usage: ./scripts/build-run-ggmlhexagon-android.sh run_llamacli <alias>
#   qwen3-2b            -> Qwen3.5-2B-Q4_0.gguf
#   qwen3-9b            -> Qwen3.5-9B-Q4_0.gguf
#   gemma4-e2b          -> gemma-4-E2B-it-Q4_0.gguf
#   gemma4-e4b          -> gemma-4-E4B_q4_0-it.gguf
#   qwen1               -> qwen1_5-1_8b-chat-q4_0.gguf
#   llama3              -> Llama-3.2-1B-Instruct-Q4_0.gguf
#   nanbeige-3b         -> Nanbeige_Nanbeige4.2-3B-Q4_0.gguf
#   minicpm5-1b         -> minicpm5-1b-q4_0.gguf
#   (default)           -> gemma-4-E2B-it-Q4_0.gguf
#   nanbeige-3b-q80     -> Nanbeige_Nanbeige4.2-3B-Q8_0.gguf
#   minicpm5-1b-q80     -> MiniCPM5-1B-Q8_0.gguf
function resolve_model_name()
{
    case "$1" in
        qwen3-2b)           echo "/sdcard/Qwen3.5-2B-Q4_0.gguf" ;;
        qwen3-9b)           echo "/sdcard/Qwen3.5-9B-Q4_0.gguf" ;;
        gemma4-e2b)         echo "/sdcard/gemma-4-E2B-it-Q4_0.gguf" ;;
        gemma4-e4b)         echo "/sdcard/gemma-4-E4B_q4_0-it.gguf" ;;
        qwen1)              echo "/sdcard/qwen1_5-1_8b-chat-q4_0.gguf" ;;
        llama3)             echo "/sdcard/Llama-3.2-1B-Instruct-Q4_0.gguf" ;;
        nanbeige-3b)        echo "/sdcard/Nanbeige_Nanbeige4.2-3B-Q4_0.gguf";;
        nanbeige-3b-q80)    echo "/sdcard/Nanbeige_Nanbeige4.2-3B-Q8_0.gguf";;
        minicpm5-1b)        echo "/sdcard/minicpm5-1b-q4_0.gguf";;
        minicpm5-1b-q80)    echo "/sdcard/MiniCPM5-1B-Q8_0.gguf";;
        *)                  echo "" ; return 1 ;;
    esac
}

PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"

running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --load-mode none -fa on --jinja -st"

######## part-3: utilities and functions ########

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
        printf "${cmd} is available on host machine\n" > /dev/null
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


#TODO:refine this function
function check_and_download_hexagon_sdk()
{
    is_hexagon_llvm_exist=1
    if [ ! -f ${TOOLCHAIN_PATH}/Hexagon_SDK/${HEXAGON_SDK_VERSION}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}/NOTICE.txt ]; then
        echo -e "${TEXT_RED}minimal-hexagon-sdk not exist...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    fi

    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        mkdir -p ${TOOLCHAIN_PATH}/Hexagon_SDK/
        if [ -f ${TOOLCHAIN_PATH}/Hexagon_SDK/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz ]; then
            echo -e "hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz already exist\n"
        else
            echo -e "begin downloading hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz \n"
            wget --no-config --quiet --show-progress -O ${TOOLCHAIN_PATH}/Hexagon_SDK/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.6.0.0/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz
            if [ $? -ne 0 ]; then
                printf "failed to download hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz\n"
                exit 1
            fi
        fi

        echo -e "begin decompressing hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz \n"
        xzcat ${TOOLCHAIN_PATH}/Hexagon_SDK/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz | tar -C ${TOOLCHAIN_PATH}/Hexagon_SDK/ -xf -
        if [ $? -ne 0 ]; then
            printf "failed to decompress hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz\n"
            exit 1
        fi
        printf "install minimal-hexagon-sdk successfully\n\n"
    fi

    if [ ! -d ${HEXAGON_SDK_PATH} ]; then
        echo -e "HEXAGON_SDK_PATH ${HEXAGON_SDK_PATH} not exist, pls install it accordingly...\n"
        exit 0
    else
        printf "Qualcomm Hexagon SDK already exist:${HEXAGON_SDK_PATH} \n\n" > /dev/null
    fi
}


#not mandatory for ggml-hexagon
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
        printf "OpenCL SDK already exist:    ${OPENCL_SDK_PATH} \n\n" > /dev/null
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
        printf "Android NDK already exist:         ${ANDROID_NDK} \n\n" > /dev/null
    fi
}


#build the mempool/FastRPC-invoke ggml-hexagon backend (default)
function build_arm64
{
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache

    # clear dspqueue skels left by a prior build_dspqueue, else detect_build_type()
    # misreports hexagon-dspqueue
    rm -f ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-*.so

    #ARMv8.7a+i8mm CPU tuning flags, moved here from CMakeLists.txt to keep it aligned with upstream master
    local arm_cpu_flags="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"

    local extra_flags="${arm_cpu_flags}"

    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DGGML_HEXAGON_USE_MEMPOOL=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} -DCMAKE_C_FLAGS="${extra_flags}" -DCMAKE_CXX_FLAGS="${extra_flags}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} -DGGML_USE_HEXAGON=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF --preset arm64-android-snapdragon-release
    cmake --build ${LOCAL_BUILD_DIR}
    #upload the new libggml-htp.so (mempool variant) on device side
    prepare_fastrpc_skels
    #push AP-side libs too: libggml-hexagon.so embeds the regenerated FastRPC stub
    #which MUST stay in sync with the DSP skel signature, otherwise FastRPC args
    #get misaligned (root cause of the 8gen3 garbled-output regression).
    update_ggml_libs
    commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        # backup for AB testing: mempool(FastRPC) AP-side libs + DSP skels
        # libggml-hexagon.so leaks different libc++ symbols between the mempool/dspqueue
        # variants, so all transitively-linked libs (libggml, libllama, libllama-common,
        # *-impl) must be swapped together to avoid symbol lookup failures at runtime
        mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${PROJECT_ROOT_PATH}/out/ab-test/libggml-hexagon-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so     ${PROJECT_ROOT_PATH}/out/ab-test/libllama-server-impl-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libmtmd.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libmtmd-fastrpc.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-fastrpc.so
        for skel in ${LOCAL_BUILD_DIR}/bin/libggml-htp-v*.so; do
            [ -f "$skel" ] || continue
            /bin/cp -fv "$skel" ${PROJECT_ROOT_PATH}/out/ab-test/$(basename "$skel" .so)-fastrpc.so
        done
        # libggml-opencl.so is optional (GGML_OPENCL=OFF by default); back up if present
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
            /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ${PROJECT_ROOT_PATH}/out/ab-test/libggml-opencl-fastrpc.so
        fi
    fi
    show_pwd

    /bin/rm -f CMakeUserPresets.json
}


#build the dspqueue ggml-hexagon backend for performance comparison
function build_arm64_dspqueue
{
    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_dspqueue

    rm -f ${LOCAL_BUILD_DIR}/.ab_test_runtime

    # clear mempool skels left by a prior fastrpc build, else detect_build_type()
    # misreports hexagon-fastrpc
    rm -f ${LOCAL_BUILD_DIR}/bin/libggml-htp-*.so

    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    #upload the new libggml-htp.so (dspqueue variant) on device side
    prepare_dspqueue_skels
    #push AP-side libs too: dspqueue build also needs to sync runtime libs
    update_ggml_libs
    # backup for AB testing: dspqueue AP-side libs + DSP skels
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${PROJECT_ROOT_PATH}/out/ab-test/libggml-hexagon-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so     ${PROJECT_ROOT_PATH}/out/ab-test/libllama-server-impl-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libmtmd.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libmtmd-dspqueue.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-dspqueue.so
        for skel in ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-v*.so; do
            [ -f "$skel" ] || continue
            /bin/cp -fv "$skel" ${PROJECT_ROOT_PATH}/out/ab-test/$(basename "$skel" .so)-dspqueue.so
        done
        # libggml-opencl.so is optional (GGML_OPENCL=OFF by default); back up if present
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
            /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ${PROJECT_ROOT_PATH}/out/ab-test/libggml-opencl-dspqueue.so
        fi
    fi
    show_pwd

    /bin/rm -f CMakeUserPresets.json

    echo "run following command to see the performance of the dspqueue ggml-hexagon backend"
    echo "./scripts/build-run-ggmlhexagon-android.sh run_llamacli"
    echo "./scripts/build-run-ggmlhexagon-android.sh run_llamabench"
}


#build Android CPU-only reference (no ggml-hexagon) for correctness check and troubleshooting trick issues
function build_armcpu()
{
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_cpu

    #ARMv8.7a+i8mm CPU tuning flags, moved here from CMakeLists.txt to keep it aligned with upstream master
    local arm_cpu_flags="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=OFF -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DCMAKE_C_FLAGS="${arm_cpu_flags}" -DCMAKE_CXX_FLAGS="${arm_cpu_flags}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cd ${LOCAL_BUILD_DIR}
    # use cmake --build so it matches whatever generator the cache was configured with
    cmake --build ${LOCAL_BUILD_DIR} -j${HOST_CPU_COUNTS}
    #remove stale hexagon artifacts from previous hexagon builds to ensure CPU-only runtime
    rm -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
    rm -f ${LOCAL_BUILD_DIR}/bin/libggml-htp-*.so
    # legacy name from before the skel unification; clean up leftovers
    rm -f ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-*.so
    # also clear dspqueue skels left in the source dir by a prior build_dspqueue, else
    # detect_build_type() falls back to them and misreports hexagon-dspqueue
    rm -f ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-*.so
    # backup CPU-only AP libs for AB switching (symmetric with build/build_dspqueue)
    # note: do NOT use "-cpu" suffix - libggml-cpu.so is the CPU backend impl, a
    # different lib from libggml.so (the core). use "-cpuonly" to avoid collision.
    mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so     ${PROJECT_ROOT_PATH}/out/ab-test/libllama-server-impl-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libmtmd.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libmtmd-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-cpuonly.so

    # fix issue CANNOT LINK EXECUTABLE "/data/local/tmp/llama-completion": library "libggml-hexagon.so" not found: needed by main executable
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/llama-bench                 ${PROJECT_ROOT_PATH}/out/ab-test/llama-bench-cpuonly
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/llama-completion            ${PROJECT_ROOT_PATH}/out/ab-test/llama-completion-cpuonly
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/llama-server                ${PROJECT_ROOT_PATH}/out/ab-test/llama-server-cpuonly

    show_pwd
}


function remove_temp_dir()
{
    if [ -d ${LOCAL_BUILD_DIR} ]; then
        echo "remove ${LOCAL_BUILD_DIR} directory"
        rm -rf ${LOCAL_BUILD_DIR}
    fi
}


function update_cfg()
{
    if [ -f ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ]; then
        adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
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
    rm -f ${LOCAL_BUILD_DIR}/.ab_test_runtime
    build_arm64
}


function build_ggml_hexagon_dspqueue()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_dspqueue
}


#push dspqueue-variant DSP skels (libggml-htp-vXX.so) to the device
function prepare_dspqueue_skels()
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


#push mempool/FastRPC-variant DSP skels (libggml-htp-vXX.so) to the device
function prepare_fastrpc_skels()
{
    if [ -f ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ]; then
        adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    fi
    for ver in ${HTP_ARCH_VERSIONS}; do
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-htp-${ver}.so ]; then
            echo "adb push ${LOCAL_BUILD_DIR}/bin/libggml-htp-${ver}.so ${REMOTE_PATH}/libggml-htp-${ver}.so"
            adb push ${LOCAL_BUILD_DIR}/bin/libggml-htp-${ver}.so ${REMOTE_PATH}/libggml-htp-${ver}.so
        fi
    done
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    adb shell ls /sdcard/${model_name} >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist on Android phone\n" > /dev/null
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

    #5.1 GiB
    check_and_download_model Qwen3.5-9B-Q4_0.gguf         https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf

    #2.9 GiB
    check_and_download_model gemma-4-E2B-it-Q4_0.gguf     https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf

    #4.9 GiB
    check_and_download_model gemma-4-E4B_q4_0-it.gguf     https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf

    #737 MiB
    check_and_download_model Llama-3.2-1B-Instruct-Q4_0.gguf     https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf

    #2.4 GiB
    check_and_download_model Nanbeige_Nanbeige4.2-3B-Q4_0.gguf   https://huggingface.com/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige_Nanbeige4.2-3B-Q4_0.gguf

    #4.2 GiB
    #check_and_download_model Nanbeige_Nanbeige4.2-3B-Q8_0.gguf   https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige_Nanbeige4.2-3B-Q8_0.gguf

    #1.1 GiB
    #check_and_download_model MiniCPM5-1B-Q8_0.gguf               https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q8_0.gguf

    #635 MiB
    check_and_download_model minicpm5-1b-q4_0.gguf               https://huggingface.co/Elmermoreno/MiniCPM5-1B-Q4_0-GGUF/resolve/main/minicpm5-1b-q4_0.gguf
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

    # check if .so exists
    if [ ! -f "$so_file" ]; then
        echo "ERROR: File not found: $so_file"
        return 1
    fi

    # get current MD5
    local current_md5
    current_md5=$(md5sum "$so_file" | awk '{print $1}')

    # FIRST RUN: no MD5 file -> save it, return CHANGED
    if [ ! -f "$md5_file" ]; then
        echo "$current_md5" > "$md5_file"
        echo "Initialized MD5 for $so_file"
        return 1
    fi

    # read previous MD5
    local last_md5
    last_md5=$(cat "$md5_file")

    # compare
    if [ "$current_md5" = "$last_md5" ]; then
        # NO CHANGE
        return 0
    else
        # CHANGED -> update MD5
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
# AP-only: does NOT push DSP skels (libggml-htp-*.so).
# Does NOT switch backend - DSP skels already on device stay as-is.
# Use update_fastrpc_libs / update_dspqueue_libs for a full backend switch (AP + DSP).
# Gotcha: bin/ reflects the last build (fastrpc or dspqueue); pushing fastrpc AP libs
# while dspqueue DSP skels are still on device leaves an AP/DSP mismatch.
function update_ggml_libs()
{
    #adb push ${LOCAL_BUILD_DIR}/bin/*.so ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-base.so                 ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so                  ${REMOTE_PATH}/
    #libggml-hexagon.so only exists in hexagon builds, not in CPU-only builds
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${REMOTE_PATH}/
    fi
    #libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so          ${REMOTE_PATH}/
    fi
    adb push ${LOCAL_BUILD_DIR}/bin/libggml.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-common.so              ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so     ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so         ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libmtmd.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so          ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama.so                     ${REMOTE_PATH}/
}


#push mempool/FastRPC runtime .so from out/ab-test/ to device, renaming *-fastrpc.so to canonical names
function update_fastrpc_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-hexagon-fastrpc.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-hexagon-fastrpc.so not found."
        echo "Run '$0 build' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-hexagon-fastrpc.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-fastrpc.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-fastrpc.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-fastrpc.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-fastrpc.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-fastrpc.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-fastrpc.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-fastrpc.so      ${REMOTE_PATH}/libllama-bench-impl.so
    # skels push as canonical libggml-htp-vXX.so, overwriting any dspqueue skels
    for skel in ${ab_test_dir}/libggml-htp-v*-fastrpc.so; do
        [ -f "$skel" ] || continue
        adb push "$skel" ${REMOTE_PATH}/$(basename "$skel" -fastrpc.so).so
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-fastrpc.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-fastrpc.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    # legacy name from before the skel unification; clean up leftovers
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"
    echo "fastrpc" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "mempool/FastRPC runtime .so pushed to device."
}


#push dspqueue runtime .so from out/ab-test/ to device, renaming *-dspqueue.so to canonical names
function update_dspqueue_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-hexagon-dspqueue.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-hexagon-dspqueue.so not found."
        echo "Run '$0 build_dspqueue' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-hexagon-dspqueue.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-dspqueue.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-dspqueue.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-dspqueue.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-dspqueue.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-dspqueue.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-dspqueue.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-dspqueue.so      ${REMOTE_PATH}/libllama-bench-impl.so
    # skels push as canonical libggml-htp-vXX.so, overwriting any fastrpc skels
    for skel in ${ab_test_dir}/libggml-htp-v*-dspqueue.so; do
        [ -f "$skel" ] || continue
        adb push "$skel" ${REMOTE_PATH}/$(basename "$skel" -dspqueue.so).so
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-dspqueue.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-dspqueue.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    # legacy name from before the skel unification; clean up leftovers
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"
    echo "dspqueue" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "dspqueue runtime .so pushed to device."
}


#push CPU-only runtime .so from out/ab-test/ to device, renaming *-cpuonly.so to canonical names
function update_cpu_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-cpuonly.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-cpuonly.so not found."
        echo "Run '$0 build_armcpu' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-cpuonly.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-cpuonly.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-cpuonly.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-cpuonly.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-cpuonly.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-cpuonly.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-cpuonly.so      ${REMOTE_PATH}/libllama-bench-impl.so

    # fix issue CANNOT LINK EXECUTABLE "/data/local/tmp/llama-completion": library "libggml-hexagon.so" not found: needed by main executable
    # because there is a different linker procedure for Android‑CPU‑only builds
    /bin/cp -f ${ab_test_dir}/llama-bench-cpuonly               ${LOCAL_BUILD_DIR}/bin/llama-bench
    /bin/cp -f ${ab_test_dir}/llama-completion-cpuonly          ${LOCAL_BUILD_DIR}/bin/llama-completion
    /bin/cp -f ${ab_test_dir}/llama-server-cpuonly              ${LOCAL_BUILD_DIR}/bin/llama-server

    adb push ${ab_test_dir}/llama-bench-cpuonly                 ${REMOTE_PATH}/llama-bench
    adb push ${ab_test_dir}/llama-completion-cpuonly            ${REMOTE_PATH}/llama-completion
    adb push ${ab_test_dir}/llama-server-cpuonly                ${REMOTE_PATH}/llama-server

    # libggml-base.so / libggml-cpu.so are shared across builds, device-side kept as-is
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
    adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    adb shell "rm -f ${REMOTE_PATH}/libggml-hexagon.so"
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"
    adb shell "rm -f ${REMOTE_PATH}/libggml-htp-*.so"
    echo "cpu" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "CPU-only runtime .so pushed to device."
}


#detect build type from build output: hexagon-fastrpc, hexagon-dspqueue, or cpu-only
#fastrpc build puts skels in bin/; dspqueue build puts them in ggml/src/ggml-hexagon/
function detect_build_type()
{
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        if ls ${LOCAL_BUILD_DIR}/bin/libggml-htp-*.so 1>/dev/null 2>&1; then
            echo "hexagon-fastrpc"
        else
            echo "hexagon-dspqueue"
        fi
    elif ls ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-*.so 1>/dev/null 2>&1; then
        echo "hexagon-dspqueue"
    else
        echo "cpu-only"
    fi
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    update_cfg

    check_prebuilt_models

    # AB test mode: if update_fastrpc_libs/update_dspqueue_libs/update_cpu_libs set the marker,
    # skip lib/skel push - user has manually set up the runtime.
    local ab_test_marker="${LOCAL_BUILD_DIR}/.ab_test_runtime"
    if [ -f "${ab_test_marker}" ]; then
        local ab_runtime=$(cat "${ab_test_marker}")
        printf "AB test mode active: runtime='%s' (skipping lib/skel push)\n\n" "${ab_runtime}"
        adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/
        adb shell ls -l ${REMOTE_PATH}/libggml-*.so
        adb shell chmod +x ${REMOTE_PATH}/${program}
        adb shell "rm -f /data/local/tmp/${program}.farf"
        adb shell "touch /data/local/tmp/${program}.farf"
        adb shell "echo 0x1c > /data/local/tmp/${program}.farf"
        return
    fi

    local current_build_type
    current_build_type=$(detect_build_type)

    local last_build_type_file="${LOCAL_BUILD_DIR}/.last_deployed_build_type"
    local last_build_type=""
    if [ -f "${last_build_type_file}" ]; then
        last_build_type=$(cat "${last_build_type_file}")
    fi

    if [ "${current_build_type}" != "${last_build_type}" ]; then
        printf "build type changed: '%s' -> '%s', force update ggml libs\n\n" "${last_build_type}" "${current_build_type}"
        update_ggml_libs
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
            commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        fi
        echo "${current_build_type}" > "${last_build_type_file}"
    else
        local need_update=0
        is_so_file_changed ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
        if [ $? -eq 0 ]; then
            printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so not changed\n"
            #reuse cached/uploaded ggml runtime libs on device side to avoid time-consuming task on host side
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
            #(user may have wiped /data/local/tmp manually, leaving host-side MD5 cache stale)
            if ! adb shell ls ${REMOTE_PATH}/libggml-cpu.so >/dev/null 2>&1; then
                printf "device-side libggml-cpu.so missing (maybe /data/local/tmp was wiped), force update ggml libs\n\n"
                need_update=1
            fi
        fi
        if [ ${need_update} -eq 0 ]; then
            printf "reuse cached/uploaded ggml runtime libs on device side\n\n"
        else
            #upload ggml runtime libs to Android phone
            update_ggml_libs
            commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
            if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
                commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
            fi
        fi
    fi

    #deploy/cleanup backend-specific libs per build type
    case "${current_build_type}" in
        hexagon-fastrpc)
            # skels push as libggml-htp-vXX.so and overwrite any dspqueue skels
            prepare_fastrpc_skels
            # legacy name from before the skel unification; clean up leftovers
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            ;;
        hexagon-dspqueue)
            # skels push as libggml-htp-vXX.so and overwrite any fastrpc skels
            prepare_dspqueue_skels
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            ;;
        cpu-only)
            adb shell rm -f ${REMOTE_PATH}/libggml-hexagon.so
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            adb shell rm -f ${REMOTE_PATH}/libggml-htp-*.so
            adb shell rm -f ${REMOTE_PATH}/libggml-opencl.so
            adb shell rm -f ${REMOTE_PATH}/libggml-vulkan.so
            ;;
    esac

    adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb shell chmod +x ${REMOTE_PATH}/${program}

    # configuration for cDSP's logcat
    # FARF bits: 0x01=LOW 0x02=MEDIUM 0x04=HIGH 0x08=ERROR 0x10=FATAL
    # 0x1f = ALL (default; chatty: HAP_compute_res logs ~6x per RPC, 26K+/test)
    # 0x1c = HIGH+ERROR+FATAL (drop LOW+MEDIUM verbose spam; keep diag)
    # 0x18 = ERROR+FATAL  (only errors)
    # 0x00 = silent
    adb shell "rm -f /data/local/tmp/${program}.farf"
    adb shell "touch /data/local/tmp/${program}.farf"
    adb shell "echo 0x1c > /data/local/tmp/${program}.farf"
    #observe cDSP's log
    #adb logcat  | grep "CDSP0"
}


function run_llamaversion()
{
    prepare_run_on_phone llama-cli

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion --version"
}


function run_llamacli()
{
    local model_name=""
    local model_path=""

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-completion

    #GGML_HEXAGON_OPPOLL is only effective for the dspqueue variant, doesn't apply to the mempool/FastRPC variant
    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""

}


function run_llamacli_all()
{
    local models=("qwen1" "minicpm5-1b" "llama3" "qwen3-2b" "gemma4-e2b" "nanbeige-3b" "gemma4-e4b" "qwen3-9b")

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


function run_llamabench()
{
    local model_name=""
    local model_path=""

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-bench

    #GGML_HEXAGON_OPPOLL is only effective for the dspqueue variant, doesn't apply to the mempool/FastRPC variant
    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${model_path}\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${model_path}"
}


function run_abtest()
{
    # mempool/FastRPC vs dspqueue performance comparison test.
    # Requires out/ab-test/ populated (run 'build' then 'build_dspqueue' first).
    # Usage: run_abtest [rounds] [model_alias]
    #   rounds:      default 3
    #   model_alias: default gemma4-e2b
    #
    # Example:
    #   $0 run_abtest
    #   $0 run_abtest 5
    #   $0 run_abtest 3 qwen3-2b
    #   $0 run_abtest 2>&1 | tee log_abtest_$(date +%Y%m%d-%H%M%S).txt

    local rounds=3
    local model_path="${GGUF_MODEL_NAME}"
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test

    if [ $# -ge 1 ]; then
        rounds=$1
    fi
    if [ $# -ge 2 ]; then
        local model_alias="$2"
        model_path=$(resolve_model_name "$model_alias")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_alias'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    fi

    # sanity check: verify out/ab-test/ has all required .so from both builds
    local missing=""
    local fastrpc_libs="libggml-hexagon-fastrpc.so libggml-fastrpc.so libllama-fastrpc.so libllama-common-fastrpc.so libllama-completion-impl-fastrpc.so libllama-server-impl-fastrpc.so libmtmd-fastrpc.so libllama-bench-impl-fastrpc.so"
    local dspqueue_libs="libggml-hexagon-dspqueue.so libggml-dspqueue.so libllama-dspqueue.so libllama-common-dspqueue.so libllama-completion-impl-dspqueue.so libllama-server-impl-dspqueue.so libmtmd-dspqueue.so libllama-bench-impl-dspqueue.so"
    for f in ${fastrpc_libs}; do
        [ ! -f ${ab_test_dir}/${f} ] && missing="${missing} ${f}"
    done
    for f in ${dspqueue_libs}; do
        [ ! -f ${ab_test_dir}/${f} ] && missing="${missing} ${f}"
    done
    if [ -n "${missing}" ]; then
        echo "ERROR: AB test backups incomplete, missing:${missing}"
        echo ""
        echo "Run these two commands first to populate ${ab_test_dir}:"
        echo "  $0 build            # builds mempool/FastRPC ggml-hexagon, backs up *-fastrpc.so"
        echo "  $0 build_dspqueue   # builds dspqueue ggml-hexagon, backs up *-dspqueue.so"
        exit 1
    fi
    # check DSP skels exist for at least one HTP arch version
    local fastrpc_skels=$(ls ${ab_test_dir}/libggml-htp-v*-fastrpc.so 2>/dev/null | wc -l)
    local dspqueue_skels=$(ls ${ab_test_dir}/libggml-htp-v*-dspqueue.so 2>/dev/null | wc -l)
    if [ ${fastrpc_skels} -eq 0 ] || [ ${dspqueue_skels} -eq 0 ]; then
        echo "ERROR: DSP skels missing in ${ab_test_dir}"
        echo "  fastrpc skels (libggml-htp-v*-fastrpc.so): ${fastrpc_skels} found"
        echo "  dspqueue skels (libggml-htp-v*-dspqueue.so): ${dspqueue_skels} found"
        echo ""
        echo "Run these two commands first:"
        echo "  $0 build            # builds mempool/FastRPC DSP skels"
        echo "  $0 build_dspqueue   # builds dspqueue DSP skels"
        exit 1
    fi

    echo "=============================================="
    echo "  AB test: mempool/FastRPC vs dspqueue, ${rounds} rounds each"
    echo "  model: ${model_path}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    # --- mempool/FastRPC phase ---
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Switching to mempool/FastRPC ==="
    adb push ${ab_test_dir}/libggml-hexagon-fastrpc.so     ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-fastrpc.so             ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-fastrpc.so            ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-fastrpc.so     ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-fastrpc.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-fastrpc.so ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-fastrpc.so             ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-fastrpc.so ${REMOTE_PATH}/libllama-bench-impl.so
    # skels push as canonical libggml-htp-vXX.so, overwriting any dspqueue skels
    for skel in ${ab_test_dir}/libggml-htp-v*-fastrpc.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/$(basename "$skel" -fastrpc.so).so
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-fastrpc.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-fastrpc.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    # legacy name from before the skel unification; clean up leftovers
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"

    echo ""
    echo "========================================"
    echo "  mempool/FastRPC test (${rounds} runs)"
    echo "========================================"
    for i in $(seq 1 ${rounds}); do
        echo ""
        echo "-------- fastrpc run ${i}/${rounds} $(date '+%H:%M:%S') --------"
        adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
        echo "-------- fastrpc run ${i} END --------"
    done

    # --- dspqueue phase ---
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Switching to dspqueue ==="
    adb push ${ab_test_dir}/libggml-hexagon-dspqueue.so     ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-dspqueue.so             ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-dspqueue.so            ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-dspqueue.so     ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-dspqueue.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-dspqueue.so ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-dspqueue.so             ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-dspqueue.so ${REMOTE_PATH}/libllama-bench-impl.so
    # skels push as canonical libggml-htp-vXX.so, overwriting any fastrpc skels
    for skel in ${ab_test_dir}/libggml-htp-v*-dspqueue.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/$(basename "$skel" -dspqueue.so).so
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-dspqueue.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-dspqueue.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    # legacy name from before the skel unification; clean up leftovers
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"

    echo ""
    echo "========================================"
    echo "  dspqueue test (${rounds} runs)"
    echo "========================================"
    for i in $(seq 1 ${rounds}); do
        echo ""
        echo "-------- dspqueue run ${i}/${rounds} $(date '+%H:%M:%S') --------"
        adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
        echo "-------- dspqueue run ${i} END --------"
    done

    echo ""
    echo "=============================================="
    echo "  AB test complete $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    # Restore current build type libs after AB test.
    # AB test leaves dspqueue libs on device (dspqueue is the last phase).
    # Without this, subsequent run_llamabench would skip pushing
    # libggml-hexagon.so (MD5 matches local fastrpc build), leaving device
    # in a mixed state: dspqueue AP lib + fastrpc DSP skels -> error 0x80000406.
    local restore_type=""
    local last_bt_file="${LOCAL_BUILD_DIR}/.last_deployed_build_type"
    if [ -f "${last_bt_file}" ]; then
        restore_type=$(cat "${last_bt_file}")
    fi
    if [ -z "${restore_type}" ]; then
        restore_type="hexagon-fastrpc"
    fi
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Restoring build type: ${restore_type} ==="
    case "${restore_type}" in
        hexagon-fastrpc)
            update_fastrpc_libs
            ;;
        hexagon-dspqueue)
            update_dspqueue_libs
            ;;
        cpu-only)
            update_cpu_libs
            ;;
    esac
}


function run_abtest_all()
{
    # Run AB test across all 8 supported models.
    # Usage: run_abtest_all [rounds]
    #   rounds: default 3 (per model); qwen3-9b is hard-capped to 1 (slow + high power, phone gets hot)
    #
    # Example:
    #   $0 run_abtest_all
    #   $0 run_abtest_all 3
    #   $0 run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt

    local rounds=3
    if [ $# -ge 1 ]; then
        rounds=$1
    fi

    local all_models="gemma4-e2b gemma4-e4b qwen3-2b nanbeige-3b qwen1 minicpm5-1b llama3 qwen3-9b"
    local total=8
    local idx=0

    for model_alias in ${all_models}; do
        idx=$((idx + 1))
        # qwen3-9b inference is slow + high power + phone gets hot; cap to 1 round (vs ${rounds} for other models)
        local model_rounds=${rounds}
        if [ "${model_alias}" = "qwen3-9b" ]; then
            model_rounds=1
            echo "  NOTE: qwen3-9b -> rounds=1 (slow inference, high power, phone gets hot)"
        fi
        echo ""
        echo "##############################################"
        echo "  AB test ${idx}/${total}: ${model_alias} (rounds=${model_rounds})"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "##############################################"
        run_abtest ${model_rounds} ${model_alias}
    done

    echo ""
    echo "##############################################"
    echo "  All AB tests complete $(date '+%Y-%m-%d %H:%M:%S')"
    echo "##############################################"
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

    echo "  $0 build                    (build the mempool/FastRPC-invoke ggml-hexagon backend for performance comparision)"
    echo "  $0 build_dspqueue           (build the dspqueue ggml-hexagon backend for performance comparison, Qualcomm's official dspqueu-based ggml-hexagon)"
    echo "  $0 build_armcpu             (build Android CPU-only reference for correctness check and troulbeshooting trick issues)"
    echo "  $0 clean"

    echo "  $0 update_fastrpc_libs      (push mempool/FastRPC runtime .so from out/ab-test/ to device, for build)"
    echo "  $0 update_dspqueue_libs     (push dspqueue runtime .so from out/ab-test/ to device, for build_dspqueue)"
    echo "  $0 update_cpu_libs          (push CPU-only runtime .so from out/ab-test/ to device, for build_armcpu)"
    echo "  $0 update_ggml_libs         (incremental: push AP-side libs from bin/ to device only; keep DSP skels as-is)"

    echo "  $0 run_llamaversion         (display llama-cpp version information, e.g. version: 0.2.0-dev (build 11120, commit 9f03708a9), built with Clang 21.0.0 for Android aarch64)"
    echo "  $0 run_testops"
    echo "  $0 run_testop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify performance of ADD/MUL_MAT)"
    echo -e "\n"

    echo "  $0 run_abtest_all [rounds]"
    echo "    Batch AB test across all 8 models (qwen1 minicpm5-1b llama3 qwen3-2b gemma4-e2b nanbeige-3b gemma4-e4b qwen3-9b)."
    echo "    rounds: default 3"
    echo "    Log capture example:"
    echo "      $0 run_abtest_all 2>&1 | tee log_abtest_all_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_llamacli     [model_alias]"
    echo "  $0 run_llamabench   [model_alias]"
    echo "  Model aliases for run_llamacli:"
    echo "    qwen3-2b      -> Qwen3.5-2B-Q4_0.gguf"
    echo "    qwen3-9b      -> Qwen3.5-9B-Q4_0.gguf"
    echo "    gemma4-e2b    -> gemma-4-E2B-it-Q4_0.gguf"
    echo "    gemma4-e4b    -> gemma-4-E4B_q4_0-it.gguf"
    echo "    qwen1         -> qwen1_5-1_8b-chat-q4_0.gguf"
    echo "    llama3        -> Llama-3.2-1B-Instruct-Q4_0.gguf"
    echo "    nanbeige-3b   -> Nanbeige_Nanbeige4.2-3B-Q4_0.gguf"
    echo "    minicpm5-1b   -> minicpm5-1b-q4_0.gguf"
    echo "    (default)     -> gemma-4-E2B-it-Q4_0.gguf"
    echo "  Examples:"
    echo "    $0 run_llamacli/run_llamabench              # run gemma4-e2b inference test on an Qualcomm mobile SoC-based Android phone"
    echo "    $0 run_llamacli/run_llamabench qwen3-2b     # test qwen3-2b"
    echo "    $0 run_llamacli/run_llamabench gemma4-e2b   # test gemma4-e2b"
    echo "    $0 run_llamacli/run_llamabench gemma4-e4b   # test gemma4-e4b"
}


######## part-4: entry point  ########

show_pwd

check_commands_in_host
check_android_phone
check_and_download_ndk
check_and_download_opencl_sdk
check_and_download_hexagon_sdk
check_prebuilt_models

if [ $# == 0 ]; then
    show_usage
    exit 1
elif [ $# == 1 ]; then
    if [ "$1" == "-h" ]; then
        show_usage
        exit 1
    elif [ "$1" == "help" ]; then
        show_usage
        exit 1
    elif [ "$1" == "update_ggml_libs" ]; then
        update_ggml_libs
        exit 1
    elif [ "$1" == "update_fastrpc_libs" ]; then
        update_fastrpc_libs
        exit 0
    elif [ "$1" == "update_dspqueue_libs" ]; then
        update_dspqueue_libs
        exit 0
    elif [ "$1" == "update_cpu_libs" ]; then
        update_cpu_libs
        exit 0
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_dspqueue" ]; then
        build_ggml_hexagon_dspqueue
        exit 0
    elif [ "$1" == "build_armcpu" ]; then
        build_armcpu
        exit 0
    elif [ "$1" == "clean" ]; then
        remove_temp_dir
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamaversion" ]; then
        run_llamaversion
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        run_llamabench
        exit 0
    elif [ "$1" == "run_llamacli_all" ]; then
        run_llamacli_all
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest
        exit 0
    elif [ "$1" == "run_abtest_all" ]; then
        run_abtest_all
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    if [ "$1" == "run_testop" ]; then
        opname=$2
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=$2
        run_perf-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen1 minicpm5-1b llama3 qwen3-2b gemma4-e2b nanbeige-3b gemma4-e4b qwen3-9b"
            show_usage
            exit 1
        fi
        run_llamacli "$2"
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen1 minicpm5-1b llama3 qwen3-2b gemma4-e2b nanbeige-3b gemma4-e4b qwen3-9b"
            show_usage
            exit 1
        fi
        run_llamabench "$2"
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest "$2"
        exit 0
    elif [ "$1" == "run_abtest_all" ]; then
        run_abtest_all "$2"
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 3 ]; then
    if [ "$1" == "run_perfop" ]; then
        opname=$2
        run_perf-op
        exit 0
    elif [ "$1" == "run_testop" ]; then
        opname=$2
        run_test-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            show_usage
            exit 1
        fi
        run_llamacli "$2"
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest "$2" "$3"
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 4 ]; then
    show_usage
    exit 1
else
    show_usage
    exit 1
fi
