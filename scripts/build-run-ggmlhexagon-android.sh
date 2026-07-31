#!/usr/bin/env bash
#
# this script will setup local dev envs automatically and docker is not needed for purpose of simplify workflow.
#
# this script is AI Agent friendly and verified with Trae AI Agent.
#
# 1. build&verify llama.cpp + JZ's ggml-hexagon backend(libggmldsp-skel.so) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 2. build&verify llama.cpp + Qualcomm's ggml-hexagon backend(libggml-htp.so) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 3. performance comparison of Qualcomm's ggml-hexagon and JZ's ggml-hexagon on Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended & verified)
#
#
set -e

######## part-1: public macros & vars ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

VERBOSE=ON

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#path of built artifacts
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android

#path of toolchain, for purpose of share same toolchain in multiple instance of ggml-hexagon
TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts
#TOOLCHAIN_PATH=/home/zhouwg/develop/ggml-hexagon/prebuilts

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
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

#Qualcomm Snapdragon 8Elite based Android phone is strongly recommended because:
#1. sometimes the same dsp codes can got the best performance on Snapdragon 8Elite based phone.
#2. DSP clock rate on 8Gen3 is slower than DSP clock rate on 8Elite.
#3. 8Elite support for LP-DDR5x memory, up to 5300 MHz; 8Gen3 support for LP-DDR5x memory, up to 4800 MHz.

#HTP_ARCH_VERSIONS="v79"                         # 8 Elite
#HTP_ARCH_VERSIONS="v79 v75"                     # 8 Elite + 8Gen3
HTP_ARCH_VERSIONS="v73 v75 v79 v81"              # all

# default HTP_ARCH
HTP_ARCH_VERSION=${HTP_ARCH_VERSIONS%% *}

######## part-2: prompt and LLM models ########

#the following LLM models has verified(works fine) with the JZ's ggml-hexagon backend on a Snapdragon 8Elite based Android phone
#1.12 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#1.2 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/Qwen3.5-2B-Q4_0.gguf

#2.9 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-4-E2B-it-Q4_0.gguf

# Model aliases for quick testing of multiple models
# Usage: ./scripts/build-run-ggmlhexagon-android.sh run_llamacli <alias>
#   qwen3       -> Qwen3.5-2B-Q4_0.gguf
#   gemma4      -> gemma-4-E2B-it-Q4_0.gguf
#   qwen1       -> qwen1_5-1_8b-chat-q4_0.gguf
#   (default)   -> gemma-4-E2B-it-Q4_0.gguf
function resolve_model_name()
{
    case "$1" in
        qwen3)      echo "/sdcard/Qwen3.5-2B-Q4_0.gguf" ;;
        gemma4)     echo "/sdcard/gemma-4-E2B-it-Q4_0.gguf" ;;
        qwen1)      echo "/sdcard/qwen1_5-1_8b-chat-q4_0.gguf" ;;
        *)          echo "" ; return 1 ;;
    esac
}

PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"

#unified command-line parameters used during inference testing for fair performance comparison of PP and TG across Qualcomm's ggml-hexagon and JZ's ggml-hexagon
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
        printf "Qualcomm Hexagon SDK already exist:${HEXAGON_SDK_PATH} \n\n"
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


function build_idl()
{
    echo "build idl"
    if [ -f ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic ]; then
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggml_dsp.idl
    fi
}


#build extra DSP skels (all HTP_ARCH_VERSIONS except the first/default)
#$1 = "debug" for debug build, anything else for release build
function build_extra_dsp_skels()
{
    local dsp_debug_flag
    if [ "$1" == "debug" ]; then
        dsp_debug_flag="-DDEBUG -Wall"
    else
        dsp_debug_flag="-DNDEBUG -Wall"
    fi

    # extras = HTP_ARCH_VERSIONS minus the first element (default)
    for extra_ver in ${HTP_ARCH_VERSIONS#* }; do
        printf "\n========== build extra DSP skel: libggmldsp-skel-${extra_ver}.so ==========\n"
        build_idl
        make -C ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ clean
        make -C ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ HTP_ARCH_VERSION=${extra_ver} HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} HEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH} DEBUG_FLAG="${dsp_debug_flag}"
        /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/libggmldsp-skel.so ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${extra_ver}.so
    done
}


#build JZ's ggml-hexagon backend for performance comparison
function build_arm64
{
    build_idl

    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache

    #ARMv8.7a+i8mm CPU tuning flags, moved here from CMakeLists.txt to keep it aligned with upstream master
    local arm_cpu_flags="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DGGML_HEXAGON_JZ=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DCMAKE_C_FLAGS="${arm_cpu_flags}" -DCMAKE_CXX_FLAGS="${arm_cpu_flags}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} -DGGML_USE_HEXAGON=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
    cd ${LOCAL_BUILD_DIR}
    make -j${HOST_CPU_COUNTS}
    #cmake POST_BUILD now builds all 4 DSP skels (v73/v75/v79/v81) in one pass; no script-side extras needed
    #build_extra_dsp_skels
    #upload the new libggmldsp-skel.so on device side
    prepare_ggmldsp
    #push AP-side libs too: libggml-hexagon.so embeds the regenerated FastRPC stub
    #which MUST stay in sync with the DSP skel signature, otherwise FastRPC args
    #get misaligned (root cause of the 8gen3 garbled-output regression).
    update_ggml_libs
    commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        # backup for AB testing: JZ's AP-side libs + DSP skels
        # libggml-hexagon.so leaks different libc++ symbols between JZ/QCOM, so all
        # transitively-linked libs (libggml, libllama, libllama-common, *-impl) must
        # be swapped together to avoid symbol lookup failures at runtime
        mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${PROJECT_ROOT_PATH}/out/ab-test/libggml-hexagon-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-jz.so
        for skel in ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-v*.so; do
            [ -f "$skel" ] || continue
            /bin/cp -fv "$skel" ${PROJECT_ROOT_PATH}/out/ab-test/
        done
    fi
    show_pwd

    cd -
}


#build Qualcomm's ggml-hexagon backend for performance comparison
function build_arm64_qcom
{
    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_qcom

    rm -f ${LOCAL_BUILD_DIR}/.ab_test_runtime

    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    #upload the new libggml-htps.so on device side
    prepare_ggmlhtp
    #push AP-side libs too: QCOM build also needs to sync runtime libs
    update_ggml_libs
    # backup for AB testing: QCOM's AP-side libs + DSP skels
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${PROJECT_ROOT_PATH}/out/ab-test/libggml-hexagon-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-qcom.so
        for skel in ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-v*.so; do
            [ -f "$skel" ] || continue
            /bin/cp -fv "$skel" ${PROJECT_ROOT_PATH}/out/ab-test/
        done
    fi
    show_pwd

    /bin/rm -f CMakeUserPresets.json

    echo "run following command to see the performance of qualcomm's official ggml-hexagon backend"
    echo "./scripts/build-run-android.sh run_llamacli"
    echo "./scripts/build-run-android.sh run_llamabench"
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
    adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
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


function build_ggml_hexagon_qcom()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_qcom
}


#for Qualcomm's open-source ggml-hexagon backend in branch self-build-jz
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


#for JZ's open-source ggml-hexagon backend in branch self-build-jz
function prepare_ggmldsp()
{
    adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    for ver in ${HTP_ARCH_VERSIONS}; do
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${ver}.so ]; then
            echo "adb push ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${ver}.so ${REMOTE_PATH}/libggmldsp-skel-${ver}.so"
            adb push ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${ver}.so ${REMOTE_PATH}/libggmldsp-skel-${ver}.so
        fi
    done
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
# AP-only: does NOT push DSP skels (libggmldsp-skel-*.so / libggml-htp-*.so).
# Does NOT switch backend - DSP skels already on device stay as-is.
# Use update_jz_libs / update_qcom_libs for a full backend switch (AP + DSP).
# Gotcha: bin/ reflects the last build (JZ or QCOM); pushing JZ AP libs while
# QCOM DSP skels are still on device leaves an AP/DSP mismatch.
function update_ggml_libs()
{
    #adb push ${LOCAL_BUILD_DIR}/bin/*.so ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-base.so                 ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so                  ${REMOTE_PATH}/
    #libggml-hexagon.so only exists in hexagon builds, not in CPU-only builds
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${REMOTE_PATH}/
    fi
    adb push ${LOCAL_BUILD_DIR}/bin/libggml.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-common.so              ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so     ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so          ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama.so                     ${REMOTE_PATH}/
}


#push JZ runtime .so from out/ab-test/ to device, renaming *-jz.so to canonical names
function update_jz_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-hexagon-jz.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-hexagon-jz.so not found."
        echo "Run '$0 build' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-hexagon-jz.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-jz.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-jz.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-jz.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-jz.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-bench-impl-jz.so      ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggmldsp-skel-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    adb shell "rm -f ${REMOTE_PATH}/libggml-htp-*.so"
    echo "jz" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "JZ runtime .so pushed to device."
}


#push QCOM runtime .so from out/ab-test/ to device, renaming *-qcom.so to canonical names
function update_qcom_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-hexagon-qcom.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-hexagon-qcom.so not found."
        echo "Run '$0 build_qcom' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-hexagon-qcom.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-qcom.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-qcom.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-qcom.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-qcom.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-bench-impl-qcom.so      ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggml-htp-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"
    echo "qcom" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "QCOM runtime .so pushed to device."
}


#detect build type from build output: hexagon-jz, hexagon-qcom, or cpu-only
function detect_build_type()
{
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        if ls ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-*.so 1>/dev/null 2>&1; then
            echo "hexagon-jz"
        else
            echo "hexagon-qcom"
        fi
    elif ls ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-*.so 1>/dev/null 2>&1; then
        echo "hexagon-qcom"
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

    # AB test mode: if update_jz_libs/update_qcom_libs set the marker,
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
        hexagon-jz)
            prepare_ggmldsp
            adb shell rm -f ${REMOTE_PATH}/libggml-htp-*.so
            ;;
        hexagon-qcom)
            prepare_ggmlhtp
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            ;;
        cpu-only)
            adb shell rm -f ${REMOTE_PATH}/libggml-hexagon.so
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            adb shell rm -f ${REMOTE_PATH}/libggml-htp-*.so
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
    #adb logcat  | grep -iE "CDSP0"
}


function run_llamacli()
{
    local model_name=""
    local model_path=""

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3, gemma4, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-completion

    echo "${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -fa 1 --ubatch-size 1024 -p 200,512,800,1024 -m ${GGUF_MODEL_NAME}\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -fa 1 --ubatch-size 1024 -p 200,512,800,1024 -m ${GGUF_MODEL_NAME}"
}


function run_llamacli_all()
{
    #local models=("gemma4" "qwen3" "qwen3-mtp" "qwen1" "llama3")
    local models=("gemma4" "qwen3" "qwen1")

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


function run_abtest()
{
    # JZ vs QCOM performance comparison test.
    # Requires out/ab-test/ populated (run 'build' then 'build_qcom' first).
    # Usage: run_abtest [rounds] [model_alias]
    #   rounds:      default 3
    #   model_alias: default gemma4
    #
    # Example:
    #   $0 run_abtest
    #   $0 run_abtest 5
    #   $0 run_abtest 3 qwen3
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
            echo "ERROR: unknown model alias '$model_alias'. Valid aliases: qwen3, gemma4, qwen1, llama3"
            exit 1
        fi
    fi

    # sanity check: verify out/ab-test/ has all required .so from both builds
    local missing=""
    local jz_libs="libggml-hexagon-jz.so libggml-jz.so libllama-jz.so libllama-common-jz.so libllama-completion-impl-jz.so libllama-bench-impl-jz.so"
    local qcom_libs="libggml-hexagon-qcom.so libggml-qcom.so libllama-qcom.so libllama-common-qcom.so libllama-completion-impl-qcom.so libllama-bench-impl-qcom.so"
    for f in ${jz_libs}; do
        [ ! -f ${ab_test_dir}/${f} ] && missing="${missing} ${f}"
    done
    for f in ${qcom_libs}; do
        [ ! -f ${ab_test_dir}/${f} ] && missing="${missing} ${f}"
    done
    if [ -n "${missing}" ]; then
        echo "ERROR: AB test backups incomplete, missing:${missing}"
        echo ""
        echo "Run these two commands first to populate ${ab_test_dir}:"
        echo "  $0 build        # builds JZ ggml-hexagon, backs up *-jz.so"
        echo "  $0 build_qcom   # builds QCOM ggml-hexagon, backs up *-qcom.so"
        exit 1
    fi
    # check DSP skels exist for at least one HTP arch version
    local jz_skels=$(ls ${ab_test_dir}/libggmldsp-skel-*.so 2>/dev/null | wc -l)
    local qcom_skels=$(ls ${ab_test_dir}/libggml-htp-*.so 2>/dev/null | wc -l)
    if [ ${jz_skels} -eq 0 ] || [ ${qcom_skels} -eq 0 ]; then
        echo "ERROR: DSP skels missing in ${ab_test_dir}"
        echo "  JZ skels (libggmldsp-skel-*.so): ${jz_skels} found"
        echo "  QCOM skels (libggml-htp-*.so): ${qcom_skels} found"
        echo ""
        echo "Run these two commands first:"
        echo "  $0 build        # builds JZ DSP skels"
        echo "  $0 build_qcom   # builds QCOM DSP skels"
        exit 1
    fi

    echo "=============================================="
    echo "  AB test: JZ vs QCOM, ${rounds} rounds each"
    echo "  model: ${model_path}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    # --- JZ phase ---
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Switching to JZ ==="
    adb push ${ab_test_dir}/libggml-hexagon-jz.so           ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-jz.so                   ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-jz.so                  ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-jz.so           ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-jz.so  ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-bench-impl-jz.so       ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggmldsp-skel-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    adb shell "rm -f ${REMOTE_PATH}/libggml-htp-*.so"

    echo ""
    echo "========================================"
    echo "  JZ test (${rounds} runs)"
    echo "========================================"
    for i in $(seq 1 ${rounds}); do
        echo ""
        echo "-------- JZ run ${i}/${rounds} $(date '+%H:%M:%S') --------"
        adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
        echo "-------- JZ run ${i} END --------"
    done

    # --- QCOM phase ---
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Switching to QCOM ==="
    adb push ${ab_test_dir}/libggml-hexagon-qcom.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-qcom.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-qcom.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-qcom.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-qcom.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-bench-impl-qcom.so      ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggml-htp-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"

    echo ""
    echo "========================================"
    echo "  QCOM test (${rounds} runs)"
    echo "========================================"
    for i in $(seq 1 ${rounds}); do
        echo ""
        echo "-------- QCOM run ${i}/${rounds} $(date '+%H:%M:%S') --------"
        adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
        echo "-------- QCOM run ${i} END --------"
    done

    echo ""
    echo "=============================================="
    echo "  AB test complete $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
}


function show_usage()
{
    echo -e "\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 update_ggml_libs  (push AP-side libs only from bin/ to device; no DSP skels, no backend switch)"
    echo "  $0 update_jz_libs   (push JZ runtime .so from out/ab-test/ to device)"
    echo "  $0 update_qcom_libs (push QCOM runtime .so from out/ab-test/ to device)"

    echo "  $0 build        (build JZ's ggml-hexagon backend)"
    echo "  $0 build_qcom   (build Qualcomm's ggml-hexagon backend for performance comparison)"
    echo "  $0 clean"

    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"

    echo "  $0 run_llamacli_all     (batch test 3 models = 3 tests)"
    echo "    Log capture example:"
    echo "      $0 run_llamacli_all 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_abtest  [rounds] [model_alias]"
    echo "    JZ vs QCOM performance comparison (requires 'build' then 'build_qcom' first)."
    echo "    rounds:       default 3"
    echo "    model_alias:  default gemma4"
    echo "    Examples:"
    echo "      $0 run_abtest                      # 3 rounds, gemma4"
    echo "      $0 run_abtest 5 gemma4             # 5 rounds, gemma4"
    echo "      $0 run_abtest 3 qwen3              # 3 rounds, qwen3"
    echo "      $0 run_abtest 3 qwen1              # 3 rounds, qwen1"
    echo "    Log capture example:"
    echo "      $0 run_abtest 2>&1 | tee log_abtest_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_llamacli   [model_alias]"
    echo "  Model aliases for run_llamacli:"
    echo "    qwen3         -> Qwen3.5-2B-Q4_0.gguf"
    echo "    gemma4        -> gemma-4-E2B-it-Q4_0.gguf"
    echo "    qwen1         -> qwen1_5-1_8b-chat-q4_0.gguf"
    echo "    (default)     -> gemma-4-E2B-it-Q4_0.gguf"
    echo "  Examples:"
    echo "    $0 run_llamacli qwen3        # test qwen3"
    echo "    $0 run_llamacli gemma4       # test gemma4"
    echo -e "\n"
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
    elif [ "$1" == "update_jz_libs" ]; then
        update_jz_libs
        exit 0
    elif [ "$1" == "update_qcom_libs" ]; then
        update_qcom_libs
        exit 0
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_qcom" ]; then
        build_ggml_hexagon_qcom
        exit 0
    elif [ "$1" == "clean" ]; then
        remove_temp_dir
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        run_llamacli
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
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    if [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3, gemma4, qwen1, llama3"
            show_usage
            exit 1
        fi
        run_llamacli "$2"
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest "$2"
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 3 ]; then
    if [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3, gemma4, qwen1, llama3"
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
