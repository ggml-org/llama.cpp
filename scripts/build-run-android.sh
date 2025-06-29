#!/bin/bash

# build llama.cpp + ggml-hexagon backend on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC
# this script will setup local dev envs automatically
#
#
set -e

######## part-1 ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

#running path on Android phone
REMOTE_PATH=/data/local/tmp/

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_NAME}

#Qualcomm QNN SDK can be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_VERSION=2.32.0.250228
QNN_SDK_VERSION=2.33.0.250327
QNN_SDK_VERSION=2.34.0.250424
QNN_SDK_VERSION=2.35.0.250530
#fully official QNN SDK, will be downloaded automatically via this script
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/qairt/2.34.0.250424/
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/qairt/2.35.0.250530/

#Qualcomm Hexagon SDK can be found at:
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
#the official Hexagon SDK, must be obtained with Qualcomm Developer Account
HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/6.2.0.1
#customized/tailored Hexagon SDK from the offcial Hexagon SDK for simplify workflow
HEXAGON_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1

#running_params="- ngl 99 -t 8 -n 256 --no-warmup -fa 1 "
running_params=" -ngl 99 -t 8 -n 256 --no-warmup "

######## part-2 ########

PROMPT_STRING="introduce the movie Once Upon a Time in America briefly.\n"

#1.12 GiB, will be downloadded automatically via this script
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#ref: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
#supported htp arch version:
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite

#8Gen2
#HTP_ARCH_VERSION=v73
#HTP_ARCH_VERSION_a=V73

#8Gen3
#HTP_ARCH_VERSION=v75
#HTP_ARCH_VERSION_a=V75

#8Elite
#HTP_ARCH_VERSION=v79
#HTP_ARCH_VERSION_a=V79

#modify the following two lines to adapt to test phone
HTP_ARCH_VERSION=v79
HTP_ARCH_VERSION_a=V79


######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "QNN_SDK_PATH:         ${QNN_SDK_PATH}"
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
    ls /usr/bin/${cmd}
    if [ $? -eq 0 ]; then
        #printf "${cmd} already exist on host machine\n"
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
}


function check_and_download_hexagon_sdk()
{
    mkdir -p ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK

    is_hexagon_llvm_exist=1
    if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/NOTICE.txt ]; then
        echo -e "${TEXT_RED}minimal-hexagon-sdk not exist...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    fi

    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        if [ -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz ]; then
            echo -e "minimal-hexagon-sdk-6.2.0.1.xz already exist\n"
        else
            echo -e "begin downloading minimal-hexagon-sdk-6.2.0.1.xz \n"
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz https://github.com/kantv-ai/toolchain/raw/refs/heads/main/minimal-hexagon-sdk-6.2.0.1.xz
            if [ $? -ne 0 ]; then
                printf "failed to download minimal-hexagon-sdk-6.2.0.1.xz\n"
                exit 1
            fi
        fi

        echo -e "begin decompressing minimal-hexagon-sdk-6.2.0.1.xz \n"
        xzcat ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz | tar -C ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/ -xf -
        if [ $? -ne 0 ]; then
            printf "failed to decompress minimal-hexagon-sdk-6.2.0.1.xz\n"
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


function check_and_download_qnn_sdk()
{
    mkdir -p ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK

    is_qnn_sdk_exist=1

    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist, download it from ${QNN_SDK_URL}...\n"
        is_qnn_sdk_exist=0
    fi

    if [ ${is_qnn_sdk_exist} -eq 0 ]; then
        if [ ! -f ${PROJECT_ROOT_PATH}/prebuild/v${QNN_SDK_VERSION}.zip ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/v${QNN_SDK_VERSION}.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_SDK_VERSION}/v${QNN_SDK_VERSION}.zip
        fi
        if [ $? -ne 0 ]; then
            printf "failed to download Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/
        unzip v${QNN_SDK_VERSION}.zip
        printf "Qualcomm QNN SDK saved to ${QNN_SDK_PATH} \n\n"
        cd ${PROJECT_ROOT_PATH}
    else
        printf "Qualcomm QNN SDK already exist:    ${QNN_SDK_PATH} \n\n"
    fi
}


function check_and_download_ndk()
{
    mkdir -p ${PROJECT_ROOT_PATH}/prebuilts

    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${PROJECT_ROOT_PATH}/prebuilts/
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


function build_arm64
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function build_arm64_debug
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/android ]; then
        echo "remove out/android directory in `pwd`"
        rm -rf out/android
    fi
}


function check_qnn_libs()
{
    set +e

    #reuse the cached qnn libs on Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    adb shell ls ${REMOTE_PATH}/libQnnGpu.so
    adb shell ls ${REMOTE_PATH}/libQnnHtp.so
    if [ $? -eq 0 ]; then
        printf "QNN runtime libs already exist on Android phone\n"
    else
        printf "QNN runtime libs not exist on Android phone\n"
        update_qnn_libs
    fi
    update_qnn_cfg

    set -e
}


function update_qnn_libs()
{
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp${HTP_ARCH_VERSION_a}Stub.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/hexagon-${HTP_ARCH_VERSION}/unsigned/libQnnHtp${HTP_ARCH_VERSION_a}Skel.so     ${REMOTE_PATH}/
}


function update_qnn_cfg()
{
    adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function build_ggml_hexagon_debug()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_debug
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
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        adb push ${PROJECT_ROOT_PATH}/models/${model_name} /sdcard/
    fi

    set -e
}


function check_prebuilt_models()
{
    set +e

    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf

    set -e
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_qnn_libs

    check_prebuilt_models

    if [ -f ./out/android/bin/libggml-cpu.so ]; then
        adb push ./out/android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/android/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg

    adb shell chmod +x ${REMOTE_PATH}/${program}
}

function run_llamacli()
{
    prepare_run_on_phone llama-cli

    echo "${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${GGUF_MODEL_NAME} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${GGUF_MODEL_NAME} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}\""
    echo "${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}"

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}"

}


function run_test-ops()
{
    prepare_run_on_phone test-backend-ops

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test"

}


function run_test-op()
{
    prepare_run_on_phone test-backend-ops

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname "

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname "

}


function show_usage()
{
    echo -e "\n\n\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 build"
    echo "  $0 build_debug (enable debug log for developers on ARM-AP side and cDSP side)"
    echo "  $0 updateqnnlib"
    echo "  $0 run_testops"
    echo "  $0 run_testop     ADD/MUL_MAT"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"

    echo -e "\n\n\n"
}


######## part-4: entry point  ########

show_pwd

check_commands_in_host
check_and_download_ndk
check_and_download_qnn_sdk
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
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_hexagon_debug
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0
    elif [ "$1" == "updateqnnlib" ]; then
        update_qnn_libs
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        qnnbackend=$2
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        qnnbackend=$2
        run_llamabench
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
    else
        show_usage
        exit 1
    fi
else
    show_usage
    exit 1
fi
