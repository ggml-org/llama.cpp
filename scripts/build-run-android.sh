#!/bin/bash

set -e

PWD=`pwd`
ANDROID_PLATFORM=android-34
ANDROID_NDK=${PWD}/android-ndk-r26c
REMOTE_PATH=/data/local/tmp/
GGUF_MODEL_NAME=/sdcard/deepseek-r1-distill-qwen-1.5b-q4_0.gguf
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#QNN SDK could be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_PATH=/opt/qcom/aistack/qairt/2.31.0.250130/

#default is QNN NPU
qnnbackend=2

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "QNN_SDK_PATH:         ${QNN_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_qnn_sdk()
{
    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist, pls check or download it from ${QNN_SDK_URL}...\n"
        exit 1
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

        if [ ! -f android-ndk-r26c-linux.zip ]; then
            wget --no-config --quiet --show-progress -O android-ndk-r26c-linux.zip  https://dl.google.com/android/repository/android-ndk-r26c-linux.zip
        fi

        unzip android-ndk-r26c-linux.zip

        if [ $? -ne 0 ]; then
            printf "failed to download android ndk to %s \n" "${ANDROID_NDK}"
            exit 1
        fi

        printf "android ndk saved to ${ANDROID_NDK} \n\n"
    else
        printf "android ndk already exist:${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Release -DGGML_USE_QNN=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_QNN=ON -DGGML_QNN_SDK_PATH=${QNN_SDK_PATH}
    cd out/android
    make -j16
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
    #reuse the cached qnn libs on Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    if [ $? -eq 0 ]; then
        printf "QNN libs already exist on Android phone\n"
    else
        update_qnn_libs
    fi
}


function update_qnn_libs()
{
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpV75Stub.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so     ${REMOTE_PATH}/
}


function build_ggml_qnn()
{
    show_pwd
    check_and_download_ndk
    check_qnn_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_qnn_libs

    if [ -f ./out/android/bin/libggml-qnn.so ]; then
        adb push ./out/android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/android/bin/${program} ${REMOTE_PATH}/
    adb shell chmod +x ${REMOTE_PATH}/${program}
}

function run_llamacli()
{
    prepare_run_on_phone llama-cli

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli -mg ${qnnbackend} -no-cnv -m ${GGUF_MODEL_NAME} -p \"introduce the movie Once Upon a Time in America briefly.\n\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench -mg ${qnnbackend} -m ${GGUF_MODEL_NAME}"

}


#refer to:https://github.com/ggml-org/llama.cpp/pull/12155
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

    qnnbackendname=qnn-cpu
    case $qnnbackend in
        0)
        qnnbackendname=qnn-cpu
        ;;
        1)
        qnnbackendname=qnn-gpu
        ;;
        2)
        qnnbackendname=qnn-npu
        ;;
        *)
        qnnbackendname=qnn-cpu
        ;;
    esac

    #debug
    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname -b $qnnbackendname "

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname -b $qnnbackendname "

}

function run_ut_add()
{
    prepare_run_on_phone ggml-qnn-ut

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggml-qnn-ut -t GGML_OP_ADD -b $qnnbackend"

}

function run_ut_mulmat()
{
    prepare_run_on_phone ggml-qnn-ut

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggml-qnn-ut -t GGML_OP_MUL_MAT -b $qnnbackend"

}

function run_ut_mul()
{
    prepare_run_on_phone ggml-qnn-ut

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggml-qnn-ut -t GGML_OP_MUL -b $qnnbackend"

}

function print_oplist()
{
oplist="DUP
    ADD
    ADD1
    ACC
    SUB
    MUL
    DIV
    SQR
    SQRT
    LOG
    SIN
    COS
    SUM
    SUM_ROWS
    MEAN
    ARGMAX
    COUNT_EQUAL
    REPEAT
    REPEAT_BACK
    CONCAT
    SILU_BACK
    NORM
    RMS_NORM
    RMS_NORM_BACK
    GROUP_NORM

    MUL_MAT
    MUL_MAT_ID
    OUT_PROD

    SCALE
    SET
    CPY
    CONT
    RESHAPE
    VIEW
    PERMUTE
    TRANSPOSE
    GET_ROWS
    GET_ROWS_BACK
    DIAG
    DIAG_MASK_INF
    DIAG_MASK_ZERO
    SOFT_MAX
    SOFT_MAX_BACK
    ROPE
    ROPE_BACK
    CLAMP
    CONV_TRANSPOSE_1D
    IM2COL
    IM2COL_BACK
    CONV_TRANSPOSE_2D
    POOL_1D
    POOL_2D
    POOL_2D_BACK
    UPSCALE
    PAD
    PAD_REFLECT_1D
    ARANGE
    TIMESTEP_EMBEDDING
    ARGSORT
    LEAKY_RELU

    FLASH_ATTN_EXT
    FLASH_ATTN_BACK
    SSM_CONV
    SSM_SCAN
    WIN_PART
    WIN_UNPART
    GET_REL_POS
    ADD_REL_POS
    RWKV_WKV6
    GATED_LINEAR_ATTN"

echo "opname list: "
echo ${oplist}
}

function show_usage()
{
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 print_oplist"
    echo "  $0 build"
    echo "  $0 updateqnnlib"
    echo "  $0 run_testops"
    echo "  $0 run_testop          [ADD/MUL/MUL_MAT]  [0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU)]"
    echo "  $0 run_ut_add          0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"
    echo "  $0 run_ut_mulmat       0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"
    echo "  $0 run_ut_mul          0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"
    echo "  $0 run_llamacli        0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"
    echo "  $0 run_llamabench      0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"

    echo -e "\n\n\n"
}


show_pwd

check_qnn_sdk

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
    elif [ "$1" == "print_oplist" ]; then
        print_oplist
        exit 1
    elif [ "$1" == "build" ]; then
        build_ggml_qnn
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0

    elif [ "$1" == "updateqnnlib" ]; then
        update_qnn_libs
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    qnnbackend=$2
    if [ ${qnnbackend} -gt 3 ]; then
        show_usage
        exit 1
    fi

    if [ "$1" == "run_llamacli" ]; then
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        run_llamabench
        exit 0
    elif [ "$1" == "run_ut_add" ]; then
        run_ut_add
        exit 0
    elif [ "$1" == "run_ut_mulmat" ]; then
        run_ut_mulmat
        exit 0
    elif [ "$1" == "run_ut_mul" ]; then
        run_ut_mul
        exit 0
    fi
elif [ $# == 3 ]; then
    opname=$2
#TODO: check opname in oplist
#opname can be found via print_oplist:

    qnnbackend=$3
    if [ ${qnnbackend} -gt 3 ]; then
        show_usage
        exit 1
    fi
    run_test-op
    exit 0
else
    show_usage
    exit 1
fi
