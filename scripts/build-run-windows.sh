#!/bin/bash
# build llama.cpp or llama.cpp + ggml-qnn for Windows with cygwin on Windows
# build llama.cpp + ggml-qnn for Snapdragon desktop SoC equipped WoA(Windows on ARM) with cygwin on Windows

# items marked TODO has not verified yet

set -e


PWD=`pwd`
PREFIX_PATH=/cygdrive/c
GGUF_MODEL_NAME=${PREFIX_PATH}/qwen1_5-1_8b-chat-q4_0.gguf
PROJECT_HOME_PATH=`pwd`

#QNN SDK could be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_PATH=${PREFIX_PATH}/qairt/2.31.0.250130/

#default is QNN NPU
qnnbackend=2

function dump_vars()
{
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

function build_windows_x86
{
    echo "build_windows_x86-without-qnn"
    cmake -H. -B./out/windows_x86 -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF
    cd out/windows_x86
    make -j16
    show_pwd

    cd -
}

function build_windows_x86_qnn
{
    echo "build_windows_x86-with-qnn"
    cmake -H. -B./out/windows_x86_qnn -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_QNN=ON -DGGML_QNN_SDK_PATH=${QNN_SDK_PATH}
    cd out/windows_x86_qnn
    make -j16
    show_pwd

    cd -
}

#TODO
function build_windows_arm64_qnn
{
    echo "build_windows_arm64 not supported now"
    return 0
    echo "cmake source dir:${PROJECT_HOME_PATH}"
    cmake -H. -B./out/windows_arm64_qnn -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_QNN=ON -DCMAKE_TOOLCHAIN_FILE=${PROJECT_HOME_PATH}/cmake/arm64-windows-llvm.cmake -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_QNN_SDK_PATH=${QNN_SDK_PATH}
    cd out/windows_arm64_qnn
    make -j16
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/windows_x86 ]; then
        echo "remove out/windows_x86 directory in `pwd`"
        rm -rf out/windows_x86
    fi
}


function check_qnn_libs()
{
    echo "do nothing"
}


function update_qnn_libs()
{
    echo "do nothing"
}

function build_x86()
{
    show_pwd
    check_qnn_sdk
    dump_vars
    #some unexpected behaviour on Windows
    #remove_temp_dir
    build_windows_x86
}

function build_x86_qnn()
{
    show_pwd
    check_qnn_sdk
    dump_vars
    #some unexpected behaviour on Windows
    #remove_temp_dir
    build_windows_x86_qnn
}

function build_arm64_qnn()
{
    show_pwd
    check_qnn_sdk
    dump_vars
    #some unexpected behaviour on Windows
    #remove_temp_dir
    build_windows_arm64_qnn
}

function run_llamacli()
{
    check_qnn_libs
    echo "not supported on Windows now"

    #llama-cli -mg ${qnnbackend} -no-cnv -m ${GGUF_MODEL_NAME} -p \"introduce the movie Once Upon a Time in America briefly.\n\"

}


function run_llamabench()
{
    check_qnn_libs
    echo "not supported on Windows now"

    #llama-bench -mg ${qnnbackend} -m ${GGUF_MODEL_NAME}"

}


function run_test-backend-ops()
{
    check_qnn_libs
    echo "not supported on Windows now"

    #test-backend-ops test"

}


function show_usage()
{
    echo "Usage:"
    echo "  $0 build_x86"
    echo "  $0 build_x86_qnn"
    echo "  $0 build_arm64_qnn"
    echo "  $0 run_testop"
    echo "  $0 run_llamacli     0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"
    echo "  $0 run_llamabench   0 (QNN_CPU) / 1 (QNN_GPU) / 2 (QNN_NPU) / 3 (ggml)"
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
    elif [ "$1" == "build_x86" ]; then
        build_x86
        exit 0
    elif [ "$1" == "build_x86_qnn" ]; then
        build_x86_qnn
        exit 0
    elif [ "$1" == "build_arm64_qnn" ]; then
        build_arm64_qnn
        exit 0

    elif [ "$1" == "run_testop" ]; then
        run_test-backend-ops
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
    fi
else
    show_usage
    exit 1
fi
