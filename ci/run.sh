#!/usr/bin/env bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with SYCL support
# GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with VULKAN support
# GG_BUILD_VULKAN=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with WebGPU support
# GG_BUILD_WEBGPU=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with MUSA support
# GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with KLEIDIAI support
# GG_BUILD_KLEIDIAI=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -f $OUT/*.log
rm -f $OUT/*.exit
rm -f $OUT/*.md

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

CMAKE_EXTRA="-DLLAMA_FATAL_WARNINGS=${LLAMA_FATAL_WARNINGS:-ON} -DLLAMA_CURL=ON -DGGML_SCHED_NO_REALLOC=ON"

if [ ! -z ${GG_BUILD_METAL} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=ON"
fi

if [ ! -z ${GG_BUILD_CUDA} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_CUDA=ON"

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$CUDA_ARCH" && "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
        else
            echo "Warning: Using fallback CUDA architectures"
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89"
        fi
    else
        echo "Error: nvidia-smi not found, cannot build with CUDA"
        exit 1
    fi
fi

if [ ! -z ${GG_BUILD_ROCM} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_HIP=ON"
    if [ -z ${GG_BUILD_AMDGPU_TARGETS} ]; then
        echo "Missing GG_BUILD_AMDGPU_TARGETS, please set it to your GPU architecture (e.g. gfx90a, gfx1100, etc.)"
        exit 1
    fi

    CMAKE_EXTRA="${CMAKE_EXTRA} -DGPU_TARGETS=${GG_BUILD_AMDGPU_TARGETS}"
fi

if [ ! -z ${GG_BUILD_SYCL} ]; then
    if [ -z ${ONEAPI_ROOT} ]; then
        echo "Not detected ONEAPI_ROOT, please install oneAPI base toolkit and enable it by:"
        echo "source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
    # Use only main GPU
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    # Enable sysman for correct memory reporting
    export ZES_ENABLE_SYSMAN=1
    # to circumvent precision issues on CPY operations
    export SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt"
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_SYCL=1 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON"
fi

if [ ! -z ${GG_BUILD_VULKAN} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_VULKAN=1"

    # if on Mac, disable METAL
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    fi

fi

if [ ! -z ${GG_BUILD_WEBGPU} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_WEBGPU=1"
fi

if [ ! -z ${GG_BUILD_MUSA} ]; then
    # Use qy1 by default (MTT S80)
    MUSA_ARCH=${MUSA_ARCH:-21}
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=${MUSA_ARCH}"
fi

if [ ! -z ${GG_BUILD_NO_SVE} ]; then
    # arm 9 and newer enables sve by default, adjust these flags depending on the cpu used
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm"
fi

if [ -n "${GG_BUILD_KLEIDIAI}" ]; then
    echo ">>===== Enabling KleidiAI support"

    CANDIDATES=(
        "armv9-a+dotprod+i8mm+sve2"
        "armv9-a+dotprod+i8mm"
        "armv8.6-a+dotprod+i8mm"
        "armv8.2-a+dotprod"
    )
    CPU=""

    for cpu in "${CANDIDATES[@]}"; do
        if echo 'int main(){}' | ${CXX:-c++} -march="$cpu" -x c++ - -c -o /dev/null >/dev/null 2>&1; then
            CPU="$cpu"
            break
        fi
    done

    if [ -z "$CPU" ]; then
        echo "ERROR: None of the required ARM baselines (armv9/armv8.6/armv8.2 + dotprod) are supported by this compiler."
        exit 1
    fi

    echo ">>===== Using ARM baseline: ${CPU}"

    CMAKE_EXTRA="${CMAKE_EXTRA:+$CMAKE_EXTRA } \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU_KLEIDIAI=ON \
        -DGGML_CPU_AARCH64=ON \
        -DGGML_CPU_ARM_ARCH=${CPU} \
        -DBUILD_SHARED_LIBS=OFF"
fi

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -c -N $url

    cd $cwd
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    # Check cmake, make and ctest are installed
    gg_check_build_requirements

    (time cmake -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                  ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -L main -E "test-opt|test-backend-ops" ) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

# ctest_release

function gg_run_ctest_release {
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    # Check cmake, make and ctest are installed
    gg_check_build_requirements

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z ${GG_BUILD_LOW_PERF} ]; then
        (time ctest --output-on-failure -L main ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest --output-on-failure -L main -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

    set +e
}

function gg_sum_ctest_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# test_scripts

function gg_run_test_scripts {
    cd ${SRC}

    set -e

    (cd ./tools/gguf-split && time bash tests.sh "$SRC/build-ci-release/bin" "$MNT/models") 2>&1 | tee -a $OUT/${ci}-scripts.log
    (cd ./tools/quantize   && time bash tests.sh "$SRC/build-ci-release/bin" "$MNT/models") 2>&1 | tee -a $OUT/${ci}-scripts.log

    set +e
}

function gg_sum_test_scripts {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs test scripts\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-scripts.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

function gg_get_model {
    local gguf_0="$MNT/models/qwen3/0.6B/ggml-model-f16.gguf"
    if [[ -s $gguf_0 ]]; then
        echo -n "$gguf_0"
    else
        echo >&2 "No model found. Can't run gg_run_ctest_with_model."
        exit 1
    fi
}

function gg_run_ctest_with_model_debug {
    cd ${SRC}

    local model; model=$(gg_get_model)
    cd build-ci-debug
    set -e

    (LLAMACPP_TEST_MODELFILE="$model" time ctest --output-on-failure -L model) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
    cd ..
}

function gg_run_ctest_with_model_release {
    cd ${SRC}

    local model; model=$(gg_get_model)
    cd build-ci-release
    set -e

    (LLAMACPP_TEST_MODELFILE="$model" time ctest --output-on-failure -L model) 2>&1 | tee -a $OUT/${ci}-ctest.log

    # test memory leaks
    #if [[ ! -z ${GG_BUILD_METAL} ]]; then
    #    # TODO: this hangs for some reason ...
    #    (time leaks -quiet -atExit -- ./bin/test-thread-safety -m $model --parallel 2 -t 2 -p "hello") 2>&1 | tee -a $OUT/${ci}-leaks.log
    #fi

    set +e
    cd ..
}

function gg_sum_ctest_with_model_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest with model files in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

function gg_sum_ctest_with_model_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest with model files in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# qwen3_0_6b

function gg_run_qwen3_0_6b {
    cd ${SRC}

    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json
    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/tokenizer.json
    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/tokenizer_config.json
   #gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/special_tokens_map.json
    gg_wget models-mnt/qwen3/0.6B/ https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors


    gg_wget models-mnt/wikitext/ https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    unzip -o models-mnt/wikitext/wikitext-2-raw-v1.zip -d models-mnt/wikitext/

    path_models="../models-mnt/qwen3/0.6B"
    path_wiki="../models-mnt/wikitext/wikitext-2-raw"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf  --outtype f16
    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-bf16.gguf --outtype bf16

    model_f16="${path_models}/ggml-model-f16.gguf"
    model_bf16="${path_models}/ggml-model-bf16.gguf"
    model_q8_0="${path_models}/ggml-model-q8_0.gguf"
    model_q4_0="${path_models}/ggml-model-q4_0.gguf"
    model_q4_1="${path_models}/ggml-model-q4_1.gguf"
    model_q5_0="${path_models}/ggml-model-q5_0.gguf"
    model_q5_1="${path_models}/ggml-model-q5_1.gguf"
    model_q2_k="${path_models}/ggml-model-q2_k.gguf"
    model_q3_k="${path_models}/ggml-model-q3_k.gguf"
    model_q4_k="${path_models}/ggml-model-q4_k.gguf"
    model_q5_k="${path_models}/ggml-model-q5_k.gguf"
    model_q6_k="${path_models}/ggml-model-q6_k.gguf"

    wiki_test="${path_wiki}/wiki.test.raw"

    ./bin/llama-quantize ${model_bf16} ${model_q8_0} q8_0 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q4_0} q4_0 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q4_1} q4_1 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q5_0} q5_0 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q5_1} q5_1 $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q2_k} q2_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q3_k} q3_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q4_k} q4_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q5_k} q5_k $(nproc)
    ./bin/llama-quantize ${model_bf16} ${model_q6_k} q6_k $(nproc)

    (time ./bin/llama-completion -no-cnv --model ${model_f16}  -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/llama-completion -no-cnv --model ${model_bf16} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-bf16.log
    (time ./bin/llama-completion -no-cnv --model ${model_q8_0} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/llama-completion -no-cnv --model ${model_q4_0} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/llama-completion -no-cnv --model ${model_q4_1} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/llama-completion -no-cnv --model ${model_q5_0} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/llama-completion -no-cnv --model ${model_q5_1} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/llama-completion -no-cnv --model ${model_q2_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q3_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q4_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q5_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/llama-completion -no-cnv --model ${model_q6_k} -ngl 99 -c 1024 -s 1234 -n 64 --ignore-eos -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/llama-perplexity --model ${model_f16}  -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    if [ -z ${GG_BUILD_NO_BF16} ]; then
        (time ./bin/llama-perplexity --model ${model_bf16} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-bf16.log
    fi
    (time ./bin/llama-perplexity --model ${model_q8_0} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/llama-perplexity --model ${model_q4_0} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/llama-perplexity --model ${model_q4_1} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/llama-perplexity --model ${model_q5_0} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/llama-perplexity --model ${model_q5_1} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/llama-perplexity --model ${model_q2_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q2_k.log
    (time ./bin/llama-perplexity --model ${model_q3_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/llama-perplexity --model ${model_q4_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/llama-perplexity --model ${model_q5_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/llama-perplexity --model ${model_q6_k} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/llama-imatrix --model ${model_f16} -f ${wiki_test} -ngl 99 -c 1024 -b 512 --chunks 2 ) 2>&1 | tee -a $OUT/${ci}-imatrix.log

    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 10 -c 1024 -fa off --no-op-offload) 2>&1 | tee -a $OUT/${ci}-save-load-state.log
    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 10 -c 1024 -fa on  --no-op-offload) 2>&1 | tee -a $OUT/${ci}-save-load-state.log
    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 99 -c 1024 -fa off                ) 2>&1 | tee -a $OUT/${ci}-save-load-state.log
    (time ./bin/llama-save-load-state --model ${model_q4_0} -ngl 99 -c 1024 -fa on                 ) 2>&1 | tee -a $OUT/${ci}-save-load-state.log

    function check_ppl {
        qnt="$1"
        ppl=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl > 20.0" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: ppl > 20.0)\n' "$qnt" "$ppl"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$ppl"
        return 0
    }

    check_ppl "f16"  "$(cat $OUT/${ci}-tg-f16.log  | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    if [ -z ${GG_BUILD_NO_BF16} ]; then
        check_ppl "bf16" "$(cat $OUT/${ci}-tg-bf16.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    fi
    check_ppl "q8_0" "$(cat $OUT/${ci}-tg-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_0" "$(cat $OUT/${ci}-tg-q4_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_1" "$(cat $OUT/${ci}-tg-q4_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_0" "$(cat $OUT/${ci}-tg-q5_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_1" "$(cat $OUT/${ci}-tg-q5_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
   #check_ppl "q2_k" "$(cat $OUT/${ci}-tg-q2_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log # note: ppl > 20.0 for this quant and model
    check_ppl "q3_k" "$(cat $OUT/${ci}-tg-q3_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_k" "$(cat $OUT/${ci}-tg-q4_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_k" "$(cat $OUT/${ci}-tg-q5_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q6_k" "$(cat $OUT/${ci}-tg-q6_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log

    cat $OUT/${ci}-imatrix.log | grep "Final" >> $OUT/${ci}-imatrix-sum.log

    set +e
}

function gg_sum_qwen3_0_6b {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Qwen3 0.6B:\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- perplexity:\n%s\n' "$(cat $OUT/${ci}-ppl.log)"
    gg_printf '- imatrix:\n```\n%s\n```\n' "$(cat $OUT/${ci}-imatrix-sum.log)"
    gg_printf '- f16:\n```\n%s\n```\n'  "$(cat $OUT/${ci}-tg-f16.log)"
    if [ -z ${GG_BUILD_NO_BF16} ]; then
        gg_printf '- bf16:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-bf16.log)"
    fi
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
    gg_printf '- q4_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_0.log)"
    gg_printf '- q4_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_1.log)"
    gg_printf '- q5_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_0.log)"
    gg_printf '- q5_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_1.log)"
    gg_printf '- q2_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q2_k.log)"
    gg_printf '- q3_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q3_k.log)"
    gg_printf '- q4_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_k.log)"
    gg_printf '- q5_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_k.log)"
    gg_printf '- q6_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q6_k.log)"
    gg_printf '- save-load-state: \n```\n%s\n```\n' "$(cat $OUT/${ci}-save-load-state.log)"
}

# bge-small

function gg_run_embd_bge_small {
    cd ${SRC}

    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/config.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/tokenizer.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/tokenizer_config.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/special_tokens_map.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/pytorch_model.bin
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/sentence_bert_config.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/vocab.txt
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/modules.json
    gg_wget models-mnt/bge-small/ https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/config.json

    gg_wget models-mnt/bge-small/1_Pooling https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/1_Pooling/config.json

    path_models="../models-mnt/bge-small"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf

    model_f16="${path_models}/ggml-model-f16.gguf"
    model_q8_0="${path_models}/ggml-model-q8_0.gguf"

    ./bin/llama-quantize ${model_f16} ${model_q8_0} q8_0

    (time ./bin/llama-embedding --model ${model_f16}  -p "I believe the meaning of life is" -ngl 99 -c 0 --no-op-offload) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/llama-embedding --model ${model_q8_0} -p "I believe the meaning of life is" -ngl 99 -c 0 --no-op-offload) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log

    set +e
}

function gg_sum_embd_bge_small {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'BGE Small (BERT):\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-f16.log)"
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
}

# rerank_tiny

function gg_run_rerank_tiny {
    cd ${SRC}

    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/config.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/tokenizer.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/tokenizer_config.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/special_tokens_map.json
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/pytorch_model.bin
    gg_wget models-mnt/rerank-tiny/ https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/raw/main/vocab.json

    path_models="../models-mnt/rerank-tiny"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert_hf_to_gguf.py ${path_models} --outfile ${path_models}/ggml-model-f16.gguf

    model_f16="${path_models}/ggml-model-f16.gguf"

    # for this model, the SEP token is "</s>"
    (time ./bin/llama-embedding --model ${model_f16} -p "what is panda?\thi\nwhat is panda?\tit's a bear\nwhat is panda?\tThe giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China." -ngl 99 -c 0 --pooling rank --embd-normalize -1 --no-op-offload --verbose-prompt) 2>&1 | tee -a $OUT/${ci}-rk-f16.log

    # sample output
    # rerank score 0:    0.029
    # rerank score 1:    0.029
    # rerank score 2:    0.135

    # check that the score is in the range [$3, $4]
    function check_score {
        qnt="$1"
        score=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$score < $3" | bc) -eq 1 ] || [ $(echo "$score > $4" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: score not in range [%s, %s])\n' "$qnt" "$score" "$3" "$4"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$score"
        return 0
    }

    check_score "rerank score 0" "$(cat $OUT/${ci}-rk-f16.log | grep "rerank score 0")" "0.00" "0.05" | tee -a $OUT/${ci}-rk-f16.log
    check_score "rerank score 1" "$(cat $OUT/${ci}-rk-f16.log | grep "rerank score 1")" "0.00" "0.05" | tee -a $OUT/${ci}-rk-f16.log
    check_score "rerank score 2" "$(cat $OUT/${ci}-rk-f16.log | grep "rerank score 2")" "0.10" "0.30" | tee -a $OUT/${ci}-rk-f16.log

    set +e
}

function gg_sum_rerank_tiny {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Rerank Tiny (Jina):\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-rk-f16.log)"
}

function gg_check_build_requirements {
    if ! command -v cmake &> /dev/null; then
        gg_printf 'cmake not found, please install'
    fi

    if ! command -v make &> /dev/null; then
        gg_printf 'make not found, please install'
    fi

    if ! command -v ctest &> /dev/null; then
        gg_printf 'ctest not found, please install'
    fi
}

## main

export LLAMA_LOG_PREFIX=1
export LLAMA_LOG_TIMESTAMPS=1

if [ -z ${GG_BUILD_LOW_PERF} ]; then
    # Create symlink: ./llama.cpp/models-mnt -> $MNT/models
    rm -rf ${SRC}/models-mnt
    mnt_models=${MNT}/models
    mkdir -p ${mnt_models}
    ln -sfn ${mnt_models} ${SRC}/models-mnt

    # Create a fresh python3 venv and enter it
    if ! python3 -m venv "$MNT/venv"; then
        echo "Error: Failed to create Python virtual environment at $MNT/venv."
        exit 1
    fi
    source "$MNT/venv/bin/activate"

    pip install -r ${SRC}/requirements.txt --disable-pip-version-check
    pip install --editable gguf-py --disable-pip-version-check
fi

ret=0

test $ret -eq 0 && gg_run ctest_debug
test $ret -eq 0 && gg_run ctest_release

if [ -z ${GG_BUILD_LOW_PERF} ]; then
    test $ret -eq 0 && gg_run embd_bge_small
    test $ret -eq 0 && gg_run rerank_tiny

    if [ -z ${GG_BUILD_CLOUD} ] || [ ${GG_BUILD_EXTRA_TESTS_0} ]; then
        test $ret -eq 0 && gg_run test_scripts
    fi

    test $ret -eq 0 && gg_run qwen3_0_6b

    test $ret -eq 0 && gg_run ctest_with_model_debug
    test $ret -eq 0 && gg_run ctest_with_model_release
fi

cat $OUT/README.md

exit $ret
