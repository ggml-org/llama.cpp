#!/bin/bash

# a script to compare MLX and GGUF models
#
# usage:
#   ./compare-mlx.sh --raw-path wiki.test.raw --no-keep
#
# TODOs
# - add QAT evals

# check if LLAMA_HOME_DIR is set
if [[ -z "$LLAMA_HOME_DIR" ]]; then
    lcpp_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../ && pwd)
else
    lcpp_dir="${LLAMA_HOME_DIR}"
fi

echo "Using llama.cpp directory: ${lcpp_dir}"

# check for convert_hf_to_gguf.py
if [[ ! -f "${lcpp_dir}/convert_hf_to_gguf.py" ]]; then
    echo "convert_hf_to_gguf.py not found in ${lcpp_dir}"
    echo "Set a LLAMA_HOME_DIR environment variable to point to your llama.cpp directory"
    exit 1
fi

set -x

# sanity checks that all Python dependencies are installed
if ! python -c "import mlx.core"; then
    echo "MLX not found. Please install MLX"
    exit 1
fi

if ! python ${lcpp_dir}/convert_hf_to_gguf.py --help; then
    echo "convert_hf_to_gguf.py not working. Please install llama.cpp python requirements"
    exit 1
fi

# by default use the system binaries (for example from brew)
llama_perplexity="llama-perplexity"

if [[ ! -z "$LLAMA_PERPLEXITY" ]]; then
    llama_perplexity="$LLAMA_PERPLEXITY"
fi

echo "Using llama-perplexity: ${llama_perplexity}"

if ! command -v "$llama_perplexity" &> /dev/null; then
    echo "llama-perplexity not found. Please install it."
    exit 1
fi

llama_quantize="llama-quantize"

if [[ ! -z "$LLAMA_QUANTIZE" ]]; then
    llama_quantize="$LLAMA_QUANTIZE"
fi

echo "Using llama-quantize: ${llama_quantize}"

if ! command -v "$llama_quantize" &> /dev/null; then
    echo "llama-quantize not found. Please install it."
    exit 1
fi

llama_batched_bench="llama-batched-bench"

if [[ ! -z "$LLAMA_BATCHED_BENCH" ]]; then
    llama_batched_bench="$LLAMA_BATCHED_BENCH"
fi

echo "Using llama-batched-bench: ${llama_batched_bench}"

if ! command -v "$llama_batched_bench" &> /dev/null; then
    echo "llama-batched-bench not found. Please install it."
    exit 1
fi

# get the size in GiB
get_size() {
    local path="$1"
    local bytes=$(du -s "$path" | awk '{print $1}')
    local res=$(echo "scale=3; ($bytes*512)/1024/1024/1024" | bc)
    echo "$res"
}

# parameters:
# --no-compute      : do not compute anything, just summarize the existing results
# --no-ppl          : do not compute ppl
# --no-perf         : do not compute performance (speed) metrics
# --no-keep         : delete intermediate model files
# --num-samples     : number of text samples to evaluate (default: 512)
# --sequence-length : sequence length of the samples in tokens (default: 512)
# --raw-path        : file with raw text (such as wikitext)

# extra agruments
args_lcpp="-t 1"

num_samples=512
sequence_length=512
raw_path=""
no_compute=false
no_ppl=false
no_perf=false
no_keep=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-compute)
            no_compute=true
            shift
            ;;
        --no-ppl)
            no_ppl=true
            shift
            ;;
        --no-perf)
            no_perf=true
            shift
            ;;
        --no-keep)
            no_keep=true
            shift
            ;;
        --num-samples)
            num_samples="$2"
            shift 2
            ;;
        --sequence-length)
            sequence_length="$2"
            shift 2
            ;;
        --raw-path)
            raw_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$raw_path" ]]; then
    echo "No raw path provided"
    echo "Recommended to use the test set of WikiText from here: https://github.com/ggml-org/llama.cpp/blob/master/scripts/get-wikitext-2.sh"
    exit 1
fi

eval_model() {
    org="$1"
    mid="$2"

    echo "Evaluating ${org}/${mid}"

    huggingface-cli download ${org}/${mid} --local-dir ${org}/${mid}

    # generate and process MLX models

    if [[ "$no_compute" == true ]]; then
        echo "Skipping computation"
    else
        rm -rfv ./${mid}-f32-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-f32-mlx --dtype float32
        get_size ./${mid}-f32-mlx > ./${mid}-f32-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-f32-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-f32-mlx-ppl.txt
        fi

        # no need for F32 perf benchmarks
        #if [[ "$no_perf" == false ]]; then
        #    mlx_lm.benchmark --model ./${mid}-f32-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-f32-mlx-perf-2048.txt
        #    mlx_lm.benchmark --model ./${mid}-f32-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-f32-mlx-perf-4096.txt
        #    mlx_lm.benchmark --model ./${mid}-f32-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-f32-mlx-perf-8192.txt
        #    mlx_lm.benchmark --model ./${mid}-f32-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-f32-mlx-perf-16384.txt
        #    mlx_lm.benchmark --model ./${mid}-f32-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-f32-mlx-perf-32768.txt
        #fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-f32-mlx
        fi

        rm -rfv ./${mid}-bf16-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-bf16-mlx --dtype bfloat16
        get_size ./${mid}-bf16-mlx > ./${mid}-bf16-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-bf16-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-bf16-mlx-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            mlx_lm.benchmark --model ./${mid}-bf16-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-bf16-mlx-perf-2048.txt
            mlx_lm.benchmark --model ./${mid}-bf16-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-bf16-mlx-perf-4096.txt
            mlx_lm.benchmark --model ./${mid}-bf16-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-bf16-mlx-perf-8192.txt
            mlx_lm.benchmark --model ./${mid}-bf16-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-bf16-mlx-perf-16384.txt
            mlx_lm.benchmark --model ./${mid}-bf16-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-bf16-mlx-perf-32768.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-bf16-mlx
        fi

        rm -rfv ./${mid}-f16-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-f16-mlx --dtype float16
        get_size ./${mid}-f16-mlx > ./${mid}-f16-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-f16-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-f16-mlx-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            mlx_lm.benchmark --model ./${mid}-f16-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-f16-mlx-perf-2048.txt
            mlx_lm.benchmark --model ./${mid}-f16-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-f16-mlx-perf-4096.txt
            mlx_lm.benchmark --model ./${mid}-f16-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-f16-mlx-perf-8192.txt
            mlx_lm.benchmark --model ./${mid}-f16-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-f16-mlx-perf-16384.txt
            mlx_lm.benchmark --model ./${mid}-f16-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-f16-mlx-perf-32768.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-f16-mlx
        fi

        rm -rfv ./${mid}-q8-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-q8-mlx --quantize --q-bits 8 --dtype float16
        get_size ./${mid}-q8-mlx > ./${mid}-q8-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-q8-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-q8-mlx-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            mlx_lm.benchmark --model ./${mid}-q8-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q8-mlx-perf-2048.txt
            mlx_lm.benchmark --model ./${mid}-q8-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q8-mlx-perf-4096.txt
            mlx_lm.benchmark --model ./${mid}-q8-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q8-mlx-perf-8192.txt
            mlx_lm.benchmark --model ./${mid}-q8-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q8-mlx-perf-16384.txt
            mlx_lm.benchmark --model ./${mid}-q8-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q8-mlx-perf-32768.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q8-mlx
        fi

        #rm -rfv ./${mid}-q6-mlx
        #mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-q6-mlx --quantize --q-bits 6 --dtype float16
        #get_size ./${mid}-q6-mlx > ./${mid}-q6-mlx-size.txt

        #if [[ "$no_ppl" == false ]]; then
        #    python ./mlx-ppl.py --model ./${mid}-q6-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-q6-mlx-ppl.txt
        #fi

        #if [[ "$no_perf" == false ]]; then
        #    mlx_lm.benchmark --model ./${mid}-q6-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q6-mlx-perf-2048.txt
        #    mlx_lm.benchmark --model ./${mid}-q6-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q6-mlx-perf-4096.txt
        #    mlx_lm.benchmark --model ./${mid}-q6-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q6-mlx-perf-8192.txt
        #    mlx_lm.benchmark --model ./${mid}-q6-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q6-mlx-perf-16384.txt
        #    mlx_lm.benchmark --model ./${mid}-q6-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q6-mlx-perf-32768.txt
        #fi

        #if [[ "$no_keep" == true ]]; then
        #    echo "Deleting intermediate model files"
        #    rm -rfv ./${mid}-q6-mlx
        #fi

        #rm -rfv ./${mid}-q5-mlx
        #mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-q5-mlx --quantize --q-bits 5 --dtype float16
        #get_size ./${mid}-q5-mlx > ./${mid}-q5-mlx-size.txt

        #if [[ "$no_ppl" == false ]]; then
        #    python ./mlx-ppl.py --model ./${mid}-q5-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-q5-mlx-ppl.txt
        #fi

        #if [[ "$no_perf" == false ]]; then
        #    mlx_lm.benchmark --model ./${mid}-q5-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q5-mlx-perf-2048.txt
        #    mlx_lm.benchmark --model ./${mid}-q5-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q5-mlx-perf-4096.txt
        #    mlx_lm.benchmark --model ./${mid}-q5-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q5-mlx-perf-8192.txt
        #    mlx_lm.benchmark --model ./${mid}-q5-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q5-mlx-perf-16384.txt
        #    mlx_lm.benchmark --model ./${mid}-q5-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q5-mlx-perf-32768.txt
        #fi

        #if [[ "$no_keep" == true ]]; then
        #    echo "Deleting intermediate model files"
        #    rm -rfv ./${mid}-q5-mlx
        #fi

        # I think this is something similar to q4_k
        rm -rfv ./${mid}-q4p-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-q4p-mlx --quantize --quant-predicate mixed_4_6 --dtype float16
        get_size ./${mid}-q4p-mlx > ./${mid}-q4p-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-q4p-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-q4p-mlx-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            mlx_lm.benchmark --model ./${mid}-q4p-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4p-mlx-perf-2048.txt
            mlx_lm.benchmark --model ./${mid}-q4p-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4p-mlx-perf-4096.txt
            mlx_lm.benchmark --model ./${mid}-q4p-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4p-mlx-perf-8192.txt
            mlx_lm.benchmark --model ./${mid}-q4p-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4p-mlx-perf-16384.txt
            mlx_lm.benchmark --model ./${mid}-q4p-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4p-mlx-perf-32768.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q4p-mlx
        fi

        rm -rfv ./${mid}-q4-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-q4-mlx --quantize --q-bits 4 --dtype float16
        get_size ./${mid}-q4-mlx > ./${mid}-q4-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-q4-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-q4-mlx-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            mlx_lm.benchmark --model ./${mid}-q4-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4-mlx-perf-2048.txt
            mlx_lm.benchmark --model ./${mid}-q4-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4-mlx-perf-4096.txt
            mlx_lm.benchmark --model ./${mid}-q4-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4-mlx-perf-8192.txt
            mlx_lm.benchmark --model ./${mid}-q4-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4-mlx-perf-16384.txt
            mlx_lm.benchmark --model ./${mid}-q4-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q4-mlx-perf-32768.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q4-mlx
        fi

        rm -rfv ./${mid}-q3-mlx
        mlx_lm.convert --hf ./${org}/${mid} --mlx-path ./${mid}-q3-mlx --quantize --q-bits 3 --dtype float16
        get_size ./${mid}-q3-mlx > ./${mid}-q3-mlx-size.txt

        if [[ "$no_ppl" == false ]]; then
            python ./mlx-ppl.py --model ./${mid}-q3-mlx --raw-path "$raw_path" --num-samples "$num_samples" --sequence-length "$sequence_length" 2>&1 | tee ./${mid}-q3-mlx-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            mlx_lm.benchmark --model ./${mid}-q3-mlx -p 2048  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q3-mlx-perf-2048.txt
            mlx_lm.benchmark --model ./${mid}-q3-mlx -p 4096  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q3-mlx-perf-4096.txt
            mlx_lm.benchmark --model ./${mid}-q3-mlx -p 8192  -g 128 --num-trials 1 2>&1 | tee ./${mid}-q3-mlx-perf-8192.txt
            mlx_lm.benchmark --model ./${mid}-q3-mlx -p 16384 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q3-mlx-perf-16384.txt
            mlx_lm.benchmark --model ./${mid}-q3-mlx -p 32768 -g 128 --num-trials 1 2>&1 | tee ./${mid}-q3-mlx-perf-32768.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q3-mlx
        fi
    fi

    # generate and process llama.cpp GGUF models

    if [[ "$no_compute" == true ]]; then
        echo "Skipping computation"
    else
        # the F32 model is the reference - we generate all other models from it
        mkdir -p ./${mid}-f32-gguf
        python ${lcpp_dir}/convert_hf_to_gguf.py ./${org}/${mid} --outtype f32 --outfile ./${mid}-f32-gguf/model.gguf
        get_size ./${mid}-f32-gguf > ./${mid}-f32-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-f32-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-f32-gguf-ppl.txt
        fi

        # no need for F32 perf benchmarks
        #if [[ "$no_perf" == false ]]; then
        #    ${llama_batched_bench} $args_lcpp -m ./${mid}-f32-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-f32-gguf-perf.txt
        #fi

        # this requires to explicitly build llama.cpp with BF16 support
        rm -rfv ./${mid}-bf16-gguf && mkdir -p ./${mid}-bf16-gguf
        ${llama_quantize} ./${mid}-f32-gguf/model.gguf ./${mid}-bf16-gguf/model.gguf bf16
        get_size ./${mid}-bf16-gguf > ./${mid}-bf16-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-bf16-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-bf16-gguf-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            ${llama_batched_bench} $args_lcpp -m ./${mid}-bf16-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-bf16-gguf-perf.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-bf16-gguf
        fi

        rm -rfv ./${mid}-f16-gguf && mkdir -p ./${mid}-f16-gguf
        ${llama_quantize} ./${mid}-f32-gguf/model.gguf ./${mid}-f16-gguf/model.gguf f16
        get_size ./${mid}-f16-gguf > ./${mid}-f16-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-f16-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-f16-gguf-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            ${llama_batched_bench} $args_lcpp -m ./${mid}-f16-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-f16-gguf-perf.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-f16-gguf
        fi

        rm -rfv ./${mid}-q8-gguf && mkdir -p ./${mid}-q8-gguf
        ${llama_quantize} ./${mid}-f32-gguf/model.gguf ./${mid}-q8-gguf/model.gguf q8_0
        get_size ./${mid}-q8-gguf > ./${mid}-q8-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-q8-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-q8-gguf-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            ${llama_batched_bench} $args_lcpp -m ./${mid}-q8-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-q8-gguf-perf.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q8-gguf
        fi

        #rm -rfv ./${mid}-q6-gguf && mkdir -p ./${mid}-q6-gguf
        #${llama_quantize}  ./${mid}-f32-gguf/model.gguf ./${mid}-q6-gguf/model.gguf q6_k
        #get_size ./${mid}-q6-gguf > ./${mid}-q6-gguf-size.txt

        #if [[ "$no_ppl" == false ]]; then
        #    ${llama_perplexity} $args_lcpp -m ./${mid}-q6-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-q6-gguf-ppl.txt
        #fi

        #if [[ "$no_perf" == false ]]; then
        #    ${llama_batched_bench} $args_lcpp -m ./${mid}-q6-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-q6-gguf-perf.txt
        #fi

        #if [[ "$no_keep" == true ]]; then
        #    echo "Deleting intermediate model files"
        #    rm -rfv ./${mid}-q6-gguf
        #fi

        #rm -rfv ./${mid}-q5-gguf && mkdir -p ./${mid}-q5-gguf
        #${llama_quantize}  ./${mid}-f32-gguf/model.gguf ./${mid}-q5-gguf/model.gguf q5_k_s
        #get_size ./${mid}-q5-gguf > ./${mid}-q5-gguf-size.txt

        #if [[ "$no_ppl" == false ]]; then
        #    ${llama_perplexity} $args_lcpp -m ./${mid}-q5-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-q5-gguf-ppl.txt
        #fi

        #if [[ "$no_perf" == false ]]; then
        #    ${llama_batched_bench} $args_lcpp -m ./${mid}-q5-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-q5-gguf-perf.txt
        #fi

        #if [[ "$no_keep" == true ]]; then
        #    echo "Deleting intermediate model files"
        #    rm -rfv ./${mid}-q5-gguf
        #fi

        rm -rfv ./${mid}-q4p-gguf && mkdir -p ./${mid}-q4p-gguf
        ${llama_quantize}  ./${mid}-f32-gguf/model.gguf ./${mid}-q4p-gguf/model.gguf q4_k
        get_size ./${mid}-q4p-gguf > ./${mid}-q4p-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-q4p-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-q4p-gguf-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            ${llama_batched_bench} $args_lcpp -m ./${mid}-q4p-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-q4p-gguf-perf.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q4p-gguf
        fi

        # note: we use --pure here to match the MLX quantization of the embeddings
        rm -rfv ./${mid}-q4-gguf && mkdir -p ./${mid}-q4-gguf
        ${llama_quantize} --pure ./${mid}-f32-gguf/model.gguf ./${mid}-q4-gguf/model.gguf q4_0
        get_size ./${mid}-q4-gguf > ./${mid}-q4-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-q4-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-q4-gguf-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            ${llama_batched_bench} $args_lcpp -m ./${mid}-q4-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-q4-gguf-perf.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q4-gguf
        fi

        rm -rfv ./${mid}-q3-gguf && mkdir -p ./${mid}-q3-gguf
        ${llama_quantize}  ./${mid}-f32-gguf/model.gguf ./${mid}-q3-gguf/model.gguf q3_k_s
        get_size ./${mid}-q3-gguf > ./${mid}-q3-gguf-size.txt

        if [[ "$no_ppl" == false ]]; then
            ${llama_perplexity} $args_lcpp -m ./${mid}-q3-gguf/model.gguf -f "$raw_path" --chunks "${num_samples}" -c "${sequence_length}" 2>&1 | tee ./${mid}-q3-gguf-ppl.txt
        fi

        if [[ "$no_perf" == false ]]; then
            ${llama_batched_bench} $args_lcpp -m ./${mid}-q3-gguf/model.gguf -c 33768 -b 2048 -ub 2048 -npp 2048,4096,8192,16384,32768 -ntg 128 -npl 1 2>&1 | tee ./${mid}-q3-gguf-perf.txt
        fi

        if [[ "$no_keep" == true ]]; then
            echo "Deleting intermediate model files"
            rm -rfv ./${mid}-q3-gguf
        fi

        # remove the f32 model at the end
        if [[ "$no_keep" == true ]]; then
            rm -rfv ./${mid}-f32-gguf
        fi
    fi

    set +x

    # analyze results

    #types=("f32" "bf16" "f16" "q8" "q6" "q5" "q4p" "q4" "q3")
    types=("f32" "bf16" "f16" "q8" "q4p" "q4" "q3")

    mlx_ppls=()
    mlx_ppl_deltas=()
    mlx_sizes=()
    mlx_pps2k=()
    mlx_tgs2k=()
    mlx_pps4k=()
    mlx_tgs4k=()
    mlx_pps8k=()
    mlx_tgs8k=()
    mlx_pps16k=()
    mlx_tgs16k=()
    mlx_pps32k=()
    mlx_tgs32k=()

    # mlx:
    for t in ${types[*]}; do
        cur_ppl="N/A"
        cur_ppl_delta="N/A"
        cur_size="N/A"
        cur_pp2k="N/A"
        cur_tg2k="N/A"
        cur_pp4k="N/A"
        cur_tg4k="N/A"
        cur_pp8k="N/A"
        cur_tg8k="N/A"
        cur_pp16k="N/A"
        cur_tg16k="N/A"
        cur_pp32k="N/A"
        cur_tg32k="N/A"

        if [[ -f ./${mid}-${t}-mlx-ppl.txt ]]; then
            cur_ppl=$(grep -o 'Perplexity: [0-9.]*' ./${mid}-${t}-mlx-ppl.txt | cut -d' ' -f2)
            cur_ppl_delta=$(grep -o 'Perplexity: [0-9.]* ± [0-9.]*' ./${mid}-${t}-mlx-ppl.txt | cut -d' ' -f4)
            cur_size=$(cat ./${mid}-${t}-mlx-size.txt)
            cur_pp2k=$(grep -o 'Averages.*prompt_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-2048.txt | cut -d'=' -f2)
            cur_tg2k=$(grep -o 'Averages.*generation_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-2048.txt | cut -d'=' -f3)
            cur_pp4k=$(grep -o 'Averages.*prompt_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-4096.txt | cut -d'=' -f2)
            cur_tg4k=$(grep -o 'Averages.*generation_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-4096.txt | cut -d'=' -f3)
            cur_pp8k=$(grep -o 'Averages.*prompt_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-8192.txt | cut -d'=' -f2)
            cur_tg8k=$(grep -o 'Averages.*generation_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-8192.txt | cut -d'=' -f3)
            cur_pp16k=$(grep -o 'Averages.*prompt_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-16384.txt | cut -d'=' -f2)
            cur_tg16k=$(grep -o 'Averages.*generation_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-16384.txt | cut -d'=' -f3)
            cur_pp32k=$(grep -o 'Averages.*prompt_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-32768.txt | cut -d'=' -f2)
            cur_tg32k=$(grep -o 'Averages.*generation_tps=[0-9.]*' ./${mid}-${t}-mlx-perf-32768.txt | cut -d'=' -f3)
        fi

        mlx_ppls+=("${cur_ppl}")
        mlx_ppl_deltas+=("${cur_ppl_delta}")
        mlx_sizes+=("${cur_size}")
        mlx_pps2k+=("${cur_pp2k}")
        mlx_tgs2k+=("${cur_tg2k}")
        mlx_pps4k+=("${cur_pp4k}")
        mlx_tgs4k+=("${cur_tg4k}")
        mlx_pps8k+=("${cur_pp8k}")
        mlx_tgs8k+=("${cur_tg8k}")
        mlx_pps16k+=("${cur_pp16k}")
        mlx_tgs16k+=("${cur_tg16k}")
        mlx_pps32k+=("${cur_pp32k}")
        mlx_tgs32k+=("${cur_tg32k}")
    done

    gguf_ppls=()
    gguf_ppl_deltas=()
    gguf_sizes=()
    gguf_pps2k=()
    gguf_tgs2k=()
    gguf_pps4k=()
    gguf_tgs4k=()
    gguf_pps8k=()
    gguf_tgs8k=()
    gguf_pps16k=()
    gguf_tgs16k=()
    gguf_pps32k=()
    gguf_tgs32k=()

    # gguf:
    for t in ${types[*]}; do
        cur_ppl="N/A"
        cur_ppl_delta="N/A"
        cur_size="N/A"
        cur_pp2k="N/A"
        cur_tg2k="N/A"
        cur_pp4k="N/A"
        cur_tg4k="N/A"
        cur_pp8k="N/A"
        cur_tg8k="N/A"
        cur_pp16k="N/A"
        cur_tg16k="N/A"
        cur_pp32k="N/A"
        cur_tg32k="N/A"

        if [[ -f ./${mid}-${t}-gguf-ppl.txt ]]; then
            cur_ppl=$(grep -o 'Final estimate: PPL = [0-9.]*' ./${mid}-${t}-gguf-ppl.txt | sed -e "s/.*Final//" | cut -d' ' -f5)
            cur_ppl_delta=$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' ./${mid}-${t}-gguf-ppl.txt | sed -e "s/.*Final//" | cut -d' ' -f7)
            cur_size=$(cat ./${mid}-${t}-gguf-size.txt)
            cur_pp2k=$(grep  -o '|  2048 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $12}')
            cur_tg2k=$(grep  -o '|  2048 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $16}')
            cur_pp4k=$(grep  -o '|  4096 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $12}')
            cur_tg4k=$(grep  -o '|  4096 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $16}')
            cur_pp8k=$(grep  -o '|  8192 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $12}')
            cur_tg8k=$(grep  -o '|  8192 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $16}')
            cur_pp16k=$(grep -o '| 16384 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $12}')
            cur_tg16k=$(grep -o '| 16384 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $16}')
            cur_pp32k=$(grep -o '| 32768 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $12}')
            cur_tg32k=$(grep -o '| 32768 |.*' ./${mid}-${t}-gguf-perf.txt | awk '{print $16}')
        fi

        gguf_ppls+=("${cur_ppl}")
        gguf_ppl_deltas+=("${cur_ppl_delta}")
        gguf_sizes+=("${cur_size}")
        gguf_pps2k+=("${cur_pp2k}")
        gguf_tgs2k+=("${cur_tg2k}")
        gguf_pps4k+=("${cur_pp4k}")
        gguf_tgs4k+=("${cur_tg4k}")
        gguf_pps8k+=("${cur_pp8k}")
        gguf_tgs8k+=("${cur_tg8k}")
        gguf_pps16k+=("${cur_pp16k}")
        gguf_tgs16k+=("${cur_tg16k}")
        gguf_pps32k+=("${cur_pp32k}")
        gguf_tgs32k+=("${cur_tg32k}")
    done

    res="${mid}-results.txt"
    echo "Results for ${org}/${mid} saved to ${res}"

    printf "\n" | tee ${res}
    printf "Model ID:        ${org}/${mid}\n" | tee -a ${res}
    #printf "Samples:         ${num_samples}\n" | tee -a ${res}
    #printf "Sequence Length: ${sequence_length}\n" | tee -a ${res}
    printf "\n" | tee -a ${res}
    printf "| Type  | MLX PPL             | GGUF PPL               | MLX Size | GGUF Size | MLX PP  2K | GGUF PP  2K | MLX TG  2K | GGUF TG  2K | MLX PP  4K | GGUF PP  4K | MLX TG  4K | GGUF TG  4K | MLX PP  8K | GGUF PP  8K | MLX TG  8K | GGUF TG  8K | MLX PP 16K | GGUF PP 16K | MLX TG 16K | GGUF TG 16K | MLX PP 32K | GGUF PP 32K | MLX TG 32K | GGUF TG 32K |\n" | tee -a ${res}
    printf "|-------|---------------------|------------------------|----------|-----------| ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- |\n" | tee -a ${res}

    for i in "${!types[@]}"; do
        printf "| %-5s | %10s ± %6s | %10s ± %9s | %8s | %9s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s | %10s | %11s |\n" \
            "${types[i]}" \
            "${mlx_ppls[i]}" \
            "${mlx_ppl_deltas[i]}" \
            "${gguf_ppls[i]}" \
            "${gguf_ppl_deltas[i]}" \
            "${mlx_sizes[i]}" \
            "${gguf_sizes[i]}" \
            "${mlx_pps2k[i]}" \
            "${gguf_pps2k[i]}" \
            "${mlx_tgs2k[i]}" \
            "${gguf_tgs2k[i]}" \
            "${mlx_pps4k[i]}" \
            "${gguf_pps4k[i]}" \
            "${mlx_tgs4k[i]}" \
            "${gguf_tgs4k[i]}" \
            "${mlx_pps8k[i]}" \
            "${gguf_pps8k[i]}" \
            "${mlx_tgs8k[i]}" \
            "${gguf_tgs8k[i]}" \
            "${mlx_pps16k[i]}" \
            "${gguf_pps16k[i]}" \
            "${mlx_tgs16k[i]}" \
            "${gguf_tgs16k[i]}" \
            "${mlx_pps32k[i]}" \
            "${gguf_pps32k[i]}" \
            "${mlx_tgs32k[i]}" \
            "${gguf_tgs32k[i]}" | tee -a ${res}
    done
}

eval_model "meta-llama" "Llama-3.2-1B"
eval_model "meta-llama" "Llama-3.2-3B"
eval_model "meta-llama" "Llama-3.1-8B"

eval_model "google" "gemma-3-270m"
eval_model "google" "gemma-3-1b-pt"
#eval_model "google" "gemma-3-4b-pt"

# the mlx-ppl.y script does not work with these models - not sure why
#eval_model "google" "gemma-3n-E2B"
#eval_model "google" "gemma-3n-E4B"

eval_model "Qwen" "Qwen3-0.6B-Base"
eval_model "Qwen" "Qwen3-1.7B-Base"
eval_model "Qwen" "Qwen3-4B-Base"
eval_model "Qwen" "Qwen3-8B-Base"
eval_model "Qwen" "Qwen3-30B-A3B-Base"
