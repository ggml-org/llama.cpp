
CONVERT=0
BUILD=0
RUN=0
MODEL_NAME=Llama-2-7b-EfficientQAT-w4g128-GPTQ
MODEL_DIR=/Users/shijie/qingtaoli-temp/models/$MODEL_NAME
if [[ $1 == *c* ]]; then
    CONVERT=1
    echo "\n ===> python convert_hf_to_gguf.py $MODEL_DIR --outtype int_n --outfile $MODEL_DIR/$MODEL_NAME.INT_N.gguf --enable-t-mac --verbose"
    python convert_hf_to_gguf.py $MODEL_DIR --outtype int_n --outfile $MODEL_DIR/$MODEL_NAME.INT_N.gguf --enable-t-mac --verbose
fi
if [[ $1 == *b* ]]; then
    BUILD=1
    cd build
    echo "\n ===> cmake --build . --target clean"
    cmake --build . --target clean
    echo "\n ===> cmake .. -DGGML_TMAC=ON -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_TMAC_RECHUNK=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
    cmake .. -DGGML_TMAC=ON -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_TMAC_RECHUNK=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    echo "\n ===> cmake --build . --target llama-cli llama-bench llama-quantize llama-perplexity --config Release"
    cmake --build . --target llama-cli llama-bench llama-quantize llama-perplexity --config Release
    cd ..
fi
if [[ $1 == *r* ]]; then
    RUN=1
    echo "\n ===> ./build/bin/llama-cli -m $MODEL_DIR/$MODEL_NAME.INT_N.gguf -n 128 -t 1 -p Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington. -ngl 0 -c 2048"
    # MallocStackLogging=1 MallocScribble=1
    ./build/bin/llama-cli -m $MODEL_DIR/$MODEL_NAME.INT_N.gguf -n 128 -t 1 -p "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington." -ngl 0 -c 2048
fi

