# Bitnet.cpp

## arm
```bash
pip install -r requirements.txt
pip install ./gguf-py
python convert-hf-to-gguf-bitnet.py ${MODEL_DIR}/bitnet_b1_58-large --outtype tl1
python convert-hf-to-gguf-bitnet.py ${MODEL_DIR}/bitnet_b1_58-large --outtype f32
./build/bin/llama-quantize ${MODEL_DIR}/bitnet_b1_58-large/ggml-model-f32.gguf ${MODEL_DIR}/bitnet_b1_58-large/ggml-model.i2_s.gguf I2_S 1


python codegen_tl1.py --BMEMD 256 --BKEMD 256 --bmEMD 32 --byEMD 8 --BMGQA 256 --BKGQA 256 --bmGQA 32 --byGQA 8
cmake -B build -DGGML_BITNET_ARM_TL1=ON
cmake --build build --target llama-cli llama-bench llama-quantize llama-eval-callback --config Release
./build/bin/llama-cli -m ${MODEL_DIR}/bitnet_b1_58-large/ggml-model-tl1.gguf -n 128 -p "Microsoft Corporation is" -ngl 0 -c 2048 -b 1 -s 0 -t 1
./build/bin/llama-cli -m ${MODEL_DIR}/bitnet_b1_58-large/ggml-model-i2_s.gguf -n 128 -p "Microsoft Corporation is" -ngl 0 -c 2048 -b 1 -s 0 -t 1
```
## x86
```bash
pip install -r requirements.txt
pip install .\gguf-py
python convert-hf-to-gguf-bitnet.py ${MODEL_DIR}\bitnet_b1_58-large --outtype tl1
python convert-hf-to-gguf-bitnet.py ${MODEL_DIR}\bitnet_b1_58-large --outtype f32
.\build\bin\Release\llama-quantize ${MODEL_DIR}\bitnet_b1_58-large\ggml-model-f32.gguf ${MODEL_DIR}\bitnet_b1_58-large\ggml-model.i2_s.gguf I2_S 1


python codegen_tl2.py --BMEMD 256 --BKEMD 96 --bmEMD 32 --byEMD 6 --BMGQA 128 --BKGQA 96 --bmGQA 32 --byGQA 4
cmake -B build -DGGML_BITNET_X86_TL2=ON
cmake --build build --target llama-cli llama-bench llama-quantize llama-eval-callback --config Release
.\build\bin\Release\llama-cli.exe -m "${MODEL_DIR}\ggml-model-tl2.gguf" -n 128 -p "Microsoft Corporation is" -ngl 0 -c 2048 -b 1 -s 0 -t 1
.\build\bin\Release\llama-cli.exe -m "${MODEL_DIR}\ggml-model-tl2.gguf" -n 128 -p "Microsoft Corporation is" -ngl 0 -c 2048 -b 1 -s 0 -t 1
```