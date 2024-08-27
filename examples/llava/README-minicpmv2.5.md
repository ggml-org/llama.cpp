## MiniCPM-Llama3-V 2.5

### Prepare models and code

Download [MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) PyTorch model from huggingface to "MiniCPM-Llama3-V-2_5" folder.

Clone llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Usage

Convert PyTorch model to gguf files (You can also download the converted [gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf) by us)

```bash
python ./examples/llava/minicpmv-surgery.py -m ../MiniCPM-Llama3-V-2_5
python ./examples/llava/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-Llama3-V-2_5 --minicpmv-projector ../MiniCPM-Llama3-V-2_5/minicpmv.projector --output-dir ../MiniCPM-Llama3-V-2_5/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version 2
python ./convert_hf_to_gguf.py ../MiniCPM-Llama3-V-2_5/model

# quantize int4 version
./llama-quantize ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
```

Build for Linux or Mac

```bash
make
make llama-minicpmv-cli
```

Inference on Linux or Mac
```
# run f16 version
./llama-minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# run quantized int4 version
./llama-minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg  -p "What is in the image?"

# or run in interactive mode
./llama-minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i
```

### Android

#### Build on Android device using Termux
We found that build on Android device would bring better runtime performance, so we recommend to build on device.

[Termux](https://github.com/termux/termux-app#installation) is a terminal app on Android device (no root required).

Install tools in Termux:
```
apt update && apt upgrade -y
apt install git make cmake
```

It's recommended to move your model inside the `~/` directory for best performance:
```
cd storage/downloads
mv model.gguf ~/
```

#### Building the Project using Android NDK
Obtain the [Android NDK](https://developer.android.com/ndk) and then build with CMake.

Execute the following commands on your computer to avoid downloading the NDK to your mobile. Alternatively, you can also do this in Termux:

```bash
mkdir build-android
cd build-android
export NDK=/your_ndk_path
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
make
```

Install [termux](https://github.com/termux/termux-app#installation) on your device and run `termux-setup-storage` to get access to your SD card (if Android 11+ then run the command twice).

Finally, copy these built `llama` binaries and the model file to your device storage. Because the file permissions in the Android sdcard cannot be changed, you can copy the executable files to the `/data/data/com.termux/files/home/bin` path, and then execute the following commands in Termux to add executable permission:

(Assumed that you have pushed the built executable files to the /sdcard/llama.cpp/bin path using `adb push`)
```
$cp -r /sdcard/llama.cpp/bin /data/data/com.termux/files/home/
$cd /data/data/com.termux/files/home/bin
$chmod +x ./*
```

Download models and push them to `/sdcard/llama.cpp/`, then move it to `/data/data/com.termux/files/home/model/`

```
$mv /sdcard/llama.cpp/ggml-model-Q4_K_M.gguf /data/data/com.termux/files/home/model/
$mv /sdcard/llama.cpp/mmproj-model-f16.gguf /data/data/com.termux/files/home/model/
```

Now, you can start chatting:
```
$cd /data/data/com.termux/files/home/bin
$./llama-minicpmv-cli -m ../model/ggml-model-Q4_K_M.gguf --mmproj ../model/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg  -p "What is in the image?"
```
