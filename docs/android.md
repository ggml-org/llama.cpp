
# Android

## Build on Android using Termux
[Termux](https://github.com/termux/termux-app#installation) is a method to execute `llama.cpp` on an Android device (no root required).
```
apt update && apt upgrade -y
apt install git make cmake
```

It's recommended to move your model inside the `~/` directory for best performance:
```
cd storage/downloads
mv model.gguf ~/
```

[Get the code](https://github.com/ggerganov/llama.cpp#get-the-code) & [follow the Linux build instructions](https://github.com/ggerganov/llama.cpp#build) to build `llama.cpp`.

## Building the Project using Android NDK
Obtain the [Android NDK](https://developer.android.com/ndk) and then build with CMake.

Execute the following commands on your computer to avoid downloading the NDK to your mobile. Alternatively, you can also do this in Termux:
```
$ mkdir build-android
$ cd build-android
$ export NDK=<your_ndk_directory>
$ cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
$ make
```

Install [termux](https://github.com/termux/termux-app#installation) on your device and run `termux-setup-storage` to get access to your SD card (if Android 11+ then run the command twice).

Finally, copy these built `llama` binaries and the model file to your device storage. Because the file permissions in the Android sdcard cannot be changed, you can copy the executable files to the `/data/data/com.termux/files/home/bin` path, and then execute the following commands in Termux to add executable permission:

(Assumed that you have pushed the built executable files to the /sdcard/llama.cpp/bin path using `adb push`)
```
$cp -r /sdcard/llama.cpp/bin /data/data/com.termux/files/home/
$cd /data/data/com.termux/files/home/bin
$chmod +x ./*
```

Download model [llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf), and push it to `/sdcard/llama.cpp/`, then move it to `/data/data/com.termux/files/home/model/`

```
$mv /sdcard/llama.cpp/llama-2-7b-chat.Q4_K_M.gguf /data/data/com.termux/files/home/model/
```

Now, you can start chatting:
```
$cd /data/data/com.termux/files/home/bin
$./llama-cli -m ../model/llama-2-7b-chat.Q4_K_M.gguf -n 128 -cml
```

Here's a demo of an interactive session running on Pixel 5 phone:

https://user-images.githubusercontent.com/271616/225014776-1d567049-ad71-4ef2-b050-55b0b3b9274c.mp4
