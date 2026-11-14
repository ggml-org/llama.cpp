#!/bin/sh
# Cross-builds the native M5-TTS pieces for arm64 Android.
#
#   ./build_android.sh [output-dir]
#
# Environment:
#   ANDROID_NDK    NDK root (default: newest under ~/Library/Android/sdk/ndk)
#   M5_ORT_ROOT    extracted onnxruntime-android.aar (headers/ + jni/<abi>/)
#   API            android platform level (default 28)
#
# Produces libm5engine.so (JNI), m5-engine (CLI), m5-server (HTTP) and the
# ring transport benchmarks.
# CMake is used when available because it links libc++ statically, so the .so
# needs nothing beyond libonnxruntime; the direct clang++ path is the fallback.
set -e

here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/build-android}
api=${API:-28}

if [ -z "$ANDROID_NDK" ]; then
    ANDROID_NDK=$(ls -d "$HOME"/Library/Android/sdk/ndk/* 2>/dev/null | sort -V | tail -1)
fi
[ -n "$ANDROID_NDK" ] || { echo "ANDROID_NDK not found" >&2; exit 1; }

if [ -z "$M5_ORT_ROOT" ]; then
    for candidate in "$here"/ort "$HOME"/ort /tmp/m5-ort/aar2; do
        if [ -f "$candidate/headers/onnxruntime_c_api.h" ]; then M5_ORT_ROOT=$candidate; break; fi
    done
fi
[ -n "$M5_ORT_ROOT" ] || { echo "M5_ORT_ROOT not set (extract onnxruntime-android.aar)" >&2; exit 1; }

echo "ndk: $ANDROID_NDK"
echo "ort: $M5_ORT_ROOT"

if command -v cmake >/dev/null 2>&1; then
    cmake -S "$here" -B "$out" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM="android-$api" \
        -DM5_ORT_ROOT="$M5_ORT_ROOT" -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$out" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" > /dev/null
    echo "built with cmake"
    ls -l "$out/libm5engine.so" "$out/m5-engine" "$out/m5-server" "$out/ring-stub" "$out/ring-bench"
    exit 0
fi

CXX=$(ls "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android"$api"-clang++ | head -1)
[ -x "$CXX" ] || { echo "no compiler at $ANDROID_NDK" >&2; exit 1; }

mkdir -p "$out"
# armv8.2-a+fp16+fp16fml unlocks the f16 depth-head dot product in m5_engine.cpp; the
# SM6650's cores all support it and older cores fall back to the f32 kernel.
flags="-std=c++17 -O2 -fPIC -march=armv8.2-a+fp16+fp16fml -I$here -I$M5_ORT_ROOT/headers"
libs="-L$M5_ORT_ROOT/jni/arm64-v8a -lonnxruntime -llog -landroid -pthread"

$CXX $flags -shared -o "$out/libm5engine.so" \
    "$here/m5_engine.cpp" "$here/m5_tokenizer.cpp" "$here/m5_jni.cpp" $libs
$CXX $flags -o "$out/m5-engine" \
    "$here/m5_engine_main.cpp" "$here/m5_engine.cpp" "$here/m5_tokenizer.cpp" $libs
$CXX $flags -I"$here/../../vendor/cpp-httplib" -o "$out/m5-server" \
    "$here/m5_server.cpp" "$here/m5_engine.cpp" "$here/m5_tokenizer.cpp" \
    "$here/../../vendor/cpp-httplib/httplib.cpp" $libs
$CXX $flags -o "$out/ring-stub" "$here/ring_stub.cpp"
$CXX $flags -o "$out/ring-bench" "$here/ring_bench.cpp"

echo "built with clang++ (needs libc++_shared.so at runtime)"
ls -l "$out"
