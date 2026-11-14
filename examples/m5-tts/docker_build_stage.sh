#!/usr/bin/env bash
# Builds the Hexagon stage binaries (backbone/depth) with the Qualcomm
# Snapdragon toolchain image. Linux x86_64 is emulated on Apple Silicon, so the
# first build takes a while; the ccache-less rebuild is a couple of minutes
# because only the stage itself changes.
#
#   ./docker_build_stage.sh [output_dir]
#
# The image ships the Hexagon SDK and the Android NDK, which is what the
# prebuilt stage binaries on the device were missing a toolchain for.
set -euo pipefail

IMAGE=ghcr.io/snapdragon-toolchain/arm64-android:v0.8
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=${1:-"$REPO/build-android-stage"}
CONTAINER=m5-stage-build

mkdir -p "$OUT"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" \
    -v "$REPO":/work/llama.cpp \
    -w /work/llama.cpp \
    "$IMAGE" sleep infinity >/dev/null

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# The image ships neither cmake nor ninja.
docker exec "$CONTAINER" bash -lc '
    command -v cmake >/dev/null && command -v ninja >/dev/null && exit 0
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build'

# PREBUILT_LIB_DIR has to be set for the top-level configure: the SDK cmake
# helper runs string(FIND ${PREBUILT_LIB_DIR} ...) while being included, and an
# empty value makes that a fatal error in CMake 3.31.
if [ ! -f "$REPO/build-android/build.ninja" ]; then
    docker exec "$CONTAINER" bash -lc 'cd /work/llama.cpp && cmake -G Ninja -B build-android \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DGGML_HEXAGON=ON \
        -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT \
        -DPREBUILT_LIB_DIR=toolv21_v73 \
        -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_CPU=ON \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF'
fi

echo "==> building libllama, ggml and the hexagon backend"
docker exec "$CONTAINER" bash -lc 'cd /work/llama.cpp/build-android && ninja -j"$(nproc)" llama ggml-hexagon'

echo "==> building the v73 HTP skel"
docker exec "$CONTAINER" bash -lc 'cd /work/llama.cpp/build-android && ninja htp-v73'

echo "==> building backbone/depth stage"
# libc++ is linked statically so the binary does not depend on the device's
# libc++_shared.so version. The ring transport is compiled out by default: it
# costs ~40% on the pipe path even when unused (see m5-tts-port-notes.md).
docker exec "$CONTAINER" bash -lc 'CC=$(ls $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android28-clang++ | head -1)
    cd /work/llama.cpp
    for variant in "" "-DM5_STAGE_RING=1"; do
        out=/work/stage$( [ -n "$variant" ] && echo -ring )
        $CC -std=c++17 -O3 -static-libstdc++ -march=armv8.2-a+fp16+fp16fml $variant -o "$out" \
            examples/m5-tts/backbone.cpp -Iinclude -Iggml/include -Iggml/src \
            -Iexamples/m5-tts \
            -Lbuild-android/bin -lllama -lggml -lggml-base -lggml-cpu -lggml-hexagon \
            -landroid -llog -pthread
    done'

docker cp "$CONTAINER":/work/stage "$OUT/backbone-resident"
docker cp "$CONTAINER":/work/stage-ring "$OUT/backbone-ring"
for lib in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so; do
    docker cp "$CONTAINER":/work/llama.cpp/build-android/bin/$lib "$OUT/$lib"
done
docker cp "$CONTAINER":/work/llama.cpp/build-android/ggml/src/ggml-hexagon/libggml-htp-v73.so "$OUT/"

cat <<EOF

Artifacts in $OUT
Copy them next to the model package on the device, keeping depth-resident a
second copy of backbone-resident (the binary picks its role from argv):

  adb push $OUT/backbone-resident /data/local/tmp/llama-m5-htp-v73/
  adb shell 'cd /data/local/tmp/llama-m5-htp-v73 && cp backbone-resident depth-resident && chmod 755 backbone-resident depth-resident'
EOF
