#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "usage: $0 M5_SOURCE ORT_ANDROID_AAR OUTPUT_DIR" >&2
    exit 2
fi
: "${JAVA_HOME:?Set JAVA_HOME to JDK 17}"
: "${ANDROID_HOME:?Set ANDROID_HOME to the Android SDK}"
source_dir="$1/android/app/src/main/java/com/m5/offline"
aar="$2"
output="$3"
here="$(cd "$(dirname "$0")" && pwd)"
android_jar="$ANDROID_HOME/platforms/android-35/android.jar"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/classes" "$output"
aar="$(cd "$(dirname "$aar")" && pwd)/$(basename "$aar")"
(cd "$scratch" && "$JAVA_HOME/bin/jar" xf "$aar" classes.jar jni/arm64-v8a)
"$JAVA_HOME/bin/javac" --release 11 -cp "$scratch/classes.jar:$android_jar" -d "$scratch/classes" \
    "$source_dir"/{M5Engine,NativeBackbone,NativeDepth,NpyMatrix,Tokenizer,WavWriter}.java \
    "$here/AndroidBenchmark.java" "$here/AndroidConvBenchmark.java"
"$JAVA_HOME/bin/jar" cf "$scratch/benchmark.jar" -C "$scratch/classes" .
"$ANDROID_HOME/build-tools/35.0.0/d8" --min-api 28 --lib "$android_jar" \
    --output "$output" "$scratch/benchmark.jar" "$scratch/classes.jar"
cp "$scratch/jni/arm64-v8a/"*.so "$output/"
