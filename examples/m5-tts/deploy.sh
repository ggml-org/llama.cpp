#!/usr/bin/env bash
# One-shot deploy of the M5-TTS server stack to an Android device over adb.
#
#   ./deploy.sh [device-dir] [--no-models]
#
# Pushes the engine/stage binaries and the web console, restarts m5-server
# with the HTP environment pinned (GGML_HEXAGON_* below is worth 2.2x - do
# NOT drop it), and (re)establishes the browser forward.
#
#   default device dir : /data/local/tmp/llama-m5-htp-v73
#   browser entry      : http://127.0.0.1:18080   (adb forward 18080 -> 8080)
#
# Local artifacts (override via env):
#   M5_BUILD   ~/m5-assets/build   m5-server, m5-engine, libm5engine.so
#   M5_STAGE   ~/m5-assets/stage   backbone-resident (== depth-resident)
#              plus the llama/ggml/hexagon .so files the stages link against
#   M5_WEB     examples/m5-tts/web/index.html -> <device-dir>/web.html
#
# The 32-wide KV pad and the direct V scatter are compiled into the stage
# libraries (no separate runtime dir / env injection). Override the depth
# command with M5_DEPTH_CMD='env M5_KV_PAD=256 ... depth-resident ...' for A/B.
#
# The device keeps the model package (model/ dir + codec_encoder.onnx) on its
# own; --no-models skips the backup pull. /data/local/tmp is wiped on device
# reboot, so re-run this script after one.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
VDIR=/data/local/tmp/llama-m5-htp-v73
SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --no-models) SKIP_MODELS=1 ;;
        --*) echo "unknown flag: $arg" >&2; exit 1 ;;
        *) VDIR=$arg ;;
    esac
done

M5_BUILD=${M5_BUILD:-$HOME/m5-assets/build}
M5_STAGE=${M5_STAGE:-$HOME/m5-assets/stage}
M5_WEB=${M5_WEB:-$HERE/web/index.html}
PORT=${PORT:-8080}
FORWARD_PORT=${FORWARD_PORT:-18080}
# Optional depth stage override for A/B. The stage binaries now carry the
# 32-wide KV pad and the direct V scatter as compiled-in defaults, so no
# separate runtime directory or env injection is needed.
M5_DEPTH_CMD=${M5_DEPTH_CMD:-}

ADB=${ADB:-$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")}
[ -x "$ADB" ] || { echo "adb not found; set ADB=" >&2; exit 1; }

M5_STAGE_LIBS="libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so libggml-htp-v73.so"

for f in "$M5_BUILD/m5-server" "$M5_BUILD/m5-engine" "$M5_STAGE/backbone-resident" "$M5_WEB"; do
    [ -f "$f" ] || { echo "missing artifact: $f" >&2; exit 1; }
done
for lib in $M5_STAGE_LIBS; do
    [ -f "$M5_STAGE/$lib" ] || { echo "missing stage library: $M5_STAGE/$lib" >&2; exit 1; }
done

echo "==> stopping old server and stages"
# pidof, not pkill -f: a pattern that matches the shell's own command line
# would kill the invoking shell (been there).
$ADB shell 'for p in $(pidof m5-server); do kill $p; done; sleep 2;
    for n in backbone-resident depth-resident; do
        for p in $(pidof $n 2>/dev/null); do kill $p; done
    done; sleep 1' || true

echo "==> device dir: $VDIR"
$ADB shell "mkdir -p $VDIR/model"

echo "==> pushing binaries"
$ADB push "$M5_BUILD/m5-server" "$VDIR/m5-server" >/dev/null
$ADB push "$M5_BUILD/m5-engine" "$VDIR/m5-engine" >/dev/null
$ADB push "$M5_BUILD/libm5engine.so" "$VDIR/libm5engine.so" >/dev/null
$ADB push "$M5_STAGE/backbone-resident" "$VDIR/backbone-resident" >/dev/null
echo "==> pushing Hexagon stage libraries"
for lib in $M5_STAGE_LIBS; do
    $ADB push "$M5_STAGE/$lib" "$VDIR/$lib" >/dev/null
done
# depth-resident is the same binary; the role comes from argv.
$ADB shell "cp $VDIR/backbone-resident $VDIR/depth-resident && chmod 755 $VDIR/m5-server $VDIR/m5-engine $VDIR/backbone-resident $VDIR/depth-resident"

echo "==> pushing web console"
$ADB push "$M5_WEB" "$VDIR/web.html" >/dev/null

if [ "$SKIP_MODELS" = 0 ]; then
    echo "==> backing up model package to ~/m5-assets/device-model-backup (skips files already there)"
    mkdir -p "$HOME/m5-assets/device-model-backup"
    for f in model/manifest.json model/voices.json model/text_embeddings.npy \
             model/audio_embeddings.npy model/tokenizer.bin model/depth-weights.f32 \
             codec_encoder.onnx codec_identity.json; do
        if [ ! -f "$HOME/m5-assets/device-model-backup/$(basename "$f")" ]; then
            $ADB pull "$VDIR/$f" "$HOME/m5-assets/device-model-backup/" >/dev/null 2>&1 \
                || echo "    (no $f on device - skipped)"
        fi
    done
fi

echo "==> restarting m5-server"
DEPTH_ARGS=""
if [ -n "$M5_DEPTH_CMD" ]; then
    DEPTH_ARGS="--depth '$M5_DEPTH_CMD'"
fi
# The GGML_HEXAGON_* knobs live in ggml/src/ggml-hexagon/ggml-hexagon.cpp:
# OPPOLL keeps the DSP clocks polled (measured 2.2x on depth), NHMX=0 avoids
# the HMX matmul path that is slower for the 512-wide heads on this silicon.
# M5_KV_PAD / M5_HVX_DIRECT_V are compiled-in defaults; prefix them in
# M5_DEPTH_CMD to override for A/B.
$ADB shell "cd $VDIR && D=\$PWD; \
    export LD_LIBRARY_PATH=\$D ADSP_LIBRARY_PATH=\$D \
           GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1; \
    nohup ./m5-server model --port $PORT --cwd \$D --threads 3 \
        $DEPTH_ARGS \
        --voice-encoder \$D/codec_encoder.onnx > \$D/m5-server.log 2>&1 &"
sleep 3
$ADB shell "pidof m5-server" >/dev/null || { echo "server did not start; see $VDIR/m5-server.log" >&2; exit 1; }

echo "==> browser forward tcp:$FORWARD_PORT -> tcp:$PORT"
$ADB forward --remove "tcp:$FORWARD_PORT" >/dev/null 2>&1 || true
$ADB forward "tcp:$FORWARD_PORT" "tcp:$PORT" >/dev/null

echo "==> health check"
curl -fsS --noproxy '*' "http://127.0.0.1:$FORWARD_PORT/health" >/dev/null && echo "ok"
cat <<EOF

m5-server is up.
  web console : http://127.0.0.1:$FORWARD_PORT
  log         : $VDIR/m5-server.log   (adb shell tail -f)
  env pinned  : GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
  compiled-in : KV pad 32 (depth) / 256 (backbone), direct V scatter
                (override with M5_KV_PAD / M5_HVX_DIRECT_V for A/B)
EOF
