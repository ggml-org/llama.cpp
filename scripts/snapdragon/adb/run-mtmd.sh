#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

adbhost=
[ "$H" != "" ] && adbhost="-H $H"

model="gemma-3-4b-it-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

mmproj="mmproj-F16.gguf"
[ "$MMPROJ" != "" ] && mmproj="$MMPROJ"

image=
[ "$IMG" != "" ] && image="$IMG"

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

experimental="GGML_HEXAGON_EXPERIMENTAL=1"
[ "$E" != "" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

# MTMD backend device for vision model (defaults to CPU if not set)
mtmd_backend=
[ "$MTMD_DEVICE" != "" ] && mtmd_backend="MTMD_BACKEND_DEVICE=$MTMD_DEVICE"

set -x

adb $adbserial $adbhost shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $experimental $sched $opmask $profile $nhvx $ndev $mtmd_backend \
      ./$branch/bin/llama-mtmd-cli --no-mmap -m $basedir/../gguf/$model      \
         --mmproj $basedir/../gguf/$mmproj                                   \
         --image $basedir/../gguf/$image                                     \
         --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1                     \
         --ctx-size 8192 --ubatch-size 256 -fa on                            \
         -ngl 99 --device $device -v $cli_opts $@                            \
"
