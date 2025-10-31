#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="Llama-3.2-3B-Instruct-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=""
[ "$V" != "" ] && verbose="$V"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

set -x

adb $adbserial shell " \
  cd $basedir;         \
  LD_LIBRARY_PATH=$basedir/$branch/lib   \
  ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $ndev $nhvx $opmask ./$branch/bin/llama-bench --device $device --mmap 0 -m $basedir/../gguf/$model \
        --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 \
        --batch-size 128 -ngl 99 $@ \
"
