#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

adbhost=
[ "$H" != "" ] && adbhost="-H $H"

model="Llama-3.2-3B-Instruct-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF" cli_opts="$cli_opts -v"

opmask=
[ "$OPSTAGE" != "" ] && opmask="GGML_HEXAGON_OPSTAGE=$OPSTAGE"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

hb=
[ "$HB" != "" ] && hb="GGML_HEXAGON_HOSTBUF=$HB"

set -x

adb $adbserial $adbhost shell " \
  cd $basedir;         \
  LD_LIBRARY_PATH=$basedir/$branch/lib   \
  ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $ndev $nhvx $opmask $verbose $profile $hb ./$branch/bin/llama-bench --device $device --mmap 0 -m $basedir/../gguf/$model \
        --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 \
        --ubatch-size 256 -fa 1 -ngl 99 $cli_opts $@    \
"
