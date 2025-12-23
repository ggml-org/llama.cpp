#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="Llama-3.2-3B-Instruct-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

device="HTP0"
[ "$D" != "" ] && device="$D"

experimental=
[ "$E" != "" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V" cli_opts="$cli_opts -v"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1" cli_opts="$cli_opts -v"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

set -x

adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $experimental $sched $opmask $profile $nhvx $ndev            \
      ./$branch/bin/llama-completion --no-mmap -m $basedir/../gguf/$model \
         --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1                  \
         --ctx-size 8192 --batch-size 128 -fa on \
         -ngl 99 -no-cnv --device $device $cli_opts $@   \
"
