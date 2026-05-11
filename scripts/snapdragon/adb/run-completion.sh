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

model="Llama-3.2-3B-Instruct-Q4_0.gguf"
[ "$M" != "" ] && model="$M"

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V" cli_opts="$cli_opts -v"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF" cli_opts="$cli_opts -v"

opmask=
[ "$OPSTAGE" != "" ] && opmask="GGML_HEXAGON_OPSTAGE=$OPSTAGE"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

hmx=
[ "$HMX" != "" ] && hmx="GGML_HEXAGON_USE_HMX=$HMX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

hb=
[ "$HB" != "" ] && hb="GGML_HEXAGON_HOSTBUF=$HB"

opbatch=
[ "$OB" != "" ] && opbatch="GGML_HEXAGON_OPBATCH=$OB"

opqueue=
[ "$OQ" != "" ] && opqueue="GGML_HEXAGON_OPQUEUE=$OQ"

opflt=
[ "$OF" != "" ] && opflt="GGML_HEXAGON_OPFILTER=$OF"

vmem=
[ "$VM" != "" ] && vmem="GGML_HEXAGON_VMEM=$VM"

mbuf=
[ "$MB" != "" ] && mbuf="GGML_HEXAGON_MBUF=$MB"

set -x

adb $adbserial $adbhost shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $sched $opmask $profile $nhvx $hmx $ndev $hb $opbatch $opqueue $opflt $vmem $mbuf \
      ./$branch/bin/llama-completion --no-mmap -m $basedir/../gguf/$model \
         --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1                  \
         --ctx-size 8192 --ubatch-size 256 -fa on                         \
         -ngl 99 -no-cnv --device $device $cli_opts $@                    \
"
