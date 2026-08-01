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

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF"

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

oppoll=
[ "$OP" != "" ] && oppoll="GGML_HEXAGON_OPPOLL=$OP"

opfuse=
[ "$OC" != "" ] && opfuse="GGML_HEXAGON_OPFUSION=$OC"

mmsel=
[ "$MM" != "" ] && mmsel="GGML_HEXAGON_MM_SELECT=$MM"

fasel=
[ "$FA" != "" ] && fasel="GGML_HEXAGON_FA_SELECT=$FA"

set -x

tool=$1; shift

adb $adbserial $adbhost shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $verbose $sched $opmask $profile $nhvx $hmx $ndev $hb $opbatch $opqueue $oppoll $opfuse $mmsel $fasel ./$branch/bin/$tool $@ \
"
