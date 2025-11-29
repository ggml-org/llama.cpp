#!/bin/bash

if [ "$1" = "quiet" ]; then
    quiet_confpthreads="quiet"
else
    quiet_confpthreads=""
fi

if [ "$quiet_confpthreads" = "" ]; then
    echo "##########################################################"
    echo "Configuring BLIS for Altra using pThreads for parallelism..."
    echo "##########################################################"
fi

. ./blis_setenv.sh $quiet_confpthreads
pushd $BLIS_HOME > /dev/null
make distclean
./configure -t pthreads --disable-pba-pools altra
popd > /dev/null

