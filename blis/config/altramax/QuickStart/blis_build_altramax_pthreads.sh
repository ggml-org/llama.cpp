#!/bin/bash
echo "#######################################################"
echo "Building pThreads version of BLIS..."
echo "#######################################################"
. ./blis_setenv.sh quiet
echo "###############################################################"
echo "Configuring BLIS for Altramax using pThreads for parallelism..."
echo "###############################################################"
. ./blis_configure_altramax_pthreads.sh quiet
echo "Switching to directory $BLIS_HOME"
pushd $BLIS_HOME > /dev/null
make -j
popd > /dev/null
if [ "$1" != "notest" ]; then
    . ./blis_test.sh quiet
fi
. ./blis_setenv.sh
echo "##########################################################"
echo "...done"
echo "##########################################################"
