#!/bin/bash
echo "#######################################################"
echo "Building standard OpenMP BLIS..."
echo "#######################################################"
. ./blis_setenv.sh quiet
echo "##########################################################"
echo "Configuring BLIS for Altra using OpenMP for parallelism..."
echo "##########################################################"
. ./blis_configure_altra.sh quiet
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
