#!/bin/bash
echo "##########################################################"
echo "Creating both OpenMP and pThread BLIS libraries..."
echo "##########################################################"
echo "First, Creating pThread library..."
echo "##########################################################"
. ./blis_build_altra_pthreads.sh notest

echo "##########################################################"
echo "Saving the pThreads build..."
echo "##########################################################"
# Temporarily move the pthreads build
mkdir $BLIS_HOME/.tempinc
mkdir $BLIS_HOME/.templib
mv $BLIS_INC/* $BLIS_HOME/.tempinc/
mv $BLIS_LIB/* $BLIS_HOME/.templib/
# And rename the pthread versions of the include and library files
#echo "##########################################################"
pushd $BLIS_HOME/.tempinc/ > /dev/null
echo "Renaming pThread-enabled blis.h -> blisP.h"
mv blis.h blisP.h
popd > /dev/null
pushd $BLIS_HOME/.templib/ > /dev/null
for f in $(ls -1); do
    destf=${f/blis/blisP}
    echo "Renaming pThread library $f -> $destf"
    mv "$f" "$destf"

    # Fix the symbolic links
    if [[ -L "$destf" ]]; then
        target=$(readlink $destf)
        target=${target/blis/blisP}
        \rm "$destf"
        ln -s "$target" "$destf"
    fi
done
popd > /dev/null
echo "##########################################################"

echo "##########################################################"
echo "Second, Creating OpenMP library..."
echo "##########################################################"
. ./blis_build_altra.sh notest

echo "##########################################################"
echo "Restoring the pThreads build..."
echo "##########################################################"
# And move the pthread versions back
mv $BLIS_HOME/.tempinc/*  $BLIS_INC/
mv $BLIS_HOME/.templib/* $BLIS_LIB/
rmdir $BLIS_HOME/.tempinc
rmdir $BLIS_HOME/.templib

. ./blis_test.sh quiet
. ./blis_setenv.sh
echo "##########################################################"
echo "Done creating BLIS libraries..."
echo "##########################################################"
