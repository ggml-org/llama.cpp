#!/bin/bash

set -e
set -x

: ${SRCDIR:=../..}

if ! [ -d test/level0 ]; then
    mkdir -p test/level0
    ln -s $SRCDIR/test/level0/* test/level0/
fi

cd test/level0
make -j2

./test_l0.x
