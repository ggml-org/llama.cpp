#!/bin/bash

cd ~/blis/testsuite 
rm -rf test_libblis.out
make clean
make -j
./test_libblis.x > test_libblis.out
echo "TEST DONE"
