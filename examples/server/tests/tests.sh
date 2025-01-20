#!/bin/bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eu

if [ $# -lt 1 ]
then
    pytest -v -x
else
    pytest "$@"
fi
