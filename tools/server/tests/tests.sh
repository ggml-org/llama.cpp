#!/usr/bin/env bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eu

if [ $# -lt 1 ]
then
    if [[ "${SLOW_TESTS:-0}" == 1 ]]; then
        pytest --durations=30 -v -x
    else
        pytest --durations=30 -v -x -m "not slow"
    fi
else
    pytest --durations=30 "$@"
fi
