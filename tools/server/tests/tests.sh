#!/usr/bin/env bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eu

WORKERS="${PYTEST_WORKERS:-auto}"

if [ $# -lt 1 ]
then
    if [[ "${SLOW_TESTS:-0}" == 1 ]]; then
        # --dist=loadfile means that all tests in the same file will be sent to the same worker.
        pytest --durations=30 -v -x -n "${WORKERS}" --dist=loadfile
    else
        pytest --durations=30 -v -x -n "${WORKERS}" --dist=loadfile -m "not slow"
    fi
else
    pytest --durations=30 -n "${WORKERS}" --dist=loadfile "$@"
fi
