#!/usr/bin/env bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eu

WORKERS="${PYTEST_WORKERS:-4}"

if [ "${WORKERS}" -eq 1 ]; then
    WORKER_FLAGS=""
else
    WORKER_FLAGS="-n ${WORKERS} --dist=worksteal"
fi

if [ $# -lt 1 ]
then
    if [[ "${SLOW_TESTS:-0}" == 1 ]]; then
        pytest --durations=30 -v -x ${WORKER_FLAGS}
    else
        pytest --durations=30 -v -x ${WORKER_FLAGS} -m "not slow"
    fi
else
    pytest --durations=30 ${WORKER_FLAGS} "$@"
fi
