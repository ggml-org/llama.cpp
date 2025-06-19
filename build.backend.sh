# force isatty-->true, so that $0 |& head -50 has colors ...
rm -f READY_backend FAILED_backend

echo "int isatty(int fd) { return 1; }" | gcc -O2 -fpic -shared -ldl -o /tmp/isatty.so -xc -
export LD_PRELOAD=/tmp/isatty.so

if [[ "${PERF_MODE:-}" ]]; then
    FLAVOR="-prod"
else
    FLAVOR=""
fi

export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)

if [[ "$FLAVOR" == "-prod" ]]; then
    cat <<EOF
###
### Building the prod flavor
###
EOF
fi

TARGETS="llama-run"
if [[ "${BENCH_MODE:-}" == "bench" ]]; then
    TARGETS="$TARGETS llama-bench"
elif [[ "${BENCH_MODE:-}" == "perf" ]]; then
    TARGETS="$TARGETS test-backend-ops"
fi

cmake --build ../build.remoting-backend$FLAVOR --parallel 8 --target $TARGETS "$@"

if [[ $? == 0 ]]; then
    touch READY_backend
else
    touch FAILED_backend
fi
