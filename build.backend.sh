# force isatty-->true, so that $0 |& head -50 has colors ...
rm -f READY_backend FAILED_backend

echo "int isatty(int fd) { return 1; }" | gcc -O2 -fpic -shared -ldl -o /tmp/isatty.so -xc -
export LD_PRELOAD=/tmp/isatty.so

cmake --build ../build.remoting-backend --parallel 8 --target llama-cli "$@"

if [[ $? == 0 ]]; then
    touch READY_backend
else
    touch FAILED_backend
fi
