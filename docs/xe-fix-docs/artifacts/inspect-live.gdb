set pagination off
set confirm off
attach 431199
add-symbol-file /mnt/nvme1/oneapi-ab/xe-investigation-20260929/trace-copies-debug.so -o 0x7c9381a8b000
python
f = gdb.newest_frame()
while f:
    if f.name() == "ggml_backend_tensor_set_async":
        try:
            t = f.read_var("tensor")
            print("LIVE_COPY name=" + t["name"].string())
            print("LIVE_COPY type=" + str(t["type"]))
            print("LIVE_COPY size=" + str(f.read_var("size")))
            print("LIVE_COPY offset=" + str(f.read_var("offset")))
            print("LIVE_COPY shape=" + str(t["ne"]))
            break
        except gdb.error:
            pass
    f = f.older()
else:
    print("LIVE_COPY metadata unavailable")
end
detach
quit
