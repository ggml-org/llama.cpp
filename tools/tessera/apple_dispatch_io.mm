// Apple dispatch_io_t bridge for the per-layer mmap (Phase 16.5,
// memopt-metal-dispatch).
//
// The legacy ``CalibPipeline`` in
// ``tools/tessera/calibration_memory.py`` uses a Python
// producer thread that calls ``np.load(mmap_mode="r")``
// synchronously.  On macOS the replacement uses libdispatch's
// ``dispatch_io_t`` to issue the read on a background GCD
// queue; the consumer's compute overlaps with the next layer's
// read.  This is the same producer/consumer pattern as the
// threaded path, but the I/O scheduler is GCD instead of the
// Python thread pool.
//
// We expose a single extern "C" entry point:
//   - tessera_dispatch_read_file(path, callback, user):
//     queues a dispatch_io_create_with_path + dispatch_io_read
//     on a process-wide GCD queue, and invokes the callback
//     with the file's bytes when the read completes.
//   - The callback runs on the GCD queue; the Python wrapper
//     is responsible for delivering the bytes back to the
//     consumer thread (via a queue.Queue).
//
// Why not use ``dispatch_io_read`` directly: the Python
// ctypes bridge can't safely retain a dispatch_data_t across
// the FFI boundary.  We allocate a buffer on the heap inside
// the bridge, copy the bytes, and hand the buffer pointer to
// the callback; the callback is responsible for freeing the
// buffer (via ``tessera_dispatch_free_buffer``).  The Python
// wrapper hands the buffer to numpy (as a memory view), then
// frees it after the numpy array is constructed.

#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The callback signature.  ``data`` is a heap-allocated
// buffer; the callback owns it after the bridge returns
// and is responsible for freeing via
// ``tessera_dispatch_free_buffer``.  ``error`` is the
// dispatch_io error code (0 for success, non-zero for
// failure); on error ``data`` is NULL and ``size`` is 0
// (the caller does NOT free a NULL buffer).  On success
// with empty data ``data`` is NULL, ``size`` is 0, and
// ``error`` is 0; the caller treats that as an empty file
// (success, not error).  The empty-file case used to be
// indistinguishable from a real error (the C bridge passed
// (NULL, 0) for both); carrying the error code through
// is the fix.
typedef void (*tessera_dispatch_read_callback_t)(
        const char * data,
        std::size_t size,
        int error,
        void * user);

// Forward declaration of the free function.
extern "C" void tessera_dispatch_free_buffer(char * data);

// A small struct that holds the path + queue + counter for
// the read completion.  The dispatch_io_t API delivers the
// data in chunks; we accumulate into a single buffer.
typedef struct {
    char * path;                  // strdup'd path
    char * buf;                   // accumulated buffer (heap-allocated)
    std::size_t size;             // bytes accumulated
    std::size_t cap;              // buffer capacity
    tessera_dispatch_read_callback_t callback;
    void * user;
} tessera_dispatch_state_t;

static void tessera_dispatch_state_free(tessera_dispatch_state_t * st) {
    if (!st) return;
    if (st->path) std::free(st->path);
    if (st->buf) std::free(st->buf);
    std::free(st);
}

// Process-wide GCD queue for the I/O.  One serial queue is
// enough: the pipeline issues at most a handful of
// concurrent reads (depth <= 4 in the test harness) and
// the GCD scheduler serialises them internally.  The
// callback runs on this queue; the Python wrapper
// marshals the result to a thread-safe queue.Queue.
static dispatch_queue_t g_io_queue = NULL;

static void tessera_dispatch_ensure_queue(void) {
    static dispatch_once_t once = 0;
    dispatch_once(&once, ^{
        // Concurrent queue with a small width; matches the
        // default thread pool width.  The I/O completion
        // handler runs on this queue, so the callback is
        // already off the producer thread.
        g_io_queue = dispatch_queue_create(
            "tessera.calibration.io",
            DISPATCH_QUEUE_CONCURRENT);
    });
}

// Append ``chunk`` of size ``chunk_size`` to ``st->buf``.
// Reallocates the buffer as needed.  Returns 0 on success,
// -1 on allocation failure.
static int tessera_dispatch_append(
        tessera_dispatch_state_t * st,
        const void * chunk,
        std::size_t chunk_size) {
    if (chunk_size == 0) return 0;
    std::size_t need = st->size + chunk_size;
    if (need > st->cap) {
        std::size_t new_cap = st->cap == 0 ? 64 * 1024 : st->cap;
        while (new_cap < need) new_cap *= 2;
        char * new_buf = (char *)std::realloc(st->buf, new_cap);
        if (!new_buf) return -1;
        st->buf = new_buf;
        st->cap = new_cap;
    }
    std::memcpy(st->buf + st->size, chunk, chunk_size);
    st->size = need;
    return 0;
}

// The per-chunk completion handler.  We accumulate the
// chunks; when ``done`` is true we hand the buffer to the
// callback.  The ``error`` parameter is the GCD error code
// (0 for success including empty data, non-zero for failure).
static void tessera_dispatch_read_handler(
        bool done,
        dispatch_data_t data,
        int error,
        tessera_dispatch_state_t * st) {
    if (error != 0) {
        // I/O error.  Free the state and notify the
        // callback with NULL/0/error so the Python wrapper
        // can surface the error.  Pass the error code
        // through so the wrapper can distinguish a real
        // error from success-with-empty (which is also
        // (NULL, 0)).
        if (st->buf) {
            std::free(st->buf);
            st->buf = NULL;
        }
        st->size = 0;
        st->callback(NULL, 0, error, st->user);
        tessera_dispatch_state_free(st);
        return;
    }
    if (data) {
        const void * bytes = NULL;
        std::size_t bytes_size = 0;
        dispatch_data_t mapped = dispatch_data_create_map(
            data, &bytes, &bytes_size);
        if (mapped) {
            if (tessera_dispatch_append(
                    st, bytes, bytes_size) != 0) {
                // Allocation failure: report as error
                // (use a synthetic non-zero error code that
                // the Python side does not need to
                // interpret; the existence of a non-zero
                // value is what matters).
                if (st->buf) {
                    std::free(st->buf);
                    st->buf = NULL;
                }
                st->size = 0;
                st->callback(NULL, 0, ENOMEM, st->user);
                tessera_dispatch_state_free(st);
                return;
            }
        }
    }
    if (done) {
        // Hand the buffer to the callback.  The callback
        // owns the buffer after this point.  Pass
        // error=0 (success) so the wrapper can distinguish
        // success-with-empty from a real error.
        char * result_buf = st->buf;
        std::size_t result_size = st->size;
        st->buf = NULL;
        st->size = 0;
        st->callback(result_buf, result_size, 0, st->user);
        tessera_dispatch_state_free(st);
    }
}

extern "C" int tessera_dispatch_read_file(
        const char * path,
        tessera_dispatch_read_callback_t callback,
        void * user) {
    if (!path || !callback) return -1;
    tessera_dispatch_ensure_queue();
    if (!g_io_queue) return -2;
    tessera_dispatch_state_t * st = (tessera_dispatch_state_t *)
        std::calloc(1, sizeof(tessera_dispatch_state_t));
    if (!st) return -3;
    st->path = strdup(path);
    if (!st->path) {
        std::free(st);
        return -3;
    }
    st->callback = callback;
    st->user = user;
    dispatch_io_t channel = dispatch_io_create_with_path(
        DISPATCH_IO_STREAM,
        path,
        O_RDONLY,
        0,
        g_io_queue,
        ^(int cleanup_error) {
            (void)cleanup_error;
        });
    if (!channel) {
        tessera_dispatch_state_free(st);
        return -4;
    }
    // Read the entire file: SIZE_MAX.  The per-layer .npz
    // bundles are bounded (a few hundred MB at the 12B
    // shape), so the full read fits in the host's RAM.
    // The async path overlaps this with the previous
    // layer's compute; the OS pages the bytes in the
    // background.
    dispatch_io_read(
        channel,
        0,
        SIZE_MAX,
        g_io_queue,
        ^(bool done, dispatch_data_t data, int error) {
            tessera_dispatch_read_handler(done, data, error, st);
        });
    return 0;
}

extern "C" void tessera_dispatch_free_buffer(char * data) {
    if (data) std::free(data);
}
