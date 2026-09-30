#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // register a callback to run after the staged tensor writes have been applied (at io teardown)
    virtual void on_commit(std::function<void()> callback) { commit_cbs.push_back(std::move(callback)); }

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);

protected:
    // run the registered on_commit callbacks (called by the concrete io after applying its writes)
    void run_commit_callbacks() {
        for (auto & cb : commit_cbs) {
            cb();
        }
        commit_cbs.clear();
    }

private:
    std::vector<std::function<void()>> commit_cbs;
};
