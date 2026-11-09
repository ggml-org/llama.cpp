#pragma once

// builders for ANE programs: MIL text and the Core ML single-file weight blob

#include <cstdint>
#include <string>
#include <vector>

using ggml_ane_shape = std::vector<int64_t>;

// weights/weight.bin: 64 B header (u32 chunk count, u32 version = 2), then per chunk a 64 B header + 64 B aligned data
struct ggml_ane_blob {
    enum dtype : uint32_t {
        DT_FP16 = 0x01,
        DT_INT8 = 0x04,
    };

    std::vector<uint8_t> data;
    uint64_t             end;   // total size, also tracked in dry mode
    bool                 dry;   // only compute offsets, do not store data
    std::vector<std::pair<std::string, uint64_t>> chunks; // (key, offset) of every chunk; equal keys = equal content

    explicit ggml_ane_blob(bool dry = false);

    // returns the chunk offset to use in BLOBFILE(..., offset=...); src is ignored in dry mode
    // key identifies the content (e.g. a weight tensor) so that programs can share one blob; "" = not shareable
    uint64_t add(const void * src, size_t size, dtype dt, const std::string & key = "");
};

struct ggml_ane_mil {
    struct value {
        std::string    name;
        std::string    type; // "fp16" or "fp32"
        ggml_ane_shape shape;
    };

    std::vector<value> inputs;
    std::vector<value> outputs;
    std::string        body;

    value input (const std::string & name, const std::string & type, const ggml_ane_shape & shape);
    void  output(const value & v);

    // emit "tensor<type, shape> name = expr[name=...]" and return the new value
    value op(const std::string & type, const ggml_ane_shape & shape, const std::string & expr, const std::string & name = "");

    std::string const_i32(const std::vector<int64_t> & vals);
    std::string const_scalar(const char * type, const std::string & literal);

    // fp16 tensor stored in the weight blob
    std::string const_blob(const ggml_ane_shape & shape, uint64_t offset);

    // conv weight [oc, ic, 1, 1] from int8 data + one fp16 scale per output row
    std::string const_int8_rows(int64_t oc, int64_t ic, uint64_t data_offset, uint64_t scale_offset);

    std::string text() const;

    // one function of a program; programs with several functions: program_header() + functions + program_footer()
    std::string func_text(const std::string & name) const;
    static std::string program_header();
    static std::string program_footer();

private:
    int n_names = 0;
    std::string new_name(const char * prefix);
};

std::string ggml_ane_shape_str(const ggml_ane_shape & shape);
