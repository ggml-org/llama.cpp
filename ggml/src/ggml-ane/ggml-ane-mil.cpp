#include "ggml-ane-mil.h"

#include <cstring>

static void put_u32(uint8_t * p, uint32_t v) { memcpy(p, &v, 4); }
static void put_u64(uint8_t * p, uint64_t v) { memcpy(p, &v, 8); }

ggml_ane_blob::ggml_ane_blob(bool dry) : data(64, 0), end(64), dry(dry) {
    put_u32(data.data() + 4, 2);
}

uint64_t ggml_ane_blob::add(const void * src, size_t size, dtype dt, const std::string & key) {
    const uint64_t pos    = end;
    chunks.push_back({ key, pos });
    const size_t   padded = (size + 63) / 64 * 64;
    end += 64 + padded;
    if (dry) {
        return pos;
    }
    data.resize(end, 0);
    uint8_t * hdr = data.data() + pos;
    put_u32(hdr + 0, 0xDEADBEEF);
    put_u32(hdr + 4, dt);
    put_u64(hdr + 8, size);
    put_u64(hdr + 16, pos + 64);
    memcpy(hdr + 64, src, size);

    uint32_t n_chunks;
    memcpy(&n_chunks, data.data(), 4);
    put_u32(data.data(), n_chunks + 1);
    return pos;
}

std::string ggml_ane_shape_str(const ggml_ane_shape & shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        s += (i ? "," : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

static std::string tensor_type(const std::string & type, const ggml_ane_shape & shape) {
    return "tensor<" + type + ", " + ggml_ane_shape_str(shape) + ">";
}

static std::string blob_file(uint64_t offset) {
    return "BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(" + std::to_string(offset) + "))";
}

std::string ggml_ane_mil::new_name(const char * prefix) {
    return prefix + std::to_string(n_names++);
}

ggml_ane_mil::value ggml_ane_mil::input(const std::string & name, const std::string & type, const ggml_ane_shape & shape) {
    inputs.push_back({ name, type, shape });
    return inputs.back();
}

void ggml_ane_mil::output(const value & v) {
    outputs.push_back(v);
}

ggml_ane_mil::value ggml_ane_mil::op(const std::string & type, const ggml_ane_shape & shape, const std::string & expr, const std::string & name) {
    const std::string n = name.empty() ? new_name("t") : name;
    body += "        " + tensor_type(type, shape) + " " + n + " = " + expr + "[name=string(\"" + n + "\")];\n";
    return { n, type, shape };
}

std::string ggml_ane_mil::const_i32(const std::vector<int64_t> & vals) {
    const std::string n = new_name("c");
    ggml_ane_shape shape = { (int64_t) vals.size() };
    std::string v = ggml_ane_shape_str(vals);
    body += "        " + tensor_type("int32", shape) + " " + n + " = const()[name=string(\"" + n + "\"), val=" + tensor_type("int32", shape) + "(" + v + ")];\n";
    return n;
}

std::string ggml_ane_mil::const_scalar(const char * type, const std::string & literal) {
    const std::string n = new_name("c");
    body += "        " + std::string(type) + " " + n + " = const()[name=string(\"" + n + "\"), val=" + type + "(" + literal + ")];\n";
    return n;
}

std::string ggml_ane_mil::const_blob(const ggml_ane_shape & shape, uint64_t offset) {
    const std::string n = new_name("w");
    body += "        " + tensor_type("fp16", shape) + " " + n + " = const()[name=string(\"" + n + "\"), val=" + tensor_type("fp16", shape) + "(" + blob_file(offset) + ")];\n";
    return n;
}

std::string ggml_ane_mil::const_int8_rows(int64_t oc, int64_t ic, uint64_t data_offset, uint64_t scale_offset) {
    const ggml_ane_shape w_shape = { oc, ic, 1, 1 };
    const ggml_ane_shape s_shape = { oc, 1, 1, 1 };
    return op("fp16", w_shape, "constexpr_blockwise_shift_scale(data=" + tensor_type("int8", w_shape) + "(" + blob_file(data_offset) + "), " +
                               "scale=" + tensor_type("fp16", s_shape) + "(" + blob_file(scale_offset) + "))").name;
}

std::string ggml_ane_mil::program_header() {
    return "program(1.3)\n"
           "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, "
           "{\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
           "{\n";
}

std::string ggml_ane_mil::program_footer() {
    return "}\n";
}

std::string ggml_ane_mil::func_text(const std::string & name) const {
    std::string s = "    func " + name + "<ios18>(";
    for (size_t i = 0; i < inputs.size(); i++) {
        s += (i ? ", " : "") + tensor_type(inputs[i].type, inputs[i].shape) + " " + inputs[i].name;
    }
    s += ") {\n" + body + "    } -> (";
    for (size_t i = 0; i < outputs.size(); i++) {
        s += (i ? ", " : "") + outputs[i].name;
    }
    return s + ");\n";
}

std::string ggml_ane_mil::text() const {
    return program_header() + func_text("main") + program_footer();
}
