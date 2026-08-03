// Implementation of the unified spec_calib.v3 telemetry record serializer
// and the v1-compat adapter. See telemetry-record.h for the schema contract.

#include "telemetry-record.h"

#include <cstdio>
#include <utility>

namespace spec_calib {

namespace {

// Append a single JSON string (number) with the given precision. Used for
// floats; integers use the std::to_string appender in the caller.
void append_float(std::string & out, double value, const char * fmt) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), fmt, value);
    out += buf;
}

// Append a single JSON number (integer or formatted float).
void append_int(std::string & out, int32_t value) {
    out += std::to_string(value);
}

// Append a parallel-arrays topk entry: [tok0, tok1, ...] / [p0, p1, ...]
// for the verifier and drafter, nested in a per-position array.
void append_topk_tokens(std::string & out, const std::vector<topk_entry> & entries, const char * field_name) {
    out += ",\"";
    out += field_name;
    out += "\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) out += ",";
        out += "[";
        for (size_t k = 0; k < entries[i].tokens.size(); ++k) {
            if (k > 0) out += ",";
            out += std::to_string(entries[i].tokens[k]);
        }
        out += "]";
    }
    out += "]";
}

void append_topk_probs(std::string & out, const std::vector<topk_entry> & entries, const char * field_name) {
    out += ",\"";
    out += field_name;
    out += "\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) out += ",";
        out += "[";
        for (size_t k = 0; k < entries[i].probs.size(); ++k) {
            if (k > 0) out += ",";
            append_float(out, (double) entries[i].probs[k], "%.6g");
        }
        out += "]";
    }
    out += "]";
}

void append_int_array(std::string & out, const char * field_name, const std::vector<int32_t> & values) {
    out += ",\"";
    out += field_name;
    out += "\":[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ",";
        out += std::to_string(values[i]);
    }
    out += "]";
}

}  // namespace

std::string build_telemetry_jsonl(const telemetry_record & rec, int topk, bool v1_compat) {
    std::string line;

    if (v1_compat) {
        // Legacy adapter. Only seq_id, drafted, accepted, confidence[].
        // All other fields on `rec` are ignored. Schema name is the
        // original v1 string so existing consumers keep working.
        line.reserve(64 + 24 * std::max<size_t>(rec.confidence.size(), 1));
        line  = "{\"schema\":\"";
        line += SCHEMA_V1_COMPAT;
        line += "\",\"seq_id\":";
        append_int(line, rec.seq_id);
        line += ",\"drafted\":";
        append_int(line, rec.drafted);
        line += ",\"accepted\":";
        append_int(line, rec.accepted);
        line += ",\"confidence\":[";
        for (size_t i = 0; i < rec.confidence.size(); ++i) {
            if (i > 0) line += ",";
            append_float(line, (double) rec.confidence[i], "%.8g");
        }
        line += "]}\n";
        return line;
    }

    // v3: strict superset of v1 + v2. Always includes confidence[] and the
    // cheap v2 token arrays. Top-k fields only when topk > 0.
    line.reserve(256 + 48 * (rec.drafted_tokens.size() + rec.accepted_tokens.size() +
                              rec.confidence.size() +
                              (topk > 0 ? (rec.verifier_topk.size() + rec.drafter_topk.size()) * 8 : 0)));

    line  = "{\"schema\":\"";
    line += SCHEMA_V3;
    line += "\",\"seq_id\":";
    append_int(line, rec.seq_id);
    line += ",\"step_idx\":";
    append_int(line, rec.step_idx);
    line += ",\"prime_token\":";
    append_int(line, rec.prime_token);
    line += ",\"drafted\":";
    append_int(line, rec.drafted);
    line += ",\"accepted\":";
    append_int(line, rec.accepted);

    append_int_array(line, "drafted_tokens",   rec.drafted_tokens);
    append_int_array(line, "accepted_tokens",  rec.accepted_tokens);

    line += ",\"confidence\":[";
    for (size_t i = 0; i < rec.confidence.size(); ++i) {
        if (i > 0) line += ",";
        append_float(line, (double) rec.confidence[i], "%.8g");
    }
    line += "]";

    if (topk > 0) {
        line += ",\"topk\":";
        append_int(line, topk);
        append_int_array(line, "verifier_argmax", rec.verifier_argmax);
        append_int_array(line, "drafter_argmax",  rec.drafter_argmax);
        append_topk_tokens(line, rec.verifier_topk, "verifier_topk_tokens");
        append_topk_probs (line, rec.verifier_topk, "verifier_topk_probs");
        append_topk_tokens(line, rec.drafter_topk,  "drafter_topk_tokens");
        append_topk_probs (line, rec.drafter_topk,  "drafter_topk_probs");
    }

    line += "}\n";
    return line;
}

}  // namespace spec_calib
