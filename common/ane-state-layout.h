// ane-state-layout.h — runtime-side reader for the ane_state_layout.v1
// manifest emitted by tools/ane-mtp/state_layout.py.
//
// This is the C++ wrapper around the C struct in ane-state.h. The C
// struct is the deserialized view of the JSON sidecar; the reader
// here does the deserialization (uses NSJSONSerialization, already in
// Foundation) and validation. The runtime (ggml/src/ggml-ane/ggml-ane.mm
// and the multifunction common/ane-mtp.mm) calls read_state_layout()
// after materializing the .mlmodelc to:
//
//   1. Allocate one state IOSurface of layout.state_size_bytes.
//   2. Pin every declared slot at its manifest offset as an
//      MLMultiArray wrapping the IOSurface with deallocator:nil.
//   3. Verify the bundle's function table matches what the .mlmodelc
//      exposes (input/output names and dtypes).
//
// The reader is header-only so it can be inlined from either
// ggml-ane or common/ane-mtp without a separate translation unit.
// The multifunction common/ane-mtp.mm will adopt this once the
// prefill / MTP / DFlash exporters emit manifests; the W0/W1 ggml
// backend already uses it (see ggml_backend_ane_program_load in
// ggml/src/ggml-ane/ggml-ane.mm for the in-tree reference).
//
// Refs:
//   common/ane-state.h: the C struct
//   tools/ane-mtp/state_layout.py: the JSON side
//   tools/ane-mtp/test_state_layout.py: the 24-test contract

#pragma once

#include "ane-state.h"

#include <cstring>
#include <string>

#import <Foundation/Foundation.h>

namespace ane_layout {

// Read a manifest JSON file into the C struct. Returns true on
// success. On failure, fills `error_out` (if non-null) with a
// human-readable reason and returns false; the caller must not
// use the layout. The reader is strict: unknown fields are
// rejected, missing required fields are rejected, and the
// version must be exactly ANE_STATE_LAYOUT_VERSION.
inline bool read_state_layout(const char * path,
                              ane_state_layout_v1_t * layout,
                              std::string * error_out = nullptr) {
    if (path == nullptr || layout == nullptr) {
        if (error_out != nullptr) {
            *error_out = "null path or layout";
        }
        return false;
    }
    std::memset(layout, 0, sizeof(*layout));
    NSError * error = nil;
    NSString * ns_path = [NSString stringWithUTF8String:path];
    NSData * data = [NSData dataWithContentsOfFile:ns_path];
    if (data == nil) {
        if (error_out != nullptr) {
            *error_out = std::string("manifest not found: ") + path;
        }
        return false;
    }
    id obj = [NSJSONSerialization JSONObjectWithData:data
                                              options:0
                                                error:&error];
    if (obj == nil || ![obj isKindOfClass:[NSDictionary class]]) {
        if (error_out != nullptr) {
            *error_out = std::string("manifest is not a JSON object: ") +
                (error != nil ? error.localizedDescription.UTF8String
                              : "unknown");
        }
        return false;
    }
    NSDictionary * root = (NSDictionary *) obj;

    // version (required, must match)
    NSNumber * version = root[@"version"];
    if (version == nil || version.unsignedIntValue != ANE_STATE_LAYOUT_VERSION) {
        if (error_out != nullptr) {
            *error_out = "manifest version mismatch (got " +
                (version != nil ? std::string(version.stringValue.UTF8String)
                                 : std::string("missing")) +
                ", expected " + std::to_string(ANE_STATE_LAYOUT_VERSION) + ")";
        }
        return false;
    }
    layout->version = version.unsignedIntValue;

    // bundle_name
    NSString * bundle_name = root[@"bundle_name"];
    if (bundle_name == nil) {
        if (error_out != nullptr) {
            *error_out = "manifest missing bundle_name";
        }
        return false;
    }
    const char * bn = bundle_name.UTF8String;
    if (bn == nullptr) {
        if (error_out != nullptr) {
            *error_out = "manifest bundle_name is not UTF-8";
        }
        return false;
    }
    std::strncpy(layout->bundle_name, bn, sizeof(layout->bundle_name) - 1);

    // state_size_bytes
    NSNumber * state_size = root[@"state_size_bytes"];
    if (state_size == nil) {
        if (error_out != nullptr) {
            *error_out = "manifest missing state_size_bytes";
        }
        return false;
    }
    layout->state_size_bytes = (size_t) state_size.unsignedLongLongValue;

    // model_type (default neural_network for back-compat)
    NSString * mt = root[@"model_type"];
    if (mt == nil || [mt isEqualToString:@"neural_network"]) {
        layout->model_type = ANE_MODEL_TYPE_NEURAL_NETWORK;
    } else if ([mt isEqualToString:@"ml_program"]) {
        layout->model_type = ANE_MODEL_TYPE_ML_PROGRAM;
    } else {
        if (error_out != nullptr) {
            *error_out = std::string("manifest model_type ") +
                (mt.UTF8String ?: "nil") + " unknown";
        }
        return false;
    }

    // slots
    NSArray * slots_arr = root[@"slots"];
    if (slots_arr == nil || ![slots_arr isKindOfClass:[NSArray class]]) {
        if (error_out != nullptr) {
            *error_out = "manifest missing or bad slots";
        }
        return false;
    }
    if (slots_arr.count > ANE_STATE_SLOTS_MAX) {
        if (error_out != nullptr) {
            *error_out = "manifest has " + std::to_string(slots_arr.count) +
                " slots, max is " + std::to_string(ANE_STATE_SLOTS_MAX);
        }
        return false;
    }
    for (NSUInteger i = 0; i < slots_arr.count; ++i) {
        NSDictionary * s = slots_arr[i];
        ane_slot_v1_t * out = &layout->slots[i];
        std::memset(out, 0, sizeof(*out));
        NSString * name = s[@"name"];
        if (name == nullptr) {
            if (error_out != nullptr) {
                *error_out = "slot " + std::to_string(i) + " has no name";
            }
            return false;
        }
        std::strncpy(out->name, name.UTF8String, sizeof(out->name) - 1);
        NSString * kind = s[@"kind"];
        if ([kind isEqualToString:@"input"])        out->kind = ANE_SLOT_KIND_INPUT;
        else if ([kind isEqualToString:@"output"])  out->kind = ANE_SLOT_KIND_OUTPUT;
        else if ([kind isEqualToString:@"state"])   out->kind = ANE_SLOT_KIND_STATE;
        else if ([kind isEqualToString:@"scratch"]) out->kind = ANE_SLOT_KIND_SCRATCH;
        else {
            if (error_out != nullptr) {
                *error_out = std::string("slot ") + out->name +
                    " has bad kind " + (kind.UTF8String ?: "nil");
            }
            return false;
        }
        NSString * dtype = s[@"dtype"];
        if ([dtype isEqualToString:@"f32"])      out->dtype = ANE_DTYPE_F32;
        else if ([dtype isEqualToString:@"f16"]) out->dtype = ANE_DTYPE_F16;
        else if ([dtype isEqualToString:@"i32"]) out->dtype = ANE_DTYPE_I32;
        else {
            if (error_out != nullptr) {
                *error_out = std::string("slot ") + out->name +
                    " has bad dtype " + (dtype.UTF8String ?: "nil");
            }
            return false;
        }
        NSArray * shape = s[@"shape"];
        if (shape == nil || shape.count < 1 || shape.count > 4) {
            if (error_out != nullptr) {
                *error_out = std::string("slot ") + out->name +
                    " shape must be 1-4 dims";
            }
            return false;
        }
        out->n_dim = (uint32_t) shape.count;
        for (NSUInteger j = 0; j < shape.count; ++j) {
            out->shape[j] = (uint32_t) [shape[j] unsignedIntValue];
        }
        out->offset = (size_t) [s[@"offset"] unsignedLongLongValue];
        out->size_bytes = (size_t) [s[@"size_bytes"] unsignedLongLongValue];
    }
    layout->n_slots = (uint32_t) slots_arr.count;

    // functions
    NSArray * funcs_arr = root[@"functions"];
    if (funcs_arr == nil || ![funcs_arr isKindOfClass:[NSArray class]]) {
        if (error_out != nullptr) {
            *error_out = "manifest missing or bad functions";
        }
        return false;
    }
    if (funcs_arr.count > ANE_STATE_FUNCTIONS_MAX) {
        if (error_out != nullptr) {
            *error_out = "manifest has " + std::to_string(funcs_arr.count) +
                " functions, max is " + std::to_string(ANE_STATE_FUNCTIONS_MAX);
        }
        return false;
    }
    for (NSUInteger i = 0; i < funcs_arr.count; ++i) {
        NSDictionary * f = funcs_arr[i];
        ane_function_v1_t * out = &layout->functions[i];
        std::memset(out, 0, sizeof(*out));
        NSString * name = f[@"name"];
        if (name == nullptr) {
            if (error_out != nullptr) {
                *error_out = "function " + std::to_string(i) + " has no name";
            }
            return false;
        }
        std::strncpy(out->name, name.UTF8String, sizeof(out->name) - 1);
        NSString * role = f[@"role"];
        if ([role isEqualToString:@"prefill"]) out->role = ANE_ROLE_PREFILL;
        else if ([role isEqualToString:@"mtp"])     out->role = ANE_ROLE_MTP;
        else if ([role isEqualToString:@"dflash"])  out->role = ANE_ROLE_DFLASH;
        else if ([role isEqualToString:@"hybrid"])  out->role = ANE_ROLE_HYBRID;
        else if ([role isEqualToString:@"sync"])    out->role = ANE_ROLE_SYNC;
        else if ([role isEqualToString:@"reset"])   out->role = ANE_ROLE_RESET;
        else if ([role isEqualToString:@"matmul"])  out->role = ANE_ROLE_MATMUL;
        else                                       out->role = ANE_ROLE_UNKNOWN;
        out->bucket = (uint32_t) [f[@"bucket"] unsignedIntValue];
        out->stateful = [f[@"stateful"] boolValue];
        out->use_ane = [f[@"use_ane"] boolValue];
        NSString * cm_name = f[@"core_ml_function_name"];
        if (cm_name != nullptr) {
            std::strncpy(out->core_ml_function_name, cm_name.UTF8String,
                         sizeof(out->core_ml_function_name) - 1);
        }
        NSArray * ins = f[@"input_slots"];
        if (ins != nil) {
            out->n_inputs = (uint32_t) std::min((NSUInteger) ANE_STATE_SLOT_IO_MAX,
                                                ins.count);
            for (NSUInteger j = 0; j < out->n_inputs; ++j) {
                NSString * sname = ins[j];
                uint32_t slot_id = UINT32_MAX;
                for (uint32_t k = 0; k < layout->n_slots; ++k) {
                    if (std::strcmp(layout->slots[k].name, sname.UTF8String) == 0) {
                        slot_id = k;
                        break;
                    }
                }
                if (slot_id == UINT32_MAX) {
                    if (error_out != nullptr) {
                        *error_out = std::string("function ") + out->name +
                            " references unknown input slot " +
                            (sname.UTF8String ?: "nil");
                    }
                    return false;
                }
                out->input_slot_ids[j] = slot_id;
            }
        }
        NSArray * outs = f[@"output_slots"];
        if (outs != nil) {
            out->n_outputs = (uint32_t) std::min((NSUInteger) ANE_STATE_SLOT_IO_MAX,
                                                 outs.count);
            for (NSUInteger j = 0; j < out->n_outputs; ++j) {
                NSString * sname = outs[j];
                uint32_t slot_id = UINT32_MAX;
                for (uint32_t k = 0; k < layout->n_slots; ++k) {
                    if (std::strcmp(layout->slots[k].name, sname.UTF8String) == 0) {
                        slot_id = k;
                        break;
                    }
                }
                if (slot_id == UINT32_MAX) {
                    if (error_out != nullptr) {
                        *error_out = std::string("function ") + out->name +
                            " references unknown output slot " +
                            (sname.UTF8String ?: "nil");
                    }
                    return false;
                }
                out->output_slot_ids[j] = slot_id;
            }
        }
    }
    layout->n_functions = (uint32_t) funcs_arr.count;

    // dependencies (best-effort; the E-core pump uses this to build
    // the per-slot lock-free state machine. The W0 case has none.)
    NSArray * deps_arr = root[@"dependencies"];
    if (deps_arr != nil && [deps_arr isKindOfClass:[NSArray class]]) {
        if (deps_arr.count > ANE_STATE_DEPS_MAX) {
            if (error_out != nullptr) {
                *error_out = "manifest has " + std::to_string(deps_arr.count) +
                    " deps, max is " + std::to_string(ANE_STATE_DEPS_MAX);
            }
            return false;
        }
        for (NSUInteger i = 0; i < deps_arr.count; ++i) {
            NSDictionary * d = deps_arr[i];
            ane_function_dep_t * out = &layout->deps[i];
            std::memset(out, 0, sizeof(*out));
            NSString * producer = d[@"producer"];
            if (producer == nullptr) {
                continue;
            }
            for (uint32_t k = 0; k < layout->n_functions; ++k) {
                if (std::strcmp(layout->functions[k].name, producer.UTF8String) == 0) {
                    out->producer_function_id = k;
                    break;
                }
            }
            NSString * slot = d[@"slot"];
            if (slot == nullptr) {
                continue;
            }
            for (uint32_t k = 0; k < layout->n_slots; ++k) {
                if (std::strcmp(layout->slots[k].name, slot.UTF8String) == 0) {
                    out->slot_id = k;
                    break;
                }
            }
            NSArray * consumers = d[@"consumers"];
            if (consumers != nil) {
                out->n_consumers = (uint32_t) std::min(
                    (NSUInteger) ANE_STATE_FUNCTIONS_MAX, consumers.count);
                for (NSUInteger j = 0; j < out->n_consumers; ++j) {
                    NSString * cn = consumers[j];
                    for (uint32_t k = 0; k < layout->n_functions; ++k) {
                        if (std::strcmp(layout->functions[k].name, cn.UTF8String) == 0) {
                            out->consumer_function_ids[j] = k;
                            break;
                        }
                    }
                }
            }
        }
        layout->n_deps = (uint32_t) deps_arr.count;
    }

    return true;
}

// Resolve the manifest path for a .mlmodelc directory. The convention
// is <bundle-stem>.ane_state.v1.json in the same directory as the
// .mlmodelc. The bundle stem is the .mlmodelc's directory name
// (e.g., "w0-256x256.mlmodelc" -> "w0-256x256").
inline std::string manifest_path_for_mlmodelc_dir(const char * mlmodelc_dir) {
    if (mlmodelc_dir == nullptr) {
        return std::string();
    }
    NSString * dir = [NSString stringWithUTF8String:mlmodelc_dir];
    NSString * dir_name = [dir lastPathComponent];
    NSString * parent = [dir stringByDeletingLastPathComponent];
    NSString * bundle_stem = [dir_name stringByDeletingPathExtension];
    NSString * manifest_path = [parent
        stringByAppendingPathComponent:
            [NSString stringWithFormat:@"%@.ane_state.v1.json", bundle_stem]];
    return std::string(manifest_path.UTF8String ?: "");
}

}  // namespace ane_layout
