// CoreML backend implementation for mtmd. Pure Objective-C++; this file is
// the only place that touches CoreML. All model-specific logic (input
// packing, output decoding) lives in adapter cpp files under coreml/models/.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "backend.h"

#include <cstring>

namespace mtmd_coreml::backend {

void * load(const char * mlmodelc_path) {
    if (!mlmodelc_path) {
        return nullptr;
    }

    NSString *      path = [NSString stringWithUTF8String:mlmodelc_path];
    NSFileManager * fm   = [NSFileManager defaultManager];

    BOOL is_dir = NO;
    if (![fm fileExistsAtPath:path isDirectory:&is_dir]) {
        NSLog(@"[mtmd-coreml] model path does not exist: %@", path);
        return nullptr;
    }
    if (!is_dir && ![path hasSuffix:@".mlmodelc"]) {
        NSLog(@"[mtmd-coreml] warning: model path is not a .mlmodelc directory: %@", path);
    }

    NSURL * url = [NSURL fileURLWithPath:path];

    MLModelConfiguration * cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsAll;

    NSError * err   = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];

    if (err || !model) {
        NSLog(@"[mtmd-coreml] failed to load CoreML model: %@",
              err ? err.localizedDescription : @"unknown error");
        return nullptr;
    }
    return (void *)CFBridgingRetain(model);
}

void unload(void * handle) {
    if (handle) {
        CFRelease(handle);
    }
}

static MLMultiArrayDataType mtmd_dtype_to_ml(dtype k) {
    switch (k) {
        case DTYPE_F32: return MLMultiArrayDataTypeFloat32;
        case DTYPE_I32: return MLMultiArrayDataTypeInt32;
    }
    return MLMultiArrayDataTypeFloat32;
}

static size_t mtmd_dtype_bytes(dtype k) {
    switch (k) {
        case DTYPE_F32: return sizeof(float);
        case DTYPE_I32: return sizeof(int32_t);
    }
    return sizeof(float);
}

// Wrap a contiguous buffer in an MLMultiArray with row-major (C-order) strides.
static MLMultiArray * make_array(const input_tensor & t) {
    NSMutableArray<NSNumber *> * shape   = [NSMutableArray arrayWithCapacity:t.shape.size()];
    NSMutableArray<NSNumber *> * strides = [NSMutableArray arrayWithCapacity:t.shape.size()];

    for (int64_t d : t.shape) {
        [shape addObject:@(d)];
    }
    int64_t s = 1;
    [strides insertObject:@(s) atIndex:0];
    for (size_t i = t.shape.size() - 1; i > 0; --i) {
        s *= t.shape[i];
        [strides insertObject:@(s) atIndex:0];
    }

    NSError * err = nil;
    return [[MLMultiArray alloc] initWithDataPointer:(void *)t.data
                                               shape:shape
                                            dataType:mtmd_dtype_to_ml(t.kind)
                                             strides:strides
                                         deallocator:nil
                                               error:&err];
}

bool predict_single_output(void *                            handle,
                           const std::vector<input_tensor> & inputs,
                           const char *                      out_name,
                           float *                           out_buf) {
    if (!handle || !out_name || !out_buf) {
        return false;
    }
    MLModel * model = (__bridge MLModel *)handle;

    NSMutableDictionary<NSString *, MLFeatureValue *> * dict =
        [NSMutableDictionary dictionaryWithCapacity:inputs.size()];

    for (const auto & in : inputs) {
        MLMultiArray * arr = make_array(in);
        if (!arr) {
            NSLog(@"[mtmd-coreml] failed to wrap input %s", in.name);
            return false;
        }
        dict[[NSString stringWithUTF8String:in.name]] =
            [MLFeatureValue featureValueWithMultiArray:arr];
    }

    NSError *                     err  = nil;
    MLDictionaryFeatureProvider * prov =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&err];
    if (err || !prov) {
        NSLog(@"[mtmd-coreml] failed to build feature provider: %@",
              err ? err.localizedDescription : @"unknown error");
        return false;
    }

    id<MLFeatureProvider> out = [model predictionFromFeatures:prov error:&err];
    if (err || !out) {
        NSLog(@"[mtmd-coreml] predict failed: %@",
              err ? err.localizedDescription : @"unknown error");
        return false;
    }

    MLFeatureValue * v = [out featureValueForName:[NSString stringWithUTF8String:out_name]];
    if (!v || v.type != MLFeatureTypeMultiArray) {
        NSLog(@"[mtmd-coreml] output '%s' missing or not a multi-array", out_name);
        return false;
    }
    MLMultiArray * arr = v.multiArrayValue;
    std::memcpy(out_buf,
                (const float *)arr.dataPointer,
                (size_t)arr.count * mtmd_dtype_bytes(DTYPE_F32));
    return true;
}

} // namespace mtmd_coreml::backend
