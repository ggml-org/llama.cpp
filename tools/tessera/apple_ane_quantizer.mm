#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace {

struct backend {
    NSURL * url;
    std::unordered_map<std::string, MLModel *> models;
    std::string error;
};

static MLMultiArray * view(
        void * data,
        NSArray<NSNumber *> * shape,
        NSArray<NSNumber *> * strides,
        MLMultiArrayDataType data_type,
        NSError ** error) {
    return [[MLMultiArray alloc]
        initWithDataPointer:data
        shape:shape
        dataType:data_type
        strides:strides
        deallocator:nil
        error:error];
}

static MLModel * model_for(backend * value, const std::string & function) {
    const auto found = value->models.find(function);
    if (found != value->models.end()) {
        return found->second;
    }

    MLModelConfiguration * configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits =
        function.find("_exact_") == std::string::npos
            ? MLComputeUnitsCPUAndNeuralEngine
            : MLComputeUnitsCPUAndGPU;
    configuration.functionName =
        [NSString stringWithUTF8String:function.c_str()];
    NSError * error = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:value->url
                                        configuration:configuration
                                                error:&error];
    if (model == nil) {
        value->error = error.localizedDescription.UTF8String ?: "Core ML load failed";
        return nil;
    }
    value->models.emplace(function, model);
    return model;
}

static int predict(
        backend * value,
        const std::string & function,
        NSDictionary<NSString *, MLFeatureValue *> * features,
        NSString * output_name,
        MLMultiArray * output) {
    MLModel * model = model_for(value, function);
    if (model == nil) {
        return 2;
    }

    NSError * error = nil;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
                                                          error:&error];
    if (provider == nil) {
        value->error = error.localizedDescription.UTF8String ?: "feature provider failed";
        return 3;
    }

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = @{ output_name: output };
    id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                         options:options
                                                           error:&error];
    if (result == nil) {
        value->error = error.localizedDescription.UTF8String ?: "Core ML prediction failed";
        return 4;
    }
    return 0;
}

static void canonical_lane_targets(
        const float * weights,
        const float * ternary,
        float * output,
        size_t rows) {
    constexpr size_t columns = 640;
    constexpr size_t lane_width = 20;
    constexpr size_t lanes = columns / lane_width;
    float magnitudes[lane_width];
    float retained[lane_width];
    for (size_t row = 0; row < rows; ++row) {
        for (size_t lane = 0; lane < lanes; ++lane) {
            const size_t offset = row * columns + lane * lane_width;
            vDSP_vabs(
                weights + offset, 1, magnitudes, 1,
                static_cast<vDSP_Length>(lane_width));
            float count = 0.0f;
            for (size_t i = 0; i < lane_width; ++i) {
                retained[i] = ternary[offset + i] == 0.0f ? 0.0f : 1.0f;
                count += retained[i];
            }
            float sum = 0.0f;
            vDSP_dotpr(
                magnitudes, 1, retained, 1, &sum,
                static_cast<vDSP_Length>(lane_width));
            output[row * lanes + lane] = count > 0.0f ? sum / count : 0.0f;
        }
    }
}

} // namespace

extern "C" {

void * tessera_ane_create(const char * compiled_model_path) {
    if (compiled_model_path == nullptr) {
        return nullptr;
    }
    @autoreleasepool {
        auto * value = new backend();
        value->url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:compiled_model_path]];
        return value;
    }
}

void tessera_ane_destroy(void * opaque) {
    delete static_cast<backend *>(opaque);
}

const char * tessera_ane_last_error(void * opaque) {
    auto * value = static_cast<backend *>(opaque);
    return value == nullptr ? "invalid ANE backend" : value->error.c_str();
}

int tessera_ane_lane_targets(
        void * opaque,
        const uint16_t * weights,
        const uint16_t * ternary,
        uint16_t * output,
        size_t rows) {
    auto * value = static_cast<backend *>(opaque);
    if (value == nullptr || weights == nullptr || ternary == nullptr ||
        output == nullptr || (rows != 64 && rows != 256 && rows != 1024)) {
        return 1;
    }
    @autoreleasepool {
        NSError * error = nil;
        NSArray * matrix_shape = @[@(rows), @640];
        NSArray * matrix_strides = @[@640, @1];
        NSArray * output_shape = @[@(rows), @32];
        NSArray * output_strides = @[@32, @1];
        MLMultiArray * weights_view = view(
            const_cast<uint16_t *>(weights), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat16, &error);
        MLMultiArray * ternary_view = view(
            const_cast<uint16_t *>(ternary), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat16, &error);
        MLMultiArray * output_view = view(
            output, output_shape, output_strides,
            MLMultiArrayDataTypeFloat16, &error);
        if (weights_view == nil || ternary_view == nil || output_view == nil) {
            value->error = error.localizedDescription.UTF8String ?: "multi-array view failed";
            return 5;
        }
        return predict(
            value,
            "lane_targets_ane_r" + std::to_string(rows),
            @{
                @"weights": [MLFeatureValue featureValueWithMultiArray:weights_view],
                @"ternary": [MLFeatureValue featureValueWithMultiArray:ternary_view],
            },
            @"div",
            output_view);
    }
}

int tessera_ane_residual_score(
        void * opaque,
        const uint16_t * weights,
        const uint16_t * ternary,
        const uint16_t * lane_scale,
        const uint16_t * importance,
        uint16_t * output,
        size_t rows) {
    auto * value = static_cast<backend *>(opaque);
    if (value == nullptr || weights == nullptr || ternary == nullptr ||
        lane_scale == nullptr || importance == nullptr || output == nullptr ||
        (rows != 64 && rows != 256 && rows != 1024)) {
        return 1;
    }
    @autoreleasepool {
        NSError * error = nil;
        NSArray * matrix_shape = @[@(rows), @640];
        NSArray * matrix_strides = @[@640, @1];
        NSArray * lane_shape = @[@(rows), @32];
        NSArray * lane_strides = @[@32, @1];
        MLMultiArray * weights_view = view(
            const_cast<uint16_t *>(weights), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat16, &error);
        MLMultiArray * ternary_view = view(
            const_cast<uint16_t *>(ternary), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat16, &error);
        MLMultiArray * lane_view = view(
            const_cast<uint16_t *>(lane_scale), lane_shape, lane_strides,
            MLMultiArrayDataTypeFloat16, &error);
        MLMultiArray * importance_view = view(
            const_cast<uint16_t *>(importance), @[@640], @[@1],
            MLMultiArrayDataTypeFloat16, &error);
        MLMultiArray * output_view = view(
            output, matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat16, &error);
        if (weights_view == nil || ternary_view == nil || lane_view == nil ||
            importance_view == nil || output_view == nil) {
            value->error = error.localizedDescription.UTF8String ?: "multi-array view failed";
            return 5;
        }
        return predict(
            value,
            "residual_score_ane_r" + std::to_string(rows),
            @{
                @"weights": [MLFeatureValue featureValueWithMultiArray:weights_view],
                @"ternary": [MLFeatureValue featureValueWithMultiArray:ternary_view],
                @"lane_scale": [MLFeatureValue featureValueWithMultiArray:lane_view],
                @"importance": [MLFeatureValue featureValueWithMultiArray:importance_view],
            },
            @"mul_3",
            output_view);
    }
}

int tessera_coreml_lane_targets_exact(
        void * opaque,
        const float * weights,
        const float * ternary,
        float * output,
        size_t rows) {
    auto * value = static_cast<backend *>(opaque);
    if (value == nullptr || weights == nullptr || ternary == nullptr ||
        output == nullptr || (rows != 64 && rows != 256 && rows != 1024)) {
        return 1;
    }
    @autoreleasepool {
        NSError * error = nil;
        NSArray * matrix_shape = @[@(rows), @640];
        NSArray * matrix_strides = @[@640, @1];
        NSArray * output_shape = @[@(rows), @32];
        NSArray * output_strides = @[@32, @1];
        MLMultiArray * weights_view = view(
            const_cast<float *>(weights), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat32, &error);
        MLMultiArray * ternary_view = view(
            const_cast<float *>(ternary), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat32, &error);
        MLMultiArray * output_view = view(
            output, output_shape, output_strides,
            MLMultiArrayDataTypeFloat32, &error);
        if (weights_view == nil || ternary_view == nil || output_view == nil) {
            value->error = error.localizedDescription.UTF8String ?: "multi-array view failed";
            return 5;
        }
        const int status = predict(
            value,
            "lane_targets_exact_r" + std::to_string(rows),
            @{
                @"weights": [MLFeatureValue featureValueWithMultiArray:weights_view],
                @"ternary": [MLFeatureValue featureValueWithMultiArray:ternary_view],
            },
            @"div",
            output_view);
        if (status == 0) {
            // Core ML may associate a 20-value reduction differently from
            // vDSP by a few ULPs.  Recompute only this tiny reduction boundary
            // so serialized page scales are bit-for-bit canonical with the
            // Accelerate quantizer path.
            canonical_lane_targets(weights, ternary, output, rows);
        }
        return status;
    }
}

int tessera_coreml_residual_score_exact(
        void * opaque,
        const float * weights,
        const float * ternary,
        const float * lane_scale,
        const float * importance,
        float * output,
        size_t rows) {
    auto * value = static_cast<backend *>(opaque);
    if (value == nullptr || weights == nullptr || ternary == nullptr ||
        lane_scale == nullptr || importance == nullptr || output == nullptr ||
        (rows != 64 && rows != 256 && rows != 1024)) {
        return 1;
    }
    @autoreleasepool {
        NSError * error = nil;
        NSArray * matrix_shape = @[@(rows), @640];
        NSArray * matrix_strides = @[@640, @1];
        NSArray * lane_shape = @[@(rows), @32];
        NSArray * lane_strides = @[@32, @1];
        MLMultiArray * weights_view = view(
            const_cast<float *>(weights), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat32, &error);
        MLMultiArray * ternary_view = view(
            const_cast<float *>(ternary), matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat32, &error);
        MLMultiArray * lane_view = view(
            const_cast<float *>(lane_scale), lane_shape, lane_strides,
            MLMultiArrayDataTypeFloat32, &error);
        MLMultiArray * importance_view = view(
            const_cast<float *>(importance), @[@640], @[@1],
            MLMultiArrayDataTypeFloat32, &error);
        MLMultiArray * output_view = view(
            output, matrix_shape, matrix_strides,
            MLMultiArrayDataTypeFloat32, &error);
        if (weights_view == nil || ternary_view == nil || lane_view == nil ||
            importance_view == nil || output_view == nil) {
            value->error = error.localizedDescription.UTF8String ?: "multi-array view failed";
            return 5;
        }
        return predict(
            value,
            "residual_score_exact_r" + std::to_string(rows),
            @{
                @"weights": [MLFeatureValue featureValueWithMultiArray:weights_view],
                @"ternary": [MLFeatureValue featureValueWithMultiArray:ternary_view],
                @"lane_scale": [MLFeatureValue featureValueWithMultiArray:lane_view],
                @"importance": [MLFeatureValue featureValueWithMultiArray:importance_view],
            },
            @"mul_3",
            output_view);
    }
}

}
