#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "ane.h"
#import "ane_minicpm4v3b_vision_f16_b1.h"
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

const void* loadModel() {
    // 新的，从 documents directionary 中加载 begin
    // 获取文件管理器实例
    NSFileManager *fileManager = [NSFileManager defaultManager];
    // 获取应用的 Documents 目录的 URL
    NSURL *documentsURL = [[fileManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    NSString *pathString = [documentsURL.absoluteString stringByAppendingString:@"ane_minicpm4v3b_vision_f16_b1.mlmodelc"];
    NSURL *modelURL = [NSURL URLWithString:pathString];

    NSLog(modelURL.absoluteString);

    const void* model = CFBridgingRetain([[ane_minicpm4v3b_vision_f16_b1 alloc] initWithContentsOfURL:modelURL error:nil]);
    return model;
}

void predictWith(const void* model, float* embed, float* encoderOutput) {
    MLMultiArray *inMultiArray = [[MLMultiArray alloc] initWithDataPointer: embed
                                                                      shape: @[@1, @1024, @1152]
                                                                   dataType: MLMultiArrayDataTypeFloat32
                                                                    strides: @[@(1179648), @(1152), @1]
                                                                deallocator: nil
                                                                      error: nil];

    ane_minicpm4v3b_vision_f16_b1Output *modelOutput = [(__bridge id)model predictionFromInput:inMultiArray error:nil];

    MLMultiArray *outMA = modelOutput.output;

    cblas_scopy((int)outMA.count,
                (float*)outMA.dataPointer, 1,
                encoderOutput, 1);
}

void closeModel(const void* model) {
    CFRelease(model);
}

#if __cplusplus
} //Extern C
#endif
