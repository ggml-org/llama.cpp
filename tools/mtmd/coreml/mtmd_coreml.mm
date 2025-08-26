#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "mtmd_coreml.h"
#import "ane_minicpmv4_vit_f16.h"
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* model_path) {
    if (!model_path) {
        NSLog(@"Error: model_path is null");
        return nullptr;
    }
    
    NSString *pathString = [NSString stringWithUTF8String:model_path];
    
    // Check if file exists
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if (![fileManager fileExistsAtPath:pathString]) {
        NSLog(@"Error: CoreML model file does not exist at path: %@", pathString);
        return nullptr;
    }
    
    // Check if it's a directory (for .mlmodelc packages)
    BOOL isDirectory;
    if ([fileManager fileExistsAtPath:pathString isDirectory:&isDirectory]) {
        if (!isDirectory && ![pathString hasSuffix:@".mlmodelc"]) {
            NSLog(@"Warning: CoreML model path should typically be a .mlmodelc directory: %@", pathString);
        }
    }
    
    NSURL *modelURL = [NSURL fileURLWithPath:pathString];
    
    NSLog(@"Loading CoreML model from: %@", modelURL.absoluteString);
    
    NSError *error = nil;
    const void* model = CFBridgingRetain([[ane_minicpmv4_vit_f16 alloc] initWithContentsOfURL:modelURL error:&error]);
    
    if (error) {
        NSLog(@"Error loading CoreML model: %@", error.localizedDescription);
        return nullptr;
    }
    
    if (!model) {
        NSLog(@"Error: Failed to create CoreML model instance");
        return nullptr;
    }
    
    NSLog(@"Successfully loaded CoreML model from: %@", pathString);
    return model;
}

void predictWith(const void* model, float* embed, float* encoderOutput) {
    MLMultiArray *inMultiArray = [[MLMultiArray alloc] initWithDataPointer: embed
                                                                      shape: @[@1, @1024, @1152]
                                                                   dataType: MLMultiArrayDataTypeFloat32
                                                                    strides: @[@(1179648), @(1152), @1]
                                                                deallocator: nil
                                                                      error: nil];

    ane_minicpmv4_vit_f16Output *modelOutput = [(__bridge id)model predictionFromInput:inMultiArray error:nil];

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
