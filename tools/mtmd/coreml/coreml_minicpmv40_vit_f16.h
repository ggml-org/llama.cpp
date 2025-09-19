//
// coreml_minicpmv40_vit_f16.h
//
// This file was automatically generated and should not be edited.
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

NS_ASSUME_NONNULL_BEGIN

/// Model Prediction Input Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface coreml_minicpmv40_vit_f16Input : NSObject<MLFeatureProvider>

/// pixel_values as 1 × 3 × 14 × 14336 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * pixel_values;

/// position_ids as 1 by 1024 matrix of 32-bit integers
@property (readwrite, nonatomic, strong) MLMultiArray * position_ids;

/// pos_embed as 1024 × 1 × 2560 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * pos_embed;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed:(MLMultiArray *)pos_embed NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface coreml_minicpmv40_vit_f16Output : NSObject<MLFeatureProvider>

/// output as 1 × 64 × 2560 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput:(MLMultiArray *)output NS_DESIGNATED_INITIALIZER;

@end

/// Class for model loading and prediction
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface coreml_minicpmv40_vit_f16 : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize coreml_minicpmv40_vit_f16 instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of coreml_minicpmv40_vit_f16.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize coreml_minicpmv40_vit_f16 instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize coreml_minicpmv40_vit_f16 instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize coreml_minicpmv40_vit_f16 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for coreml_minicpmv40_vit_f16.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize coreml_minicpmv40_vit_f16 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for coreml_minicpmv40_vit_f16.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Construct coreml_minicpmv40_vit_f16 instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid coreml_minicpmv40_vit_f16 instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmv40_vit_f16 * _Nullable model, NSError * _Nullable error))handler;

/**
    Construct coreml_minicpmv40_vit_f16 instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid coreml_minicpmv40_vit_f16 instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmv40_vit_f16 * _Nullable model, NSError * _Nullable error))handler;

/**
    Make a prediction using the standard interface
    @param input an instance of coreml_minicpmv40_vit_f16Input to predict from
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as coreml_minicpmv40_vit_f16Output
*/
- (nullable coreml_minicpmv40_vit_f16Output *)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of coreml_minicpmv40_vit_f16Input to predict from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as coreml_minicpmv40_vit_f16Output
*/
- (nullable coreml_minicpmv40_vit_f16Output *)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of coreml_minicpmv40_vit_f16Input to predict from
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input completionHandler:(void (^)(coreml_minicpmv40_vit_f16Output * _Nullable output, NSError * _Nullable error))completionHandler API_AVAILABLE(macos(14.0), ios(17.0), watchos(10.0), tvos(17.0)) __attribute__((visibility("hidden")));

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of coreml_minicpmv40_vit_f16Input to predict from
    @param options prediction options
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(coreml_minicpmv40_vit_f16Output * _Nullable output, NSError * _Nullable error))completionHandler API_AVAILABLE(macos(14.0), ios(17.0), watchos(10.0), tvos(17.0)) __attribute__((visibility("hidden")));

/**
    Make a prediction using the convenience interface
    @param pixel_values 1 × 3 × 14 × 14336 4-dimensional array of floats
    @param position_ids 1 by 1024 matrix of 32-bit integers
    @param pos_embed 1024 × 1 × 2560 3-dimensional array of floats
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as coreml_minicpmv40_vit_f16Output
*/
- (nullable coreml_minicpmv40_vit_f16Output *)predictionFromPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed:(MLMultiArray *)pos_embed error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Batch prediction
    @param inputArray array of coreml_minicpmv40_vit_f16Input instances to obtain predictions from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the predictions as NSArray<coreml_minicpmv40_vit_f16Output *>
*/
- (nullable NSArray<coreml_minicpmv40_vit_f16Output *> *)predictionsFromInputs:(NSArray<coreml_minicpmv40_vit_f16Input*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
