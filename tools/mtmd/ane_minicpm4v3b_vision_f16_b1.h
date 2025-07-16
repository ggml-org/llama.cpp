//
// ane_minicpm4v3b_vision_f16_b1.h
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
@interface ane_minicpm4v3b_vision_f16_b1Input : NSObject<MLFeatureProvider>

/// input as 1 × 1024 × 1152 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * input;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithInput:(MLMultiArray *)input NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface ane_minicpm4v3b_vision_f16_b1Output : NSObject<MLFeatureProvider>

/// output as 1 × 1024 × 1152 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput:(MLMultiArray *)output NS_DESIGNATED_INITIALIZER;

@end

/// Class for model loading and prediction
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface ane_minicpm4v3b_vision_f16_b1 : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of ane_minicpm4v3b_vision_f16_b1.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for ane_minicpm4v3b_vision_f16_b1.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for ane_minicpm4v3b_vision_f16_b1.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Construct ane_minicpm4v3b_vision_f16_b1 instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid ane_minicpm4v3b_vision_f16_b1 instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1 * _Nullable model, NSError * _Nullable error))handler;

/**
    Construct ane_minicpm4v3b_vision_f16_b1 instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid ane_minicpm4v3b_vision_f16_b1 instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1 * _Nullable model, NSError * _Nullable error))handler;

/**
    Make a prediction using the standard interface
    @param input an instance of ane_minicpm4v3b_vision_f16_b1Input to predict from
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as ane_minicpm4v3b_vision_f16_b1Output
*/
- (nullable ane_minicpm4v3b_vision_f16_b1Output *)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of ane_minicpm4v3b_vision_f16_b1Input to predict from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as ane_minicpm4v3b_vision_f16_b1Output
*/
- (nullable ane_minicpm4v3b_vision_f16_b1Output *)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of ane_minicpm4v3b_vision_f16_b1Input to predict from
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1Output * _Nullable output, NSError * _Nullable error))completionHandler API_AVAILABLE(macos(14.0), ios(17.0), watchos(10.0), tvos(17.0)) __attribute__((visibility("hidden")));

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of ane_minicpm4v3b_vision_f16_b1Input to predict from
    @param options prediction options
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1Output * _Nullable output, NSError * _Nullable error))completionHandler API_AVAILABLE(macos(14.0), ios(17.0), watchos(10.0), tvos(17.0)) __attribute__((visibility("hidden")));

/**
    Make a prediction using the convenience interface
    @param input 1 × 1024 × 1152 3-dimensional array of floats
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as ane_minicpm4v3b_vision_f16_b1Output
*/
- (nullable ane_minicpm4v3b_vision_f16_b1Output *)predictionFromInput:(MLMultiArray *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Batch prediction
    @param inputArray array of ane_minicpm4v3b_vision_f16_b1Input instances to obtain predictions from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the predictions as NSArray<ane_minicpm4v3b_vision_f16_b1Output *>
*/
- (nullable NSArray<ane_minicpm4v3b_vision_f16_b1Output *> *)predictionsFromInputs:(NSArray<ane_minicpm4v3b_vision_f16_b1Input*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
