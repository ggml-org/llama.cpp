//
// ane_minicpm4v3b_vision_f16_b1.m
//
// This file was automatically generated and should not be edited.
//

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "ane_minicpm4v3b_vision_f16_b1.h"

@implementation ane_minicpm4v3b_vision_f16_b1Input

- (instancetype)initWithInput:(MLMultiArray *)input {
    self = [super init];
    if (self) {
        _input = input;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"input"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.input];
    }
    return nil;
}

@end

@implementation ane_minicpm4v3b_vision_f16_b1Output

- (instancetype)initWithOutput:(MLMultiArray *)output {
    self = [super init];
    if (self) {
        _output = output;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"output"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"output"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.output];
    }
    return nil;
}

@end

@implementation ane_minicpm4v3b_vision_f16_b1


/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"ane_minicpm4v3b_vision_f16_b1" ofType:@"mlmodelc"];
    if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load ane_minicpm4v3b_vision_f16_b1.mlmodelc in the bundle resource"); return nil; }
    return [NSURL fileURLWithPath:assetPath];
}


/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of ane_minicpm4v3b_vision_f16_b1.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model {
    if (model == nil) {
        return nil;
    }
    self = [super init];
    if (self != nil) {
        _model = model;
    }
    return self;
}


/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance with the model in this bundle.
*/
- (nullable instancetype)init {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}


/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
}


/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for ane_minicpm4v3b_vision_f16_b1.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Initialize ane_minicpm4v3b_vision_f16_b1 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for ane_minicpm4v3b_vision_f16_b1.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Construct ane_minicpm4v3b_vision_f16_b1 instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid ane_minicpm4v3b_vision_f16_b1 instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1 * _Nullable model, NSError * _Nullable error))handler {
    [self loadContentsOfURL:(NSURL * _Nonnull)[self URLOfModelInThisBundle]
              configuration:configuration
          completionHandler:handler];
}


/**
    Construct ane_minicpm4v3b_vision_f16_b1 instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid ane_minicpm4v3b_vision_f16_b1 instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1 * _Nullable model, NSError * _Nullable error))handler {
    [MLModel loadContentsOfURL:modelURL
                 configuration:configuration
             completionHandler:^(MLModel *model, NSError *error) {
        if (model != nil) {
            ane_minicpm4v3b_vision_f16_b1 *typedModel = [[ane_minicpm4v3b_vision_f16_b1 alloc] initWithMLModel:model];
            handler(typedModel, nil);
        } else {
            handler(nil, error);
        }
    }];
}

- (nullable ane_minicpm4v3b_vision_f16_b1Output *)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self predictionFromFeatures:input options:[[MLPredictionOptions alloc] init] error:error];
}

- (nullable ane_minicpm4v3b_vision_f16_b1Output *)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [self.model predictionFromFeatures:input options:options error:error];
    if (!outFeatures) { return nil; }
    return [[ane_minicpm4v3b_vision_f16_b1Output alloc] initWithOutput:(MLMultiArray *)[outFeatures featureValueForName:@"output"].multiArrayValue];
}

- (void)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1Output * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            ane_minicpm4v3b_vision_f16_b1Output *output = [[ane_minicpm4v3b_vision_f16_b1Output alloc] initWithOutput:(MLMultiArray *)[prediction featureValueForName:@"output"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (void)predictionFromFeatures:(ane_minicpm4v3b_vision_f16_b1Input *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(ane_minicpm4v3b_vision_f16_b1Output * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input options:options completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            ane_minicpm4v3b_vision_f16_b1Output *output = [[ane_minicpm4v3b_vision_f16_b1Output alloc] initWithOutput:(MLMultiArray *)[prediction featureValueForName:@"output"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (nullable ane_minicpm4v3b_vision_f16_b1Output *)predictionFromInput:(MLMultiArray *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    ane_minicpm4v3b_vision_f16_b1Input *input_ = [[ane_minicpm4v3b_vision_f16_b1Input alloc] initWithInput:input];
    return [self predictionFromFeatures:input_ error:error];
}

- (nullable NSArray<ane_minicpm4v3b_vision_f16_b1Output *> *)predictionsFromInputs:(NSArray<ane_minicpm4v3b_vision_f16_b1Input*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLBatchProvider> inBatch = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
    id<MLBatchProvider> outBatch = [self.model predictionsFromBatch:inBatch options:options error:error];
    if (!outBatch) { return nil; }
    NSMutableArray<ane_minicpm4v3b_vision_f16_b1Output*> *results = [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
    for (NSInteger i = 0; i < outBatch.count; i++) {
        id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
        ane_minicpm4v3b_vision_f16_b1Output * result = [[ane_minicpm4v3b_vision_f16_b1Output alloc] initWithOutput:(MLMultiArray *)[resultProvider featureValueForName:@"output"].multiArrayValue];
        [results addObject:result];
    }
    return results;
}

@end
