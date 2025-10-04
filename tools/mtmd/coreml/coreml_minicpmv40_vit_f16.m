//
// coreml_minicpmv40_vit_f16.m
//
// This file was automatically generated and should not be edited.
//

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "coreml_minicpmv40_vit_f16.h"

@implementation coreml_minicpmv40_vit_f16Input

- (instancetype)initWithPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed:(MLMultiArray *)pos_embed {
    self = [super init];
    if (self) {
        _pixel_values = pixel_values;
        _position_ids = position_ids;
        _pos_embed = pos_embed;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"pixel_values", @"position_ids", @"pos_embed"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"pixel_values"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.pixel_values];
    }
    if ([featureName isEqualToString:@"position_ids"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.position_ids];
    }
    if ([featureName isEqualToString:@"pos_embed"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.pos_embed];
    }
    return nil;
}

@end

@implementation coreml_minicpmv40_vit_f16Output

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

@implementation coreml_minicpmv40_vit_f16


/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"coreml_minicpmv40_vit_f16" ofType:@"mlmodelc"];
    if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load coreml_minicpmv40_vit_f16.mlmodelc in the bundle resource"); return nil; }
    return [NSURL fileURLWithPath:assetPath];
}


/**
    Initialize coreml_minicpmv40_vit_f16 instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of coreml_minicpmv40_vit_f16.
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
    Initialize coreml_minicpmv40_vit_f16 instance with the model in this bundle.
*/
- (nullable instancetype)init {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}


/**
    Initialize coreml_minicpmv40_vit_f16 instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
}


/**
    Initialize coreml_minicpmv40_vit_f16 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for coreml_minicpmv40_vit_f16.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Initialize coreml_minicpmv40_vit_f16 instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for coreml_minicpmv40_vit_f16.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Construct coreml_minicpmv40_vit_f16 instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid coreml_minicpmv40_vit_f16 instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmv40_vit_f16 * _Nullable model, NSError * _Nullable error))handler {
    [self loadContentsOfURL:(NSURL * _Nonnull)[self URLOfModelInThisBundle]
              configuration:configuration
          completionHandler:handler];
}


/**
    Construct coreml_minicpmv40_vit_f16 instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid coreml_minicpmv40_vit_f16 instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmv40_vit_f16 * _Nullable model, NSError * _Nullable error))handler {
    [MLModel loadContentsOfURL:modelURL
                 configuration:configuration
             completionHandler:^(MLModel *model, NSError *error) {
        if (model != nil) {
            coreml_minicpmv40_vit_f16 *typedModel = [[coreml_minicpmv40_vit_f16 alloc] initWithMLModel:model];
            handler(typedModel, nil);
        } else {
            handler(nil, error);
        }
    }];
}

- (nullable coreml_minicpmv40_vit_f16Output *)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self predictionFromFeatures:input options:[[MLPredictionOptions alloc] init] error:error];
}

- (nullable coreml_minicpmv40_vit_f16Output *)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [self.model predictionFromFeatures:input options:options error:error];
    if (!outFeatures) { return nil; }
    return [[coreml_minicpmv40_vit_f16Output alloc] initWithOutput:(MLMultiArray *)[outFeatures featureValueForName:@"output"].multiArrayValue];
}

- (void)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input completionHandler:(void (^)(coreml_minicpmv40_vit_f16Output * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            coreml_minicpmv40_vit_f16Output *output = [[coreml_minicpmv40_vit_f16Output alloc] initWithOutput:(MLMultiArray *)[prediction featureValueForName:@"output"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (void)predictionFromFeatures:(coreml_minicpmv40_vit_f16Input *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(coreml_minicpmv40_vit_f16Output * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input options:options completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            coreml_minicpmv40_vit_f16Output *output = [[coreml_minicpmv40_vit_f16Output alloc] initWithOutput:(MLMultiArray *)[prediction featureValueForName:@"output"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (nullable coreml_minicpmv40_vit_f16Output *)predictionFromPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed:(MLMultiArray *)pos_embed error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    coreml_minicpmv40_vit_f16Input *input_ = [[coreml_minicpmv40_vit_f16Input alloc] initWithPixel_values:pixel_values position_ids:position_ids pos_embed:pos_embed];
    return [self predictionFromFeatures:input_ error:error];
}

- (nullable NSArray<coreml_minicpmv40_vit_f16Output *> *)predictionsFromInputs:(NSArray<coreml_minicpmv40_vit_f16Input*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLBatchProvider> inBatch = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
    id<MLBatchProvider> outBatch = [self.model predictionsFromBatch:inBatch options:options error:error];
    if (!outBatch) { return nil; }
    NSMutableArray<coreml_minicpmv40_vit_f16Output*> *results = [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
    for (NSInteger i = 0; i < outBatch.count; i++) {
        id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
        coreml_minicpmv40_vit_f16Output * result = [[coreml_minicpmv40_vit_f16Output alloc] initWithOutput:(MLMultiArray *)[resultProvider featureValueForName:@"output"].multiArrayValue];
        [results addObject:result];
    }
    return results;
}

@end
