#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* model_path);
void closeModel(const void* model);
void predictWith(const void* model, float* embed, float* encoderOutput);

#if __cplusplus
}   // Extern C
#endif
