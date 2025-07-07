package com.example.llama.data.remote

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

interface HuggingFaceRemoteDataSource {
    suspend fun searchModels(
        query: String? = "gguf",
        filter: String? = "text-generation", // Only generative models,
        sort: String? = "downloads",
        direction: String? = "-1",
        limit: Int? = 20
    ): List<HuggingFaceModel>

    suspend fun getModelDetails(modelId: String): HuggingFaceModelDetails

    suspend fun downloadModelFile(modelId: String, filePath: String, outputFile: File): Result<File>
}

@Singleton
class HuggingFaceRemoteDataSourceImpl @Inject constructor(
    private val apiService: HuggingFaceApiService
) : HuggingFaceRemoteDataSource {

    override suspend fun searchModels(
        query: String?,
        filter: String?,
        sort: String?,
        direction: String?,
        limit: Int?
    ) = withContext(Dispatchers.IO) {
        apiService.getModels(
            search = query,
            filter = filter,
            sort = sort,
            direction = direction,
            limit = limit
        )
    }

    override suspend fun getModelDetails(
        modelId: String
    ) = withContext(Dispatchers.IO) {
        apiService.getModelDetails(modelId)
    }

    override suspend fun downloadModelFile(
        modelId: String,
        filePath: String,
        outputFile: File
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            val response = apiService.downloadModelFile(modelId, filePath)

            // Create parent directories if needed
            outputFile.parentFile?.mkdirs()

            // Save the file
            response.byteStream().use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            Result.success(outputFile)
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading file $filePath: ${e.message}")
            Result.failure(e)
        }
    }

    companion object {
        private val TAG = HuggingFaceRemoteDataSourceImpl::class.java.simpleName
    }
}
