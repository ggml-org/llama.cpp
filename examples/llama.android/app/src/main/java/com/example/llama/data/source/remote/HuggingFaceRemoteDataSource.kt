package com.example.llama.data.source.remote

import android.content.Context
import com.example.llama.monitoring.MemoryMetrics


/*
 * HuggingFace Search API
 */
private const val QUERY_Q4_0_GGUF = "gguf q4_0"
private const val FILTER_TEXT_GENERATION = "text-generation"
private const val SORT_BY_DOWNLOADS = "downloads"
private const val SEARCH_RESULT_LIMIT = 30

private val INVALID_KEYWORDS = arrayOf("-of-", "split", "70B", "30B", "27B", "14B", "13B", "12B")

interface HuggingFaceRemoteDataSource {

    suspend fun fetchPreselectedModels(
        memoryUsage: MemoryMetrics,
        parallelCount: Int = 3,
        quorum: Float = 0.5f,
    ): List<HuggingFaceModelDetails>

    /**
     * Query openly available Q4_0 GGUF models on HuggingFace
     */
    suspend fun searchModels(
        query: String? = QUERY_Q4_0_GGUF,
        filter: String? = FILTER_TEXT_GENERATION,
        sort: String? = SORT_BY_DOWNLOADS,
        direction: String? = "-1",
        limit: Int? = SEARCH_RESULT_LIMIT,
        full: Boolean = true,
        invalidKeywords: Array<String> = INVALID_KEYWORDS
    ): Result<List<HuggingFaceModel>>

    suspend fun getModelDetails(modelId: String): HuggingFaceModelDetails

    /**
     * Obtain selected HuggingFace model's GGUF file size from HTTP header
     */
    suspend fun getFileSize(modelId: String, filePath: String): Result<Long>

    /**
     * Download selected HuggingFace model's GGUF file via DownloadManager
     */
    suspend fun downloadModelFile(
        context: Context,
        downloadInfo: HuggingFaceDownloadInfo,
    ): Result<Long>
}

