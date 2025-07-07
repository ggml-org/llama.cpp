package com.example.llama.data.remote

import android.app.DownloadManager
import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

private const val QUERY_Q4_0_GGUF = "gguf q4_0"
private const val FILTER_TEXT_GENERATION = "text-generation"
private const val SORT_BY_DOWNLOADS = "downloads"
private const val SEARCH_RESULT_LIMIT = 20

interface HuggingFaceRemoteDataSource {
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
    ): List<HuggingFaceModel>

    suspend fun getModelDetails(modelId: String): HuggingFaceModelDetails

    /**
     * Obtain selected HuggingFace model's GGUF file size from HTTP header
     */
    suspend fun getFileSize(modelId: String, filePath: String): Long?

    /**
     * Download selected HuggingFace model's GGUF file via DownloadManager
     */
    suspend fun downloadModelFile(
        context: Context,
        downloadInfo: HuggingFaceDownloadInfo,
    ): Result<Unit>
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
        limit: Int?,
        full: Boolean,
    ) = withContext(Dispatchers.IO) {
        apiService.getModels(
            search = query,
            filter = filter,
            sort = sort,
            direction = direction,
            limit = limit,
            full = full,
        ).filter { it.gated != true && it.private != true }
    }

    override suspend fun getModelDetails(
        modelId: String
    ) = withContext(Dispatchers.IO) {
        apiService.getModelDetails(modelId)
    }

    override suspend fun getFileSize(
        modelId: String,
        filePath: String
    ): Long? = withContext(Dispatchers.IO) {
        try {
            apiService.getModelFileHeader(modelId, filePath).let {
                if (it.isSuccessful) {
                    it.headers()[HTTP_HEADER_CONTENT_LENGTH]?.toLongOrNull()
                } else {
                    null
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting file size for $modelId: ${e.message}")
            null
        }
    }

    override suspend fun downloadModelFile(
        context: Context,
        downloadInfo: HuggingFaceDownloadInfo,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val downloadManager =
                context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            val request = DownloadManager.Request(downloadInfo.uri).apply {
                setTitle("HuggingFace model download")
                setDescription("Downloading ${downloadInfo.filename}")
                setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                setDestinationInExternalPublicDir(
                    Environment.DIRECTORY_DOWNLOADS,
                    downloadInfo.filename
                )
                setAllowedNetworkTypes(
                    DownloadManager.Request.NETWORK_WIFI or DownloadManager.Request.NETWORK_MOBILE
                )
                setAllowedOverMetered(true)
                setAllowedOverRoaming(false)
            }
            Log.d(TAG, "Enqueuing download request for: ${downloadInfo.modelId}")
            val downloadId = downloadManager.enqueue(request)

            delay(DOWNLOAD_MANAGER_DOUBLE_CHECK_DELAY)

            val cursor = downloadManager.query(DownloadManager.Query().setFilterById(downloadId))
            if (cursor != null && cursor.moveToFirst()) {
                val statusIndex = cursor.getColumnIndex(DownloadManager.COLUMN_STATUS)
                if (statusIndex >= 0) {
                    val status = cursor.getInt(statusIndex)
                    cursor.close()

                    when (status) {
                        DownloadManager.STATUS_FAILED -> {
                            // Get failure reason if available
                            val reasonIndex = cursor.getColumnIndex(DownloadManager.COLUMN_REASON)
                            val reason = if (reasonIndex >= 0) cursor.getInt(reasonIndex) else -1
                            val errorMessage = when (reason) {
                                DownloadManager.ERROR_HTTP_DATA_ERROR -> "HTTP error"
                                DownloadManager.ERROR_INSUFFICIENT_SPACE -> "Insufficient storage"
                                DownloadManager.ERROR_TOO_MANY_REDIRECTS -> "Too many redirects"
                                DownloadManager.ERROR_UNHANDLED_HTTP_CODE -> "Unhandled HTTP code"
                                DownloadManager.ERROR_CANNOT_RESUME -> "Cannot resume download"
                                DownloadManager.ERROR_FILE_ERROR -> "File error"
                                else -> "Unknown error"
                            }
                            Result.failure(Exception(errorMessage))
                        }
                        else -> {
                            // Download is pending, paused, or running
                            Result.success(Unit)
                        }
                    }
                } else {
                    // Assume success if we can't check status
                    cursor.close()
                    Result.success(Unit)
                }
            } else {
                // Assume success if cursor is empty
                cursor?.close()
                Result.success(Unit)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enqueue download: ${e.message}")
            Result.failure(e)
        }
    }

    companion object {
        private val TAG = HuggingFaceRemoteDataSourceImpl::class.java.simpleName

        private const val HTTP_HEADER_CONTENT_LENGTH = "content-length"
        private const val DOWNLOAD_MANAGER_DOUBLE_CHECK_DELAY = 500L
    }
}
