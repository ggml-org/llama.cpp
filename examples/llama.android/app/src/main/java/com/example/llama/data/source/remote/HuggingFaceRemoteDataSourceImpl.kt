package com.example.llama.data.source.remote

import android.app.DownloadManager
import android.content.Context
import android.os.Environment
import android.util.Log
import com.example.llama.monitoring.MemoryMetrics
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.supervisorScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import retrofit2.HttpException
import java.io.FileNotFoundException
import java.io.IOException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.collections.contains
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.ceil


/*
 * Preselected models: sized <2GB
 */
private val PRESELECTED_MODEL_IDS_SMALL = listOf(
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "unsloth/gemma-3-1b-it-GGUF",
    "bartowski/granite-3.0-2b-instruct-GGUF",
)

/*
 * Preselected models: sized 2~3GB
 */
private val PRESELECTED_MODEL_IDS_MEDIUM = listOf(
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "unsloth/gemma-3n-E2B-it-GGUF",
    "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "gaianet/Phi-4-mini-instruct-GGUF",
    "unsloth/gemma-3-4b-it-GGUF",
)

/*
 * Preselected models: sized 4~6B
 */
private val PRESELECTED_MODEL_IDS_LARGE = listOf(
    "unsloth/gemma-3n-E4B-it-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
)

@Singleton
class HuggingFaceRemoteDataSourceImpl @Inject constructor(
    private val apiService: HuggingFaceApiService
) : HuggingFaceRemoteDataSource {

    override suspend fun fetchPreselectedModels(
        memoryUsage: MemoryMetrics,
        parallelCount: Int,
        quorum: Float,
    ): List<HuggingFaceModelDetails> = withContext(Dispatchers.IO) {
        val ids: List<String> = when {
            memoryUsage.availableGB >= 7f ->
                PRESELECTED_MODEL_IDS_MEDIUM + PRESELECTED_MODEL_IDS_LARGE + PRESELECTED_MODEL_IDS_SMALL
            memoryUsage.availableGB >= 5f ->
                PRESELECTED_MODEL_IDS_SMALL + PRESELECTED_MODEL_IDS_MEDIUM + PRESELECTED_MODEL_IDS_LARGE
            memoryUsage.availableGB >= 3f ->
                PRESELECTED_MODEL_IDS_SMALL + PRESELECTED_MODEL_IDS_MEDIUM
            else ->
                PRESELECTED_MODEL_IDS_SMALL
        }

        val sem = Semaphore(parallelCount)
        val results = supervisorScope {
            ids.map { id ->
                async {
                    sem.withPermit {
                        try {
                            Result.success(getModelDetails(id))
                        } catch (t: CancellationException) {
                            Result.failure(t)
                        }
                    }
                }
            }.awaitAll()
        }

        val successes = results.mapNotNull { it.getOrNull() }
        val failures = results.mapNotNull { it.exceptionOrNull() }

        val total = ids.size
        val failed = failures.size
        val ok = successes.size
        val shouldThrow = failed >= ceil(total * quorum).toInt()

        if (!shouldThrow) return@withContext successes.toList()

        // 1. No Network
        if (failures.count { it is UnknownHostException } >= ceil(failed * 0.5).toInt()) {
            throw UnknownHostException()
        }

        // 2. Time out
        if (failures.count { it is SocketTimeoutException } >= ceil(failed * 0.5).toInt()) {
            throw SocketTimeoutException()
        }

        // 3. known error codes: 404/410/204
        val http404ish = failures.count { (it as? HttpException)?.code() in listOf(404, 410, 204) }
        if (ok == 0 && (failed > 0) && (http404ish >= ceil(failed * 0.5).toInt() || failed == total)) {
            throw FileNotFoundException()
        }

        // 4. Unknown issues
        val ioMajority = failures.count {
            it is IOException && it !is UnknownHostException && it !is SocketTimeoutException
        } >= ceil(failed * 0.5).toInt()
        if (ioMajority) {
            throw IOException(failures.first { it is IOException }.message)
        }

        successes
    }

    override suspend fun searchModels(
        query: String?,
        filter: String?,
        sort: String?,
        direction: String?,
        limit: Int?,
        full: Boolean,
        invalidKeywords: Array<String>,
    ) = withContext(Dispatchers.IO) {
        try {
            apiService.getModels(
                search = query,
                filter = filter,
                sort = sort,
                direction = direction,
                limit = limit,
                full = full,
            )
                .filterNot { it.gated || it.private }
                .filterNot {
                    it.getGgufFilename().let { filename ->
                        filename.isNullOrBlank() || invalidKeywords.any {
                            filename.contains(it, ignoreCase = true)
                        }
                    }
                }.let {
                    if (it.isEmpty()) Result.failure(FileNotFoundException())
                    else Result.success(it)
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error searching for models on HuggingFace: ${e.message}")
            Result.failure(e)
        }
    }

    override suspend fun getModelDetails(
        modelId: String
    ) = withContext(Dispatchers.IO) {
        apiService.getModelDetails(modelId)
    }

    override suspend fun getFileSize(
        modelId: String,
        filePath: String
    ): Result<Long> = withContext(Dispatchers.IO) {
        try {
            apiService.getModelFileHeader(modelId, filePath).let { resp ->
                if (resp.isSuccessful) {
                    resp.headers()[HTTP_HEADER_CONTENT_LENGTH]?.toLongOrNull()?.let {
                        Result.success(it)
                    } ?: Result.failure(IOException("Content-Length header missing"))
                } else {
                    Result.failure(
                        when (resp.code()) {
                            401 -> SecurityException("Model requires authentication")
                            404 -> FileNotFoundException("Model file not found")
                            else -> IOException("Failed to get file info: HTTP ${resp.code()}")
                        }
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting file size for $modelId: ${e.message}")
            Result.failure(e)
        }
    }

    override suspend fun downloadModelFile(
        context: Context,
        downloadInfo: HuggingFaceDownloadInfo,
    ): Result<Long> = withContext(Dispatchers.IO) {
        try {
            val downloadManager =
                context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            val request = DownloadManager.Request(downloadInfo.uri).apply {
                setTitle(downloadInfo.filename)
                setDescription("Downloading directly from HuggingFace")
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
                            Result.success(downloadId)
                        }
                    }
                } else {
                    // Assume success if we can't check status
                    cursor.close()
                    Result.success(downloadId)
                }
            } else {
                // Assume success if cursor is empty
                cursor?.close()
                Result.success(downloadId)
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
