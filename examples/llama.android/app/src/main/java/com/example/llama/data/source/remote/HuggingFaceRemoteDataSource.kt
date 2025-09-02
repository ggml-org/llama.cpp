package com.example.llama.data.source.remote

import android.app.DownloadManager
import android.content.Context
import android.os.Environment
import android.util.Log
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
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.ceil

private const val QUERY_Q4_0_GGUF = "gguf q4_0"
private const val FILTER_TEXT_GENERATION = "text-generation"
private const val SORT_BY_DOWNLOADS = "downloads"
private const val SEARCH_RESULT_LIMIT = 30

private val INVALID_KEYWORDS = arrayOf("-of-", "split", "70B", "30B", "27B", "14B", "13B", "12B")

private val PRESELECTED_MODEL_IDS = listOf(
    "unsloth/gemma-3-1b-it-GGUF",
    "unsloth/gemma-3-4b-it-GGUF",
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "gaianet/Phi-4-mini-instruct-GGUF",
    "bartowski/granite-3.0-2b-instruct-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
)

interface HuggingFaceRemoteDataSource {

    suspend fun fetchPreselectedModels(
        ids: List<String> = PRESELECTED_MODEL_IDS,
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

@Singleton
class HuggingFaceRemoteDataSourceImpl @Inject constructor(
    private val apiService: HuggingFaceApiService
) : HuggingFaceRemoteDataSource {

    override suspend fun fetchPreselectedModels(
        ids: List<String>,
        parallelCount: Int,
        quorum: Float,
    ): List<HuggingFaceModelDetails> = withContext(Dispatchers.IO) {
        val successes = mutableListOf<HuggingFaceModelDetails>()
        val failures = mutableListOf<Throwable>()

        val sem = Semaphore(parallelCount)
        supervisorScope {
            ids.map { id ->
                async {
                    sem.withPermit {
                        runCatching { getModelDetails(id) }
                            .onSuccess { synchronized(successes) { successes += it } }
                            .onFailure { t ->
                                if (t is CancellationException) throw t
                                synchronized(failures) { failures += t }
                            }
                    }
                }
            }.awaitAll()
        }

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
                        filename.isNullOrBlank() || INVALID_KEYWORDS.any {
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
