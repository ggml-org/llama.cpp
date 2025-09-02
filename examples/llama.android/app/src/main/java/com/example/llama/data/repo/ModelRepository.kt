package com.example.llama.data.repo

import android.content.Context
import android.llama.cpp.gguf.GgufMetadataReader
import android.llama.cpp.gguf.InvalidFileFormatException
import android.net.Uri
import android.os.StatFs
import android.util.Log
import com.example.llama.data.db.dao.ModelDao
import com.example.llama.data.db.entity.ModelEntity
import com.example.llama.data.model.GgufMetadata
import com.example.llama.data.model.ModelInfo
import com.example.llama.data.repo.ModelRepository.ImportProgressTracker
import com.example.llama.data.source.local.LocalFileDataSource
import com.example.llama.data.source.remote.HuggingFaceDownloadInfo
import com.example.llama.data.source.remote.HuggingFaceModel
import com.example.llama.data.source.remote.HuggingFaceModelDetails
import com.example.llama.data.source.remote.HuggingFaceRemoteDataSource
import com.example.llama.monitoring.StorageMetrics
import com.example.llama.util.formatFileByteSize
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for managing available models on local device.
 */
interface ModelRepository {
    /**
     * Obtain the current status of local storage and available models.
     */
    fun getStorageMetrics(): Flow<StorageMetrics>
    fun getModels(): Flow<List<ModelInfo>>
    suspend fun getModelById(id: String): ModelInfo?

    /**
     * Import a local model file from device storage.
     */
    suspend fun importModel(
        uri: Uri,
        name: String? = null,
        size: Long? = null,
        progressTracker: ImportProgressTracker? = null
    ): ModelInfo

    fun interface ImportProgressTracker {
        fun onProgress(progress: Float) // 0.0f to 1.0f
    }

    /**
     * Cancels any ongoing local model import operation.
     *
     * @return null if no import is in progress,
     *         true if successfully canceled,
     *         false if cancellation failed
     */
    suspend fun cancelImport(): Boolean?

    /**
     * Update a model's last used timestamp.
     */
    suspend fun updateModelLastUsed(modelId: String)

    /**
     * Delete a model or in batches
     */
    suspend fun deleteModel(modelId: String)
    suspend fun deleteModels(modelIds: List<String>)

    /**
     * Fetch details of preselected models
     */
    suspend fun fetchPreselectedHuggingFaceModels(): List<HuggingFaceModelDetails>

    /**
     * Search models on HuggingFace
     */
    suspend fun searchHuggingFaceModels(limit: Int): Result<List<HuggingFaceModel>>

    /**
     * Obtain the model details from HuggingFace
     */
    suspend fun getHuggingFaceModelDetails(modelId: String): HuggingFaceModelDetails

    /**
     * Obtain the model's size from HTTP response header
     */
    suspend fun getHuggingFaceModelFileSize(downloadInfo: HuggingFaceDownloadInfo): Result<Long>

    /**
     * Download a HuggingFace model via system download manager
     */
    suspend fun downloadHuggingFaceModel(
        downloadInfo: HuggingFaceDownloadInfo,
        actualSize: Long,
    ): Result<Long>
}

class InsufficientStorageException(message: String) : IOException(message)

@Singleton
class ModelRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelDao: ModelDao,
    private val localFileDataSource: LocalFileDataSource,
    private val huggingFaceRemoteDataSource: HuggingFaceRemoteDataSource,
    private val ggufMetadataReader: GgufMetadataReader,
) : ModelRepository {

    private val modelsDir = File(context.filesDir, INTERNAL_STORAGE_PATH)

    init {
        if (!modelsDir.exists()) { modelsDir.mkdirs() }
    }

    val modelsSizeBytes: Long
        get() = modelsDir.listFiles()?.fold(0L) { totalSize, file ->
            totalSize + if (file.isFile) file.length() else 0
        } ?: 0L

    val availableSpaceBytes: Long
        get() = StatFs(context.filesDir.path).availableBytes

    val totalSpaceBytes: Long
        get() = StatFs(context.filesDir.path).totalBytes

    override fun getStorageMetrics(): Flow<StorageMetrics> = flow {
        while (true) {
            emit(
                StorageMetrics(
                    usedGB = modelsSizeBytes / BYTES_IN_GB,
                    availableGB = availableSpaceBytes / BYTES_IN_GB
                )
            )
            delay(STORAGE_METRICS_UPDATE_INTERVAL)
        }
    }

    override fun getModels(): Flow<List<ModelInfo>> =
        modelDao.getAllModels()
            .map { entities ->
                entities.filter {
                    val file = File(it.path)
                    file.exists() && file.isFile
                }.map {
                    it.toModelInfo()
                }
            }

    override suspend fun getModelById(id: String) =
        modelDao.getModelById(id)?.toModelInfo()


    private var importJob: Job? = null
    private var currentModelFile: File? = null

    override suspend fun importModel(
        uri: Uri,
        name: String?,
        size: Long?,
        progressTracker: ImportProgressTracker?
    ): ModelInfo = withContext(Dispatchers.IO) {
        if (importJob != null && importJob?.isActive == true) {
            throw IllegalStateException("Another import is already in progress!")
        }

        // Check file info
        val fileInfo = localFileDataSource.getFileInfo(uri)
        val fileSize = size ?: fileInfo?.size ?: throw FileNotFoundException("File size N/A")
        val fileName = name ?: fileInfo?.name ?: throw FileNotFoundException("File name N/A")
        if (!ggufMetadataReader.ensureSourceFileFormat(context, uri)) {
            throw InvalidFileFormatException()
        }

        // Check for enough storage
        if (!hasEnoughSpaceForImport(fileSize)) {
            throw InsufficientStorageException(
                "Not enough storage space! " +
                    "Required: ${formatFileByteSize(fileSize)}, " +
                    "Available: ${formatFileByteSize(availableSpaceBytes)}"
            )
        }
        val modelFile = File(modelsDir, fileName)
        importJob = coroutineContext[Job]
        currentModelFile = modelFile

        try {
            localFileDataSource.copyFile(
                sourceUri = uri,
                destinationFile = modelFile,
                fileSize = fileSize,
                onProgress = { progress ->
                    progressTracker?.let {
                        withContext(Dispatchers.Main) {
                            it.onProgress(progress)
                        }
                    }
                }
            ).getOrThrow()

            // Extract GGUF metadata if possible
            val metadata = try {
                Log.i(TAG, "Extracting GGUF Metadata from ${modelFile.absolutePath}")
                modelFile.inputStream().buffered().use {
                    GgufMetadata.fromDomain(ggufMetadataReader.readStructuredMetadata(it))
                }
            } catch (e: Exception) {
                Log.e(TAG, "Cannot extract GGUF metadata: ${e.message}", e)
                throw e
            }

            // Create model entity and save via DAO
            ModelEntity(
                id = UUID.randomUUID().toString(),
                name = fileName.substringBeforeLast('.'),
                path = modelFile.absolutePath,
                sizeInBytes = modelFile.length(),
                metadata = metadata,
                dateAdded = System.currentTimeMillis(),
                dateLastUsed = null
            ).let {
                modelDao.insertModel(it)
                it.toModelInfo()
            }

        } catch (e: CancellationException) {
            Log.i(TAG, "Import was cancelled for $fileName: ${e.message}")
            localFileDataSource.cleanupPartialFile(modelFile)
            throw e

        } catch (e: Exception) {
            Log.e(TAG, "Import failed for $fileName: ${e.message}")
            localFileDataSource.cleanupPartialFile(modelFile)
            throw e

        } finally {
            importJob = null
            currentModelFile = null
        }
    }

    // Add this method to ModelRepositoryImpl.kt
    private fun hasEnoughSpaceForImport(fileSize: Long): Boolean {
        val availableSpace = availableSpaceBytes
        val requiredSpace = (fileSize * MODEL_IMPORT_SPACE_BUFFER_SCALE).toLong()
        return availableSpace >= requiredSpace
    }

    override suspend fun cancelImport(): Boolean? = withContext(Dispatchers.IO) {
        val job = importJob
        val file = currentModelFile

        return@withContext when {
            // No import in progress
            job == null -> null

            // Job already completed or cancelled
            !job.isActive -> {
                importJob = null
                currentModelFile = null
                null
            }

            // Job in progress
            else -> try {
                // Attempt to cancel the job
                job.cancel("Import cancelled by user")

                // Give the job a moment to clean up
                delay(CANCEL_LOCAL_MODEL_IMPORT_TIMEOUT)

                // Clean up the partial file
                file?.let { localFileDataSource.cleanupPartialFile(it) }

                // Reset state
                importJob = null
                currentModelFile = null

                true // Successfully cancelled
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cancel import: ${e.message}")
                false
            }
        }
    }

    override suspend fun updateModelLastUsed(modelId: String) = withContext(Dispatchers.IO) {
        modelDao.updateLastUsed(modelId, System.currentTimeMillis())
    }

    override suspend fun deleteModel(modelId: String) = withContext(Dispatchers.IO) {
        modelDao.getModelById(modelId)?.let { model ->
            File(model.path).let {
                if (it.exists()) { it.delete() }
            }
            modelDao.deleteModel(model)
        } ?: Unit
    }

    override suspend fun deleteModels(modelIds: List<String>) = withContext(Dispatchers.IO) {
        modelDao.getModelsByIds(modelIds).let { models ->
            models.forEach { model ->
                File(model.path).let {
                    if (it.exists()) { it.delete() }
                }
            }
            modelDao.deleteModels(models)
        }
    }

    override suspend fun fetchPreselectedHuggingFaceModels() = withContext(Dispatchers.IO) {
        huggingFaceRemoteDataSource.fetchPreselectedModels()
    }

    override suspend fun searchHuggingFaceModels(
        limit: Int
    ) = withContext(Dispatchers.IO) {
        huggingFaceRemoteDataSource.searchModels(limit = limit)
    }

    override suspend fun getHuggingFaceModelDetails(
        modelId: String
    ) = withContext(Dispatchers.IO) {
        huggingFaceRemoteDataSource.getModelDetails(modelId)
    }

    override suspend fun getHuggingFaceModelFileSize(
        downloadInfo: HuggingFaceDownloadInfo,
    ): Result<Long> = withContext(Dispatchers.IO) {
        huggingFaceRemoteDataSource.getFileSize(downloadInfo.modelId, downloadInfo.filename)
    }

    override suspend fun downloadHuggingFaceModel(
        downloadInfo: HuggingFaceDownloadInfo,
        actualSize: Long,
    ): Result<Long> = withContext(Dispatchers.IO) {
        if (!hasEnoughSpaceForImport(actualSize)) {
            throw InsufficientStorageException(
                "Not enough storage space! " +
                    "Estimated required: ${formatFileByteSize(actualSize)}, " +
                    "Available: ${formatFileByteSize(availableSpaceBytes)}"
            )
        }

        try {
            huggingFaceRemoteDataSource.downloadModelFile(
                context = context,
                downloadInfo = downloadInfo,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Import failed: ${e.message}")
            Result.failure(e)
        }
    }

    companion object {
        private val TAG = ModelRepository::class.java.simpleName

        private const val INTERNAL_STORAGE_PATH = "models"

        private const val STORAGE_METRICS_UPDATE_INTERVAL = 10_000L
        private const val BYTES_IN_GB = 1024f * 1024f * 1024f

        private const val MODEL_IMPORT_SPACE_BUFFER_SCALE = 1.2f
        private const val CANCEL_LOCAL_MODEL_IMPORT_TIMEOUT = 500L
    }
}

