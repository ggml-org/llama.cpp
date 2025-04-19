package com.example.llama.revamp.data.repository

import android.content.Context
import android.net.Uri
import android.os.StatFs
import android.util.Log
import com.example.llama.revamp.data.local.ModelDao
import com.example.llama.revamp.data.local.ModelEntity
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository.ImportProgressTracker
import com.example.llama.revamp.util.GgufMetadataReader
import com.example.llama.revamp.util.copyWithBuffer
import com.example.llama.revamp.util.copyWithChannels
import com.example.llama.revamp.util.formatSize
import com.example.llama.revamp.util.getFileNameFromUri
import com.example.llama.revamp.util.getFileSizeFromUri
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
import java.io.FileOutputStream
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
}

class InsufficientStorageException(message: String) : IOException(message)

@Singleton
class ModelRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelDao: ModelDao,
) : ModelRepository {

    private val modelsDir = File(context.filesDir, INTERNAL_STORAGE_PATH)

    init {
        if (!modelsDir.exists()) { modelsDir.mkdirs() }
    }

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

        val fileSize = size ?: getFileSizeFromUri(context, uri) ?: throw FileNotFoundException("File size N/A")
        if (!hasEnoughSpaceForImport(fileSize)) {
            throw InsufficientStorageException(
                "Not enough storage space. Required: ${formatSize(fileSize)}, Available: ${formatSize(availableSpaceBytes)}"
            )
        }

        val fileName = name ?: getFileNameFromUri(context, uri) ?: throw FileNotFoundException("Filename N/A")
        val modelFile = File(modelsDir, fileName)

        importJob = coroutineContext[Job]
        currentModelFile = modelFile

        try {
            val inputStream = context.contentResolver.openInputStream(uri)
                ?: throw IOException("Failed to open input stream")
            val outputStream = FileOutputStream(modelFile)

            if (fileSize > LARGE_MODEL_THRESHOLD_SIZE) {
                Log.i(TAG, "Copying $fileName (size: $fileSize) via NIO...")

                // Use NIO channels for large models
                copyWithChannels(
                    input = inputStream,
                    output = outputStream,
                    totalSize = fileSize,
                    bufferSize = NIO_BUFFER_SIZE,
                    yieldSize = NIO_YIELD_SIZE
                ) { progress ->
                    progressTracker?.let {
                        withContext(Dispatchers.Main) {
                            it.onProgress(progress)
                        }
                    }
                }
            } else {
                Log.i(TAG, "Copying $fileName (size: $fileSize) via buffer...")

                // Default copy with buffer for small models
                copyWithBuffer(
                    input = inputStream,
                    output = outputStream,
                    totalSize = fileSize,
                    bufferSize = DEFAULT_BUFFER_SIZE,
                    yieldSize = DEFAULT_YIELD_SIZE
                ) { progress ->
                    progressTracker?.let {
                        withContext(Dispatchers.Main) {
                            it.onProgress(progress)
                        }
                    }
                }
            }

            // Extract GGUF metadata if possible
            val metadata = try {
                val filePath = modelFile.absolutePath
                Log.i(TAG, "Extracting GGUF Metadata from $filePath")
                GgufMetadataReader().readStructuredMetadata(filePath)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to extract GGUF metadata: ${e.message}", e)
                null
            }

            // Create model entity and save via DAO
            ModelEntity(
                id = UUID.randomUUID().toString(),
                name = fileName.substringBeforeLast('.'),
                path = modelFile.absolutePath,
                sizeInBytes = modelFile.length(),
                // TODO-han.yin: add metadata here
                dateAdded = System.currentTimeMillis(),
                lastUsed = null
            ).let {
                modelDao.insertModel(it)

                importJob = null
                currentModelFile = null

                it.toModelInfo()
            }

        } catch (e: CancellationException) {
            Log.i(TAG, "Import was cancelled for $fileName: ${e.message}")
            cleanupPartialFile(modelFile)
            throw e

        } catch (e: Exception) {
            Log.e(TAG, "Import failed for $fileName: ${e.message}")
            cleanupPartialFile(modelFile)
            throw e

        } finally {
            importJob = null
            currentModelFile = null
        }
    }

    // Add this method to ModelRepositoryImpl.kt
    private fun hasEnoughSpaceForImport(fileSize: Long): Boolean {
        val availableSpace = availableSpaceBytes
        val requiredSpace = (fileSize * MODEL_IMPORT_SPACE_BUFFER_SCALE ).toLong()
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

                // Clean up the partial file (as a safety measure)
                cleanupPartialFile(file)

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

    private fun cleanupPartialFile(file: File?) {
        try {
            if (file?.exists() == true && !file.delete()) {
                Log.e(TAG, "Failed to delete partial file: ${file.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up partial file: ${e.message}")
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

    val modelsSizeBytes: Long
        get() = modelsDir.listFiles()?.fold(0L) { totalSize, file ->
            totalSize + if (file.isFile) file.length() else 0
        } ?: 0L

    val availableSpaceBytes: Long
        get() = StatFs(context.filesDir.path).availableBytes

    val totalSpaceBytes: Long
        get() = StatFs(context.filesDir.path).totalBytes

    companion object {
        private val TAG = ModelRepository::class.java.simpleName

        private const val INTERNAL_STORAGE_PATH = "models"

        private const val STORAGE_METRICS_UPDATE_INTERVAL = 10_000L
        private const val BYTES_IN_GB = 1024f * 1024f * 1024f

        private const val MODEL_IMPORT_SPACE_BUFFER_SCALE = 1.2f
        private const val LARGE_MODEL_THRESHOLD_SIZE = 1024 * 1024 * 1024
        private const val NIO_BUFFER_SIZE = 32 * 1024 * 1024
        private const val NIO_YIELD_SIZE = 128 * 1024 * 1024
        private const val DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
        private const val DEFAULT_YIELD_SIZE = 16 * 1024 * 1024
        private const val CANCEL_LOCAL_MODEL_IMPORT_TIMEOUT = 500L

        private const val DEFAULT_CONTEXT_SIZE = 8192

    }
}

data class StorageMetrics(
    val usedGB: Float,
    val availableGB: Float
)
