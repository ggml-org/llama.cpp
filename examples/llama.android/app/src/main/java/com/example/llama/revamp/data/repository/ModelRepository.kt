package com.example.llama.revamp.data.repository

import android.content.Context
import android.net.Uri
import android.os.StatFs
import android.provider.OpenableColumns
import android.util.Log
import com.example.llama.revamp.data.local.ModelDao
import com.example.llama.revamp.data.local.ModelEntity
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository.ImportProgressTracker
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.channels.ReadableByteChannel
import java.nio.channels.WritableByteChannel
import java.util.Locale
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for managing available models on local device.
 */
interface ModelRepository {
    fun getStorageMetrics(): Flow<StorageMetrics>
    fun getModels(): Flow<List<ModelInfo>>

    suspend fun importModel(uri: Uri, progressTracker: ImportProgressTracker? = null): ModelInfo

    suspend fun deleteModel(modelId: String)
    suspend fun deleteModels(modelIds: List<String>)

    fun interface ImportProgressTracker {
        fun onProgress(progress: Float) // 0.0f to 1.0f
    }
}

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
                    totalGB = availableSpaceBytes / BYTES_IN_GB
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

    override suspend fun importModel(
        uri: Uri,
        progressTracker: ImportProgressTracker?
    ): ModelInfo = withContext(Dispatchers.IO) {
        val fileName = getFileNameFromUri(uri) ?: throw FileNotFoundException("Filename N/A")
        val fileSize = getFileSizeFromUri(uri) ?: throw FileNotFoundException("File size N/A")
        val modelFile = File(modelsDir, fileName)

        try {
            val inputStream = context.contentResolver.openInputStream(uri)
                ?: throw IOException("Failed to open input stream")
            val outputStream = FileOutputStream(modelFile)

            if (fileSize > LARGE_MODEL_THRESHOLD_SIZE) {
                Log.i(TAG, "Copying $fileName (size: $fileSize) via NIO...")

                // Use NIO channels for large models
                copyWithChannels(inputStream, outputStream, fileSize, progressTracker)
            } else {
                Log.i(TAG, "Copying $fileName (size: $fileSize) via buffer...")

                // Default copy with buffer for small models
                val bufferedInput = BufferedInputStream(inputStream, DEFAULT_BUFFER_SIZE)
                val bufferedOutput = BufferedOutputStream(outputStream, DEFAULT_BUFFER_SIZE)
                copyWithBuffer(bufferedInput, bufferedOutput, fileSize, progressTracker)

                // Close streams
                bufferedOutput.flush()
                bufferedOutput.close()
                bufferedInput.close()
            }

            // Extract model parameters from filename
            val modelType = extractModelTypeFromFilename(fileName)
            val parameters = extractParametersFromFilename(fileName)
            val quantization = extractQuantizationFromFilename(fileName)

            // Create model entity and save via DAO
            ModelEntity(
                id = UUID.randomUUID().toString(),
                name = fileName.substringBeforeLast('.'),
                path = modelFile.absolutePath,
                sizeInBytes = modelFile.length(),
                parameters = parameters,
                quantization = quantization,
                type = modelType,
                contextLength = DEFAULT_CONTEXT_SIZE,
                lastUsed = null,
                dateAdded = System.currentTimeMillis()
            ).let {
                modelDao.insertModel(it)
                it.toModelInfo()
            }
        } catch (e: Exception) {
            // Clean up partially downloaded file if error occurs
            if (modelFile.exists()) {
                modelFile.delete()
            }
            throw e
        }
    }

    private suspend fun copyWithChannels(
        input: InputStream,
        output: OutputStream,
        totalSize: Long,
        progressTracker: ImportProgressTracker?
    ) {
        val inChannel: ReadableByteChannel = Channels.newChannel(input)
        val outChannel: WritableByteChannel = Channels.newChannel(output)

        val buffer = ByteBuffer.allocateDirect(NIO_BUFFER_SIZE)
        var totalBytesRead = 0L

        while (inChannel.read(buffer) != -1) {
            buffer.flip()
            while (buffer.hasRemaining()) {
                outChannel.write(buffer)
            }
            totalBytesRead += buffer.position()
            buffer.clear()

            // Report progress
            progressTracker?.let {
                val progress = totalBytesRead.toFloat() / totalSize
                withContext(Dispatchers.Main) {
                    it.onProgress(progress)
                }
            }

            if (totalBytesRead % (NIO_YIELD_SIZE) == 0L) {
                yield()
            }
        }

        inChannel.close()
        outChannel.close()
        output.close()
        input.close()
    }

    private suspend fun copyWithBuffer(
        input: BufferedInputStream,
        output: BufferedOutputStream,
        totalSize: Long,
        progressTracker: ImportProgressTracker?
    ) {
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)

        var bytesRead: Int
        var totalBytesRead = 0L

        while (input.read(buffer).also { bytesRead = it } != -1) {
            output.write(buffer, 0, bytesRead)
            totalBytesRead += bytesRead

            // Report progress
            if (progressTracker != null) {
                val progress = totalBytesRead.toFloat() / totalSize
                withContext(Dispatchers.Main) {
                    progressTracker.onProgress(progress)
                }
            }

            // Yield less frequently with larger buffers
            if (totalBytesRead % (DEFAULT_YIELD_SIZE) == 0L) { // Every 64MB
                yield()
            }
        }

        output.close()
        input.close()
    }

    override suspend fun deleteModel(modelId: String) {
        modelDao.getModelById(modelId)?.let { model ->
            File(model.path).let {
                if (it.exists()) { it.delete() }
            }
            modelDao.deleteModel(model)
        }
    }

    override suspend fun deleteModels(modelIds: List<String>) {
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

    private fun getFileNameFromUri(uri: Uri): String? =
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME).let { nameIndex ->
                    if (nameIndex != -1) cursor.getString(nameIndex) else null
                }
            } else {
                null
            }
        } ?: uri.lastPathSegment

    /**
     * Gets the file size from a content URI, or returns 0 if size is unknown.
     */
    private fun getFileSizeFromUri(uri: Uri): Long? =
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                cursor.getColumnIndex(OpenableColumns.SIZE).let { sizeIndex ->
                    if (sizeIndex != -1) cursor.getLong(sizeIndex) else null
                }
            } else {
                null
            }
        }

    /**
     * Try to extract parameters by looking for patterns like 7B, 13B, etc.
     *
     * TODO-han.yin: Enhance and move into a utility object for unit testing
     */
    private fun extractParametersFromFilename(filename: String): String? =
        Regex("([0-9]+(\\.[0-9]+)?)[bB]").find(filename)?.value?.uppercase()

    /**
     * Try to extract quantization by looking for patterns like Q4_0, Q5_K_M, etc.
     */
    private fun extractQuantizationFromFilename(filename: String) =
        listOf(
            Regex("[qQ][0-9]_[0-9]"),
            Regex("[qQ][0-9]_[kK]_[mM]"),
            Regex("[qQ][0-9]_[kK]"),
            Regex("[qQ][0-9][fF](16|32)")
        ).firstNotNullOfOrNull {
            it.find(filename)?.value?.uppercase()
        }

    /**
     * Try to extract model type (Llama, Mistral, etc.)
     *
     * TODO-han.yin: Replace with GGUF parsing, also to be moved into the util object
     */
    private fun extractModelTypeFromFilename(filename: String): String? {
        val lowerFilename = filename.lowercase()
        return listOf("llama", "mistral", "phi", "qwen", "falcon", "mpt")
            .firstNotNullOfOrNull { type ->
                if (lowerFilename.contains(type)) {
                    type.replaceFirstChar {
                        if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString()
                    }
                } else { null }
            }
    }

    companion object {
        private val TAG = ModelRepository::class.java.simpleName

        private const val INTERNAL_STORAGE_PATH = "models"

        private const val BYTES_IN_GB = 1024f * 1024f * 1024f

        private const val LARGE_MODEL_THRESHOLD_SIZE = 1024 * 1024 * 1024
        private const val NIO_BUFFER_SIZE = 32 * 1024 * 1024
        private const val NIO_YIELD_SIZE = 128 * 1024 * 1024
        private const val DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
        private const val DEFAULT_YIELD_SIZE = 16 * 1024 * 1024

        private const val STORAGE_METRICS_UPDATE_INTERVAL = 5_000L
        private const val DEFAULT_CONTEXT_SIZE = 8192

    }
}

data class StorageMetrics(
    val usedGB: Float,
    val totalGB: Float
)
