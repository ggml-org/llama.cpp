package com.example.llama.revamp.data.repository

import android.content.Context
import android.net.Uri
import android.os.StatFs
import android.provider.OpenableColumns
import com.example.llama.revamp.data.local.ModelDao
import com.example.llama.revamp.data.local.ModelEntity
import com.example.llama.revamp.data.model.ModelInfo
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
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

    suspend fun importModel(uri: Uri): ModelInfo
    suspend fun deleteModel(modelId: String)
    suspend fun deleteModels(modelIds: Collection<String>)
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

    override suspend fun importModel(uri: Uri): ModelInfo {
        // Obtain the local model's file via provided URI
        val fileName = getFileNameFromUri(uri)
        val modelFile = File(modelsDir, fileName)

        // Copy file to app's internal storage
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            FileOutputStream(modelFile).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        } ?: throw IOException("Failed to open input stream")

        // Extract model parameters from filename
        val modelType = extractModelTypeFromFilename(fileName) ?: "unknown"
        val parameters = extractParametersFromFilename(fileName) ?: "unknown"
        val quantization = extractQuantizationFromFilename(fileName) ?: "unknown"

        // Create model entity and save via DAO
        val modelEntity = ModelEntity(
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
        )
        modelDao.insertModel(modelEntity)
        return modelEntity.toModelInfo()
    }

    override suspend fun deleteModel(modelId: String) {
        modelDao.getModelById(modelId)?.let { model ->
            File(model.path).let {
                if (it.exists()) { it.delete() }
            }
            modelDao.deleteModel(model)
        }
    }

    override suspend fun deleteModels(modelIds: Collection<String>) {
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

    private fun getFileNameFromUri(uri: Uri): String =
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME).let { nameIndex ->
                    if (nameIndex != -1) cursor.getString(nameIndex) else null
                }
            } else {
                null
            }
        } ?: uri.lastPathSegment ?: "unknown_model.gguf"

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
        private const val INTERNAL_STORAGE_PATH = "models"
        private const val BYTES_IN_GB = 1024f * 1024f * 1024f

        private const val STORAGE_METRICS_UPDATE_INTERVAL = 5_000L
        private const val DEFAULT_CONTEXT_SIZE = 8192

    }
}

data class StorageMetrics(
    val usedGB: Float,
    val totalGB: Float
)
