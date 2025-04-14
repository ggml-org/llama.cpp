package com.example.llama.revamp.data.repository

import android.content.Context
import android.net.Uri
import com.example.llama.revamp.data.model.ModelInfo
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

interface ModelRepository {
    suspend fun getModels(): List<ModelInfo>
    suspend fun deleteModel(modelId: String)
    suspend fun deleteModels(modelIds: Collection<String>)
    suspend fun importModel(uri: Uri): ModelInfo
    suspend fun getStorageMetrics(): StorageMetrics
}

@Singleton
class ModelRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    // TODO-han.yin: Add model DAO
) : ModelRepository {

    override suspend fun getModels(): List<ModelInfo> {
        // In a real implementation, this would load from database
        return ModelInfo.getSampleModels()
    }

    override suspend fun deleteModel(modelId: String) {
        // Stub - would delete from filesystem and database
    }

    override suspend fun deleteModels(modelIds: Collection<String>) {
        // Stub - would delete from filesystem and database
    }

    override suspend fun importModel(uri: Uri): ModelInfo {
        // Stub - would copy file and extract metadata
        return ModelInfo.getSampleModels().first()
    }

    override suspend fun getStorageMetrics(): StorageMetrics {
        // Stub - would calculate from actual storage
        return StorageMetrics(14.6f, 32.0f)
    }
}

data class StorageMetrics(
    val usedGB: Float,
    val totalGB: Float
)
