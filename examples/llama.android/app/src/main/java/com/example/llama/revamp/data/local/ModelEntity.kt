package com.example.llama.revamp.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.llama.revamp.data.model.ModelInfo

@Entity(tableName = "models")
data class ModelEntity(
    @PrimaryKey
    val id: String,
    val name: String,
    val path: String,
    val sizeInBytes: Long,
    val parameters: String?,
    val quantization: String?,
    val type: String?,
    val contextLength: Int?,
    val lastUsed: Long?,
    val dateAdded: Long
) {
    fun toModelInfo() = ModelInfo(
        id = id,
        name = name,
        path = path,
        sizeInBytes = sizeInBytes,
        parameters = parameters,
        quantization = quantization,
        type = type,
        contextLength = contextLength,
        lastUsed = lastUsed
    )
}
