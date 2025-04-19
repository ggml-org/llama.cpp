package com.example.llama.revamp.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.llama.revamp.data.model.ModelInfo

// TODO-han.yin: Add GgufMetaData

@Entity(tableName = "models")
data class ModelEntity(
    @PrimaryKey
    val id: String,
    val name: String,
    val path: String,
    val sizeInBytes: Long,
    val dateAdded: Long,
    val lastUsed: Long?
) {
    fun toModelInfo() = ModelInfo(
        id = id,
        name = name,
        path = path,
        sizeInBytes = sizeInBytes,
        lastUsed = lastUsed,
    )
}
