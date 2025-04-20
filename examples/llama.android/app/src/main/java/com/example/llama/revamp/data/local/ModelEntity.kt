package com.example.llama.revamp.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverters
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.util.GgufMetadata
import com.example.llama.revamp.util.GgufMetadataConverters


@Entity(tableName = "models")
data class ModelEntity(
    @PrimaryKey
    val id: String,
    val name: String,
    val path: String,
    val sizeInBytes: Long,
    @field:TypeConverters(GgufMetadataConverters::class)
    val metadata: GgufMetadata,
    val dateAdded: Long,
    val lastUsed: Long?
) {
    fun toModelInfo() = ModelInfo(
        id = id,
        name = name,
        path = path,
        sizeInBytes = sizeInBytes,
        metadata = metadata,
        lastUsed = lastUsed,
    )
}
