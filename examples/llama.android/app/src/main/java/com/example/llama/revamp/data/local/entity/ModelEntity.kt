package com.example.llama.revamp.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverters
import com.example.llama.revamp.data.local.converter.GgufMetadataConverters
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.util.GgufMetadata


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
    val dateLastUsed: Long?
) {
    fun toModelInfo() = ModelInfo(
        id = id,
        name = name,
        path = path,
        sizeInBytes = sizeInBytes,
        metadata = metadata,
        dateAdded = dateAdded,
        dateLastUsed = this@ModelEntity.dateLastUsed,
    )
}
