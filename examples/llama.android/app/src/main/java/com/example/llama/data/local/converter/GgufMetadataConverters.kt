package com.example.llama.data.local.converter

import androidx.room.TypeConverter
import com.example.llama.util.GgufMetadata
import kotlinx.serialization.json.Json

class GgufMetadataConverters {
    private val json = Json { encodeDefaults = false; ignoreUnknownKeys = true }

    @TypeConverter
    fun toJson(value: GgufMetadata?): String? =
        value?.let { json.encodeToString(GgufMetadata.serializer(), it) }

    @TypeConverter
    fun fromJson(value: String?): GgufMetadata? =
        value?.let { json.decodeFromString(GgufMetadata.serializer(), it) }
}
