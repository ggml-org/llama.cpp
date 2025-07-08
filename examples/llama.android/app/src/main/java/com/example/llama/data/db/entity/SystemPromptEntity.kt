package com.example.llama.data.db.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.llama.data.model.SystemPrompt

/**
 * Database entity for storing system prompts.
 */
@Entity(tableName = "system_prompts")
data class SystemPromptEntity(
    @PrimaryKey
    val id: String,
    val content: String,
    val name: String?,
    val timestamp: Long,
    val isPreset: Boolean
) {
    /**
     * Convert to domain model.
     */
    fun toDomainModel(): SystemPrompt {
        return if (isPreset) {
            SystemPrompt.Preset(
                id = id,
                content = content,
                name = name ?: "Unnamed Preset",
                timestamp = timestamp
            )
        } else {
            SystemPrompt.Custom(
                id = id,
                content = content,
                timestamp = timestamp
            )
        }
    }

    companion object {
        /**
         * Create an entity from a domain model.
         */
        fun fromDomainModel(prompt: SystemPrompt): SystemPromptEntity {
            return when (prompt) {
                is SystemPrompt.Preset -> SystemPromptEntity(
                    id = prompt.id,
                    content = prompt.content,
                    name = prompt.name,
                    timestamp = prompt.timestamp ?: System.currentTimeMillis(),
                    isPreset = true
                )
                is SystemPrompt.Custom -> SystemPromptEntity(
                    id = prompt.id,
                    content = prompt.content,
                    name = null,
                    timestamp = prompt.timestamp,
                    isPreset = false
                )
            }
        }
    }
}
