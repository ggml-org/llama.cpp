package com.example.llama.revamp.data.repository

import com.example.llama.revamp.data.local.SystemPromptDao
import com.example.llama.revamp.data.local.SystemPromptEntity
import com.example.llama.revamp.data.model.SystemPrompt
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for managing system prompts.
 */
interface SystemPromptRepository {
    fun getPresetPrompts(): Flow<List<SystemPrompt>>
    fun getRecentPrompts(): Flow<List<SystemPrompt>>
    suspend fun savePromptToRecents(prompt: SystemPrompt)
    suspend fun saveCustomPrompt(content: String): SystemPrompt
    suspend fun deletePrompt(id: String)
    suspend fun deleteAllPrompts()
}

@Singleton
class SystemPromptRepositoryImpl @Inject constructor(
    private val systemPromptDao: SystemPromptDao,
) : SystemPromptRepository {

    /**
     * Get all preset prompts.
     */
    override fun getPresetPrompts(): Flow<List<SystemPrompt>> {
        // For now, we'll just return the static list since we don't store presets in the database
        return kotlinx.coroutines.flow.flowOf(SystemPrompt.STUB_PRESETS)
    }

    /**
     * Get recent prompts from the database.
     */
    override fun getRecentPrompts(): Flow<List<SystemPrompt>> {
        return systemPromptDao.getRecentPrompts(MAX_RECENT_PROMPTS)
            .map { entities ->
                entities.map { it.toDomainModel() }
            }
    }

    /**
     * Save a prompt to the recents list.
     * If it's already in recents, just update the timestamp.
     */
    override suspend fun savePromptToRecents(prompt: SystemPrompt) {
        // Check if this prompt already exists
        val existingPrompt = systemPromptDao.getPromptById(prompt.id)

        if (existingPrompt != null) {
            // Update the timestamp to mark it as recently used
            systemPromptDao.updatePromptTimestamp(prompt.id, System.currentTimeMillis())
        } else {
            // Insert as a new prompt
            systemPromptDao.insertPrompt(SystemPromptEntity.fromDomainModel(prompt))

            // Check if we need to trim the list
            pruneOldPrompts()
        }
    }

    /**
     * Create and save a custom prompt.
     */
    override suspend fun saveCustomPrompt(content: String): SystemPrompt {
        val customPrompt = SystemPrompt.Custom(
            id = UUID.randomUUID().toString(),
            content = content
        )

        systemPromptDao.insertPrompt(SystemPromptEntity.fromDomainModel(customPrompt))

        // Check if we need to trim the list
        pruneOldPrompts()

        return customPrompt
    }

    /**
     * Remove prompts if we exceed the maximum count.
     */
    private suspend fun pruneOldPrompts() {
        val count = systemPromptDao.getPromptCount()
        if (count > MAX_RECENT_PROMPTS) {
            // Get all prompts and delete the oldest ones
            val allPrompts = systemPromptDao.getAllPrompts().first()
            val promptsToDelete = allPrompts
                .sortedByDescending { it.timestamp }
                .drop(MAX_RECENT_PROMPTS)

            promptsToDelete.forEach {
                systemPromptDao.deletePrompt(it)
            }
        }
    }

    /**
     * Delete a prompt by ID.
     */
    override suspend fun deletePrompt(id: String) = systemPromptDao.deletePromptById(id)

    /**
     * Delete all prompts.
     */
    override suspend fun deleteAllPrompts() = systemPromptDao.deleteAllPrompts()

    companion object {
        // Maximum number of recent prompts to keep
        private const val MAX_RECENT_PROMPTS = 10
    }
}
