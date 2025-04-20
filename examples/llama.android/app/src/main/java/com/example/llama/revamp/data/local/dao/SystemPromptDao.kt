package com.example.llama.revamp.data.local.dao

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.llama.revamp.data.local.entity.SystemPromptEntity
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for System Prompts.
 */
@Dao
interface SystemPromptDao {

    @Query("SELECT * FROM system_prompts ORDER BY timestamp DESC")
    fun getAllPrompts(): Flow<List<SystemPromptEntity>>

    @Query("SELECT * FROM system_prompts WHERE id = :id")
    suspend fun getPromptById(id: String): SystemPromptEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPrompt(prompt: SystemPromptEntity)

    @Delete
    suspend fun deletePrompt(prompt: SystemPromptEntity)

    @Query("DELETE FROM system_prompts WHERE id = :id")
    suspend fun deletePromptById(id: String)

    @Query("DELETE FROM system_prompts")
    suspend fun deleteAllPrompts()

    @Query("SELECT COUNT(*) FROM system_prompts")
    suspend fun getPromptCount(): Int

    // Get the most recent prompts, limited by count
    @Query("SELECT * FROM system_prompts ORDER BY timestamp DESC LIMIT :count")
    fun getRecentPrompts(count: Int): Flow<List<SystemPromptEntity>>

    // Update the timestamp of an existing prompt to make it the most recent
    @Query("UPDATE system_prompts SET timestamp = :timestamp WHERE id = :id")
    suspend fun updatePromptTimestamp(id: String, timestamp: Long)
}
